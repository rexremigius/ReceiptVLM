"""Framework-free core shared by the FastAPI backend and the Gradio Space.

serve.py grew this logic inline, which was fine while Streamlit-over-HTTP was the only
front end. A Hugging Face Space runs a single process, so there is no uvicorn to call and
the Gradio app needs the same confidence badges, spend aggregation and category rollups
as plain function calls. Rather than fork them -- the calibrated confidence score is the
project's headline result, so two drifting copies would be a genuine hazard -- they live
here and serve.py imports them.

Nothing in this module imports torch, mlx, fastapi or gradio, so it stays cheap to import
from either stack. The actual model lives in backend_hf.py (transformers/CUDA) or behind
mlx_vlm in serve.py.

One deliberate difference from serve.py: the uploaded-receipt list is not a module global
here. serve.py kept LIVE_RECEIPTS at module scope, which is correct for a single-user
localhost demo but wrong on a public Space -- every visitor would pool into one dashboard
and see each other's receipts. Callers own that list and pass it in, so the Gradio app can
keep it in per-session state.
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

# confidence.py and categorize.py import eval by bare name (run-as-script style), so src/
# has to be importable in its own right, not just as the src package. Done here rather
# than in each caller so this module is self-sufficient however it is imported.
_SRC_DIR = Path(__file__).resolve().parent
for _p in (str(_SRC_DIR.parent), str(_SRC_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.confidence import (
    apply_platt, arithmetic_consistency, line_item_consistency,
    line_item_logprob_feature, line_item_raw_score, logprob_feature, raw_score,
    run as run_confidence,
)
from src.categorize import infer_category
from src.eval import SCALAR_FIELDS, normalize_num, normalize_text

REPO_ROOT = Path(__file__).resolve().parent.parent
PROC_ROOT = REPO_ROOT / "data" / "processed"

PREDICTION_FIELDS = SCALAR_FIELDS + ["line_items"]

_CONF_TAG = "finetuned"
_CONF_TAG_LOGPROB = "finetuned_logprob"

# Minimal date parser for the dashboard's month bucketing.
_DATE_RE = re.compile(r"(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2,4})")

CATEGORY_ORDER = ["dining", "grocery", "fuel", "retail", "transport", "misc", "other"]

# A null `store` is almost always a real miss (every receipt has a store name), so it gets
# a "missing" badge; a null in any other field is usually correct absence, so it gets a
# neutral "na" badge rather than a red 0.00 implying a failed extraction.
FIELDS_WHERE_NULL_IS_LIKELY_A_MISS = {"store"}


def month_bucket(date_str) -> str:
    if not date_str:
        return "unknown"
    m = _DATE_RE.search(str(date_str))
    if not m:
        return "unknown"
    mo, _, y = (int(g) for g in m.groups())
    if y < 100:
        y += 2000
    if not (1 <= mo <= 12):
        return "unknown"
    return f"{y}-{mo:02d}"


def load_jsonl(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    out = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                out[rec["image_id"]] = rec
    return out


# --- confidence badges -------------------------------------------------------------

def _band(score: float) -> dict:
    """Confidence badge for a score in [0, 1], with a level colour."""

    score = max(0.0, min(1.0, score))
    level = "green" if score >= 0.75 else "amber" if score >= 0.4 else "red"
    return {"score": round(score, 2), "level": level}


def _na_badge() -> dict:
    """Neutral badge: the field is null/empty, so there is no value to judge (e.g. a null
    tip on a receipt that has no tip)."""

    return {"score": None, "level": "na"}


def _unscored_badge() -> dict:
    """Badge for a field that is present but has no confidence signal (e.g. a line item
    with no price)."""

    return {"score": None, "level": "unscored"}


def _missing_badge() -> dict:
    """Badge for a field that should have been present and was not."""

    return {"score": None, "level": "missing"}


def _null_field_badge(field: str) -> dict:
    return _missing_badge() if field in FIELDS_WHERE_NULL_IS_LIKELY_A_MISS else _na_badge()


def _params_by_field(conf_results: dict) -> dict[str, np.ndarray]:
    """Platt calibration parameters per field, from a confidence sweep's results."""

    return {f: np.array(r["calibration_weights"] + [r["calibration_bias"]])
            for f, r in conf_results["fields"].items() if "calibration_weights" in r}


class ConfidenceScorer:
    """Holds the Platt calibrations and turns a prediction record into badges.

    Two calibrations, matching serve.py: a 2-signal fit (format validity + arithmetic
    consistency) for cached predictions, and a 3-signal fit (+ token logprob) for live
    inference. Both are fit lazily on first use rather than at import, so importing this
    module stays cheap for callers that only want the aggregation helpers.
    """

    def __init__(self, calib_frac: float = 0.5, seed: int = 0) -> None:
        self._calib_frac = calib_frac
        self._seed = seed
        self._platt: dict[str, np.ndarray] | None = None
        self._platt_3sig: dict[str, np.ndarray] | None = None

    @property
    def platt(self) -> dict[str, np.ndarray]:
        if self._platt is None:
            results = run_confidence(_CONF_TAG, "test", seed=self._seed,
                                     calib_frac=self._calib_frac, quiet=True)
            self._platt = _params_by_field(results)
        return self._platt

    @property
    def platt_3sig(self) -> dict[str, np.ndarray]:
        """3-signal calibration, or {} when the logprob-enabled prediction file is absent
        (callers then fall back to the 2-signal fit)."""

        if self._platt_3sig is None:
            if (PROC_ROOT / f"{_CONF_TAG_LOGPROB}_test.jsonl").exists():
                results = run_confidence(_CONF_TAG_LOGPROB, "test", seed=self._seed,
                                         calib_frac=self._calib_frac, quiet=True)
                self._platt_3sig = _params_by_field(results)
            else:
                self._platt_3sig = {}
        return self._platt_3sig

    def line_item_badges(self, record: dict, platt_params: dict,
                         use_logprob: bool = False) -> dict:
        """Per-item badges plus an aggregate for the whole line-items list.

        Falls back to the raw heuristic when no Platt calibration was fit for line items
        (too sparse even in a bigger sample).
        """

        items = record.get("line_items") or []
        if not items:
            return {"aggregate": _missing_badge(), "items": []}

        consistency = line_item_consistency(items, record.get("subtotal"),
                                            record.get("total"), record.get("tax"),
                                            record.get("tip"))
        params = platt_params.get("line_items")
        item_badges, scored = [], []
        for idx, item in enumerate(items):
            if item.get("price") is None:
                item_badges.append(_na_badge())
                continue
            heuristic = line_item_raw_score(item, consistency.get(idx))
            if use_logprob:
                lp_feat = line_item_logprob_feature(idx, record)
                if heuristic is None and lp_feat is None:
                    item_badges.append(_unscored_badge())
                    continue
                features = [heuristic if heuristic is not None else 0.5,
                            lp_feat if lp_feat is not None else 0.5]
            else:
                if heuristic is None:
                    item_badges.append(_unscored_badge())
                    continue
                features = [heuristic]
            score = (float(apply_platt(np.array(features), params)[0])
                     if params is not None else features[0])
            item_badges.append(_band(score))
            scored.append(score)

        if scored:
            aggregate = _band(sum(scored) / len(scored))
        elif any(item.get("price") is not None for item in items):
            aggregate = _unscored_badge()
        else:
            aggregate = _na_badge()
        return {"aggregate": aggregate, "items": item_badges}

    def field_confidence(self, record: dict) -> dict[str, Any]:
        """2-signal badges, for cached predictions."""

        consistent = arithmetic_consistency(record)
        platt = self.platt
        out: dict[str, Any] = {}
        for field in SCALAR_FIELDS:
            if record.get(field) is None:
                out[field] = _null_field_badge(field)
                continue
            score = raw_score(field, record, consistent)
            if field in platt:
                score = float(apply_platt(np.array([score]), platt[field])[0])
            out[field] = _band(score)
        out["line_items"] = self.line_item_badges(record, platt)
        return out

    def field_confidence_live(self, record: dict) -> dict[str, Any]:
        """3-signal badges (+ token logprob), for freshly inferred predictions.

        Degrades to the 2-signal fit, then to the raw heuristic, per field -- `tip` is
        typically too sparse to calibrate even in a bigger sample.
        """

        consistent = arithmetic_consistency(record)
        platt, platt_3sig = self.platt, self.platt_3sig
        out: dict[str, Any] = {}
        for field in SCALAR_FIELDS:
            if record.get(field) is None:
                out[field] = _null_field_badge(field)
                continue
            heuristic = raw_score(field, record, consistent)
            lp_feat = logprob_feature(field, record)
            features = [heuristic, lp_feat if lp_feat is not None else 0.5]
            if field in platt_3sig:
                score = float(apply_platt(np.array(features), platt_3sig[field])[0])
            elif field in platt:
                score = float(apply_platt(np.array([heuristic]), platt[field])[0])
            else:
                score = heuristic
            out[field] = _band(score)
        if "line_items" in platt_3sig:
            out["line_items"] = self.line_item_badges(record, platt_3sig, use_logprob=True)
        else:
            out["line_items"] = self.line_item_badges(record, platt)
        return out


# --- spend aggregation -------------------------------------------------------------

def aggregate_spend(records: list[dict]) -> dict:
    """Store/month totals from a list of prediction dicts."""

    total_spend = 0.0
    n_priced = 0
    by_store: dict[str, float] = defaultdict(float)
    by_store_label: dict[str, str] = {}
    by_month: dict[str, float] = defaultdict(float)

    for rec in records:
        total = normalize_num(rec.get("total"))
        if total is None:
            continue
        total_spend += total
        n_priced += 1

        store_key = normalize_text(rec.get("store")) or "unknown"
        by_store[store_key] += total
        by_store_label.setdefault(store_key, rec.get("store") or "Unknown")

        by_month[month_bucket(rec.get("date"))] += total

    top_stores = sorted(by_store.items(), key=lambda kv: kv[1], reverse=True)[:10]
    return {
        "n_receipts": len(records),
        "n_priced": n_priced,
        "total_spend": round(total_spend, 2),
        "by_store": [{"store": by_store_label[k], "spend": round(v, 2)}
                     for k, v in top_stores],
        "by_month": [{"month": k, "spend": round(v, 2)}
                     for k, v in sorted(by_month.items())],
    }


def dashboard_payload(live_receipts: list[dict]) -> dict:
    """Dashboard summary over the receipts this session uploaded.

    `live_receipts` entries are {"timestamp", "filename", "prediction"}. Category is
    computed here rather than client-side because it needs the line items too, so a
    storeless receipt still categorizes from what was bought instead of falling to
    "other".
    """

    agg = aggregate_spend([r["prediction"] for r in live_receipts])
    agg["recent"] = [
        {"timestamp": r["timestamp"], "filename": r["filename"],
         "store": r["prediction"].get("store"), "date": r["prediction"].get("date"),
         "total": r["prediction"].get("total"),
         "category": infer_category(r["prediction"])}
        for r in reversed(live_receipts[-20:])
    ]
    agg["caveat"] = "predicted totals, not manually verified"
    return agg


def categories_payload(live_receipts: list[dict]) -> dict:
    """Receipts grouped by heuristic (store-name keyword) category."""

    records = [r["prediction"] for r in live_receipts]
    buckets: dict[str, dict] = defaultdict(
        lambda: {"count": 0, "spend": 0.0, "n_priced": 0})
    for rec in records:
        b = buckets[infer_category(rec)]
        b["count"] += 1
        total = normalize_num(rec.get("total"))
        if total is not None:
            b["spend"] += total
            b["n_priced"] += 1

    ordered = ([c for c in CATEGORY_ORDER if c in buckets]
               + [c for c in buckets if c not in CATEGORY_ORDER])
    out = []
    for cat in ordered:
        b = buckets[cat]
        out.append({
            "category": cat,
            "count": b["count"],
            "share": round(100 * b["count"] / max(len(records), 1), 1),
            "total_spend": round(b["spend"], 2),
            "avg_total": round(b["spend"] / b["n_priced"], 2) if b["n_priced"] else None,
        })
    return {"n_receipts": len(records),
            "basis": "uploaded receipts this session; heuristic categories",
            "categories": out}
