from __future__ import annotations

import datetime
import json
import logging
import re
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pillow_heif

pillow_heif.register_heif_opener()  # /infer needs its own HEIC opener (registration isn't cross-process)
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

# repo root + src/ on path: the src submodules import each other by bare name (run-as-script style)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.eval import normalize_num, normalize_text, SCALAR_FIELDS
from src.confidence import (
    arithmetic_consistency, raw_score, logprob_feature, apply_platt, run as run_confidence,
    line_item_raw_score, line_item_consistency, line_item_logprob_feature,
)
from src.repair import repair_json
from src.categorize import infer_category
from src.zeroshot import (
    generate_with_logprobs, field_avg_logprob, line_item_avg_logprob,
    normalize as normalize_prediction,
)
from src.train import DEFAULT_MODEL, PROMPT

try:
    from mlx_vlm import load as load_vlm
    from mlx_vlm.prompt_utils import apply_chat_template
    _MLX_AVAILABLE = True
except ImportError:
    _MLX_AVAILABLE = False

PROC_ROOT = Path(__file__).resolve().parent.parent / "data" / "processed"
IMG_ROOT = Path(__file__).resolve().parent.parent / "data" / "wildreceipt"
CKPT_PATH = Path(__file__).resolve().parent.parent / "checkpoints" / "final"
PRED_FILE = PROC_ROOT / "finetuned_test.jsonl"
GT_FILE = PROC_ROOT / "test.jsonl"

_CONF_TAG = "finetuned"
_CONF_TAG_LOGPROB = "finetuned_logprob"

# minimal date parser for the dashboard's month bucketing
_DATE_RE = re.compile(r"(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2,4})")


def _month_bucket(date_str) -> str:
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


PREDICTIONS = load_jsonl(PRED_FILE)
GROUND_TRUTH = load_jsonl(GT_FILE)

# Confidence is computed on every receipt but not shown in the UI — logged here instead.
LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
_conf_logger = logging.getLogger("receiptvlm.confidence")
_conf_logger.setLevel(logging.INFO)
if not _conf_logger.handlers:
    # mode="w" truncates on startup, so the log holds only the current backend session.
    _fh = logging.FileHandler(LOG_DIR / "confidence.jsonl", mode="w")
    _fh.setFormatter(logging.Formatter("%(message)s"))
    _conf_logger.addHandler(_fh)


def _log_confidence(ref: str, source: str, confidence: dict) -> None:
    li = confidence.get("line_items", {})
    entry = {
        "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source": source, "ref": ref,
        "fields": {f: {"level": confidence.get(f, {}).get("level"),
                       "score": confidence.get(f, {}).get("score")} for f in SCALAR_FIELDS},
        "line_items_aggregate": {"level": li.get("aggregate", {}).get("level"),
                                 "score": li.get("aggregate", {}).get("score")},
        "line_items": [{"level": b.get("level"), "score": b.get("score")}
                       for b in li.get("items", [])],
    }
    _conf_logger.info(json.dumps(entry))


def _params_by_field(conf_results: dict) -> dict[str, np.ndarray]:
    return {f: np.array(r["calibration_weights"] + [r["calibration_bias"]])
            for f, r in conf_results["fields"].items() if "calibration_weights" in r}


# 2-signal Platt calibration (format validity + arithmetic consistency) for the cached
# predictions, fit once at startup against the same file this API serves.
_CONF_RESULTS = run_confidence(_CONF_TAG, "test", seed=0, calib_frac=0.5, quiet=True)
_PLATT = _params_by_field(_CONF_RESULTS)

# Separate 3-signal (+ token logprob) calibration for /infer's live predictions; falls
# back to 2-signal only if the logprob-enabled prediction file isn't present.
_LOGPROB_PRED_FILE = PROC_ROOT / f"{_CONF_TAG_LOGPROB}_test.jsonl"
if _LOGPROB_PRED_FILE.exists():
    _CONF_RESULTS_3SIG = run_confidence(_CONF_TAG_LOGPROB, "test", seed=0,
                                        calib_frac=0.5, quiet=True)
    _PLATT_3SIG = _params_by_field(_CONF_RESULTS_3SIG)
else:
    _PLATT_3SIG = {}


def _band(score: float) -> dict:
    score = max(0.0, min(1.0, score))
    level = "green" if score >= 0.75 else "amber" if score >= 0.4 else "red"
    return {"score": round(score, 2), "level": level}


# A null `store` is almost always a real miss (every receipt has a store name), so it
# gets a "missing" badge; a null in any other field is usually correct absence, so it
# gets a neutral "na" badge rather than a red/0.00 that implies a failed extraction.
FIELDS_WHERE_NULL_IS_LIKELY_A_MISS = {"store"}


def _na_badge() -> dict:
    return {"score": None, "level": "na"}


def _unscored_badge() -> dict:
    # Distinct from `_na_badge()`: "na" means there's no *value* to have an opinion
    # about (null price); "unscored" means there IS a predicted item, its name format
    # looks fine, but no consistency signal exists to judge it against (no subtotal
    # to reconcile, or this item wasn't clearly implicated) — see
    # `line_item_raw_score`'s docstring. Showing this as a numeric badge (the old
    # behavior) is what made every item look identically confident.
    return {"score": None, "level": "unscored"}


def _missing_badge() -> dict:
    return {"score": None, "level": "missing"}


def _null_field_badge(field: str) -> dict:
    return _missing_badge() if field in FIELDS_WHERE_NULL_IS_LIKELY_A_MISS else _na_badge()


def _line_item_badges(record: dict, platt_params: dict, use_logprob: bool = False) -> dict:
    """Per-item confidence. Each item's raw score comes from name format validity plus
    whether its price is implicated in a subtotal mismatch; null-price items get an
    `_na_badge()` (a missing price isn't always a miss). Items with a price but no
    consistency signal get `_unscored_badge()` rather than a fabricated middling number
    — unless `use_logprob` is set (live inference), where per-item token logprob scores
    them so every item's badge can differ. `_unscored_badge()` still applies if logprob
    is also empty (e.g. a truncated generation).

    The aggregate is the mean of the actually-scored items' calibrated probabilities,
    so it reflects the real spread instead of repeating one constant onto every
    receipt; if nothing was scorable it falls back to `_unscored_badge()`/`_na_badge()`
    the same way a single item would.
    """
    items = record.get("line_items") or []
    if not items:
        return {"aggregate": _missing_badge(), "items": []}

    consistency = line_item_consistency(items, record.get("subtotal"),
                                        record.get("total"), record.get("tax"), record.get("tip"))
    params = platt_params.get("line_items")
    item_badges = []
    scored = []
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
        score = float(apply_platt(np.array(features), params)[0]) if params is not None else features[0]
        item_badges.append(_band(score))
        scored.append(score)

    if scored:
        aggregate = _band(sum(scored) / len(scored))
    elif any(item.get("price") is not None for item in items):
        aggregate = _unscored_badge()
    else:
        aggregate = _na_badge()
    return {"aggregate": aggregate, "items": item_badges}


def field_confidence(record: dict) -> dict[str, Any]:
    """#8's 2-signal calibrated composite for cached dataset predictions: format
    validity + arithmetic consistency, Platt-scaled against actual outcomes on this
    same prediction file. No token-logprob signal here — these predictions were
    generated without `--capture-logprobs`. Fields with too few predicted values to
    calibrate (tip: ~4-5% of receipts) fall back to the raw uncalibrated heuristic.
    """
    consistent = arithmetic_consistency(record)
    out = {}
    for field in SCALAR_FIELDS:
        if record.get(field) is None:
            out[field] = _null_field_badge(field)
            continue
        score = raw_score(field, record, consistent)
        if field in _PLATT:
            score = float(apply_platt(np.array([score]), _PLATT[field])[0])
        out[field] = _band(score)
    out["line_items"] = _line_item_badges(record, _PLATT)
    return out


def field_confidence_live(record: dict) -> dict[str, Any]:
    """#8's full 3-signal calibrated composite for a just-generated prediction (live
    inference always has token logprobs available, via `record["_field_logprobs"]`
    and, since 2026-07-26, `record["_line_item_logprobs"]`). Falls back to the
    2-signal calibration, then to the raw uncalibrated heuristic, if the 3-signal
    calibration wasn't fit for a given field (e.g. tip: too sparse even in a bigger
    sample) or doesn't exist at all on this machine. Line items get the 3-signal
    (`_PLATT_3SIG`) calibration too now, with per-item logprob as the feature that
    actually breaks the identical-badge problem when consistency has nothing to say
    (see `_line_item_badges`'s docstring) — falls back to `_PLATT`'s 1-feature fit if
    the 3-signal file's "line_items" entry wasn't calibrated for some reason.
    """
    consistent = arithmetic_consistency(record)
    out = {}
    for field in SCALAR_FIELDS:
        if record.get(field) is None:
            out[field] = _null_field_badge(field)
            continue
        heuristic = raw_score(field, record, consistent)
        lp_feat = logprob_feature(field, record)
        features = [heuristic, lp_feat if lp_feat is not None else 0.5]
        if field in _PLATT_3SIG:
            score = float(apply_platt(np.array(features), _PLATT_3SIG[field])[0])
        elif field in _PLATT:
            score = float(apply_platt(np.array([heuristic]), _PLATT[field])[0])
        else:
            score = heuristic
        out[field] = _band(score)
    if "line_items" in _PLATT_3SIG:
        out["line_items"] = _line_item_badges(record, _PLATT_3SIG, use_logprob=True)
    else:
        out["line_items"] = _line_item_badges(record, _PLATT)
    return out


class ReceiptSummary(BaseModel):
    image_id: str
    store: str | None
    date: str | None
    total: str | None


class ReceiptDetail(BaseModel):
    image_id: str
    prediction: dict
    # Scalar fields are {"score": float|None, "level": "green"|"amber"|"red"|"na"|
    # "missing"}; "line_items" is instead {"aggregate": {...same shape...}, "items":
    # [{...per-item...}, ...]} — genuinely different shapes per key, so this is a
    # plain dict rather than a single Pydantic model every value must fit.
    confidence: dict[str, Any]
    ground_truth: dict | None = None
    repair_status: str


app = FastAPI(title="Receipt-to-JSON API", version="0.1.0-stopgap")


@app.get("/health")
def health():
    return {"status": "ok", "n_predictions": len(PREDICTIONS)}


@app.get("/receipts", response_model=list[ReceiptSummary])
def list_receipts(limit: int = 500):
    out = []
    for image_id, rec in list(PREDICTIONS.items())[:limit]:
        out.append(ReceiptSummary(image_id=image_id, store=rec.get("store"),
                                   date=rec.get("date"), total=rec.get("total")))
    return out


@app.get("/receipts/{image_id:path}/image")
def get_receipt_image(image_id: str):
    path = IMG_ROOT / image_id
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"no image at {image_id}")
    return FileResponse(path)


# NB: this route MUST come after the more specific "/image" route above — Starlette's
# `:path` converter is greedy (matches everything including further slashes), so had
# this route been registered first, `.../<image_id>/image` would satisfy this route's
# `{image_id:path}` too (image_id = "<image_id>/image", a string PREDICTIONS never
# has) and always win as the first match, silently 404ing every image request. Route
# *order* is the fix, not the converter — found via curl testing after WildReceipt
# images landed on this machine, since a 404 that's ALWAYS present looks identical to
# one that only started once real images were expected to exist.
@app.get("/receipts/{image_id:path}", response_model=ReceiptDetail)
def get_receipt(image_id: str, include_gt: bool = False):
    rec = PREDICTIONS.get(image_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"no prediction for {image_id}")
    confidence = field_confidence(rec)
    _log_confidence(image_id, "cached", confidence)
    return ReceiptDetail(
        image_id=image_id,
        prediction={k: rec.get(k) for k in SCALAR_FIELDS + ["line_items"]},
        confidence=confidence,
        ground_truth=GROUND_TRUTH.get(image_id) if include_gt else None,
        # #9 (src/repair.py) exists now and runs at generation time inside zeroshot.py's
        # inference loop, on the raw text the model emits — but this endpoint serves
        # already-parsed cached predictions, not raw completions, so there's nothing
        # left here to repair.
        repair_status="handled_upstream_at_generation",
    )


# --- live inference: model + adapter loaded once at startup ------------------------
# Not wrapped in module-load-time try/except beyond the mlx_vlm import check above —
# if mlx_vlm imported fine but the model/adapter fails to load (e.g. no network, no
# checkpoint), that should surface loudly at startup, not silently degrade a
# user-facing inference endpoint into always-503.
_MODEL = _PROCESSOR = _PROMPT = None
if _MLX_AVAILABLE and CKPT_PATH.exists():
    _MODEL, _PROCESSOR = load_vlm(DEFAULT_MODEL, adapter_path=str(CKPT_PATH),
                                  processor_config={"trust_remote_code": True})
    _PROMPT = apply_chat_template(_PROCESSOR, _MODEL.config.__dict__, PROMPT, num_images=1)
LIVE_INFERENCE_AVAILABLE = _MODEL is not None

# Receipts actually uploaded and analyzed via /infer this run — the real thing a
# "spending dashboard" should reflect, unlike the old /dashboard which aggregated the
# static WildReceipt eval set (472 research receipts, mixed currencies, a checkpoint's
# own extraction errors baked in) and produced a number that needed a permanent
# "not a validated spend total" caveat just to not be misleading. In-memory only —
# resets on API restart, and shared across every client hitting this server (fine for
# a single-user local demo tool, not a multi-tenant design). A deliberate, documented
# scope choice, not an oversight: durable storage would need a real datastore and
# per-user separation, neither of which this deliverable asked for.
LIVE_RECEIPTS: list[dict] = []


class InferResult(BaseModel):
    prediction: dict
    confidence: dict[str, Any]
    repair_status: str


@app.post("/infer", response_model=InferResult)
async def infer(file: UploadFile = File(...)):
    """Real image -> JSON: generate with the fine-tuned checkpoint, repair the raw
    completion (#9), score confidence with the full 3-signal composite (#8) since a
    live generation always has token logprobs available. Unlike /receipts/{image_id},
    this has no ground truth to compare against — it's someone's own photo. Also
    records the result into LIVE_RECEIPTS so /dashboard can reflect it.
    """
    if not LIVE_INFERENCE_AVAILABLE:
        raise HTTPException(status_code=503,
                            detail="live inference unavailable on this server (mlx_vlm "
                                   "or the checkpoint isn't present)")
    suffix = Path(file.filename or "upload.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp.flush()
        raw, chunks = generate_with_logprobs(
            _MODEL, _PROCESSOR, _PROMPT, image=tmp.name,
            max_tokens=1536, temperature=0.0, resize_shape=(768, 1024),
        )
    parsed, status = repair_json(raw)
    record = normalize_prediction(parsed)
    record["_field_logprobs"] = {f: field_avg_logprob(f, raw, chunks) for f in SCALAR_FIELDS}
    record["_line_item_logprobs"] = [
        line_item_avg_logprob(i, raw, chunks) for i in range(len(record.get("line_items") or []))
    ]
    prediction = {k: record.get(k) for k in SCALAR_FIELDS + ["line_items"]}
    LIVE_RECEIPTS.append({
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "filename": file.filename,
        "prediction": prediction,
    })
    confidence = field_confidence_live(record)
    _log_confidence(file.filename or "upload", "infer", confidence)
    return InferResult(
        prediction=prediction,
        confidence=confidence,
        repair_status=status,
    )


def _aggregate_spend(records: list[dict]) -> dict:
    """Shared by /dashboard: store/month totals from a list of prediction dicts."""
    total_spend = 0.0
    n_priced = 0
    by_store = defaultdict(float)
    by_store_label: dict[str, str] = {}
    by_month = defaultdict(float)

    for rec in records:
        total = normalize_num(rec.get("total"))
        if total is None:
            continue
        total_spend += total
        n_priced += 1

        store_key = normalize_text(rec.get("store")) or "unknown"
        by_store[store_key] += total
        by_store_label.setdefault(store_key, rec.get("store") or "Unknown")

        by_month[_month_bucket(rec.get("date"))] += total

    top_stores = sorted(by_store.items(), key=lambda kv: kv[1], reverse=True)[:10]
    return {
        "n_receipts": len(records),
        "n_priced": n_priced,
        "total_spend": round(total_spend, 2),
        "by_store": [{"store": by_store_label[k], "spend": round(v, 2)} for k, v in top_stores],
        "by_month": [{"month": k, "spend": round(v, 2)} for k, v in sorted(by_month.items())],
    }


@app.get("/dashboard")
def dashboard():
    """Real-time spending dashboard over receipts actually uploaded and analyzed via
    /infer this server run — not the static WildReceipt eval set the old version of
    this endpoint used (a genuinely misleading number: mixed currencies, a
    checkpoint's own extraction errors, and 472 research receipts nobody actually
    bought anything on). Empty until at least one receipt has been analyzed.
    """
    agg = _aggregate_spend([r["prediction"] for r in LIVE_RECEIPTS])
    # category computed server-side (needs store + line items) so storeless receipts
    # still categorize from what was bought instead of falling to "other".
    agg["recent"] = [
        {"timestamp": r["timestamp"], "filename": r["filename"],
         "store": r["prediction"].get("store"), "date": r["prediction"].get("date"),
         "total": r["prediction"].get("total"),
         "category": infer_category(r["prediction"])}
        for r in reversed(LIVE_RECEIPTS[-20:])
    ]
    agg["caveat"] = "predicted totals, not manually verified"
    return agg


@app.get("/categories")
def categories():
    """Merchant-category breakdown over the receipts uploaded this server run (same
    LIVE_RECEIPTS set /dashboard uses). Categories are heuristic (store-name keywords)."""
    records = [r["prediction"] for r in LIVE_RECEIPTS]
    buckets: dict[str, dict] = defaultdict(lambda: {"count": 0, "spend": 0.0, "n_priced": 0})
    for rec in records:
        b = buckets[infer_category(rec)]
        b["count"] += 1
        total = normalize_num(rec.get("total"))
        if total is not None:
            b["spend"] += total
            b["n_priced"] += 1
    order = ["dining", "grocery", "fuel", "retail", "other"]
    out = []
    for cat in [c for c in order if c in buckets] + [c for c in buckets if c not in order]:
        b = buckets[cat]
        out.append({
            "category": cat,
            "count": b["count"],
            "share": round(100 * b["count"] / max(len(records), 1), 1),
            "total_spend": round(b["spend"], 2),
            "avg_total": round(b["spend"] / b["n_priced"], 2) if b["n_priced"] else None,
        })
    return {"n_receipts": len(records),
            "basis": "uploaded receipts this session; heuristic categories", "categories": out}
