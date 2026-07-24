"""Deliverable #10 (API half) — FastAPI serving layer (Track C).

UPDATED 2026-07-23: this machine now has mlx_vlm + the base model + WildReceipt images
installed (see PROGRESS.md — it was set up mid-session, no longer a stopgap-only
analysis clone). `/receipts/{image_id}` still serves *cached* predictions from
`data/processed/finetuned_test.jsonl` (fast, and matches what #5/#6's reported numbers
were scored against) with #8's real 2-signal calibrated confidence (format validity +
arithmetic consistency — those cached predictions were never generated with
`--capture-logprobs`, so no token-logprob signal exists for them). The new `POST
/infer` endpoint is genuinely live: an uploaded image is run through the model, #9's
repair layer, and #8's *3-signal* calibration (now that generation-time logprobs are
available), calibrated against a real (if modest, n=80) logprob-enabled run. Two
separate calibration fits, not one, because the two endpoints have different signals
available — documented at each `_PLATT_*` definition below.

Run (from repo root):
    uvicorn src.serve:app --reload --port 8000
"""
from __future__ import annotations

import datetime
import json
import re
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pillow_heif

# mlx_vlm's own image loader (mlx_vlm/utils.py: load_image) goes through
# PIL.Image.open, same as app/streamlit_app.py's preview path — stock Pillow has no
# HEIC/HEIF decoder at all, and this registration is process-global but NOT shared
# across processes, so /infer needs its own copy of this call, not just the one in
# the Streamlit app.
pillow_heif.register_heif_opener()
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# confidence.py's own `import eval as ev` (and zeroshot.py's `from train import ...`)
# assume they're run as a script (`python src/confidence.py`), which puts src/ itself
# on sys.path — true for every other consumer of eval.py in this project, but not for
# serve.py importing them as submodules of the `src` package. Adding src/ here (in
# addition to the repo root above) covers both without changing their import style.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.eval import normalize_num, normalize_text, SCALAR_FIELDS  # noqa: E402  (reuse #5's parser)
from src.confidence import (  # noqa: E402  (reuse #8's real calibrated composite)
    arithmetic_consistency, raw_score, logprob_feature, apply_platt, run as run_confidence,
    line_item_raw_score, line_item_consistency, line_item_logprob_feature,
)
from src.repair import repair_json  # noqa: E402  (#9)
from src.zeroshot import (  # noqa: E402
    generate_with_logprobs, field_avg_logprob, line_item_avg_logprob,
    normalize as normalize_prediction,
)
from src.train import DEFAULT_MODEL, PROMPT  # noqa: E402  (the exact schema-guided prompt #4/#3 train/eval against)

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

# eval.py (#5) has no date-normalization helper — it only needs text/numeric equality
# for scoring, never a canonical calendar value — so the dashboard's month bucketing
# is its own minimal parser rather than a borrowed one that doesn't actually exist.
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


def _params_by_field(conf_results: dict) -> dict[str, np.ndarray]:
    return {f: np.array(r["calibration_weights"] + [r["calibration_bias"]])
            for f, r in conf_results["fields"].items() if "calibration_weights" in r}


# Fit #8's Platt calibration once at startup against the same predictions this API
# serves — cheap (472 receipts, sub-second) and keeps confidence live/reproducible
# rather than depending on a possibly-stale cached _confidence_*.json on disk. This is
# the 2-signal (format validity + arithmetic consistency) fit: /receipts/{image_id}'s
# cached predictions were never generated with --capture-logprobs, so no logprob
# signal exists for them to calibrate against.
_CONF_RESULTS = run_confidence(_CONF_TAG, "test", seed=0, calib_frac=0.5, quiet=True)
_PLATT = _params_by_field(_CONF_RESULTS)

# Separate 3-signal (+ token logprob) calibration for /infer's live predictions, fit
# against the one logprob-enabled prediction file that exists (a modest n=80 subset —
# see PROGRESS.md). Falls back to None (2-signal only) if that file hasn't been
# generated on whatever machine this runs on.
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


# A null prediction means two very different things depending on the field, checked
# against real ground truth (test.jsonl) rather than assumed (see PROGRESS.md
# 2026-07-25): for `store`, a null prediction is *always* a genuine miss in this
# dataset (0/23 null predictions have a null gold value too — every receipt has a
# store name) — showing that as a neutral "not applicable" badge would hide a real
# problem. For every other scalar field, null is usually the model *correctly*
# predicting legitimate absence (78-97% of null predictions match a null gold value,
# tip most starkly at 97%) — showing that as red/0.00 falsely implies a failed
# extraction on what's actually a correct answer. So: `store`'s nulls get a distinct
# "missing" badge (still a real concern, but no fabricated numeric score), everything
# else gets a neutral "na" badge (no score, nothing to be confident or unconfident
# about since there's no value on display).
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
    """Real per-item confidence, replacing the old hardcoded 0.6-for-any-nonempty-list
    constant (see PROGRESS.md 2026-07-25 for the before/after distribution this used
    to produce vs. now). Each item's raw score comes from #8's `line_item_raw_score`
    (name format validity + whether this item's price is implicated in a subtotal
    mismatch); items with a null price get `_na_badge()` directly — per real data,
    ~12% of *gold* line items also have no price, so it's not always a miss (same
    principle as the scalar-field fix above, one level down).

    Items that DO have a price and a valid-looking name, but no consistency signal
    exists for this receipt, get `_unscored_badge()` instead of a fabricated middling
    number — this used to render as an identical "amber · 0.72"-ish badge on 86.8% of
    multi-item receipts (see PROGRESS.md 2026-07-26), which looked like real per-item
    confidence but wasn't. When `use_logprob` is set (live inference, which always has
    per-item token logprobs available — see `line_item_logprob_feature`), that no-
    consistency-signal case is instead SCORED using logprob alone (heuristic imputed
    neutral), because logprob varies per item even when consistency has nothing to
    say — this is what actually breaks the identical-badge problem for live receipts,
    not just relabels it more honestly. `_unscored_badge()` still applies when even
    logprob comes back empty (e.g. a truncated generation whose array span wasn't
    found).

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
    return ReceiptDetail(
        image_id=image_id,
        prediction={k: rec.get(k) for k in SCALAR_FIELDS + ["line_items"]},
        confidence=field_confidence(rec),
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
    return InferResult(
        prediction=prediction,
        confidence=field_confidence_live(record),
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
    agg["recent"] = [
        {"timestamp": r["timestamp"], "filename": r["filename"],
         "store": r["prediction"].get("store"), "date": r["prediction"].get("date"),
         "total": r["prediction"].get("total")}
        for r in reversed(LIVE_RECEIPTS[-20:])
    ]
    agg["caveat"] = ("figures reflect the model's raw predicted `total` per receipt — "
                     "real extraction, not manually verified; check each receipt's "
                     "confidence badges in the Receipt viewer tab before trusting a "
                     "number here")
    return agg
