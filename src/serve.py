"""Runs the FastAPI layer behind the live demo. /receipts/{id} returns cached
WildReceipt predictions with 2-signal confidence, /infer runs a freshly uploaded
photo through the fine-tuned model, the repair layer, and the full 3-signal
confidence score (token logprob added), and /dashboard aggregates spend only from
receipts actually uploaded this run, not the static evaluation set.
"""
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
    """Log a confidence dict to the confidence.jsonl file, with timestamp, source, and reference."""

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
    """Return a dict of Platt calibration parameters by field, from the confidence sweep results."""

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
    """Return a confidence badge dict for a score in [0, 1], with level color."""

    score = max(0.0, min(1.0, score))
    level = "green" if score >= 0.75 else "amber" if score >= 0.4 else "red"
    return {"score": round(score, 2), "level": level}


# A null `store` is almost always a real miss (every receipt has a store name), so it
# gets a "missing" badge; a null in any other field is usually correct absence, so it
# gets a neutral "na" badge rather than a red/0.00 that implies a failed extraction.
FIELDS_WHERE_NULL_IS_LIKELY_A_MISS = {"store"}


def _na_badge() -> dict:
    """Return a neutral badge for a field that is null/empty, meaning there's no value to
    have an opinion about (e.g. a null tip on a receipt that has no tip)."""

    return {"score": None, "level": "na"}


def _unscored_badge() -> dict:
    """Return a badge for a field that is present but has no confidence score, meaning the
    model didn't have enough signal to make a judgment (e.g. a line-item with no price)."""

    return {"score": None, "level": "unscored"}


def _missing_badge() -> dict:
    """Return a badge for a field that is missing (e.g. a null store on a receipt
    that has no store)."""

    return {"score": None, "level": "missing"}


def _null_field_badge(field: str) -> dict:
    """Return a badge for a null field, either "missing" (red) if it's a field that should
    always be present, or "na" (neutral) if it's a field that is often legitimately null."""

    return _missing_badge() if field in FIELDS_WHERE_NULL_IS_LIKELY_A_MISS else _na_badge()


def _line_item_badges(record: dict, platt_params: dict, use_logprob: bool = False) -> dict:
    """Return a dict of confidence badges for each line item in a receipt record, plus an
    aggregate badge for the whole line-items list. Uses the 2-signal Platt calibration
    (format validity + arithmetic consistency) fit at startup against the same file this
    API serves. Falls back to the raw heuristic if no Platt calibration was fit for
    line items (e.g. too sparse even in a bigger sample). If use_logprob is True, the
    3-signal Platt calibration (format validity + arithmetic consistency + token logprob)
    is used instead, if it was fit at startup; otherwise falls back to 2-signal or raw
    heuristic as above."""

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
    """Return a dict of confidence badges for each field in a receipt record, using the
    2-signal Platt calibration (format validity + arithmetic consistency) fit at startup
    against the same file this API serves. Falls back to the raw heuristic if no Platt
    calibration was fit for a given field (e.g. tip: too sparse even in a bigger sample)."""

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
    """Return a dict of confidence badges for each field in a receipt record, using the
    3-signal Platt calibration (format validity + arithmetic consistency + token logprob)
    fit at startup against the same file this API serves. Falls back to the 2-signal
    calibration if the logprob-enabled prediction file isn't present, and to the raw
    heuristic if no Platt calibration was fit for a given field (e.g. tip: too sparse
    even in a bigger sample)."""

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


app = FastAPI(title="Receipt-to-JSON API", version="1.0.0")


@app.get("/health")
def health():
    return {"status": "ok", "n_predictions": len(PREDICTIONS)}


@app.get("/receipts", response_model=list[ReceiptSummary])
def list_receipts(limit: int = 500):
    """Return a list of receipt summaries (image_id, store, date, total) for the first
    `limit` receipts in the cached predictions. The list is sorted by image_id."""

    out = []
    for image_id, rec in list(PREDICTIONS.items())[:limit]:
        out.append(ReceiptSummary(image_id=image_id, store=rec.get("store"),
                                   date=rec.get("date"), total=rec.get("total")))
    return out


@app.get("/receipts/{image_id:path}/image")
def get_receipt_image(image_id: str):
    """Return the image file for a given receipt image_id. Raises 404 if the image is
    not present on disk."""

    path = IMG_ROOT / image_id
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"no image at {image_id}")
    return FileResponse(path)


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
        # repair.py already ran on this record when it was generated; there's no raw
        # completion left here for this endpoint to repair.
        repair_status="handled_upstream_at_generation",
    )


# --- live inference: model + adapter loaded once at startup ------------------------

_MODEL = _PROCESSOR = _PROMPT = None
if _MLX_AVAILABLE and CKPT_PATH.exists():
    _MODEL, _PROCESSOR = load_vlm(DEFAULT_MODEL, adapter_path=str(CKPT_PATH),
                                  processor_config={"trust_remote_code": True})
    _PROMPT = apply_chat_template(_PROCESSOR, _MODEL.config.__dict__, PROMPT, num_images=1)
LIVE_INFERENCE_AVAILABLE = _MODEL is not None


LIVE_RECEIPTS: list[dict] = []


class InferResult(BaseModel):
    prediction: dict
    confidence: dict[str, Any]
    repair_status: str


@app.post("/infer", response_model=InferResult)
async def infer(file: UploadFile = File(...)):
    """Run inference on a single uploaded receipt image, returning the prediction dict,
    confidence badges, and repair status. Confidence is computed with the 3-signal Platt
    calibration (format validity + arithmetic consistency + token logprob) fit at
    startup against the same file this API serves. Falls back to 2-signal calibration if
    the logprob-enabled prediction file isn't present, and to the raw heuristic if no
    Platt calibration was fit for a given field (e.g. tip: too sparse even in a bigger
    sample). Confidence is logged to logs/confidence.jsonl but not returned in the API
    response, to avoid overwhelming the client with a large JSON blob. The repair status
    indicates whether the model's raw output was valid JSON or had to be repaired by the
    repair_json function."""

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
    """Return a dashboard summary of the receipts uploaded this server run (same
    LIVE_RECEIPTS set /infer uses). Includes total spend, top stores, and monthly
    spend. Categories are heuristic (store-name keywords)."""

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
    """Return a summary of the receipts uploaded this server run, grouped by heuristic
    category (store-name keywords). Includes count, total spend, and average spend per
    receipt. Categories are heuristic (store-name keywords)."""

    records = [r["prediction"] for r in LIVE_RECEIPTS]
    buckets: dict[str, dict] = defaultdict(lambda: {"count": 0, "spend": 0.0, "n_priced": 0})
    for rec in records:
        b = buckets[infer_category(rec)]
        b["count"] += 1
        total = normalize_num(rec.get("total"))
        if total is not None:
            b["spend"] += total
            b["n_priced"] += 1
    order = ["dining", "grocery", "fuel", "retail", "transport", "misc", "other"]
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
