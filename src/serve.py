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
import sys
import tempfile
from pathlib import Path
from typing import Any

import pillow_heif

pillow_heif.register_heif_opener()  # /infer needs its own HEIC opener (registration isn't cross-process)
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

# repo root + src/ on path: the src submodules import each other by bare name (run-as-script style)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.eval import SCALAR_FIELDS
# Confidence badges, spend aggregation and category rollups live in pipeline.py so the
# Gradio Space can call them without a FastAPI process. Single definition on purpose --
# the calibrated confidence score is the headline result, so a second copy would drift.
from src.pipeline import (
    ConfidenceScorer, PREDICTION_FIELDS, categories_payload, dashboard_payload,
    load_jsonl,
)
from src.repair import repair_json
from src.zeroshot import (
    generate_with_logprobs, field_avg_logprob, line_item_avg_logprob,
    normalize as normalize_prediction,
)
from src.schema import DEFAULT_MODEL, PROMPT

try:
    from mlx_vlm import load as load_vlm
    from mlx_vlm.prompt_utils import apply_chat_template
    _MLX_AVAILABLE = True
except ImportError:
    _MLX_AVAILABLE = False

PROC_ROOT = Path(__file__).resolve().parent.parent / "data" / "processed"
IMG_ROOT = Path(__file__).resolve().parent.parent / "data" / "wildreceipt"
CKPT_PATH = Path(__file__).resolve().parent.parent / "checkpoints" / "final"
# The transformers-format adapter, used when mlx_vlm isn't available. A separate file
# because an adapter only works against the base it was fit to.
PEFT_CKPT_PATH = Path(__file__).resolve().parent.parent / "checkpoints" / "final_peft"
PRED_FILE = PROC_ROOT / "finetuned_test.jsonl"
GT_FILE = PROC_ROOT / "test.jsonl"

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


# The 2-signal (format validity + arithmetic consistency) and 3-signal (+ token logprob)
# Platt calibrations now live in pipeline.py, fit lazily on first badge request rather
# than at import. Behaviour is unchanged; the definitions are just shared with the Space.
_SCORER = ConfidenceScorer(calib_frac=0.5, seed=0)


def field_confidence(record: dict) -> dict[str, Any]:
    """2-signal confidence badges for a cached prediction."""

    return _SCORER.field_confidence(record)


def field_confidence_live(record: dict) -> dict[str, Any]:
    """3-signal confidence badges (+ token logprob) for a live prediction."""

    return _SCORER.field_confidence_live(record)


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
    return {"status": "ok", "n_predictions": len(PREDICTIONS),
            "live_inference": LIVE_INFERENCE_AVAILABLE, "backend": _BACKEND}


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
        prediction={k: rec.get(k) for k in PREDICTION_FIELDS},
        confidence=confidence,
        ground_truth=GROUND_TRUTH.get(image_id) if include_gt else None,
        # repair.py already ran on this record when it was generated; there's no raw
        # completion left here for this endpoint to repair.
        repair_status="handled_upstream_at_generation",
    )


# --- live inference: model + adapter loaded once at startup ------------------------
#
# Two backends, because the two adapters are not interchangeable. mlx_vlm drives
# checkpoints/final (MLX 4-bit) and is roughly 6x faster on Apple Silicon, so it wins when
# available. Everywhere else -- a Linux GPU box, or a Mac without mlx installed --
# backend_hf drives checkpoints/final_peft (retrained for transformers) on cuda, mps or
# cpu. Loading the MLX adapter through transformers is the 0.288 micro-F1 failure
# documented in scripts/validate_peft_adapter.py, hence one adapter per backend rather
# than one shared file.

_MODEL = _PROCESSOR = _PROMPT = None
_HF_MODEL = None
_BACKEND = None

if _MLX_AVAILABLE and CKPT_PATH.exists():
    _MODEL, _PROCESSOR = load_vlm(DEFAULT_MODEL, adapter_path=str(CKPT_PATH),
                                  processor_config={"trust_remote_code": True})
    _PROMPT = apply_chat_template(_PROCESSOR, _MODEL.config.__dict__, PROMPT, num_images=1)
    _BACKEND = "mlx"
elif PEFT_CKPT_PATH.exists():
    try:
        from src.backend_hf import ReceiptModel

        _HF_MODEL = ReceiptModel(adapter_path=PEFT_CKPT_PATH).load()
        _BACKEND = "transformers"
    except Exception as exc:  # torch/peft absent, or no usable device
        logging.getLogger("receiptvlm").warning(
            "transformers backend unavailable (%s: %s); serving cached predictions only",
            type(exc).__name__, exc)

LIVE_INFERENCE_AVAILABLE = _BACKEND is not None


def _generate(image_path: str) -> tuple[str, list]:
    """Run the active backend, returning (raw_text, [(chunk_text, token_logprob), ...]).

    Both backends return the same shape: the chunk texts concatenate to exactly raw_text,
    which is what zeroshot.field_avg_logprob needs to map character spans back to tokens.
    """

    if _BACKEND == "mlx":
        return generate_with_logprobs(
            _MODEL, _PROCESSOR, _PROMPT, image=image_path,
            max_tokens=1536, temperature=0.0, resize_shape=(768, 1024),
        )
    # backend_hf applies the same 768x1024 aspect-preserving resize and greedy decoding
    # internally, so the call sites stay equivalent.
    from PIL import Image

    with Image.open(image_path) as img:
        return _HF_MODEL.generate_with_logprobs(img)


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
        raise HTTPException(
            status_code=503,
            detail="live inference unavailable on this server: neither mlx_vlm with "
                   f"{CKPT_PATH.name} nor a transformers backend with "
                   f"{PEFT_CKPT_PATH.name} could be loaded")
    suffix = Path(file.filename or "upload.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp.flush()
        raw, chunks = _generate(tmp.name)
    parsed, status = repair_json(raw)
    record = normalize_prediction(parsed)
    record["_field_logprobs"] = {f: field_avg_logprob(f, raw, chunks) for f in SCALAR_FIELDS}
    record["_line_item_logprobs"] = [
        line_item_avg_logprob(i, raw, chunks) for i in range(len(record.get("line_items") or []))
    ]
    prediction = {k: record.get(k) for k in PREDICTION_FIELDS}
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


@app.get("/dashboard")
def dashboard():
    """Dashboard summary over the receipts uploaded this server run (the same
    LIVE_RECEIPTS set /infer appends to), not the static evaluation set."""

    return dashboard_payload(LIVE_RECEIPTS)


@app.get("/categories")
def categories():
    """Receipts uploaded this server run, grouped by heuristic (store-name) category."""

    return categories_payload(LIVE_RECEIPTS)
