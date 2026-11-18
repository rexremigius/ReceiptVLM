"""Picks the fastest inference backend this host can run.

Preference order: mlx_vlm + checkpoints/final (INT4, Apple Silicon), then
transformers + checkpoints/final_peft (fp16, CUDA/MPS/CPU), then None. 4-bit wins
on MLX (6.2s vs 9.6s) but loses on CUDA, where bitsandbytes dequantizes on every
matmul (23.5s vs 18.0s). The two backends need different adapters: an adapter only
works against the base it was fit to. See RESULTS.md section 5.
"""

from __future__ import annotations

import datetime
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.eval import SCALAR_FIELDS
from src.pipeline import PREDICTION_FIELDS, ConfidenceScorer
from src.repair import repair_json
from src.schema import DEFAULT_MODEL, PROMPT
from src.zeroshot import field_avg_logprob, line_item_avg_logprob, normalize

MLX_ADAPTER = REPO_ROOT / "checkpoints" / "final"
PEFT_ADAPTER = REPO_ROOT / "checkpoints" / "final_peft"

IMAGE_RESIZE = (768, 1024)
MAX_NEW_TOKENS = 1536

_log = logging.getLogger("receiptvlm.backend")


class MLXBackend:
    """mlx_vlm against the 4-bit base and the MLX adapter. Apple Silicon only."""

    name = "mlx"
    precision = "INT4"
    device = "metal"

    def __init__(
        self, base_model: str = DEFAULT_MODEL, adapter_path: Path = MLX_ADAPTER
    ) -> None:
        self.base_model = base_model
        self.adapter_path = adapter_path
        self._model = self._processor = self._prompt = None

    def load(self) -> "MLXBackend":
        from mlx_vlm import load as load_vlm
        from mlx_vlm.prompt_utils import apply_chat_template

        self._model, self._processor = load_vlm(
            self.base_model,
            adapter_path=str(self.adapter_path),
            processor_config={"trust_remote_code": True},
        )
        self._prompt = apply_chat_template(
            self._processor, self._model.config.__dict__, PROMPT, num_images=1
        )
        return self

    def generate_with_logprobs(self, image_path: str | Path) -> tuple[str, list]:
        from src.zeroshot import generate_with_logprobs

        return generate_with_logprobs(
            self._model,
            self._processor,
            self._prompt,
            image=str(image_path),
            max_tokens=MAX_NEW_TOKENS,
            temperature=0.0,
            resize_shape=IMAGE_RESIZE,
        )


class TransformersBackend:
    """transformers + peft against the fp16 base and the retrained adapter.

    Thin wrapper over backend_hf.ReceiptModel so both backends take an image path; the
    underlying model works on a PIL image.
    """

    name = "transformers"

    def __init__(self, adapter_path: Path = PEFT_ADAPTER) -> None:
        self.adapter_path = adapter_path
        self._model = None

    def load(self) -> "TransformersBackend":
        from src.backend_hf import ReceiptModel

        self._model = ReceiptModel(adapter_path=self.adapter_path).load()
        return self

    @property
    def precision(self) -> str:
        return "NF4" if getattr(self._model, "load_4bit", False) else "fp16"

    @property
    def device(self) -> str:
        return getattr(self._model, "device", "unknown")

    def generate_with_logprobs(self, image_path: str | Path) -> tuple[str, list]:
        from PIL import Image

        with Image.open(image_path) as img:
            return self._model.generate_with_logprobs(img)


def select(prefer_mlx: bool = True):
    """Return a loaded backend, or None if neither stack is usable here.

    Failures are logged rather than raised: a host without torch, or without either
    checkpoint, should still serve the cached sample predictions instead of refusing to
    start. /health reports which backend won.
    """

    if prefer_mlx and MLX_ADAPTER.exists():
        try:
            backend = MLXBackend().load()
            _log.info("using MLX backend (INT4) with %s", MLX_ADAPTER.name)
            return backend
        except ImportError:
            pass  # not Apple Silicon, or mlx_vlm not installed - expected off-Mac
        except Exception as exc:
            _log.warning(
                "MLX backend failed to load (%s: %s); trying transformers",
                type(exc).__name__,
                exc,
            )

    if PEFT_ADAPTER.exists():
        try:
            backend = TransformersBackend().load()
            _log.info(
                "using transformers backend (%s, %s) with %s",
                backend.precision,
                backend.device,
                PEFT_ADAPTER.name,
            )
            return backend
        except Exception as exc:
            _log.warning(
                "transformers backend unavailable (%s: %s); serving cached "
                "predictions only",
                type(exc).__name__,
                exc,
            )

    _log.warning("no inference backend available; serving cached predictions only")
    return None


def describe(backend) -> dict:
    """Small summary for /health and the UI."""

    if backend is None:
        return {"backend": None, "precision": None, "device": None}
    return {
        "backend": backend.name,
        "precision": backend.precision,
        "device": backend.device,
    }


def analyze(
    backend,
    scorer: ConfidenceScorer,
    image_path: str | Path,
    filename: str = "upload.jpg",
) -> dict:
    """Run one receipt end to end: generate, repair, normalize, score confidence.

    Backend-agnostic - both stacks return (raw_text, [(chunk_text, logprob), ...]) whose
    chunk texts concatenate to raw_text, which is what the per-field logprob spans need.
    """

    raw, chunks = backend.generate_with_logprobs(image_path)
    parsed, status = repair_json(raw)
    record = normalize(parsed)
    record["_field_logprobs"] = {
        f: field_avg_logprob(f, raw, chunks) for f in SCALAR_FIELDS
    }
    record["_line_item_logprobs"] = [
        line_item_avg_logprob(i, raw, chunks)
        for i in range(len(record.get("line_items") or []))
    ]
    return {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "filename": filename,
        "prediction": {k: record.get(k) for k in PREDICTION_FIELDS},
        "confidence": scorer.field_confidence_live(record),
        "repair_status": status,
    }
