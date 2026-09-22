"""Transformers/CUDA inference backend, used wherever mlx_vlm is unavailable.

Matches the MLX run on the three things that silently degrade output if wrong:
aspect-preserving 768x1024 sizing, greedy decoding with no repetition penalty, and
(chunk_text, logprob) pairs whose text concatenates to the completion exactly.
fp16 rather than 4-bit; bitsandbytes NF4 is slower on CUDA and scored lower.
"""

from __future__ import annotations

import datetime
import os
from pathlib import Path

import pillow_heif
import torch
from PIL import Image

pillow_heif.register_heif_opener()  # stock Pillow has no HEIC decoder at all

from src.eval import SCALAR_FIELDS
from src.pipeline import PREDICTION_FIELDS, ConfidenceScorer
from src.repair import repair_json
from src.schema import PROMPT
from src.zeroshot import field_avg_logprob, line_item_avg_logprob, normalize

REPO_ROOT = Path(__file__).resolve().parent.parent

HF_BASE_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_ADAPTER = REPO_ROOT / "checkpoints" / "final_peft"

# Must match train.py's --image-resize default; the production checkpoint was trained here.
IMAGE_RESIZE = (768, 1024)
MAX_NEW_TOKENS = 1536


def fit_within(img: Image.Image, max_w: int, max_h: int) -> Image.Image:
    """Port of mlx_vlm.utils.resize_image: scale to fit the box, aspect preserved.

    Does not clamp the ratio to <= 1, matching upstream - a receipt smaller than the box
    gets upscaled.
    """

    ratio = min(max_w / img.width, max_h / img.height)
    return img.resize((int(img.width * ratio), int(img.height * ratio)))


def _pick_device_dtype() -> tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        return "cuda", torch.float16
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


class ReceiptModel:
    """The fine-tuned extractor, loaded once and reused.

    ZeroGPU requires the model to be placed on cuda at module import rather than lazily
    inside the @spaces.GPU function - outside that function a CUDA emulation mode is
    active, and placements made during startup are what get optimized. So callers should
    construct and load this at module scope.
    """

    def __init__(
        self,
        base_model: str = HF_BASE_MODEL,
        adapter_path: Path | str | None = DEFAULT_ADAPTER,
        load_4bit: bool | None = None,
    ) -> None:
        self.base_model = base_model
        self.adapter_path = Path(adapter_path) if adapter_path else None
        self.device, self.dtype = _pick_device_dtype()
        # An adapter only works against the base it was fit to - that is the whole
        # lesson of the fp16 transfer failure (see scripts/validate_peft_adapter.py's
        # header). An adapter retrained in NF4 must be served in NF4, so this is
        # switchable, defaulting from the environment so a Space can set it as a variable.
        if load_4bit is None:
            load_4bit = bool(os.environ.get("RECEIPTVLM_LOAD_4BIT"))
        self.load_4bit = load_4bit
        self.model = None
        self.processor = None

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def load(self) -> "ReceiptModel":
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        kwargs = {"dtype": self.dtype}
        if self.load_4bit:
            if self.device != "cuda":
                raise RuntimeError(
                    "RECEIPTVLM_LOAD_4BIT is set but no CUDA device is available; "
                    "bitsandbytes has no MPS backend."
                )
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=self.dtype,
            )

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.base_model, **kwargs
        )
        self.processor = AutoProcessor.from_pretrained(self.base_model)

        if self.adapter_path is not None:
            if not self.adapter_path.exists():
                raise FileNotFoundError(
                    f"No adapter at {self.adapter_path}. Run "
                    "scripts/convert_mlx_adapter_to_peft.py to produce it from the MLX "
                    "checkpoint, or pass adapter_path=None to serve the base model."
                )
            from peft import PeftModel

            model = PeftModel.from_pretrained(
                model, str(self.adapter_path), is_trainable=False
            )
            n_injected = sum(
                1
                for _, m in model.named_modules()
                if hasattr(getattr(m, "lora_A", None), "keys")
            )
            if n_injected == 0:
                raise RuntimeError(
                    "The adapter injected no LoRA modules, so this is the base model "
                    "wearing a no-op adapter. target_modules in adapter_config.json does "
                    "not match this transformers version's layout - re-run the converter "
                    "with the other --layout."
                )

        # bitsandbytes has already placed the quantized weights; .to() on a 4-bit model
        # raises.
        self.model = model.eval() if self.load_4bit else model.to(self.device).eval()
        return self

    def generate_with_logprobs(self, image: Image.Image) -> tuple[str, list]:
        """Return (raw_text, [(chunk_text, token_logprob), ...]).

        The concatenation of chunk texts equals raw_text exactly, which is the invariant
        zeroshot.field_avg_logprob relies on to map character spans back to tokens.
        """

        if not self.loaded:
            raise RuntimeError("ReceiptModel.load() has not been called.")

        image = fit_within(image.convert("RGB"), *IMAGE_RESIZE)
        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": PROMPT}],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=[text], images=[image], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
            )

        prompt_len = inputs["input_ids"].shape[1]
        generated = out.sequences[0, prompt_len:]

        # Trim at the first EOS so the terminator does not contribute an empty chunk with
        # a logprob attached, which would perturb the trailing field's average.
        tokenizer = self.processor.tokenizer
        eos_ids = {
            tokenizer.eos_token_id,
            *(tokenizer.convert_tokens_to_ids(t) for t in ("<|im_end|>",)),
        }
        eos_ids.discard(None)
        keep = len(generated)
        for i, tok in enumerate(generated.tolist()):
            if tok in eos_ids:
                keep = i
                break

        chunks: list[tuple[str, float]] = []
        for step in range(keep):
            token_id = int(generated[step])
            # Greedy decoding with no logits processors means scores are the raw logits.
            step_logprobs = torch.log_softmax(out.scores[step][0].float(), dim=-1)
            piece = tokenizer.decode([token_id], skip_special_tokens=True)
            chunks.append((piece, float(step_logprobs[token_id])))

        return "".join(p for p, _ in chunks), chunks


def analyze(
    model: ReceiptModel,
    scorer: ConfidenceScorer,
    image: Image.Image,
    filename: str = "upload.jpg",
) -> dict:
    """Run one receipt end to end: generate, repair, normalize, score confidence.

    Mirrors serve.py's /infer, including the 3-signal confidence path, but returns a
    plain dict and takes the caller's scorer so session state stays with the caller.
    """

    raw, chunks = model.generate_with_logprobs(image)
    parsed, status = repair_json(raw)
    record = normalize(parsed)
    record["_field_logprobs"] = {
        f: field_avg_logprob(f, raw, chunks) for f in SCALAR_FIELDS
    }
    record["_line_item_logprobs"] = [
        line_item_avg_logprob(i, raw, chunks)
        for i in range(len(record.get("line_items") or []))
    ]
    prediction = {k: record.get(k) for k in PREDICTION_FIELDS}
    return {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "filename": filename,
        "prediction": prediction,
        "confidence": scorer.field_confidence_live(record),
        "repair_status": status,
    }
