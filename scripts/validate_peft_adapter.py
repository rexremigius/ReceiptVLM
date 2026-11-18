"""Accuracy gate: does the converted adapter still work on a transformers base?

The MLX run trained by QLoRA against a 4-bit base, so replaying it elsewhere is not
bit-identical by construction. Scores eval.py's micro-F1 against the MLX run on the
same receipts. Nothing downstream is worth building until this passes.

Image sizing, decoding and prompt are matched to the MLX run; peft silently ignores
state-dict keys that match no module, so a non-zero lora_B after loading is the
check that the trained weights actually landed.

Usage:
    python scripts/validate_peft_adapter.py --adapter checkpoints/final_peft --limit 30
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from src.eval import bootstrap_micro_f1, evaluate, paired_bootstrap_test
from src.repair import repair_json
from src.schema import PROMPT
from src.zeroshot import normalize

DATA_ROOT = REPO_ROOT / "data" / "wildreceipt"
PROC_ROOT = REPO_ROOT / "data" / "processed"
HF_BASE_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"

# Production MLX run (full 472-receipt test split) used as the paired reference. Scoring
# is restricted to whichever ids this run covers, so the comparison stays apples-to-apples.
DEFAULT_REFERENCE = PROC_ROOT / "finetuned_test.jsonl"


def fit_within(img: Image.Image, max_w: int, max_h: int) -> Image.Image:
    """Port of mlx_vlm.utils.resize_image: scale to fit the box, aspect preserved.

    Deliberately does not clamp ratio to <= 1, matching upstream: a receipt smaller than
    the box is upscaled, which is what the checkpoint was trained on.
    """

    ratio = min(max_w / img.width, max_h / img.height)
    return img.resize((int(img.width * ratio), int(img.height * ratio)))


def pick_dtype_and_device(device_arg: str, dtype_arg: str) -> tuple[torch.dtype, str]:
    if device_arg == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = device_arg

    if dtype_arg == "auto":
        dtype = torch.float32 if device == "cpu" else torch.float16
    else:
        dtype = getattr(torch, dtype_arg)
    return dtype, device


def count_lora_modules(model) -> tuple[int, int]:
    """Return (n modules carrying lora_A, n whose lora_B is non-zero).

    peft zero-initializes lora_B, so n_loaded < n_injected means some trained weights
    never landed - the silent-no-op failure this gate exists to catch.
    """

    n_injected = n_loaded = 0
    for _, module in model.named_modules():
        lora_A = getattr(module, "lora_A", None)
        lora_B = getattr(module, "lora_B", None)
        if lora_A is None or lora_B is None or not hasattr(lora_A, "keys"):
            continue
        n_injected += 1
        if any(float(p.detach().abs().max()) > 0 for p in lora_B.parameters()):
            n_loaded += 1
    return n_injected, n_loaded


def load_model(
    base: str,
    adapter: Path | None,
    dtype: torch.dtype,
    device: str,
    expect_modules: int,
    load_4bit: bool = False,
):
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    kwargs = {"dtype": dtype}
    if load_4bit:
        # An adapter is only meaningful against the base it was fit to. The MLX adapter
        # was trained on a 4-bit base, and scoring it on fp16 is what produced the 0.288
        # in this file's header - so a retrain done in NF4 has to be scored in NF4.
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )

    print(f"loading {base} ({dtype}, {device}" f"{', nf4' if load_4bit else ''})")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(base, **kwargs)
    processor = AutoProcessor.from_pretrained(base)

    if adapter is not None:
        from peft import PeftModel

        print(f"applying adapter {adapter}")
        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
        n_injected, n_loaded = count_lora_modules(model)
        print(
            f"  lora modules injected: {n_injected}   with trained weights: {n_loaded}"
        )
        if n_injected == 0:
            raise SystemExit(
                "No LoRA modules were injected. target_modules in the adapter config "
                "matched nothing - most likely the wrong layout. Re-run the converter "
                "with the other --layout and compare against this model's paths:\n  "
                + "\n  ".join(
                    sorted(
                        {
                            n.rsplit(".", 1)[0]
                            for n, _ in model.named_modules()
                            if n.endswith("self_attn.q_proj")
                        }
                    )[:4]
                )
            )
        if n_loaded != n_injected:
            raise SystemExit(
                f"{n_injected - n_loaded} of {n_injected} injected modules still have a "
                "zero lora_B, meaning peft did not load their trained weights. The "
                "tensor keys in adapter_model.safetensors do not line up with the "
                "injected module paths."
            )
        if n_injected != expect_modules:
            print(
                f"  WARNING: expected {expect_modules} modules, found {n_injected}. "
                "target_modules may be matching more (or less) than the MLX run did."
            )

    if load_4bit:
        # bitsandbytes already placed the quantized weights; .to() on a 4-bit model
        # raises.
        model.eval()
    else:
        model.to(device).eval()
    return model, processor


def build_inputs(processor, image: Image.Image, device: str):
    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": PROMPT}],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    return {k: v.to(device) for k, v in inputs.items()}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--adapter",
        type=Path,
        default=Path("checkpoints/final_peft"),
        help="peft adapter dir; pass --no-adapter to score the base model",
    )
    ap.add_argument(
        "--no-adapter",
        action="store_true",
        help="skip the adapter (zero-shot sanity run)",
    )
    ap.add_argument("--base-model", default=HF_BASE_MODEL)
    ap.add_argument("--split", choices=["train", "test"], default="test")
    ap.add_argument(
        "--limit",
        type=int,
        default=30,
        help="receipts to score; 30 is enough to see a collapse",
    )
    ap.add_argument(
        "--image-resize",
        type=int,
        nargs=2,
        default=[768, 1024],
        help="must match train.py's default",
    )
    ap.add_argument("--max-new-tokens", type=int, default=1536)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    ap.add_argument(
        "--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"]
    )
    ap.add_argument(
        "--load-4bit",
        action="store_true",
        help="load the base in NF4 via bitsandbytes (CUDA only). Use this to "
        "score an adapter that was trained in NF4, e.g. the output of "
        "notebooks/kaggle_retrain_qlora.ipynb - scoring it on fp16 "
        "repeats the base-mismatch this script's header documents.",
    )
    ap.add_argument(
        "--expect-modules",
        type=int,
        default=252,
        help="LoRA module count the MLX adapter carried",
    )
    ap.add_argument(
        "--reference",
        type=Path,
        default=DEFAULT_REFERENCE,
        help="MLX prediction jsonl to compare against",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="where to write predictions (default: data/processed/peft_<split>.subset<N>.jsonl)",
    )
    args = ap.parse_args()

    gt_path = PROC_ROOT / f"{args.split}.jsonl"
    if not gt_path.exists():
        raise SystemExit(f"Missing ground truth {gt_path}; run src/prep.py first.")
    gold = {
        json.loads(l)["image_id"]: json.loads(l) for l in gt_path.open() if l.strip()
    }
    image_ids = list(gold)[: args.limit] if args.limit else list(gold)

    missing = [i for i in image_ids if not (DATA_ROOT / i).exists()]
    if missing:
        raise SystemExit(
            f"{len(missing)} images missing under {DATA_ROOT}, "
            f"e.g. {missing[0]}. Download WildReceipt first."
        )

    dtype, device = pick_dtype_and_device(args.device, args.dtype)
    if args.load_4bit and device != "cuda":
        raise SystemExit(
            f"--load-4bit needs CUDA; this resolved to device={device!r}. bitsandbytes "
            "has no MPS backend, so an NF4-trained adapter cannot be scored on Apple "
            "Silicon - run this on the same GPU host you trained on."
        )
    adapter = None if args.no_adapter else args.adapter
    if adapter is not None and not adapter.exists():
        raise SystemExit(
            f"Missing adapter {adapter}; "
            "run scripts/convert_mlx_adapter_to_peft.py first."
        )
    model, processor = load_model(
        args.base_model,
        adapter,
        dtype,
        device,
        args.expect_modules,
        load_4bit=args.load_4bit,
    )

    max_w, max_h = args.image_resize
    records, parse_failures = [], 0
    repair_counts: dict[str, int] = {}
    t_start = time.time()

    for i, image_id in enumerate(image_ids):
        img = Image.open(DATA_ROOT / image_id).convert("RGB")
        img = fit_within(img, max_w, max_h)
        inputs = build_inputs(processor, img, device)

        t0 = time.time()
        with torch.inference_mode():
            out = model.generate(
                **inputs, max_new_tokens=args.max_new_tokens, do_sample=False
            )
        raw = processor.batch_decode(
            out[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )[0]

        parsed, status = repair_json(raw)
        repair_counts[status] = repair_counts.get(status, 0) + 1
        if parsed is None:
            parse_failures += 1
        records.append({"image_id": image_id, **normalize(parsed)})
        print(
            f"  [{i + 1}/{len(image_ids)}] {time.time() - t0:.1f}s  "
            f"({len(raw)} chars, {status})  {image_id}",
            flush=True,
        )

    elapsed = time.time() - t_start
    print(
        f"\n{len(records)} receipts in {elapsed:.0f}s "
        f"({elapsed / max(len(records), 1):.1f}s each, {parse_failures} parse failures)"
    )
    print(f"repair breakdown: {repair_counts}")

    out_path = args.out or (
        PROC_ROOT / f"peft_{args.split}"
        f"{f'.subset{args.limit}' if args.limit else ''}.jsonl"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"wrote {out_path}")

    # eval.evaluate returns (per_field, micro, n_ids) and every prf value is a
    # (precision, recall, f1) tuple.
    pred = {r["image_id"]: r for r in records}
    scored_gold = {k: gold[k] for k in pred}
    per_field, micro, n_scored = evaluate(scored_gold, pred)
    lo, hi = bootstrap_micro_f1(scored_gold, pred, n=1000)
    print(
        f"\n{'PEFT / fp16':<24} micro-F1 {micro[2]:.3f}  "
        f"95% CI [{lo:.3f}, {hi:.3f}]  (n={n_scored})"
    )
    for field, (p, r, f1) in sorted(per_field.items()):
        print(f"  {field:<14} P {p:.3f}  R {r:.3f}  F1 {f1:.3f}")

    if not args.reference or not args.reference.exists():
        print(f"\nNo reference at {args.reference}; skipped paired comparison.")
        return

    ref = {
        json.loads(l)["image_id"]: json.loads(l)
        for l in args.reference.open()
        if l.strip()
    }
    shared = [i for i in pred if i in ref]
    if not shared:
        print(
            f"\n{args.reference.name} shares no receipts with this run; "
            "skipped paired comparison."
        )
        return

    ref_sub = {i: ref[i] for i in shared}
    pred_sub = {i: pred[i] for i in shared}
    gold_sub = {i: gold[i] for i in shared}
    _, ref_micro, _ = evaluate(gold_sub, ref_sub)
    _, own_micro, _ = evaluate(gold_sub, pred_sub)
    # paired_bootstrap_test reports pred_b - pred_a, so the MLX run goes in as A to make
    # a positive delta mean "the port is better".
    result = paired_bootstrap_test(gold_sub, ref_sub, pred_sub, n=1000)

    print(f"\nPaired against {args.reference.name} on {len(shared)} shared receipts:")
    print(f"  MLX reference  micro-F1 {ref_micro[2]:.3f}")
    print(f"  PEFT / fp16    micro-F1 {own_micro[2]:.3f}")
    print(
        f"  delta {result['mean_diff']:+.3f}  "
        f"95% CI [{result['ci'][0]:+.3f}, {result['ci'][1]:+.3f}]  "
        f"p={result['p_approx']:.3f}"
    )

    regressed = result["mean_diff"] < 0 and result["p_approx"] < 0.05
    print(
        "\n"
        + (
            "GATE FAIL: significant regression - inspect before porting the UI."
            if regressed
            else "GATE PASS: no significant regression vs. the MLX run."
        )
    )


if __name__ == "__main__":
    main()
