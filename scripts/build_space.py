"""Assembles the deployable Hugging Face Space from this repo.

The repo is not itself a Space: Spaces want a `requirements.txt` and a README with SDK
frontmatter at the root, and this repo's versions of both describe the MLX stack
(transformers==4.49.0, `uvicorn`/`streamlit` run instructions). Rather than fork the
project or break the on-device setup docs, this copies the subset a Space actually needs
into a staging directory that can be pushed to the Space remote as its own history.

What it deliberately leaves out:

  * The MLX pipeline modules (train/prep/baseline/quantize/sweep/taxonomy/serve). Nothing
    the Space does touches them and they would drag mlx and datasets into the image.
  * The full WildReceipt dataset. Only the images for the receipts app.py actually offers
    as samples are copied, which is a few MB rather than 179.
  * checkpoints/final (the MLX adapter). The Space needs the converted peft adapter.

Usage:
    python scripts/convert_mlx_adapter_to_peft.py        # if not done yet
    python scripts/build_space.py --out build/space
    cd build/space && git init && git add -A && git commit -m "ReceiptVLM Space"
    git remote add origin https://huggingface.co/spaces/<user>/receiptvlm
    git push -u origin main
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from src import samples
from src.pipeline import load_jsonl

# Only what app.py's import graph reaches. zeroshot.py is included for its field/line-item
# logprob span walkers (confidence.py's third signal) -- its mlx import is guarded, and it
# no longer pulls in train.py now that the prompt lives in schema.py.
SRC_MODULES = [
    "backend_hf.py",
    "categorize.py",
    "confidence.py",
    "display_format.py",
    "eval.py",
    "pipeline.py",
    "render.py",
    "repair.py",
    "samples.py",
    "schema.py",
    "zeroshot.py",
]

# confidence.run(tag, split) reads {tag}_{split}.jsonl plus the {split}.jsonl ground truth,
# for both the 2-signal and 3-signal calibrations.
DATA_FILES = [
    "finetuned_test.jsonl",
    "finetuned_logprob_test.jsonl",
    "test.jsonl",
]

README_TEMPLATE = """---
title: ReceiptVLM
emoji: 🧾
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: {sdk_version}
python_version: "3.12.12"
app_file: app.py
pinned: false
license: mit
short_description: Receipt photo to structured JSON, with calibrated confidence
---

# ReceiptVLM

Turn a photo of a receipt into structured data: merchant, date, tax, tip, subtotal,
total, and the individual line items, each with a calibrated confidence score.

Extraction is done by **Qwen2.5-VL-3B fine-tuned with QLoRA** on the WildReceipt
dataset. Built as a CS6140 (Machine Learning) project at Northeastern University.

## Results

Per-field micro-F1 on the 472-receipt WildReceipt test set (95% bootstrap CI):

| Model | micro-F1 |
|---|---|
| OCR + regex baseline (Tesseract) | 0.115 |
| Qwen2.5-VL-3B zero-shot | 0.212 |
| **Qwen2.5-VL-3B + QLoRA (ours)** | **0.781** [0.756, 0.802] |

Fine-tuning beats zero-shot ~3.7x and the OCR baseline ~6.8x (paired bootstrap, p~0).

## About this Space

The adapter was trained on Apple Silicon through MLX against a 4-bit base, then converted
to peft format for CUDA (`scripts/convert_mlx_adapter_to_peft.py` in the source repo).
This Space runs it in fp16 on ZeroGPU -- the INT4 quantization in the original project was
a memory/latency tradeoff for on-device use, which a 48 GB GPU slice does not need.

**Upload** a receipt photo to run the model, or pick one of the bundled **sample
receipts** to see cached predictions without using GPU quota. The spending overview
aggregates only what you upload in your own session.

Confidence dots combine token logprob, arithmetic consistency
(`subtotal + tax + tip ~ total`) and format validity, Platt-calibrated on a held-out
split. Grey means the model abstained rather than scored low -- hover any dot for the
number.
"""

GITATTRIBUTES = "*.safetensors filter=lfs diff=lfs merge=lfs -text\n"


def copy_samples(out: Path, max_samples: int) -> list[str]:
    """Copy images for exactly the receipts app.py will offer."""

    predictions = load_jsonl(REPO_ROOT / "data" / "processed" / "finetuned_test.jsonl")
    image_root = REPO_ROOT / "data" / "wildreceipt"
    chosen = samples.pick(predictions, image_root, max_samples)
    if not chosen:
        print("  WARNING: no sample images found under data/wildreceipt -- the Space "
              "will offer no samples. Download WildReceipt first if you want them.")
    for image_id in chosen:
        dest = out / "data" / "wildreceipt" / image_id
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_root / image_id, dest)
    return chosen


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "build" / "space")
    ap.add_argument("--adapter", type=Path,
                    default=REPO_ROOT / "checkpoints" / "final_peft",
                    help="peft adapter to bundle")
    ap.add_argument("--samples", type=int, default=samples.DEFAULT_MAX_SAMPLES)
    ap.add_argument("--sdk-version", default="5.49.1",
                    help="must match the gradio pin in requirements-spaces.txt")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing output directory")
    args = ap.parse_args()

    out = args.out
    if out.exists():
        if not args.force:
            raise SystemExit(f"{out} already exists; pass --force to overwrite.")
        shutil.rmtree(out)
    out.mkdir(parents=True)

    # Entry point + dependency set under the names Spaces expects.
    shutil.copy2(REPO_ROOT / "app.py", out / "app.py")
    shutil.copy2(REPO_ROOT / "requirements-spaces.txt", out / "requirements.txt")

    (out / "src").mkdir()
    for name in SRC_MODULES:
        shutil.copy2(REPO_ROOT / "src" / name, out / "src" / name)

    (out / "data" / "processed").mkdir(parents=True)
    missing = []
    for name in DATA_FILES:
        src = REPO_ROOT / "data" / "processed" / name
        if src.exists():
            shutil.copy2(src, out / "data" / "processed" / name)
        else:
            missing.append(name)
    if missing:
        raise SystemExit(
            f"Missing prediction/ground-truth files: {missing}. The confidence "
            "calibration is fit from these at startup, so the Space cannot run without "
            "them."
        )

    if not args.adapter.exists():
        raise SystemExit(
            f"No peft adapter at {args.adapter}. Run "
            "scripts/convert_mlx_adapter_to_peft.py first."
        )
    shutil.copytree(args.adapter, out / "checkpoints" / args.adapter.name)

    chosen = copy_samples(out, args.samples)

    (out / "README.md").write_text(
        README_TEMPLATE.format(sdk_version=args.sdk_version))
    (out / ".gitattributes").write_text(GITATTRIBUTES)

    total_bytes = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    n_files = sum(1 for p in out.rglob("*") if p.is_file())
    print(f"Staged Space in {out}")
    print(f"  src modules   {len(SRC_MODULES)}")
    print(f"  data files    {len(DATA_FILES)}")
    print(f"  adapter       {args.adapter.name}")
    print(f"  samples       {len(chosen)}")
    print(f"  total         {n_files} files, {total_bytes / 1e6:.1f} MB")
    print("\nServing precision: fp16 (the default). Do NOT set RECEIPTVLM_LOAD_4BIT --\n"
          "the re-trained adapter scores higher on fp16 than on the NF4 base it was\n"
          "trained on (0.760 vs 0.722 micro-F1), and fp16 keeps bitsandbytes out of the\n"
          "image, which ZeroGPU does not reliably support.")
    print(f"\nSanity-check it before pushing:\n  cd {out} && python app.py")


if __name__ == "__main__":
    main()
