"""Deliverable #3 — zero-shot VLM baseline (no LoRA adapter), Track B.

Runs the base model (same prompt/schema `train.py` fine-tunes towards) on WildReceipt
test-split receipts, so there's a real "before fine-tuning" number to compare the QLoRA
checkpoint against in eval.py. Output is the same schema/JSONL convention as prep.py (#1)
and baseline.py (#2) — a prediction file eval.py can consume identically regardless of
which pipeline produced it. See CLAUDE.md / SKILL.md #3.

Also doubles as the fine-tuned-model prediction generator (--adapter-path): same prompt,
same generation code, so zero-shot and fine-tuned predictions are directly comparable —
only difference is whether LoRA adapters are applied on load.

Usage:
    python src/zeroshot.py --limit 20                                  # quick subset check
    python src/zeroshot.py                                             # full test split, base model
    python src/zeroshot.py --adapter-path checkpoints/final --tag finetuned  # fine-tuned predictions
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template

from train import DEFAULT_MODEL, PROMPT, SCHEMA_KEYS  # reuse train.py's exact prompt/schema

DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "wildreceipt"
OUT_ROOT = Path(__file__).resolve().parent.parent / "data" / "processed"
SCALAR_KEYS = [k for k in SCHEMA_KEYS if k != "line_items"]

CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def extract_json(text: str) -> dict | None:
    """Pull a JSON object out of the model's raw completion.

    Zero-shot models reliably wrap JSON in a ```json fence even when asked not to;
    fall back to the first {...} span if there's no fence, and give up (rather than
    guess) on anything that still doesn't parse — that's #9's job (JSON repair layer),
    not this script's.
    """
    fenced = CODE_FENCE_RE.search(text)
    candidate = fenced.group(1) if fenced else text
    start, end = candidate.find("{"), candidate.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        return json.loads(candidate[start:end + 1])
    except json.JSONDecodeError:
        return None


def normalize(parsed: dict | None) -> dict:
    if parsed is None:
        return {k: None for k in SCALAR_KEYS} | {"line_items": []}
    record = {k: parsed.get(k) for k in SCALAR_KEYS}
    items = parsed.get("line_items")
    record["line_items"] = items if isinstance(items, list) else []
    return record


def load_image_ids(split: str, limit: int | None) -> list[str]:
    suffix = f".subset{limit}" if limit else ""
    exact = OUT_ROOT / f"{split}{suffix}.jsonl"
    src = exact if exact.exists() else OUT_ROOT / f"{split}.jsonl"
    with src.open() as f:
        records = [json.loads(line) for line in f if line.strip()]
    if limit:
        records = records[:limit]
    return [r["image_id"] for r in records]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], default="test")
    ap.add_argument("--limit", type=int, default=None, help="cap receipts processed")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--adapter-path", default=None,
                     help="path to a trained LoRA adapter dir (e.g. checkpoints/final); "
                          "omit for the zero-shot base model")
    ap.add_argument("--tag", default=None,
                     help="output file tag; defaults to 'finetuned' if --adapter-path is "
                          "set, else 'zeroshot'")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--image-resize", type=int, nargs=2, default=[448, 448])
    args = ap.parse_args()
    tag = args.tag or ("finetuned" if args.adapter_path else "zeroshot")

    image_ids = load_image_ids(args.split, args.limit)
    print(f"loading {args.model}" + (f" + adapter {args.adapter_path}" if args.adapter_path else ""))
    model, processor = load(args.model, adapter_path=args.adapter_path,
                             processor_config={"trust_remote_code": True})
    config = model.config.__dict__
    prompt = apply_chat_template(processor, config, PROMPT, num_images=1)
    resize_shape = tuple(args.image_resize)

    records, parse_failures = [], 0
    t_start = time.time()
    for i, image_id in enumerate(image_ids):
        raw = generate(
            model, processor, prompt, image=str(DATA_ROOT / image_id),
            max_tokens=args.max_tokens, temperature=0.0, resize_shape=resize_shape,
            verbose=False,
        )
        parsed = extract_json(raw)
        if parsed is None:
            parse_failures += 1
        record = {"image_id": image_id, **normalize(parsed)}
        records.append(record)
        if (i + 1) % 10 == 0 or i == len(image_ids) - 1:
            elapsed = time.time() - t_start
            print(f"{i + 1}/{len(image_ids)} ({elapsed:.0f}s, "
                  f"{parse_failures} parse failures so far)")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    suffix = f".subset{args.limit}" if args.limit else ""
    out_path = OUT_ROOT / f"{tag}_{args.split}{suffix}.jsonl"
    with out_path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    field_present = {k: sum(1 for r in records if r[k]) for k in SCALAR_KEYS}
    li_counts = [len(r["line_items"]) for r in records]
    summary = {
        "split": args.split,
        "model": args.model,
        "adapter_path": args.adapter_path,
        "receipts": len(records),
        "parse_failures": parse_failures,
        "field_present": field_present,
        "field_missing": {k: len(records) - v for k, v in field_present.items()},
        "line_items_total": sum(li_counts),
        "line_items_per_receipt_avg": round(sum(li_counts) / max(len(records), 1), 2),
        "receipts_with_no_line_items": sum(1 for c in li_counts if c == 0),
        "elapsed_seconds": round(time.time() - t_start, 1),
    }
    summ_path = OUT_ROOT / f"_{tag}_summary_{args.split}{suffix}.json"
    summ_path.write_text(json.dumps(summary, indent=2))
    print(f"\n=== {args.split} -> {out_path} ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
