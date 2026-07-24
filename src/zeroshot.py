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
import time
from pathlib import Path

from mlx_vlm import generate, load, stream_generate
from mlx_vlm.prompt_utils import apply_chat_template

from train import DEFAULT_MODEL, PROMPT, SCHEMA_KEYS  # reuse train.py's exact prompt/schema
from repair import repair_json  # #9 — extraction + trailing-comma/literal/truncation fixups

DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "wildreceipt"
OUT_ROOT = Path(__file__).resolve().parent.parent / "data" / "processed"
SCALAR_KEYS = [k for k in SCHEMA_KEYS if k != "line_items"]


def normalize(parsed: dict | None) -> dict:
    if parsed is None:
        return {k: None for k in SCALAR_KEYS} | {"line_items": []}
    record = {k: parsed.get(k) for k in SCALAR_KEYS}
    items = parsed.get("line_items")
    record["line_items"] = items if isinstance(items, list) else []
    return record


def generate_with_logprobs(model, processor, prompt, image, **kwargs) -> tuple[str, list]:
    """Like mlx_vlm.generate(), but also keeps the per-step (text_chunk, token_logprob)
    pairs that plain generate() discards — #8's third confidence signal (token logprob)
    needs them, and there's no way to recover them after the fact from just the final
    text. `stream_generate`'s GenerationResult already carries a full-vocab logprob
    vector plus which token was actually chosen at each step; this just keeps
    `logprobs[token]` (the log-probability of what the model actually emitted) instead
    of throwing it away like `generate()` does.
    """
    import numpy as np
    chunks = []
    for response in stream_generate(model, processor, prompt, image=image, **kwargs):
        # mx.array shapes/dtypes for `token`/`logprobs` are an internal mlx_vlm detail
        # (int vs 0-d array, (vocab,) vs (1,vocab)) that shifted across versions in
        # practice — flattening through numpy sidesteps guessing the exact shape.
        vocab_logprobs = np.asarray(response.logprobs).reshape(-1)
        token_id = int(np.asarray(response.token).reshape(-1)[0])
        chunks.append((response.text, float(vocab_logprobs[token_id])))
    return "".join(c for c, _ in chunks), chunks


def field_avg_logprob(field: str, text: str, chunks: list) -> float | None:
    """Average token logprob over the characters spanning `field`'s value in the raw
    JSON text — a coarse (chunk-level, not exact-token-level) but simple mapping from
    the streamed per-step logprobs back to one specific field's value. Returns None if
    the field's key isn't found in the raw text at all (e.g. the model never emitted it).
    """
    key_pos = text.find(f'"{field}"')
    if key_pos == -1:
        return None
    colon = text.find(":", key_pos)
    if colon == -1:
        return None
    start = colon + 1
    while start < len(text) and text[start] in " \t\n":
        start += 1

    depth, in_string, escape, end = 0, False, False, len(text)
    i = start
    while i < len(text):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
        elif ch == "," and depth == 0:
            end = i
            break
        i += 1

    pos, logprobs = 0, []
    for chunk_text, lp in chunks:
        chunk_start, chunk_end = pos, pos + len(chunk_text)
        if chunk_end > start and chunk_start < end:
            logprobs.append(lp)
        pos = chunk_end
    return sum(logprobs) / len(logprobs) if logprobs else None


def line_item_spans(text: str) -> list[tuple[int, int]]:
    """Character (start, end) spans of each `{...}` object inside the top-level
    `"line_items"` array in the raw JSON text — the array-element analog of
    `field_avg_logprob`'s single-value span-finding, needed because a line item isn't
    one value at one position, it's a whole sub-object. Returns spans in array order,
    so `spans[i]` lines up with `parsed["line_items"][i]` as long as the model didn't
    truncate mid-array (repair.py's job, not this one's).
    """
    key_pos = text.find('"line_items"')
    if key_pos == -1:
        return []
    arr_start = text.find("[", key_pos)
    if arr_start == -1:
        return []

    spans = []
    depth, in_string, escape, obj_start = 0, False, False, None
    i = arr_start + 1
    while i < len(text):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                obj_start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and obj_start is not None:
                spans.append((obj_start, i + 1))
                obj_start = None
        elif ch == "]" and depth == 0:
            break
        i += 1
    return spans


def line_item_avg_logprob(idx: int, text: str, chunks: list) -> float | None:
    """Average token logprob over the characters spanning the `idx`-th line item's
    `{...}` object in the raw JSON text — #8's third signal, extended one level down
    from per-field to per-item (see PROGRESS.md 2026-07-26). None if the array wasn't
    found or has fewer than `idx + 1` objects (e.g. truncated generation).
    """
    spans = line_item_spans(text)
    if idx >= len(spans):
        return None
    start, end = spans[idx]
    pos, logprobs = 0, []
    for chunk_text, lp in chunks:
        chunk_start, chunk_end = pos, pos + len(chunk_text)
        if chunk_end > start and chunk_start < end:
            logprobs.append(lp)
        pos = chunk_end
    return sum(logprobs) / len(logprobs) if logprobs else None


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
    # 512 truncated mid-JSON on receipts with many line items (p99 is 21 items, max is 50;
    # 50 items alone need ~500-600 tokens before scalar fields/JSON structure even start).
    # 1536 covers the observed max with headroom.
    ap.add_argument("--max-tokens", type=int, default=1536)
    # NB: tried repetition_penalty=1.3 to fight a repeat-loop degeneration on illegible
    # long receipts — it wrecked accuracy across the board (micro-F1 0.525 -> 0.069, parse
    # failures 10 -> 236/472) because JSON syntax is *inherently* repetitive (every line
    # item repeats `{"name":`/`"price":`/`},{`, every receipt repeats the same field-name
    # keys) — penalizing repeated tokens fights the correct structure, not just the
    # pathological content loop. Do not re-enable for this task without a much lower
    # value and a smaller context window tested on a subset first.
    ap.add_argument("--repetition-penalty", type=float, default=None)
    ap.add_argument("--repetition-context-size", type=int, default=20)
    # must match train.py's default — the production checkpoint was trained at 768x1024
    ap.add_argument("--image-resize", type=int, nargs=2, default=[768, 1024])
    ap.add_argument("--capture-logprobs", action="store_true",
                    help="record per-field average token logprob (#8's 3rd confidence "
                         "signal) alongside each prediction; off by default since it "
                         "changes nothing about the prediction itself, only adds "
                         "bookkeeping most callers (#5/#6) don't need")
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
    repair_counts = {"clean": 0, "repaired_trailing_comma": 0,
                      "repaired_python_literal": 0, "repaired_truncation": 0,
                      "hard_failure": 0}
    t_start = time.time()
    for i, image_id in enumerate(image_ids):
        gen_kwargs = dict(
            max_tokens=args.max_tokens, temperature=0.0, resize_shape=resize_shape,
            repetition_penalty=args.repetition_penalty,
            repetition_context_size=args.repetition_context_size,
        )
        t_receipt = time.time()
        if args.capture_logprobs:
            raw, chunks = generate_with_logprobs(
                model, processor, prompt, image=str(DATA_ROOT / image_id), **gen_kwargs)
        else:
            raw = generate(model, processor, prompt, image=str(DATA_ROOT / image_id),
                           verbose=False, **gen_kwargs)
        print(f"  [{i + 1}/{len(image_ids)}] {time.time() - t_receipt:.1f}s  "
              f"({len(raw)} chars raw)  {image_id}", flush=True)
        parsed, status = repair_json(raw)
        repair_counts[status] += 1
        if parsed is None:
            parse_failures += 1
        record = {"image_id": image_id, **normalize(parsed)}
        if args.capture_logprobs:
            record["_field_logprobs"] = {f: field_avg_logprob(f, raw, chunks)
                                         for f in SCALAR_KEYS}
            record["_line_item_logprobs"] = [
                line_item_avg_logprob(i, raw, chunks) for i in range(len(record["line_items"]))
            ]
        records.append(record)
    print(f"{len(image_ids)}/{len(image_ids)} done in {time.time() - t_start:.0f}s "
          f"({parse_failures} parse failures)")

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
        "repair_breakdown": repair_counts,
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
