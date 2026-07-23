"""Deliverable #4 — QLoRA fine-tune of Qwen2.5-VL-3B (fallback SmolVLM2-2.2B) via MLX-VLM.

Loads a 4-bit-quantized MLX checkpoint, freezes it, and trains LoRA adapters on top
(mlx_vlm.trainer.LoRaLayer wraps the frozen QuantizedLinear layers directly — that's
QLoRA, not LoRA-then-downgrade). Loss is masked to the assistant's JSON completion only
(train_on_completions), so the model isn't trained to predict the prompt/image tokens.
See CLAUDE.md / SKILL.md #4.

Environment note: requires transformers==4.49.0 (not the latest). Newer transformers
(4.5x) made Qwen2's image processor "fast"-only, which hard-requires PyTorch tensors —
incompatible with mlx_vlm's own tensor backend. See requirements.txt.

Usage:
    python src/train.py --limit 20                  # tiny-subset validation run (spec-required first step)
    python src/train.py                              # full QLoRA fine-tune on data/processed/train.jsonl
    python src/train.py --model mlx-community/SmolVLM2-2.2B-Instruct-4bit  # fallback if 3B has no headroom
"""
from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from datasets import Dataset as HFDataset
from mlx.utils import tree_flatten, tree_map
from mlx_vlm import load
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.trainer import Dataset, Trainer, save_adapter
from mlx_vlm.trainer.utils import find_all_linear_names, get_peft_model
from mlx_vlm.utils import load_image_processor

DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "wildreceipt"
PROC_ROOT = Path(__file__).resolve().parent.parent / "data" / "processed"
CKPT_ROOT = Path(__file__).resolve().parent.parent / "checkpoints"

SCHEMA_KEYS = ["store", "date", "tax", "tip", "subtotal", "total", "line_items"]
DEFAULT_MODEL = "mlx-community/Qwen2.5-VL-3B-Instruct-4bit"
PROMPT = ("Extract the receipt fields as JSON with keys store, date, tax, tip, "
          "subtotal, total, line_items (each {name, price}). Use null for missing "
          "scalar fields and [] for no line items.")


def target_json(record: dict) -> str:
    return json.dumps({k: record[k] for k in SCHEMA_KEYS}, ensure_ascii=False)


def to_example(record: dict) -> dict:
    # content must be a plain string here: mlx_vlm's apply_chat_template (called with
    # return_messages=True in build_dataset) does its own content-list wrapping
    # (image token + text parts) from a bare string. Passing an already-wrapped list
    # gets it wrapped *again*, so the assistant target the model actually saw was a
    # corrupted repr of the wrapper around the JSON, not the clean JSON itself — this
    # was silently training the model on garbage until caught by inspecting raw
    # fine-tuned generations.
    messages = [
        {"role": "user", "content": PROMPT},
        {"role": "assistant", "content": target_json(record)},
    ]
    return {"messages": messages, "images": [str(DATA_ROOT / record["image_id"])]}


def load_split(limit: int | None, val_frac: float, seed: int) -> tuple[list[dict], list[dict]]:
    """Train/val split over data/processed/train.jsonl, at receipt granularity.

    Each JSONL line is already one whole receipt (prep.py grouped boxes by image_id),
    so a plain shuffle-and-slice over records is a receipt-level split, not a box-level
    one — a receipt's line items never end up split across train and val.
    """
    records = [json.loads(line) for line in (PROC_ROOT / "train.jsonl").open() if line.strip()]
    rng = random.Random(seed)
    rng.shuffle(records)
    if limit:
        records = records[:limit]
    n_val = max(1, round(len(records) * val_frac)) if len(records) > 1 else 0
    val, train = records[:n_val], records[n_val:]
    return train, val


def build_dataset(records, processor, config, image_processor, image_resize_shape) -> Dataset:
    hf_ds = HFDataset.from_list([to_example(r) for r in records])

    def process_data(ex):
        ex["messages"] = apply_chat_template(
            config=config, processor=processor, prompt=ex["messages"], return_messages=True
        )
        return ex

    hf_ds = hf_ds.map(process_data)
    return Dataset(hf_ds, config, processor, image_processor=image_processor,
                    image_resize_shape=image_resize_shape)


def resolve_assistant_id(processor, dataset: Dataset) -> int:
    """Token id marking the start of the assistant turn, for completion-only loss masking.

    Computed from the tokenizer rather than hard-coded, since the SmolVLM2 fallback has
    a different vocabulary. Verified against an actual tokenized example — if it's not
    found there, train_on_completions would silently mask nothing and train on the whole
    sequence (prompt included) instead of just the JSON completion.
    """
    tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    candidates = tok.encode("assistant", add_special_tokens=False)
    assistant_id = candidates[-1] if candidates else None
    sample_ids = dataset[0]["input_ids"].tolist()
    flat = sample_ids[0] if isinstance(sample_ids[0], list) else sample_ids
    if assistant_id is None or assistant_id not in flat:
        raise ValueError(
            "Could not locate an 'assistant' role token in a tokenized training example "
            f"(tried token id {assistant_id}). Completion-only loss masking would silently "
            "train on the full prompt instead of just the JSON completion."
        )
    return assistant_id


def guarded_train_step(trainer: Trainer, batch) -> tuple[float, bool]:
    """Trainer.train_step, but skips the optimizer update if loss/grads are NaN/Inf.

    Found by direct reproduction: a specific training image, only at higher resize
    resolutions (768x1024), produces an unusually large vision-tower patch grid
    (74x52=3848 patches vs. 768 at 448x448) that triggers NaN somewhere in the forward
    pass — confirmed the raw preprocessed pixel values themselves are clean (no NaN/Inf),
    so it's a numerical edge case inside the model at that scale, not a data bug. Since
    Adam's moving averages absorb a NaN update permanently, one bad image silently wrecked
    an entire multi-hour run (loss went NaN at step ~6 of 2280 and never recovered). This
    checks loss/grads *before* they reach the optimizer, so a single pathological example
    costs one skipped step instead of the whole run.
    """
    loss_and_grad_fn = nn.value_and_grad(trainer.model, trainer.loss_fn)
    loss, grads = loss_and_grad_fn(trainer.model, batch)
    mx.eval(loss)
    bad = bool(mx.isnan(loss).item() or mx.isinf(loss).item())
    if not bad:
        bad = any(bool(mx.any(mx.isnan(g) | mx.isinf(g)).item())
                   for _, g in tree_flatten(grads))
    if bad:
        return loss, False
    if trainer.clip_gradients is not None:
        grads = tree_map(lambda g: mx.clip(g, -trainer.clip_gradients, trainer.clip_gradients), grads)
    trainer.optimizer.update(trainer.model, grads)
    return loss, True


def val_loss(trainer: Trainer, dataset: Dataset, max_batches: int | None = None) -> float:
    """Mean val loss, excluding NaN/Inf examples (same forward-pass edge case
    guarded_train_step protects against — a single poisoned example previously
    turned the whole aggregate NaN even though the model itself was fine)."""
    n = len(dataset) if max_batches is None else min(max_batches, len(dataset))
    if n == 0:
        return float("nan")
    losses = [trainer.loss_fn(trainer.model, dataset[i]).item() for i in range(n)]
    clean = [v for v in losses if not (math.isnan(v) or math.isinf(v))]
    return sum(clean) / len(clean) if clean else float("nan")


def save_checkpoint(model, step: int, keep_last: int, ckpt_root: Path) -> Path:
    ckpt_root.mkdir(parents=True, exist_ok=True)
    ckpt_dir = ckpt_root / f"step_{step}"
    ckpt_dir.mkdir(exist_ok=True)
    save_adapter(model, ckpt_dir / "adapters.safetensors")

    ckpts = sorted(ckpt_root.glob("step_*"), key=lambda p: int(p.name.split("_")[1]))
    for old in ckpts[:-keep_last]:
        shutil.rmtree(old)
    return ckpt_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--limit", type=int, default=None,
                     help="cap training receipts (tiny-subset validation run)")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--iters", type=int, default=None,
                     help="total optimizer steps; default len(train) * epochs")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-rank", type=int, default=8)
    # NB: mlx_vlm's LoRaLayer applies `alpha` as a raw multiplier on the LoRA update
    # (not the usual alpha/rank scaling), so this needs to stay near their own CLI
    # default (0.1) — alpha=16 (the "usual" convention) blew the loss up to NaN within
    # 15 steps on the tiny-subset validation run.
    ap.add_argument("--lora-alpha", type=float, default=1.0)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    # 448x448 was the original default; receipts with many line items (tall/narrow images,
    # e.g. 1040x1536) got squashed illegibly at that size, causing severe under-extraction
    # of line items (confirmed via taxonomy.py + direct inspection). 768x1024 fixed it:
    # micro-F1 0.525 -> 0.724 on the full test set. Keep this in sync with zeroshot.py's
    # default so training and inference always use the same resolution.
    ap.add_argument("--image-resize", type=int, nargs=2, default=[768, 1024])
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--print-every", type=int, default=5)
    ap.add_argument("--keep-last", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt-root", default=str(CKPT_ROOT),
                     help="where to write checkpoints; override for sweep trials so they "
                          "don't clobber the production checkpoints/final")
    args = ap.parse_args()
    ckpt_root = Path(args.ckpt_root)

    train_records, val_records = load_split(args.limit, args.val_frac, args.seed)
    print(f"train={len(train_records)} val={len(val_records)} receipts, model={args.model}")

    model, processor = load(args.model, processor_config={"trust_remote_code": True})
    config = model.config.__dict__
    # mlx_vlm's trainer.Dataset expects "image_token_index"; Qwen2.5-VL's own config
    # names it "image_token_id" — alias so both names resolve.
    config.setdefault("image_token_index", config.get("image_token_id"))
    image_processor = load_image_processor(args.model)
    resize_shape = tuple(args.image_resize)

    train_ds = build_dataset(train_records, processor, config, image_processor, resize_shape)
    val_ds = (build_dataset(val_records, processor, config, image_processor, resize_shape)
              if val_records else None)

    linear_names = find_all_linear_names(model.language_model)
    model = get_peft_model(model, linear_names, rank=args.lora_rank,
                            alpha=args.lora_alpha, dropout=args.lora_dropout)
    optimizer = optim.Adam(learning_rate=args.lr)
    assistant_id = resolve_assistant_id(processor, train_ds)
    trainer = Trainer(model, optimizer, train_on_completions=True,
                       assistant_id=assistant_id, clip_gradients=1.0)
    model.train()

    total_steps = args.iters or len(train_ds) * args.epochs
    order = list(range(len(train_ds)))
    rng = random.Random(args.seed)
    history = []
    skipped_steps = 0
    t_start = time.time()

    for step in range(total_steps):
        pos = step % len(order)
        if pos == 0:
            rng.shuffle(order)
        loss, applied = guarded_train_step(trainer, train_ds[order[pos]])
        mx.eval(model.parameters(), optimizer.state)
        loss_val = loss.item()
        entry = {"step": step, "loss": loss_val, "applied": applied}
        if not applied:
            skipped_steps += 1
            print(f"step {step}/{total_steps} SKIPPED (NaN/Inf loss or grad, "
                  f"record={train_records[order[pos]]['image_id']})")

        if step % args.print_every == 0:
            print(f"step {step}/{total_steps} loss {loss_val:.4f} ({time.time() - t_start:.1f}s)"
                  + (f"  [{skipped_steps} skipped so far]" if skipped_steps else ""))

        if val_ds is not None and args.eval_every and step > 0 and step % args.eval_every == 0:
            v = val_loss(trainer, val_ds)
            entry["val_loss"] = v
            print(f"  val_loss {v:.4f}")

        if args.save_every and step > 0 and step % args.save_every == 0:
            ckpt = save_checkpoint(model, step, args.keep_last, ckpt_root)
            print(f"  checkpoint -> {ckpt}")

        history.append(entry)

    final_dir = ckpt_root / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    save_adapter(model, final_dir / "adapters.safetensors")
    (final_dir / "training_log.json").write_text(json.dumps(history, indent=2))
    print(f"\nDone in {time.time() - t_start:.1f}s. Final adapter -> {final_dir}")


if __name__ == "__main__":
    main()
