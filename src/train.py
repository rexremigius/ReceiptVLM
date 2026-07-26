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
    """Return a JSON string of the record, including only the schema fields. Ensures
    consistent key order and no ASCII-escaping of non-ASCII characters."""

    return json.dumps({k: record[k] for k in SCHEMA_KEYS}, ensure_ascii=False)


def to_example(record: dict) -> dict:
    """Convert a receipt record to a training example dict with messages and images for
    mlx_vlm.trainer.Dataset. The prompt is the user message, and the JSON completion
    is the assistant message. The image is the receipt image."""
    
    messages = [
        {"role": "user", "content": PROMPT},
        {"role": "assistant", "content": target_json(record)},
    ]
    return {"messages": messages, "images": [str(DATA_ROOT / record["image_id"])]}


def load_split(limit: int | None, val_frac: float, seed: int) -> tuple[list[dict], list[dict]]:
    """Load the train split of receipts, optionally limiting the number of records and
    reserving a fraction for validation. Returns (train_records, val_records)."""

    records = [json.loads(line) for line in (PROC_ROOT / "train.jsonl").open() if line.strip()]
    rng = random.Random(seed)
    rng.shuffle(records)
    if limit:
        records = records[:limit]
    n_val = max(1, round(len(records) * val_frac)) if len(records) > 1 else 0
    val, train = records[:n_val], records[n_val:]
    return train, val


def build_dataset(records, processor, config, image_processor, image_resize_shape) -> Dataset:
    """Build a mlx_vlm.trainer.Dataset from receipt records, applying the chat template
    to the prompt + JSON completion. Returns a Dataset with input_ids, attention_mask,
    pixel_values, and labels for training."""

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
    """Locate the token ID of the 'assistant' role in a tokenized training example, for
    use in Trainer.train_on_completions. Raises ValueError if not found."""

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
    """Run a single train step, returning (loss, applied). If the loss or any gradient
    is NaN or Inf, the step is skipped and applied=False. Otherwise, the optimizer is
    updated and applied=True."""

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
    """Compute the average loss on a validation dataset, optionally limiting the number of
    batches. Returns NaN if the dataset is empty or all losses are NaN/Inf."""

    n = len(dataset) if max_batches is None else min(max_batches, len(dataset))
    if n == 0:
        return float("nan")
    losses = [trainer.loss_fn(trainer.model, dataset[i]).item() for i in range(n)]
    clean = [v for v in losses if not (math.isnan(v) or math.isinf(v))]
    return sum(clean) / len(clean) if clean else float("nan")


def save_checkpoint(model, step: int, keep_last: int, ckpt_root: Path) -> Path:
    """Save a LoRA adapter checkpoint for the given step, keeping only the last N checkpoints.
    Returns the path to the saved checkpoint directory."""

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
    ap.add_argument("--lora-alpha", type=float, default=1.0)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
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
