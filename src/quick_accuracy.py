"""Quick, rough per-field accuracy check — NOT the real eval harness.

Person A owns the actual #5 eval.py (per-field micro-F1, bootstrap CI, paired
significance test). This script exists only to give an early accuracy signal on the
already-generated baseline/zero-shot/fine-tuned predictions against #1's ground truth,
using plain exact-match — no CI, no significance test, no formal line-item matching
algorithm. Treat these numbers as directional, not final.

Usage:
    python src/quick_accuracy.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

PROC_ROOT = Path(__file__).resolve().parent.parent / "data" / "processed"
SCALAR_KEYS = ["store", "date", "tax", "tip", "subtotal", "total"]

MONEY_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def normalize(field: str, value):
    if value is None:
        return None
    value = str(value).strip()
    if not value:
        return None
    if field in ("tax", "tip", "subtotal", "total"):
        m = MONEY_RE.search(value.replace(",", ""))
        if not m:
            return value.lower()
        try:
            return round(float(m.group(0)), 2)
        except ValueError:
            return value.lower()
    return value.lower()


def load_jsonl(path: Path) -> dict[str, dict]:
    records = {}
    for line in path.open():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        records[rec["image_id"]] = rec
    return records


def score(pred_path: Path, gt: dict[str, dict]) -> dict:
    preds = load_jsonl(pred_path)
    field_correct = {k: 0 for k in SCALAR_KEYS}
    field_total = {k: 0 for k in SCALAR_KEYS}
    li_price_hits, li_true_total, li_pred_total = 0, 0, 0

    for image_id, truth in gt.items():
        pred = preds.get(image_id)
        if pred is None:
            continue
        for k in SCALAR_KEYS:
            t, p = normalize(k, truth.get(k)), normalize(k, pred.get(k))
            field_total[k] += 1
            if t == p:
                field_correct[k] += 1

        true_prices = [normalize("total", i.get("price")) for i in truth.get("line_items", [])]
        pred_prices = [normalize("total", i.get("price")) for i in pred.get("line_items", [])]
        true_prices = [p for p in true_prices if p is not None]
        pred_prices = [p for p in pred_prices if p is not None]
        li_true_total += len(true_prices)
        li_pred_total += len(pred_prices)
        remaining = list(pred_prices)
        for tp in true_prices:
            if tp in remaining:
                remaining.remove(tp)
                li_price_hits += 1

    accuracy = {k: round(field_correct[k] / field_total[k], 3) if field_total[k] else None
                for k in SCALAR_KEYS}
    li_precision = round(li_price_hits / li_pred_total, 3) if li_pred_total else None
    li_recall = round(li_price_hits / li_true_total, 3) if li_true_total else None
    return {
        "n": len(preds),
        "field_accuracy": accuracy,
        "line_item_price_precision": li_precision,
        "line_item_price_recall": li_recall,
    }


def main():
    gt = load_jsonl(PROC_ROOT / "test.jsonl")
    print(f"ground truth: {len(gt)} receipts\n")

    for tag in ["baseline", "zeroshot", "finetuned"]:
        path = PROC_ROOT / f"{tag}_test.jsonl"
        if not path.exists():
            print(f"=== {tag}: no predictions file found, skipping ===\n")
            continue
        result = score(path, gt)
        print(f"=== {tag} ===")
        print(json.dumps(result, indent=2))
        print()


if __name__ == "__main__":
    main()
