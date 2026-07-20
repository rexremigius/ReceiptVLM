"""Deliverable #1 — WildReceipt boxes -> per-receipt JSON in our target schema.

Reads WildReceipt annotation lines (one JSON object per receipt), maps the 25 KIE
categories onto {store, date, tax, tip, subtotal, total, line_items:[{name,price}]},
and writes one JSONL record per receipt. Unmapped categories are dropped and counted
(never guessed). See CLAUDE.md / SKILL.md #1.

Usage:
    python src/prep.py --split both              # full train + test
    python src/prep.py --split train --limit 20  # ~20-receipt validation subset
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "wildreceipt"
OUT_ROOT = Path(__file__).resolve().parent.parent / "data" / "processed"

# WildReceipt label-id -> our scalar schema field. Only *_value categories that
# correspond to a schema field appear here; everything else is dropped + logged.
SCALAR_MAP = {
    1: "store",     # Store_name_value
    7: "date",      # Date_value
    17: "subtotal", # Subtotal_value
    19: "tax",      # Tax_value
    21: "tip",      # Tips_value  (sparse in data — see CLAUDE.md)
    23: "total",    # Total_value
}
ITEM_LABEL = 11   # Prod_item_value  -> line_item name
PRICE_LABEL = 15  # Prod_price_value -> line_item price
MONEY_FIELDS = {"subtotal", "tax", "tip", "total"}
SCHEMA_KEYS = ["store", "date", "tax", "tip", "subtotal", "total"]  # line_items added separately


def load_class_names(path: Path) -> dict[int, str]:
    names = {}
    for line in path.read_text().splitlines():
        line = line.rstrip()
        if not line:
            continue
        idx, name = line.split(maxsplit=1)
        names[int(idx)] = name.strip()
    return names


def box_center(box: list[float]) -> tuple[float, float]:
    """Center (y, x) of the 4-corner polygon. y first because we sort y-then-x."""
    xs = box[0::2]
    ys = box[1::2]
    return (sum(ys) / len(ys), sum(xs) / len(xs))


def box_height(box: list[float]) -> float:
    ys = box[1::2]
    return max(ys) - min(ys)


def clean_money(text: str) -> str | None:
    """Extract the numeric portion of a money string ('$12.34' -> '12.34')."""
    # [\d,]* before an optional dot so leading-decimal amounts (".49") keep their
    # point instead of being read as 49; \d+ requires at least one trailing digit.
    m = re.search(r"-?[\d,]*\.?\d+", text.replace(" ", ""))
    if not m:
        return None
    return m.group(0).replace(",", "")


def _median(xs: list[float]) -> float:
    s = sorted(xs)
    return s[len(s) // 2] if s else 0.0


def build_line_items(anns: list[dict]) -> list[dict]:
    """Pair each Prod_item_value name with its Prod_price_value price by y-proximity.

    WildReceipt puts names in a left column and prices in a right column, but the
    price box sits on a systematically offset baseline from its name (seen: price
    ~35-50px below the name, or slightly above it) — never the same y. So a naive
    same-row match drops most prices. Instead we walk items top-down (y-then-x
    sorted, since source box order is not reading order) and give each item the
    nearest still-unused price. The search band is scaled to the median gap between
    consecutive items: that gap is much larger than the name->price offset, so the
    band bridges the offset without ever crossing into an adjacent item's price.
    """
    items, prices, heights = [], [], []
    for a in anns:
        if a["label"] not in (ITEM_LABEL, PRICE_LABEL):
            continue
        y, x = box_center(a["box"])
        heights.append(box_height(a["box"]))
        (items if a["label"] == ITEM_LABEL else prices).append(
            {"y": y, "x": x, "text": a["text"]})
    if not items:
        return []

    items.sort(key=lambda b: (b["y"], b["x"]))
    prices.sort(key=lambda b: (b["y"], b["x"]))

    if len(items) >= 2:
        gaps = [items[i + 1]["y"] - items[i]["y"] for i in range(len(items) - 1)]
        band = 0.6 * _median(gaps)
    else:
        # Single item: no inter-item gap to scale from; fall back to a generous
        # multiple of median box height so its lone price still gets matched.
        band = 5 * (_median(heights) or 1.0)

    used = [False] * len(prices)
    line_items = []
    for it in items:
        best, best_d = None, band
        for j, p in enumerate(prices):
            if used[j]:
                continue
            d = abs(p["y"] - it["y"])
            if d <= best_d:
                best, best_d = j, d
        price = None
        if best is not None:
            used[best] = True
            price = clean_money(prices[best]["text"])
        line_items.append({"name": it["text"], "price": price})
    return line_items


def build_record(receipt: dict, class_names: dict[int, str],
                 dropped: Counter) -> dict:
    anns = receipt["annotations"]

    # Collect scalar-field boxes; a field can span multiple boxes (e.g. "$" + number).
    scalar_boxes: dict[str, list[dict]] = defaultdict(list)
    for a in anns:
        label = a["label"]
        if label in SCALAR_MAP:
            y, x = box_center(a["box"])
            scalar_boxes[SCALAR_MAP[label]].append({"y": y, "x": x, "text": a["text"]})
        elif label not in (ITEM_LABEL, PRICE_LABEL):
            dropped[class_names.get(label, f"label_{label}")] += 1

    record: dict = {"image_id": receipt["file_name"]}
    for field in SCHEMA_KEYS:
        parts = scalar_boxes.get(field)
        if not parts:
            record[field] = None
            continue
        parts.sort(key=lambda b: (b["y"], b["x"]))  # reading order within the field
        joined = " ".join(p["text"] for p in parts).strip()
        record[field] = clean_money(joined) if field in MONEY_FIELDS else joined
    record["line_items"] = build_line_items(anns)
    return record


def validate(record: dict) -> None:
    for key in SCHEMA_KEYS + ["line_items"]:
        assert key in record, f"missing key {key} in {record.get('image_id')}"
    assert isinstance(record["line_items"], list)


def process_split(split: str, class_names: dict[int, str], limit: int | None):
    src = DATA_ROOT / f"{split}.txt"
    records, dropped = [], Counter()
    with src.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = build_record(json.loads(line), class_names, dropped)
            validate(rec)
            records.append(rec)
            if limit and len(records) >= limit:
                break

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    suffix = f".subset{limit}" if limit else ""
    out_path = OUT_ROOT / f"{split}{suffix}.jsonl"
    with out_path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    # Coverage summary: how often each field is present, line-item stats, drops.
    field_present = {k: sum(1 for r in records if r[k]) for k in SCHEMA_KEYS}
    li_counts = [len(r["line_items"]) for r in records]
    summary = {
        "split": split,
        "receipts": len(records),
        "field_present": field_present,
        "field_missing": {k: len(records) - v for k, v in field_present.items()},
        "line_items_total": sum(li_counts),
        "line_items_per_receipt_avg": round(sum(li_counts) / max(len(records), 1), 2),
        "receipts_with_no_line_items": sum(1 for c in li_counts if c == 0),
        "dropped_categories": dict(dropped.most_common()),
        "dropped_total": sum(dropped.values()),
    }
    summ_path = OUT_ROOT / f"_prep_summary_{split}{suffix}.json"
    summ_path.write_text(json.dumps(summary, indent=2))
    return records, summary, out_path, summ_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test", "both"], default="both")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap receipts per split (validation subset)")
    args = ap.parse_args()

    class_names = load_class_names(DATA_ROOT / "class_list.txt")
    splits = ["train", "test"] if args.split == "both" else [args.split]

    for split in splits:
        records, summary, out_path, summ_path = process_split(split, class_names, args.limit)
        print(f"\n=== {split} -> {out_path} ===")
        print(json.dumps({k: summary[k] for k in
                          ["receipts", "field_present", "field_missing",
                           "line_items_per_receipt_avg", "receipts_with_no_line_items",
                           "dropped_total"]}, indent=2))
        print("dropped categories:", summary["dropped_categories"])
        print(f"--- {min(4, len(records))} sample records ---")
        for rec in records[:4]:
            print(json.dumps(rec, indent=2))


if __name__ == "__main__":
    main()
