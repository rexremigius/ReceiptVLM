"""Deliverable #2 — OCR + regex extraction baseline (Track A).

Runs Tesseract OCR over the same receipts #1 (prep.py) already processed, then pulls
{store, date, tax, tip, subtotal, total, line_items:[{name,price}]} out of the raw OCR
text with keyword-anchored regexes — no layout/box information, unlike prep.py which
reads WildReceipt's ground-truth boxes. This is the non-ML floor #5 (eval.py) compares
the fine-tuned VLM against. See CLAUDE.md / SKILL.md #2.

Usage:
    python src/baseline.py --split test               # full test split
    python src/baseline.py --split train --limit 20   # ~20-receipt validation subset
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pytesseract
from PIL import Image, ImageOps

DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "wildreceipt"
OUT_ROOT = Path(__file__).resolve().parent.parent / "data" / "processed"
SCHEMA_KEYS = ["store", "date", "tax", "tip", "subtotal", "total"]

MONEY_RE = re.compile(r"\$?\s*(-?\d[\d,]*\.\d{2})")
SUBTOTAL_RE = re.compile(r"\bsub[\s-]?total\b", re.I)
TAX_RE = re.compile(r"\b(tax|vat|gst|hst)\b", re.I)
TIP_RE = re.compile(r"\b(tip|gratuity)\b", re.I)
TOTAL_RE = re.compile(r"\btotal\b", re.I)  # "subtotal" never matches: no \b before its "total"
DATE_NUMERIC_RE = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b")
DATE_MONTH_RE = re.compile(
    r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{2,4}\b",
    re.I,
)
FIELD_KEYWORD_RES = [SUBTOTAL_RE, TAX_RE, TIP_RE, TOTAL_RE]  # exclude these lines from line_items


def ocr_lines(image_path: Path) -> list[str]:
    """OCR the receipt and return non-empty lines in reading order.

    Grayscale + 2x upscale turned a mostly-unreadable scan into one where SUBTOTAL/TAX/
    TOTAL amounts all came through cleanly in testing — cheap, generic preprocessing,
    not per-receipt tuning. Tried also forcing `--psm 6` (uniform block of text) on top
    of that: it helped that one receipt but wrecked OCR on a sparser-layout receipt
    (legible text became noise), so we leave psm on Tesseract's automatic mode.
    """
    img = ImageOps.grayscale(Image.open(image_path))
    img = img.resize((img.width * 2, img.height * 2))
    text = pytesseract.image_to_string(img)
    return [line.strip() for line in text.splitlines() if line.strip()]


def last_money_match(line: str) -> tuple[str | None, int]:
    """Rightmost money amount on a line, and where it starts (for name/price splits)."""
    matches = list(MONEY_RE.finditer(line))
    if not matches:
        return None, -1
    m = matches[-1]
    return m.group(1).replace(",", ""), m.start()


def find_field(lines: list[str], keyword_re: re.Pattern) -> str | None:
    """Keyword-anchored amount for one scalar money field.

    Fields like TOTAL are often printed more than once (a running tally, then the
    final tender line). We take the *last* keyword match in the receipt — the
    bottom-most occurrence is consistently the authoritative one — falling back to
    the following line if the keyword line itself has no amount (some layouts put
    the label and value on separate lines).
    """
    candidate = None
    for i, line in enumerate(lines):
        if not keyword_re.search(line):
            continue
        amount, _ = last_money_match(line)
        if amount is None and i + 1 < len(lines):
            amount, _ = last_money_match(lines[i + 1])
        if amount is not None:
            candidate = amount
    return candidate


def find_date(lines: list[str]) -> str | None:
    for line in lines:
        m = DATE_NUMERIC_RE.search(line) or DATE_MONTH_RE.search(line)
        if m:
            return m.group(0)
    return None


def find_line_items(lines: list[str]) -> list[dict]:
    """Line + trailing-price heuristic: rightmost money match on a line is the price,
    the text before it is the name. Lines already claimed by a scalar field keyword
    are excluded so subtotal/tax/tip/total don't double as line items.
    """
    items = []
    for line in lines:
        if any(r.search(line) for r in FIELD_KEYWORD_RES):
            continue
        price, start = last_money_match(line)
        if price is None:
            continue
        name = line[:start].strip(" -:$\t")
        if not name or name.replace(" ", "").isdigit():
            continue
        items.append({"name": name, "price": price})
    return items


def build_record(image_id: str) -> dict:
    lines = ocr_lines(DATA_ROOT / image_id)
    record = {"image_id": image_id, "store": lines[0] if lines else None}
    record["date"] = find_date(lines)
    record["subtotal"] = find_field(lines, SUBTOTAL_RE)
    record["tax"] = find_field(lines, TAX_RE)
    record["tip"] = find_field(lines, TIP_RE)
    record["total"] = find_field(lines, TOTAL_RE)
    record["line_items"] = find_line_items(lines)
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


def process_split(split: str, limit: int | None):
    image_ids = load_image_ids(split, limit)
    records = [build_record(image_id) for image_id in image_ids]

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    suffix = f".subset{limit}" if limit else ""
    out_path = OUT_ROOT / f"baseline_{split}{suffix}.jsonl"
    with out_path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

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
    }
    summ_path = OUT_ROOT / f"_baseline_summary_{split}{suffix}.json"
    summ_path.write_text(json.dumps(summary, indent=2))
    return records, summary, out_path, summ_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test", "both"], default="test")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap receipts processed (validation subset)")
    args = ap.parse_args()

    splits = ["train", "test"] if args.split == "both" else [args.split]
    for split in splits:
        records, summary, out_path, summ_path = process_split(split, args.limit)
        print(f"\n=== {split} -> {out_path} ===")
        print(json.dumps(summary, indent=2))
        print(f"--- {min(4, len(records))} sample records ---")
        for rec in records[:4]:
            print(json.dumps(rec, indent=2))


if __name__ == "__main__":
    main()
