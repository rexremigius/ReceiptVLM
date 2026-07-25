from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval as ev  # reuse load / score_receipt / prf

PROC_ROOT = Path(__file__).resolve().parent.parent / "data" / "processed"

# Known chains, checked first (store name only) — unambiguous, settle cases generic
# keywords get wrong (a Costco selling food items is grocery, not dining).
KNOWN_CHAINS: list[tuple[str, str]] = [
    ("grocery", "costco"), ("grocery", "safeway"), ("grocery", "kroger"),
    ("grocery", "metro"), ("grocery", "tesco"), ("grocery", "publix"),
    ("grocery", "morrison"), ("grocery", "bhandar"), ("grocery", "aldi"),
    ("grocery", "trader joe"), ("grocery", "whole foods"), ("grocery", "ralphs"),
    ("grocery", "start market"), ("grocery", "vons"), ("grocery", "akshar"),
    ("grocery", "miramar"), ("grocery", "harris teeter"), ("grocery", "foodland"),
    ("grocery", "food mart"), ("grocery", "h mart"),
    ("fuel", "speedway"), ("fuel", "shell"), ("fuel", "chevron"), ("fuel", "exxon"),
    ("fuel", "texaco"), ("fuel", "citgo"),
    ("dining", "mcdonald"), ("dining", "starbucks"), ("dining", "chick-fil-a"),
    ("dining", "pizzahut"), ("dining", "dominos"), ("dining", "cheesecake factory"),
    ("dining", "honeygrow"), ("dining", "chipotle"), ("dining", "subway"),
    ("dining", "wendy's"), ("dining", "burger king"), ("dining", "kfc"),
    ("dining", "taco bell"), ("dining", "panda express"), ("dining", "dunkin"),
    ("dining", "panera"), ("dining", "olive garden"), ("dining", "red lobster"),
    ("dining", "buffalo wild wings"), ("dining", "outback steakhouse"),
    ("dining", "applebee's"), ("dining", "chili's"), ("dining", "longhorn steakhouse"),
    ("dining", "bahama breeze"), ("dining", "taco stand"),
    ("retail", "cvs"), ("retail", "walgreens"), ("retail", "rite aid"), ("retail", "boots"),
    ("retail", "sephora"), ("retail", "macy's"), ("retail", "nordstrom"), ("retail", "kohls"),
    ("retail", "jcpenny"), ("retail", "best buy"), ("retail", "home depot"),
    ("retail", "lowe's"), ("retail", "target"), ("retail", "walmart"), ("retail", "wal-mart"),
    ("retail", "wal★mart"), ("retail", "ikea"), ("retail", "dollar tree"),
    ("retail", "dollar general"), ("retail", "big lots"), ("retail", "tj maxx"),
    ("retail", "marshalls"), ("retail", "ross dress for less"),
    ("retail", "burlington coat factory"), ("retail", "five below"), ("retail", "homegoods"),
    ("transport", "uber"), ("transport", "lyft"),
    ("misc", "amazon"), ("misc", "ebay"), ("misc", "etsy"), ("misc", "paypal"),
    ("misc", "grubhub"), ("misc", "doordash"), ("misc", "instacart"), ("misc", "postmates"),
    ("misc", "seamless"), ("misc", "uber eats"), ("misc", "usps"), ("misc", "fedex"),
    ("misc", "ups"), ("misc", "dhl"), ("misc", "supercuts"), ("misc", "great clips"),
    ("misc", "sport clips"), ("misc", "regal cinemas"), ("misc", "amc theaters"),
    ("misc", "cinemark"), ("misc", "cineplex"), ("misc", "landmark theaters"),
    ("misc", "arcade"), ("misc", "bowling"), ("misc", "laser tag"), ("misc", "mini golf"),
]

# Store-name keyword rules (plain substrings; concatenated OCR names defeat word bounds).
# dining claims "gastro" before the fuel "gas" rule so "GASTROBAREL" stays dining.
STORE_RULES: list[tuple[str, list[str]]] = [
    ("dining", ["gastro", "restaurant", "rican rest", "ricanrest", "nrest", "cafe", "caffe",
                "café", "coffee", "espresso", "grill", "diner", "tavern", "pizz", "pasta",
                "sushi", "soba", "ramen", "noodle", "wok", "dumpling", "kitchen", "bistro",
                "deli", "steak", "burger", "bakery", "patisserie", "boulangerie", "creperie",
                "eatery", "trattoria", "ristorante", "osteria", "brasserie", "cantina",
                "churrasc", "kebab", "shawarma", "curry", "gelato", "hotel", "resort",
                "taco", "thai", "japanese", "korean", "chinese", "indian", "mexican",
                "bar", "pub", "bbq", "grille", "juice", "smoothie", "creamery",
                "sandwich", "salad", "seafood", "poke", "waffle", "pancake", "donut",
                "bagel", "essen", "trinken"]),
    ("fuel", ["gas", "fuel", "petrol", "diesel", "unleaded", "carwash", "car wash", "lube",
            "arco", "express lane", "servicestation", "service station"]),
    ("grocery", ["market", "supermarket", "grocery", "aldi", "trader joe", "whole foods",
                "ralphs", "start market", "vons", "akshar", "miramar", "harris teeter",
                "foodland", "food mart", "h mart", "sainsbury", "wegman", "mart", "foods",
                "bhandar", "provision"]),
    ("retail", ["pharmacy", "chemist", "bookshop", "book", "liquor", "wine", "hardware",
                "depot", "target", "shop", "boutique", "centre", "center", "store",
                "duty free", "outlet", "apparel", "kitchen", "home", "furniture",
                "electronics", "clothing", "fashion", "jewelry", "accessories", "cosmetics",
                "beauty", "toys", "games", "sports", "fitness", "outdoors", "garden", "pet",
                "baby", "kids", "shoes", "health", "stationery"]),
]

# Fallback for storeless receipts: infer from purchased items.
ITEM_HINTS: list[tuple[str, list[str]]] = [
    ("fuel", ["diesel", "unleaded", "regular ca", "fuel", "petrol"]),
    ("dining", ["burger", "pizza", "coffee", "latte", "beer", "wine", "salad", "chicken",
                "rice", "soup", "sandwich", "dessert", "cola", "coke", "fries", "taco",
                "burrito", "pasta", "sushi", "noodle", "ramen", "dumpling", "kebab",
                "shawarma", "curry", "gelato", "sliders", "waffle", "pancake", "donut",
                "bagel", "ice cream", "smoothie"]),
    ("grocery", ["bag fee", "grocery", "bag tax", "bread", "eggs", "cheese", "butter",
                "produce", "vegetable", "fruit", "rice", "cereal", "milk", "yogurt",
                "meat", "poultry", "fish", "seafood", "snack", "candy", "chocolate"]),
]


def _match(keywords, text):
    for kw in keywords:
        if kw.startswith(r"\b"):
            if re.search(kw, text):
                return True
        elif kw in text:
            return True
    return False


def infer_category(record):
    store = (record.get("store") or "").lower()
    for category, needle in KNOWN_CHAINS:
        if needle in store:
            return category
    for category, keywords in STORE_RULES:
        if _match(keywords, store):
            return category
    # Storeless: score item hints by hit count (a grocery basket with one taco should be
    # grocery, not dining off the lone taco). Ties favor grocery/fuel over generic food.
    items = " ".join((it.get("name") or "") for it in (record.get("line_items") or [])).lower()
    pref = {"grocery": 3, "fuel": 2, "dining": 1}
    scored = [(sum(1 for kw in kws if _match([kw], items)), pref.get(cat, 0), cat)
            for cat, kws in ITEM_HINTS]
    scored.sort(reverse=True)
    return scored[0][2] if scored[0][0] > 0 else "other"


def score_subset(gold, pred, ids):
    counts = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    n = 0
    for i in ids:
        if i in pred:
            ev.score_receipt(gold[i], pred[i], counts)
            n += 1
    micro = {"tp": 0, "fp": 0, "fn": 0}
    for c in counts.values():
        for k in micro:
            micro[k] += c[k]
    return ev.prf(micro)[2], n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default=str(PROC_ROOT / "test.jsonl"))
    ap.add_argument("--pred", default=str(PROC_ROOT / "finetuned_test.jsonl"))
    ap.add_argument("--basis", choices=["gold", "pred"], default="gold")
    ap.add_argument("--dump", action="store_true")
    args = ap.parse_args()

    gold = ev.load(args.gold)
    pred = ev.load(args.pred)
    basis = gold if args.basis == "gold" else pred

    by_cat = defaultdict(list)
    for rid, rec in basis.items():
        by_cat[infer_category(rec)].append(rid)

    order = ["dining", "grocery", "fuel", "retail", "transport", "misc", "other"]
    total = sum(len(v) for v in by_cat.values())
    print(f"categorized {total} receipts (basis: {args.basis}, HEURISTIC)\n")
    print(f"{'category':10}{'count':>7}{'share':>8}{'micro_F1':>10}")
    for cat in [c for c in order if c in by_cat] + [c for c in by_cat if c not in order]:
        ids = by_cat[cat]
        f1, _ = score_subset(gold, pred, ids)
        print(f"{cat:10}{len(ids):>7}{100 * len(ids) / total:>7.1f}%{f1:>10.3f}")

    if args.dump:
        tags = {rid: infer_category(rec) for rid, rec in basis.items()}
        out = PROC_ROOT / "_category_tags.json"
        out.write_text(json.dumps(tags, indent=2))
        print(f"\nper-receipt tags -> {out}")


if __name__ == "__main__":
    main()
