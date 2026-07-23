"""Failure taxonomy for receipt KIE predictions.

Classifies every field-level disagreement between gold and a prediction file
into a specific failure type (not just "wrong"), then reports counts per
field x category plus a handful of example records per category.

Reuses eval.py's normalization, matching, and line-item alignment so a value
is never counted as an "error" here under different rules than it was
scored under there — one shared definition of match, in one place.

Usage:
    python taxonomy.py --gold gold.jsonl --pred finetuned.jsonl
    python taxonomy.py                     # synthetic smoke test
"""

import argparse
import difflib
import json
import re
from collections import defaultdict

import eval as ev

# --- category definitions ----------------------------------------------------
MISSING = "missing"                    # gold has value, pred omitted it
HALLUCINATED = "hallucinated"          # pred has value, gold has none
DIGIT_TRANSPOSED = "digit_transposed"  # same digits, different order/position
NUMERIC_NEAR = "numeric_near_miss"     # numeric, wrong, but close (<10% off)
NUMERIC_FAR = "numeric_far_miss"       # numeric, wrong, way off
TEXT_PARTIAL = "text_partial_overlap"  # text, wrong, but shares real overlap
TEXT_UNRELATED = "text_unrelated"      # text, wrong, no meaningful overlap
LINE_MISSING = "line_item_missing"     # gold row never matched to a pred row
LINE_EXTRA = "line_item_hallucinated"  # pred row never matched to a gold row

NEAR_MISS_REL_THRESHOLD = 0.10   # <10% relative error => "near miss"
TEXT_PARTIAL_THRESHOLD = 0.3     # token-Jaccard above this => "partial overlap"
MAX_EXAMPLES_PER_CATEGORY = 5    # keep the report skimmable, not a data dump


def _digits_only(v):
    return re.sub(r"\D", "", str(v))


def is_digit_transposition(gold, pred):
    """Same multiset of digits, different arrangement (e.g. 49.99 vs 94.99)."""
    gd, pd = _digits_only(gold), _digits_only(pred)
    return bool(gd) and gd != pd and sorted(gd) == sorted(pd)


def token_overlap(a, b):
    """Same scoring as eval.py's desc_score, duplicated here (not imported) so
    classify() and the line-item alignment step never disagree about
    closeness under two different rules.
    """
    na, nb = ev.normalize_text(a), ev.normalize_text(b)
    ta = set(re.findall(r"\w+", na or ""))
    tb = set(re.findall(r"\w+", nb or ""))
    if not ta or not tb:
        return 0.0
    word_score = len(ta & tb) / len(ta | tb)
    if len(ta) == 1 and len(tb) == 1:
        char_score = difflib.SequenceMatcher(None, na or "", nb or "").ratio()
        if char_score >= ev.CHAR_FALLBACK_THRESHOLD:
            return max(word_score, char_score)
    return word_score


def classify(field, gold_val, pred_val):
    """Return a failure category string, or None if the pair is a match."""
    g_has, p_has = gold_val is not None, pred_val is not None
    if not g_has and not p_has:
        return None
    if g_has and not p_has:
        return MISSING
    if p_has and not g_has:
        return HALLUCINATED
    if ev.match(field, gold_val, pred_val):
        return None  # agrees under eval.py's rule -> not an error

    leaf = field.rsplit(".", 1)[-1]

    if leaf in ev.NUMERIC_FIELDS:
        if is_digit_transposition(gold_val, pred_val):
            return DIGIT_TRANSPOSED
        g, p = ev.normalize_num(gold_val), ev.normalize_num(pred_val)
        if g is not None and p is not None and g != 0:
            rel_err = abs(g - p) / abs(g)
            if rel_err < NEAR_MISS_REL_THRESHOLD:
                return NUMERIC_NEAR
        return NUMERIC_FAR
    else:
        gn, pn = ev.normalize_text(gold_val), ev.normalize_text(pred_val)
        substring_match = bool(gn) and bool(pn) and (gn in pn or pn in gn)
        overlaps = substring_match or token_overlap(gold_val, pred_val) >= TEXT_PARTIAL_THRESHOLD
        return TEXT_PARTIAL if overlaps else TEXT_UNRELATED


def analyze_receipt(rid, gold, pred, counts, examples):
    def record(field, category, g, p):
        counts[field][category] += 1
        if len(examples[(field, category)]) < MAX_EXAMPLES_PER_CATEGORY:
            examples[(field, category)].append((rid, g, p))

    for f in ev.SCALAR_FIELDS:
        g, p = gold.get(f), pred.get(f)
        cat = classify(f, g, p)
        if cat:
            record(f, cat, g, p)

    g_items = gold.get(ev.LINE_ITEM_FIELD) or []
    p_items = pred.get(ev.LINE_ITEM_FIELD) or []
    pairs, g_lo, p_lo = ev.align_line_items(g_items, p_items)
    for gi, pj in pairs:
        for k in ev.LINE_ITEM_KEYS:
            f = f"{ev.LINE_ITEM_FIELD}.{k}"
            g, p = g_items[gi].get(k), p_items[pj].get(k)
            cat = classify(f, g, p)
            if cat:
                record(f, cat, g, p)
    for gi in g_lo:
        row = g_items[gi]
        record(ev.LINE_ITEM_FIELD, LINE_MISSING, row.get(ev.ALIGN_KEY), None)
    for pj in p_lo:
        row = p_items[pj]
        record(ev.LINE_ITEM_FIELD, LINE_EXTRA, None, row.get(ev.ALIGN_KEY))


def build_taxonomy(gold_by_id, pred_by_id):
    counts = defaultdict(lambda: defaultdict(int))
    examples = defaultdict(list)
    ids = [i for i in gold_by_id if i in pred_by_id]
    for rid in ids:
        analyze_receipt(rid, gold_by_id[rid], pred_by_id[rid], counts, examples)
    return counts, examples, len(ids)


CATEGORY_ORDER = [MISSING, HALLUCINATED, DIGIT_TRANSPOSED, NUMERIC_NEAR,
                   NUMERIC_FAR, TEXT_PARTIAL, TEXT_UNRELATED, LINE_MISSING, LINE_EXTRA]


def print_report(name, counts, examples, n):
    print(f"\n=== failure taxonomy: {name}  (n={n} receipts) ===\n")
    col_w = max(len(c) for c in CATEGORY_ORDER) + 2
    header = f"{'field':<24}" + "".join(f"{c:>{col_w}}" for c in CATEGORY_ORDER)
    print(header)
    total_by_cat = defaultdict(int)
    field_totals = []
    for field in sorted(counts, key=lambda f: -sum(counts[f].values())):
        row = counts[field]
        line = f"{field:<24}" + "".join(f"{row.get(c, 0):>{col_w}}" for c in CATEGORY_ORDER)
        print(line)
        for c in CATEGORY_ORDER:
            total_by_cat[c] += row.get(c, 0)
        field_totals.append((field, sum(row.values())))
    print("-" * (24 + col_w * len(CATEGORY_ORDER)))
    print(f"{'TOTAL':<24}" + "".join(f"{total_by_cat.get(c, 0):>{col_w}}" for c in CATEGORY_ORDER))

    field_totals.sort(key=lambda x: -x[1])
    print("\ntop offending fields:")
    for f, t in field_totals[:5]:
        print(f"  {f:<24}{t} errors")

    print("\nexample errors (up to {} per category):".format(MAX_EXAMPLES_PER_CATEGORY))
    for (field, cat), exs in examples.items():
        if not exs:
            continue
        print(f"\n  [{field} / {cat}]")
        for rid, g, p in exs:
            print(f"    {rid}: gold={g!r}  pred={p!r}")


def run(gold_path, pred_path):
    gold = ev.load(gold_path)
    pred = ev.load(pred_path)
    counts, examples, n = build_taxonomy(gold, pred)
    print_report(pred_path, counts, examples, n)
    return counts, examples


def _smoke():
    gold = [
        {"image_id": "r1", "store": "Trader Joes", "date": "2026-01-05",
         "subtotal": "10.00", "tax": "0.80", "total": "10.80",
         "line_items": [{"name": "whole milk", "price": "3.50"},
                        {"name": "eggs", "price": "6.50"}]},
        {"image_id": "r2", "store": "CVS Pharmacy", "date": "2026-01-06",
         "subtotal": "5.00", "tax": "0.40", "tip": None, "total": "5.40",
         "line_items": [{"name": "advil", "price": "5.00"}]},
        {"image_id": "r3", "store": "Target", "date": "2026-01-07",
         "subtotal": "20.00", "tax": "1.60", "total": "49.99",
         "line_items": [{"name": "socks", "price": "20.00"}]},
        {"image_id": "r4", "store": "Chipotle", "date": "2026-01-08",
         "subtotal": "12.00", "tax": "0.96", "tip": "2.00", "total": "14.96",
         "line_items": [{"name": "burrito bowl", "price": "12.00"}]},
    ]
    pred = [
        # r1: hallucinated tip, near-miss tax rounding, hallucinated extra line item
        {"image_id": "r1", "store": "Trader Joes", "date": "2026-01-05",
         "subtotal": "10.00", "tax": "0.75", "tip": "1.00", "total": "10.80",
         "line_items": [{"name": "whole milk", "price": "3.50"},
                        {"name": "eggs", "price": "6.50"},
                        {"name": "bread", "price": "4.00"}]},
        # r2: unrelated store misread, dropped line item entirely
        {"image_id": "r2", "store": "Walgreens", "date": "2026-01-06",
         "subtotal": "5.00", "tax": "0.40", "total": "5.40",
         "line_items": []},
        # r3: classic digit transposition on total (49.99 gold vs 94.99 pred)
        {"image_id": "r3", "store": "Target", "date": "2026-01-07",
         "subtotal": "20.00", "tax": "1.60", "total": "94.99",
         "line_items": [{"name": "socks", "price": "20.00"}]},
        # r4: partial text overlap on store, far-miss on subtotal, missing tax
        {"image_id": "r4", "store": "Chipotle Mexican Grill #4021", "date": "2026-01-08",
         "subtotal": "21.00", "tip": "2.00", "total": "14.96",
         "line_items": [{"name": "burrito bowl", "price": "12.00"}]},
    ]
    import tempfile, os
    d = tempfile.mkdtemp()
    paths = {}
    for nm, data in [("gold", gold), ("pred", pred)]:
        p = os.path.join(d, f"{nm}.jsonl")
        with open(p, "w") as fh:
            for rec in data:
                fh.write(json.dumps(rec) + "\n")
        paths[nm] = p
    run(paths["gold"], paths["pred"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold")
    ap.add_argument("--pred")
    args = ap.parse_args()
    if not args.gold:
        print("no --gold given, running synthetic smoke test\n")
        _smoke()
    else:
        run(args.gold, args.pred)
