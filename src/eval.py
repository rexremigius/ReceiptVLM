"""The shared per-field evaluation harness every other script scores
against. Computes micro-F1 across the six scalar fields plus line items (matched by
name-similarity alignment, not position), a bootstrap confidence interval, and a
paired significance test between two prediction files.
"""
from __future__ import annotations
import argparse
import difflib
import json
import random
import re
from collections import defaultdict

# schema contract: matches prep.py's actual output (see prep.py SCALAR_MAP)
SCALAR_FIELDS = ["store", "date", "subtotal", "tax", "tip", "total"]
LINE_ITEM_FIELD = "line_items"
LINE_ITEM_KEYS = ["name", "price"]        # prep.py has no quantity field
NUMERIC_FIELDS = {"subtotal", "tax", "tip", "total", "price"}
NUM_TOL = 0.005                # abs tolerance for numeric match (cent-level)
LINE_MATCH_THRESHOLD = 0.4     # min desc_score to align two line items
ALIGN_KEY = LINE_ITEM_KEYS[0]  # field used to match gold/pred line items by similarity
CHAR_FALLBACK_THRESHOLD = 0.75  # high bar: only trust char-ratio on single-token names


def normalize_text(v):
    """Return a lowercased, whitespace-normalized string, or None if v is None/empty."""
    if v is None:
        return None
    s = re.sub(r"\s+", " ", str(v).strip().lower())
    return s or None


def normalize_num(v):
    """Return a float if v is a number, or None if not. Strips $, commas, letters."""
    if v is None:
        return None
    s = re.sub(r"[^\d.\-]", "", str(v))  # strip $, commas, letters
    try:
        return float(s)
    except ValueError:
        return None


QUANTITY_PREFIX_RE = re.compile(r"^\s*\d+\s*x\s*", re.IGNORECASE)


def strip_quantity_prefix(text):
    """Strip a leading quantity prefix (e.g. "2x ") from a line-item name, if present."""

    if text is None:
        return text
    return QUANTITY_PREFIX_RE.sub("", text)


LINE_ITEM_NAME_FIELD = f"{LINE_ITEM_FIELD}.{ALIGN_KEY}"


def match(field, gold, pred):
    """True if two non-null values agree under the field's comparison rule."""

    leaf = field.rsplit(".", 1)[-1]  # "line_items.price" -> "price"
    if leaf in NUMERIC_FIELDS:
        g, p = normalize_num(gold), normalize_num(pred)
        if g is None or p is None:
            return normalize_text(gold) == normalize_text(pred)
        return abs(g - p) <= NUM_TOL
    if field == LINE_ITEM_NAME_FIELD:
        gold, pred = strip_quantity_prefix(gold), strip_quantity_prefix(pred)
    gold_n, pred_n = normalize_text(gold), normalize_text(pred)
    if leaf == "store" and gold_n and pred_n:
        # gold is often just the brand ("Chipotle") while the model reads the
        # full printed name ("Chipotle Mexican Grill #4021") - that's a correct
        # read, not an error, so credit containment either direction.
        return gold_n in pred_n or pred_n in gold_n
    return gold_n == pred_n


def desc_score(a, b):
    """Return a similarity score in [0,1] for two line-item names, ignoring quantity prefixes."""

    a, b = strip_quantity_prefix(a), strip_quantity_prefix(b)
    na, nb = normalize_text(a), normalize_text(b)
    ta = set(re.findall(r"\w+", na or ""))
    tb = set(re.findall(r"\w+", nb or ""))
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    word_score = len(ta & tb) / len(ta | tb)
    if len(ta) == 1 and len(tb) == 1:
        char_score = difflib.SequenceMatcher(None, na or "", nb or "").ratio()
        if char_score >= CHAR_FALLBACK_THRESHOLD:
            return max(word_score, char_score)
    return word_score


def align_line_items(gold_items, pred_items):
    """Align gold/pred line items by similarity of their names, returning a list of
    (gold_index, pred_index) pairs, plus lists of leftover gold/pred indices that
    didn't match anything. Each gold item is aligned to at most one pred item, and vice
    versa, with a minimum similarity threshold for alignment.
    """
    pairs = []
    used = set()
    gold_leftover, pred_leftover = [], list(range(len(pred_items)))
    for gi, g in enumerate(gold_items):
        best_j, best_s = None, LINE_MATCH_THRESHOLD
        for pj, p in enumerate(pred_items):
            if pj in used:
                continue
            s = desc_score(g.get(ALIGN_KEY), p.get(ALIGN_KEY))
            if s > best_s:
                best_j, best_s = pj, s
        if best_j is None:
            gold_leftover.append(gi)
        else:
            used.add(best_j)
            pairs.append((gi, best_j))
    pred_leftover = [j for j in pred_leftover if j not in used]
    return pairs, gold_leftover, pred_leftover


def count_pair(field, gold, pred, counts):
    """Increment counts[field] for a single gold/pred pair, using the field's comparison rule."""

    g_has, p_has = gold is not None, pred is not None
    if g_has and p_has:
        if match(field, gold, pred):
            counts[field]["tp"] += 1
        else:
            counts[field]["fp"] += 1  # wrong value = one FP + one FN
            counts[field]["fn"] += 1
    elif g_has and not p_has:
        counts[field]["fn"] += 1
    elif p_has and not g_has:
        counts[field]["fp"] += 1


def score_receipt(gold, pred, counts):
    """Accumulate per-field counts for one receipt into `counts`."""

    for f in SCALAR_FIELDS:
        count_pair(f, gold.get(f), pred.get(f), counts)

    g_items = gold.get(LINE_ITEM_FIELD) or []
    p_items = pred.get(LINE_ITEM_FIELD) or []
    pairs, g_lo, p_lo = align_line_items(g_items, p_items)
    for gi, pj in pairs:
        for k in LINE_ITEM_KEYS:
            fld = f"{LINE_ITEM_FIELD}.{k}"
            count_pair(fld, g_items[gi].get(k), p_items[pj].get(k), counts)
    for gi in g_lo:                       # unmatched gold rows -> all FN
        for k in LINE_ITEM_KEYS:
            if g_items[gi].get(k) is not None:
                counts[f"{LINE_ITEM_FIELD}.{k}"]["fn"] += 1
    for pj in p_lo:                       # spurious pred rows -> all FP
        for k in LINE_ITEM_KEYS:
            if p_items[pj].get(k) is not None:
                counts[f"{LINE_ITEM_FIELD}.{k}"]["fp"] += 1


def prf(c):
    tp, fp, fn = c["tp"], c["fp"], c["fn"]
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1


def evaluate(gold_by_id, pred_by_id):
    """Return per-field PRF + micro PRF over the intersection of ids."""

    counts = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    ids = [i for i in gold_by_id if i in pred_by_id]
    for i in ids:
        score_receipt(gold_by_id[i], pred_by_id[i], counts)
    micro = {"tp": 0, "fp": 0, "fn": 0}
    per_field = {}
    for f, c in counts.items():
        per_field[f] = prf(c)
        for k in micro:
            micro[k] += c[k]
    return per_field, prf(micro), len(ids)


def _receipt_micro_counts(gold_by_id, pred_by_id, ids):
    """Return a dict keyed by receipt ID, with values being (tp, fp, fn) counts for that receipt."""

    out = {}
    for i in ids:
        c = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
        score_receipt(gold_by_id[i], pred_by_id[i], c)
        out[i] = tuple(sum(fc[k] for fc in c.values()) for k in ("tp", "fp", "fn"))
    return out


def bootstrap_micro_f1(gold_by_id, pred_by_id, n=1000, seed=0):
    """Percentile bootstrap CI for the micro-F1, resampling receipts."""
    rng = random.Random(seed)
    ids = [i for i in gold_by_id if i in pred_by_id]
    if not ids:
        return (0.0, 0.0)
    counts = _receipt_micro_counts(gold_by_id, pred_by_id, ids)
    samples = []
    for _ in range(n):
        tp = fp = fn = 0
        for _ in ids:
            c_tp, c_fp, c_fn = counts[rng.choice(ids)]
            tp += c_tp
            fp += c_fp
            fn += c_fn
        samples.append(prf({"tp": tp, "fp": fp, "fn": fn})[2])
    samples.sort()
    lo = samples[int(0.025 * (n - 1))]
    hi = samples[int(0.975 * (n - 1))]
    return lo, hi


def paired_bootstrap_test(gold_by_id, pred_a_by_id, pred_b_by_id, n=1000, seed=0):
    """Paired bootstrap test for the difference in micro-F1 between two prediction sets.
    Returns a dict with keys "mean_diff", "ci" (95% percentile CI),
    and "p_approx" (two-sided p-value approximation), or None if there are no receipts
    in common between the two prediction sets and the gold set.
    """
    rng = random.Random(seed)
    ids = [i for i in gold_by_id if i in pred_a_by_id and i in pred_b_by_id]
    if not ids:
        return None
    counts_a = _receipt_micro_counts(gold_by_id, pred_a_by_id, ids)
    counts_b = _receipt_micro_counts(gold_by_id, pred_b_by_id, ids)
    diffs = []
    for _ in range(n):
        tp_a = fp_a = fn_a = tp_b = fp_b = fn_b = 0
        for _ in ids:
            i = rng.choice(ids)
            a_tp, a_fp, a_fn = counts_a[i]
            b_tp, b_fp, b_fn = counts_b[i]
            tp_a += a_tp; fp_a += a_fp; fn_a += a_fn
            tp_b += b_tp; fp_b += b_fp; fn_b += b_fn
        micro_a = {"tp": tp_a, "fp": fp_a, "fn": fn_a}
        micro_b = {"tp": tp_b, "fp": fp_b, "fn": fn_b}
        diffs.append(prf(micro_b)[2] - prf(micro_a)[2])
    diffs.sort()
    mean_diff = sum(diffs) / len(diffs)
    lo = diffs[int(0.025 * (n - 1))]
    hi = diffs[int(0.975 * (n - 1))]
    frac_le_zero = sum(1 for d in diffs if d <= 0) / n
    frac_ge_zero = sum(1 for d in diffs if d >= 0) / n
    p_approx = min(2 * min(frac_le_zero, frac_ge_zero), 1.0)
    return {"mean_diff": mean_diff, "ci": (lo, hi), "p_approx": p_approx}


def load(path):
    """Load a JSON list or JSONL file into {id: fields_dict}."""
    with open(path) as fh:
        text = fh.read().strip()
    records = []
    if text.startswith("["):
        records = json.loads(text)
    else:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    out = {}
    for rec in records:
        rid = rec.get("id", rec.get("image_id"))
        fields = rec.get("fields", rec)
        out[rid] = fields
    return out


def print_report(name, per_field, micro, n, ci):
    """Print a human-readable report of per-field and micro-F1 scores, plus bootstrap CI."""
    p, r, f1 = micro
    print(f"\n=== {name}  (n={n} receipts) ===")
    print(f"{'field':<24}{'P':>8}{'R':>8}{'F1':>8}")
    order = SCALAR_FIELDS + [f"{LINE_ITEM_FIELD}.{k}" for k in LINE_ITEM_KEYS]
    for f in order:
        if f in per_field:
            fp, fr, ff = per_field[f]
            print(f"{f:<24}{fp:>8.3f}{fr:>8.3f}{ff:>8.3f}")
    print(f"{'-'*48}")
    print(f"{'MICRO':<24}{p:>8.3f}{r:>8.3f}{f1:>8.3f}")
    print(f"micro-F1 95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]")


def run(gold_path, pred_paths, boot=1000):
    """Run evaluation of one or more prediction files against a gold file, printing per-field and
    micro-F1 scores, plus bootstrap CIs and paired significance tests."""
    gold = load(gold_path)
    results = {}
    for pp in pred_paths:
        pred = load(pp)
        per_field, micro, n = evaluate(gold, pred)
        ci = bootstrap_micro_f1(gold, pred, n=boot)
        results[pp] = (per_field, micro, n, ci)
        print_report(pp, per_field, micro, n, ci)
    if len(pred_paths) > 1:
        print("\n=== micro-F1 comparison ===")
        for pp in pred_paths:
            _, micro, _, ci = results[pp]
            print(f"{pp:<28}  {micro[2]:.3f}  [{ci[0]:.3f}, {ci[1]:.3f}]")

        print("\n=== paired significance tests ===")
        for i in range(len(pred_paths)):
            for j in range(i + 1, len(pred_paths)):
                a_path, b_path = pred_paths[i], pred_paths[j]
                a_by_id, b_by_id = load(a_path), load(b_path)
                test = paired_bootstrap_test(gold, a_by_id, b_by_id, n=boot)
                if test is None:
                    continue
                sig = "significant" if test["p_approx"] < 0.05 else "not significant"
                print(f"{b_path} vs {a_path}:")
                print(f"  delta micro-F1 = {test['mean_diff']:+.3f}  "
                      f"95% CI [{test['ci'][0]:+.3f}, {test['ci'][1]:+.3f}]  "
                      f"p~={test['p_approx']:.3f} ({sig} at alpha=0.05)")
    return results


def _smoke():
    gold = [
        {"image_id": "r1", "store": "Trader Joes", "date": "2026-01-05",
         "subtotal": "10.00", "tax": "0.80", "total": "10.80",
         "line_items": [{"name": "milk", "price": "3.50"},
                        {"name": "eggs", "price": "6.50"}]},
        {"image_id": "r2", "store": "CVS", "date": "2026-01-06",
         "subtotal": "5.00", "tax": "0.40", "total": "5.40",
         "line_items": [{"name": "advil", "price": "5.00"}]},
        {"image_id": "r3", "store": "Target", "date": "2026-01-07",
         "subtotal": "20.00", "tax": "1.60", "total": "21.60", "tip": None,
         "line_items": [{"name": "socks", "price": "20.00"}]},
    ]
    # baseline: misses tax, wrong total on r1, drops a line item
    baseline = [
        {"image_id": "r1", "store": "Trader Joes", "date": "2026-01-05",
         "subtotal": "10.00", "total": "10.08",
         "line_items": [{"name": "milk", "price": "3.50"}]},
        {"image_id": "r2", "store": "cvs", "date": "2026-01-06",
         "subtotal": "5.00", "total": "5.40",
         "line_items": [{"name": "advil", "price": "5.00"}]},
        {"image_id": "r3", "store": "Targett", "total": "21.60",
         "line_items": [{"name": "socks", "price": "20.00"}]},
    ]
    # finetuned: near-perfect, minor casing only
    finetuned = [dict(g) for g in gold]

    import tempfile, os
    d = tempfile.mkdtemp()
    paths = {}
    for nm, data in [("gold", gold), ("baseline", baseline), ("finetuned", finetuned)]:
        p = os.path.join(d, f"{nm}.jsonl")
        with open(p, "w") as fh:
            for rec in data:
                fh.write(json.dumps(rec) + "\n")
        paths[nm] = p
    run(paths["gold"], [paths["baseline"], paths["finetuned"]], boot=500)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold")
    ap.add_argument("--pred", nargs="+")
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()
    if not args.gold:
        print("no --gold given, running synthetic smoke test\n")
        _smoke()
    else:
        run(args.gold, args.pred, boot=args.boot)
