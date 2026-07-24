"""Deliverable #8 — calibrated per-field confidence (Track C, headline result).

SKILL.md's composite: token logprob + arithmetic consistency (subtotal+tax+tip ~=
total) + format validity — all three real now, scoped to the 6 scalar fields
(store/date/tax/tip/subtotal/total) plus line items. Line items went through three
rounds (see PROGRESS.md 2026-07-25/2026-07-26 for before/after distributions at each):
real per-item format-validity + subtotal-consistency signals replacing an old
hardcoded 0.6 constant every item shared; then an honest `None` (rendered as an
"unscored" badge upstream, not a fabricated number) when neither signal fires;
then per-item token logprob (`line_item_logprob_feature`, mirroring
`logprob_feature` one level down) as a genuine third signal that varies per item
even when consistency has nothing to say, plus a total-minus-tax-minus-tip fallback
anchor (`line_item_consistency`'s `total`/`tax`/`tip` params) for the common case
where subtotal itself was never predicted.

Calibration split: `test.jsonl` (WildReceipt held-out, 472 receipts) is split in half
by receipt, seeded, into a calibration half (fits Platt scaling) and a report half —
everything below (ECE, reliability diagram, risk-coverage curve) is measured only on
the report half. This is deliberately NOT the same use of test.jsonl that #5 (eval.py)
reports F1 on: fitting and reporting calibration quality on the same data it was fit
against would overstate how well-calibrated the result actually is, the same leakage
concern eval.py itself is careful about between its own bootstrap resamples.

Reuses eval.py's `match`/`normalize_num`/`normalize_text`/`load` — one definition of
"correct" and "parses", shared with #5/#6, never re-derived here under different rules.

Usage:
    python src/confidence.py                     # finetuned predictions, seed 0
    python src/confidence.py --tag zeroshot       # score a different prediction file
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

import eval as ev

PROC_ROOT = Path(__file__).resolve().parent.parent / "data" / "processed"
MONEY_FIELDS = {"subtotal", "tax", "tip", "total"}
ARITH_TOL = 0.02


def arithmetic_consistency(pred: dict) -> bool | None:
    """Whether the PREDICTED subtotal+tax+tip lands within 2c of the PREDICTED total.
    Ground truth is never available at serving time, so this can only check the
    model's values against themselves, never against the answer key — that's exactly
    why it's a confidence signal and not a scoring one. None if there isn't enough of
    the receipt present to check (e.g. subtotal never extracted at all).
    """
    sub, tax, tot = (ev.normalize_num(pred.get(f)) for f in ("subtotal", "tax", "total"))
    if sub is None or tax is None or tot is None:
        return None
    tip = ev.normalize_num(pred.get("tip")) or 0.0
    return abs((sub + tax + tip) - tot) <= ARITH_TOL


def format_valid(field: str, value) -> bool:
    if value is None:
        return False
    if field in MONEY_FIELDS:
        return ev.normalize_num(value) is not None
    return ev.normalize_text(value) is not None


def raw_score(field: str, pred: dict, consistent: bool | None) -> float:
    """Heuristic raw score in [0, 1]: format validity, plus (money fields only)
    arithmetic consistency. Deliberately uncalibrated — `fit_platt`/`apply_platt` below
    turn it into a real probability against actual outcomes. This is one of up to two
    features fed to calibration; `logprob_feature` below is the other, when available.
    """
    if not format_valid(field, pred.get(field)):
        return 0.05
    if field not in MONEY_FIELDS:
        return 0.6  # no arithmetic signal applies to store/date
    if consistent is None:
        return 0.5
    return 0.8 if consistent else 0.2


def logprob_feature(field: str, pred: dict) -> float | None:
    """exp(mean token logprob) for this field's value in the model's raw completion —
    a natural [0, 1]-ish quantity (geometric-mean per-token probability of what the
    model actually emitted), #8's third signal. None unless the prediction file was
    generated with `zeroshot.py --capture-logprobs` (opt-in there, since it adds
    bookkeeping most callers don't need) and the field's key was actually found in the
    raw text this receipt's `_field_logprobs` was computed from.
    """
    lp = (pred.get("_field_logprobs") or {}).get(field)
    return float(np.exp(lp)) if lp is not None else None


# --- line-item confidence -----------------------------------------------------------
# Real per-item signal, replacing the old hardcoded "0.6 if any items else 0.0"
# constant (which gave every line item on every receipt the identical badge — see
# PROGRESS.md 2026-07-25 for the before/after distribution this actually produces).

# Two narrow, low-false-positive name-noise checks, kept deliberately small: a broader
# "stray punctuation ratio" or "asterisk-wrapped" rule was tried first and rejected
# after checking real `finetuned_test.jsonl` line items — receipts legitimately use
# parenthetical modifiers ("Houji-cha(Hot)", "PinotBlanc(g)") and asterisk-wrapped POS
# annotations ("*SPECPREP*", "*33CLZERO*") that a punctuation-ratio rule flags as noise
# but are real, correct extractions. What actually survives as unambiguous noise in the
# real data: a quote character directly touching a digit (real contractions/possessives
# never do this — "Maker'sMark" and "Owners'Wash" don't trigger it, but the one clear
# real noise example "'AAA'2PK" does), and a `$` immediately followed by a digit inside
# the name (a price fragment that leaked into the name field).
_QUOTE_DIGIT_RE = re.compile(r"[\"'](?=\d)|(?<=\d)[\"']")
_PRICE_LEAK_RE = re.compile(r"\$\d")


def format_valid_line_item_name(name) -> bool:
    if name is None:
        return False
    s = str(name)
    if len(s.strip()) <= 1:
        return False
    if _QUOTE_DIGIT_RE.search(s) or _PRICE_LEAK_RE.search(s):
        return False
    return True


def line_item_logprob_feature(idx: int, pred: dict) -> float | None:
    """exp(mean token logprob) for the idx-th line item's `{...}` object in the raw
    completion — the line-item analog of `logprob_feature` above, one level down
    (see PROGRESS.md 2026-07-26). None unless the prediction file was generated with
    `--capture-logprobs` (which now also records `_line_item_logprobs`, one entry per
    predicted item — see zeroshot.py's `line_item_avg_logprob`) or this item's span
    wasn't found (e.g. truncated generation).
    """
    lps = pred.get("_line_item_logprobs")
    if lps is None or idx >= len(lps) or lps[idx] is None:
        return None
    return float(np.exp(lps[idx]))


def line_item_consistency(items: list[dict], subtotal, total=None, tax=None,
                          tip=None) -> dict[int, bool | None]:
    """Per-item consistency: does sum(predicted item prices) reconcile with the
    predicted subtotal, and if not, which single item looks most like the culprit?
    Checked against real ground truth first (see PROGRESS.md): a null line-item price
    is NOT always a model miss — ~12% of *gold* line items have no price either (a
    genuine WildReceipt annotation gap), so a null price gets no consistency opinion
    at all here, same principle as the tip fix below applied one level down.

    Falls back to an IMPLIED subtotal (`total - tax - tip`) when `subtotal` itself is
    missing/unparseable but `total` is present — recovers a real anchor for 121 of 134
    real receipts that have line items but no predicted subtotal (see PROGRESS.md
    2026-07-26), which previously got no consistency opinion for any item at all. tax
    and tip default to 0 only when genuinely absent from the prediction (a receipt can
    legitimately have zero of either), matching `arithmetic_consistency`'s own tip
    convention above.

    Returns {item_index: True} for every priced item once the aggregate already
    reconciles (within 2c); {item_index: False} for the one item whose removal would
    best explain the gap, when removing it cuts the discrepancy by at least half;
    {item_index: None} (no opinion, discrepancy exists but isn't clearly this item's
    fault) otherwise. Items with no price aren't included in the returned dict at all.
    """
    target = ev.normalize_num(subtotal) if subtotal is not None else None
    if target is None:
        implied_total = ev.normalize_num(total) if total is not None else None
        if implied_total is not None:
            tax_v = ev.normalize_num(tax) if tax is not None else 0.0
            tip_v = ev.normalize_num(tip) if tip is not None else 0.0
            target = implied_total - (tax_v or 0.0) - (tip_v or 0.0)
    priced = [(i, ev.normalize_num(it.get("price"))) for i, it in enumerate(items)]
    priced = [(i, p) for i, p in priced if p is not None]
    if target is None or not priced:
        return {}
    s = sum(p for _, p in priced)
    diff = s - target
    if abs(diff) <= ARITH_TOL:
        return {i: True for i, _ in priced}
    best_i, best_residual = None, abs(diff)
    for i, p in priced:
        residual = abs(diff - p)
        if residual < best_residual:
            best_residual, best_i = residual, i
    out = {}
    for i, _ in priced:
        if i == best_i and best_residual <= abs(diff) * 0.5:
            out[i] = False
        else:
            out[i] = None
    return out


def line_item_raw_score(item: dict, consistency_flag: bool | None) -> float | None:
    """Raw heuristic score in [0, 1] for one predicted line item, or None if there's
    nothing this signal can say about it. Two distinct reasons collapse to None here
    (callers tell them apart by checking `item.get("price")` themselves — see
    `_line_item_badges`): no price to have an opinion about at all (its own state,
    see `line_item_consistency`'s docstring), or a price *is* present and the name
    format is fine, but no consistency signal exists for this receipt/item (the old
    code returned a flat 0.6 here, which is what made every item on 86.8% of
    multi-item receipts show an identical badge — see PROGRESS.md 2026-07-26 for the
    before/after; the honest answer when there's genuinely no signal is "no opinion",
    not a fabricated middling number).
    """
    if item.get("price") is None:
        return None
    if not format_valid_line_item_name(item.get("name")) or ev.normalize_num(item.get("price")) is None:
        return 0.05
    if consistency_flag is True:
        return 0.8
    if consistency_flag is False:
        return 0.2
    return None


# --- Platt scaling: logistic regression, raw feature(s) -> calibrated probability ---
# Generalized from a single raw_score to however many features are available (2 once
# token logprob is present) — same idea as classic 1D Platt scaling, just fit as a
# small multi-feature logistic regression instead of a 1-coefficient one.

def fit_platt(X: np.ndarray, outcomes: np.ndarray) -> np.ndarray:
    """Fit P(correct) = sigmoid(X @ w + b) by maximum likelihood. No closed form for
    logistic regression (unlike OLS), so this minimizes negative log-likelihood
    directly via scipy — a handful of parameters doesn't need an sklearn dependency.
    Returns the parameter vector [w..., b].
    """
    X = np.atleast_2d(X)
    n_features = X.shape[1]

    def nll(params):
        w, b = params[:-1], params[-1]
        z = X @ w + b
        log_p = -np.logaddexp(0, -z)      # numerically stable log(sigmoid(z))
        log_1mp = -np.logaddexp(0, z)      # log(1 - sigmoid(z))
        return -np.sum(outcomes * log_p + (1 - outcomes) * log_1mp)

    result = minimize(nll, x0=np.concatenate([np.ones(n_features), [0.0]]),
                      method="Nelder-Mead")
    return result.x


def apply_platt(X: np.ndarray, params: np.ndarray) -> np.ndarray:
    X = np.atleast_2d(X)
    w, b = params[:-1], params[-1]
    return 1.0 / (1.0 + np.exp(-(X @ w + b)))


# --- reliability diagram / ECE ------------------------------------------------------

def reliability_diagram(probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 10):
    """Bin calibrated probabilities into `n_bins` equal-width buckets; ECE is the
    n-weighted average gap between each bucket's mean predicted probability and its
    empirical (actual) accuracy — the standard calibration-quality summary number."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    ece = 0.0
    n = len(probs)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (probs >= lo) & (probs < hi) if i < n_bins - 1 else (probs >= lo) & (probs <= hi)
        cnt = int(mask.sum())
        if cnt == 0:
            rows.append({"bin": [round(float(lo), 2), round(float(hi), 2)], "n": 0,
                        "mean_confidence": None, "empirical_accuracy": None})
            continue
        mean_conf = float(probs[mask].mean())
        emp_acc = float(outcomes[mask].mean())
        rows.append({"bin": [round(float(lo), 2), round(float(hi), 2)], "n": cnt,
                    "mean_confidence": round(mean_conf, 4),
                    "empirical_accuracy": round(emp_acc, 4)})
        ece += (cnt / n) * abs(mean_conf - emp_acc)
    return rows, round(ece, 4)


# --- risk-coverage curve ------------------------------------------------------------

def risk_coverage(probs: np.ndarray, outcomes: np.ndarray,
                  levels=(1.0, 0.9, 0.8, 0.7, 0.6, 0.5)):
    """'At X% coverage (keeping only the X% most-confident predictions and treating
    the rest as needing manual review), what fraction of the kept ones are actually
    correct?' — the headline number SKILL.md asks for ("90% coverage -> X% precision").
    Sorted descending by calibrated probability; a stable sort keeps ties in original
    (receipt) order so the coverage cutoffs are deterministic across runs.
    """
    order = np.argsort(-probs, kind="stable")
    sorted_outcomes = outcomes[order]
    n = len(outcomes)
    out = []
    for lvl in levels:
        k = max(1, int(round(lvl * n)))
        precision = float(sorted_outcomes[:k].mean())
        out.append({"coverage": lvl, "n_kept": k, "precision": round(precision, 4)})
    return out


# --- data assembly -------------------------------------------------------------------

def collect(gold_by_id: dict, pred_by_id: dict, ids: list[str], use_logprob: bool,
           use_li_logprob: bool = False):
    """Per scalar field: (receipt_id, feature_vector, is_correct) for every receipt
    where the model predicted a non-null value. Confidence only applies to values the
    model actually displayed — a field the model left blank isn't a "should you trust
    this" question, it's a recall miss for #5/#6 to count, not this module.

    `feature_vector` is `[raw_score]` normally, or `[raw_score, logprob_feature]` when
    this prediction file was generated with `--capture-logprobs`. A field the model
    predicted but whose key text couldn't be located for logprob extraction (rare —
    see `field_avg_logprob`'s note) gets a neutral 0.5 imputed rather than being
    dropped, so a single edge case doesn't shrink that field's sample size.

    Line items are folded into the same per_field dict under the key "line_items",
    one row per *predicted* item (not per receipt) with the same (id, features,
    correct) shape — "correct" reuses eval.py's own `align_line_items` alignment plus
    its `match()` on both name and price, the identical definition #5/#6 score
    against, so "correct" never means something different here than it does there.
    Items with a null price are excluded outright — nothing to have an opinion about
    at all (see `line_item_raw_score`'s docstring). Items with a price but no
    heuristic signal (no consistency check possible for this receipt) are excluded
    too UNLESS `use_li_logprob` is set, in which case per-item token logprob (real
    once `--capture-logprobs` was used to generate this file — see
    `line_item_logprob_feature`) becomes a genuine second feature, imputing a neutral
    0.5 for whichever half is missing on a given item, the same convention as scalar
    fields above. This is what stops line-item confidence from degenerating to one
    shared badge per receipt even when subtotal-consistency has nothing to say (see
    PROGRESS.md 2026-07-26).
    """
    per_field = defaultdict(list)
    for rid in ids:
        gold, pred = gold_by_id[rid], pred_by_id[rid]
        consistent = arithmetic_consistency(pred)
        for field in ev.SCALAR_FIELDS:
            pv = pred.get(field)
            if pv is None:
                continue
            gv = gold.get(field)
            correct = gv is not None and ev.match(field, gv, pv)
            features = [raw_score(field, pred, consistent)]
            if use_logprob:
                lp_feat = logprob_feature(field, pred)
                features.append(lp_feat if lp_feat is not None else 0.5)
            per_field[field].append((rid, features, correct))

        pred_items = pred.get("line_items") or []
        gold_items = gold.get("line_items") or []
        pairs, _, _ = ev.align_line_items(gold_items, pred_items)
        correct_idx = set()
        for gi, pj in pairs:
            name_ok = ev.match(f"{ev.LINE_ITEM_FIELD}.name",
                              gold_items[gi].get("name"), pred_items[pj].get("name"))
            price_ok = ev.match(f"{ev.LINE_ITEM_FIELD}.price",
                               gold_items[gi].get("price"), pred_items[pj].get("price"))
            if name_ok and price_ok:
                correct_idx.add(pj)
        li_consistency = line_item_consistency(pred_items, pred.get("subtotal"),
                                               pred.get("total"), pred.get("tax"), pred.get("tip"))
        for idx, item in enumerate(pred_items):
            if item.get("price") is None:
                continue  # nothing to evaluate at all, regardless of any signal
            heuristic = line_item_raw_score(item, li_consistency.get(idx))
            if use_li_logprob:
                lp_feat = line_item_logprob_feature(idx, pred)
                if heuristic is None and lp_feat is None:
                    continue  # still genuinely nothing to score this item on
                features = [heuristic if heuristic is not None else 0.5,
                           lp_feat if lp_feat is not None else 0.5]
            else:
                if heuristic is None:
                    continue
                features = [heuristic]
            per_field["line_items"].append((f"{rid}#{idx}", features, idx in correct_idx))
    return per_field


def run(tag: str, split: str, seed: int, calib_frac: float, quiet: bool = False):
    gold_by_id = ev.load(str(PROC_ROOT / f"{split}.jsonl"))
    pred_by_id = ev.load(str(PROC_ROOT / f"{tag}_{split}.jsonl"))
    ids = sorted(i for i in gold_by_id if i in pred_by_id)
    use_logprob = any(pred_by_id[i].get("_field_logprobs") is not None for i in ids)
    use_li_logprob = any(pred_by_id[i].get("_line_item_logprobs") is not None for i in ids)

    rng = random.Random(seed)
    shuffled = ids[:]
    rng.shuffle(shuffled)
    split_at = int(len(shuffled) * calib_frac)
    calib_ids, report_ids = shuffled[:split_at], shuffled[split_at:]

    calib_data = collect(gold_by_id, pred_by_id, calib_ids, use_logprob, use_li_logprob)
    report_data = collect(gold_by_id, pred_by_id, report_ids, use_logprob, use_li_logprob)

    signals = ["format_validity+arithmetic_consistency"] + (
        ["token_logprob"] if use_logprob else [])
    results = {"model": tag, "n_calib_receipts": len(calib_ids),
              "n_report_receipts": len(report_ids), "signals": signals, "fields": {}}

    for field in ev.SCALAR_FIELDS + ["line_items"]:
        c_rows = calib_data.get(field, [])
        r_rows = report_data.get(field, [])
        if len(c_rows) < 10 or len(r_rows) < 10:
            results["fields"][field] = {
                "note": f"too few predicted values to calibrate "
                        f"(calib={len(c_rows)}, report={len(r_rows)})"}
            continue

        c_X = np.array([f for _, f, _ in c_rows])
        c_out = np.array([1.0 if ok else 0.0 for _, _, ok in c_rows])
        params = fit_platt(c_X, c_out)

        r_X = np.array([f for _, f, _ in r_rows])
        r_out = np.array([1.0 if ok else 0.0 for _, _, ok in r_rows])
        r_probs = apply_platt(r_X, params)

        diagram, ece = reliability_diagram(r_probs, r_out)
        rc = risk_coverage(r_probs, r_out)

        results["fields"][field] = {
            "n_predicted_calib": len(c_rows), "n_predicted_report": len(r_rows),
            "calibration_weights": [round(float(w), 4) for w in params[:-1]],
            "calibration_bias": round(float(params[-1]), 4),
            "ece": ece,
            "reliability_diagram": diagram,
            "risk_coverage": rc,
        }

    out_path = PROC_ROOT / f"_confidence_{tag}_{split}.json"
    out_path.write_text(json.dumps(results, indent=2))
    if not quiet:
        report(tag, split, calib_ids, report_ids, results, out_path)
    return results


def report(tag, split, calib_ids, report_ids, results, out_path):
    print(f"=== confidence calibration: {tag} on {split} "
          f"(calib n={len(calib_ids)}, report n={len(report_ids)}, "
          f"signals={results['signals']}) ===\n")
    for field, r in results["fields"].items():
        if "note" in r:
            print(f"{field}: {r['note']}")
            continue
        print(f"{field}  (n_report={r['n_predicted_report']}, ECE={r['ece']})")
        print("  risk-coverage:  " + "  ".join(
            f"{rc['coverage']:.0%}->{rc['precision']:.1%}" for rc in r["risk_coverage"]))
    print(f"\n-> {out_path.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="finetuned", help="prediction file tag to score")
    ap.add_argument("--split", default="test", help="ground-truth split")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--calib-frac", type=float, default=0.5,
                    help="fraction of receipts used to fit calibration, the rest "
                         "reserved for reporting ECE/risk-coverage")
    args = ap.parse_args()
    run(args.tag, args.split, args.seed, args.calib_frac)


if __name__ == "__main__":
    main()
