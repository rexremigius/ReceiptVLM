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
    """Whether the predicted subtotal+tax+tip lands within 2c of the predicted total,
    or None if there isn't enough of the receipt present to check (e.g. subtotal never
    extracted at all)."""

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
    """Raw heuristic score in [0, 1] for one predicted scalar field, or None when there's
    nothing to say — either no value at all, or a valid value but no arithmetic signal
    (e.g. store/date). Callers tell the two apart via `pred.get(field)`.
    """
    if not format_valid(field, pred.get(field)):
        return 0.05
    if field not in MONEY_FIELDS:
        return 0.6  # no arithmetic signal applies to store/date
    if consistent is None:
        return 0.5
    return 0.8 if consistent else 0.2


def logprob_feature(field: str, pred: dict) -> float | None:
    """exp(mean token logprob) for the predicted field's key text span, or None if the
    prediction file wasn't generated with `--capture-logprobs` or the key text span
    wasn't found.
    """

    lp = (pred.get("_field_logprobs") or {}).get(field)
    return float(np.exp(lp)) if lp is not None else None


# --- line-item-specific features ----------------------

_QUOTE_DIGIT_RE = re.compile(r"[\"'](?=\d)|(?<=\d)[\"']")
_PRICE_LEAK_RE = re.compile(r"\$\d")


def format_valid_line_item_name(name) -> bool:
    """Whether a line-item name is non-empty, not just a single quote or digit, and
    doesn't contain a quoted digit or a leaked price (e.g. "2x $3.99")."""
    
    if name is None:
        return False
    s = str(name)
    if len(s.strip()) <= 1:
        return False
    if _QUOTE_DIGIT_RE.search(s) or _PRICE_LEAK_RE.search(s):
        return False
    return True


def line_item_logprob_feature(idx: int, pred: dict) -> float | None:
    """exp(mean token logprob) for the predicted line item's name+price text span, or
    None if the prediction file wasn't generated with `--capture-logprobs` or the
    line-item index is out of range or the logprobs for that item are missing.
    """
    lps = pred.get("_line_item_logprobs")
    if lps is None or idx >= len(lps) or lps[idx] is None:
        return None
    return float(np.exp(lps[idx]))


def line_item_consistency(items: list[dict], subtotal, total=None, tax=None,
                          tip=None) -> dict[int, bool | None]:
    """For each line item with a price, whether the predicted subtotal+tax+tip lands
    within 2c of the predicted total, or None if there isn't enough of the receipt
    present to check (e.g. subtotal never extracted at all). Returns a dict keyed by
    line-item index, with values being True/False/None for each item with a price.
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
    """Raw heuristic score in [0, 1] for one predicted line item, or None when there's
    nothing to say — either no price at all, or a valid price+name but no consistency
    signal for this receipt. Callers tell the two apart via `item.get("price")`.
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
    """Fit a logistic regression to map raw feature(s) to calibrated probability.
    `X` is shape (n_samples, n_features), `outcomes` is shape (n_samples,) with 0/1
    values. Returns a 1D array of length n_features+1: the first n_features are the
    learned weights, the last is the learned bias term. The caller can then compute
    calibrated probabilities for new data using `apply_platt`."""
    
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
    """For each coverage level in `levels`, compute the precision (empirical accuracy)
    of the top-k predicted values, where k is the number of predictions needed to cover
    that fraction of the dataset. Returns a list of dicts with keys "coverage",
    "n_kept", and "precision" (rounded to 4 decimal places)."""

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
    """Collect raw features and correctness labels for each predicted field/line item
    across a set of receipts, for later calibration and evaluation. Returns a dict
    keyed by field name, with values being lists of tuples (receipt_id, feature_vector, correct_bool)."""

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
