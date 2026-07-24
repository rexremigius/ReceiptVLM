"""Deliverable #7 — FP16/INT8/INT4 quantization sweep, per field (Track B).

SKILL.md's spec: confirm FP16 fits standalone first (if not, INT8 is the real
ceiling — report that as a finding, not a failure); reuse #5's eval harness; record
F1, latency/receipt, and peak memory; and produce one clear side-by-side table/chart
of precision level x field, as its own labeled figure — not folded into a paragraph.

Same LoRA adapter (`checkpoints/final`, trained once in full precision) is applied on
top of each precision tier of the SAME base model — only the frozen base weights'
representation changes between FP16/INT8/INT4. Backing repos (checked against what
mlx-community actually publishes, not assumed):
    FP16 -> mlx-community/Qwen2.5-VL-3B-Instruct-bf16
    INT8 -> mlx-community/Qwen2.5-VL-3B-Instruct-8bit
    INT4 -> mlx-community/Qwen2.5-VL-3B-Instruct-4bit  (what #3/#4/#8/#9/#10 use)

Each tier runs in its OWN subprocess (`--tier <name>` mode), not just a fresh model
object in one long-lived process — mlx's peak-memory counter is process-global and
doesn't reset between model loads, so measuring three tiers back-to-back in one
process would let an earlier tier's high-water mark contaminate a later tier's
reading. A subprocess per tier is the actual fix; it's not extra ceremony.

Usage:
    python src/quantize.py                     # orchestrates all 3 tiers + report
    python src/quantize.py --tier FP16          # just one tier (used internally too)
    python src/quantize.py --limit 60           # receipts per tier (default 60 —
                                                 # a full 472 x 3 tiers is ~5 hours)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt

from train import DEFAULT_MODEL, PROMPT, SCHEMA_KEYS  # noqa: F401 (DEFAULT_MODEL unused here on purpose — see PRECISIONS)
from repair import repair_json
from zeroshot import normalize, load_image_ids
import eval as ev

THIS_FILE = Path(__file__).resolve()
DATA_ROOT = THIS_FILE.parent.parent / "data" / "wildreceipt"
OUT_ROOT = THIS_FILE.parent.parent / "data" / "processed"
ADAPTER_PATH = THIS_FILE.parent.parent / "checkpoints" / "final"

PRECISIONS = {
    "FP16": "mlx-community/Qwen2.5-VL-3B-Instruct-bf16",
    "INT8": "mlx-community/Qwen2.5-VL-3B-Instruct-8bit",
    "INT4": "mlx-community/Qwen2.5-VL-3B-Instruct-4bit",
}
TIER_ORDER = ["FP16", "INT8", "INT4"]  # report table column order

SCALAR_KEYS = [k for k in SCHEMA_KEYS if k != "line_items"]


def _peak_memory_gb() -> float:
    import mlx.core as mx
    # mx.metal.get_peak_memory() is deprecated in favor of mx.get_peak_memory() as of
    # mlx 0.29 (seen in this project's own generation logs) — support both since the
    # exact cutover version isn't pinned in requirements.txt.
    fn = getattr(mx, "get_peak_memory", None) or mx.metal.get_peak_memory
    return fn() / 1e9


def run_one_tier(tier: str, image_ids: list[str], resize_shape: tuple[int, int],
                 max_tokens: int) -> dict:
    """Load exactly one precision tier, generate over `image_ids`, write predictions
    + a meta file. Returns the meta dict. A load failure (OOM or otherwise) is
    recorded as `fit: false`, not raised — SKILL.md frames "doesn't fit" as an
    expected, reportable outcome for FP16 in particular, not a bug.
    """
    from mlx_vlm import generate, load
    from mlx_vlm.prompt_utils import apply_chat_template

    repo_id = PRECISIONS[tier]
    print(f"=== {tier} ({repo_id}) ===", flush=True)
    try:
        model, processor = load(repo_id, adapter_path=str(ADAPTER_PATH),
                                processor_config={"trust_remote_code": True})
    except Exception as e:  # noqa: BLE001 — any load/allocation failure here is the
                            # exact finding SKILL.md asks this script to surface
        meta = {"tier": tier, "repo_id": repo_id, "fit": False,
                "error": f"{type(e).__name__}: {e}"}
        print(f"  {tier} did not fit / failed to load: {meta['error']}")
        (OUT_ROOT / f"_quantize_{tier}_meta.json").write_text(json.dumps(meta, indent=2))
        return meta

    config = model.config.__dict__
    prompt = apply_chat_template(processor, config, PROMPT, num_images=1)

    records, latencies = [], []
    for i, image_id in enumerate(image_ids):
        t0 = time.time()
        raw = generate(model, processor, prompt, image=str(DATA_ROOT / image_id),
                       max_tokens=max_tokens, temperature=0.0, resize_shape=resize_shape,
                       verbose=False)
        latencies.append(time.time() - t0)
        parsed, _ = repair_json(raw)
        records.append({"image_id": image_id, **normalize(parsed)})
        if (i + 1) % 20 == 0 or i == len(image_ids) - 1:
            print(f"  {i + 1}/{len(image_ids)}", flush=True)

    pred_path = OUT_ROOT / f"quant_{tier}_test.jsonl"
    with pred_path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    meta = {
        "tier": tier, "repo_id": repo_id, "fit": True,
        "n_receipts": len(records),
        "avg_latency_s": round(sum(latencies) / len(latencies), 2) if latencies else None,
        "peak_memory_gb": round(_peak_memory_gb(), 2),
    }
    (OUT_ROOT / f"_quantize_{tier}_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"  -> {pred_path.name}  (avg {meta['avg_latency_s']}s/receipt, "
          f"peak {meta['peak_memory_gb']}GB)")
    return meta


def report(limit: int, split: str):
    gold = ev.load(str(OUT_ROOT / f"{split}.jsonl"))
    field_order = SCALAR_KEYS + [f"line_items.{k}" for k in ("name", "price")]

    metas, f1_table, preds_by_tier = {}, {}, {}
    for tier in TIER_ORDER:
        meta_path = OUT_ROOT / f"_quantize_{tier}_meta.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        metas[tier] = meta
        if not meta["fit"]:
            continue
        pred = ev.load(str(OUT_ROOT / f"quant_{tier}_test.jsonl"))
        preds_by_tier[tier] = pred
        per_field, micro, n = ev.evaluate(gold, pred)
        f1_table[tier] = {f: per_field.get(f, (0, 0, 0))[2] for f in field_order}
        metas[tier]["micro_f1"] = round(micro[2], 4)
        ci = ev.bootstrap_micro_f1(gold, pred, n=1000)
        metas[tier]["micro_f1_ci95"] = [round(ci[0], 4), round(ci[1], 4)]

    # --- console table (SKILL.md: F1 per field, not just an aggregate) -------------
    print(f"\n=== quantize sweep report (n={limit} receipts, split={split}) ===\n")
    header = "field".ljust(18) + "".join(t.ljust(10) for t in TIER_ORDER)
    print(header)
    for f in field_order:
        row = f.ljust(18)
        for t in TIER_ORDER:
            row += (f"{f1_table[t][f]:.3f}".ljust(10) if t in f1_table else "n/a".ljust(10))
        print(row)
    print()
    for t in TIER_ORDER:
        m = metas.get(t)
        if m is None:
            print(f"{t}: not run")
        elif not m["fit"]:
            print(f"{t}: DID NOT FIT — {m['error']}")
        else:
            print(f"{t}: micro-F1={m['micro_f1']} 95%CI={m['micro_f1_ci95']}  "
                  f"avg_latency={m['avg_latency_s']}s/receipt  "
                  f"peak_memory={m['peak_memory_gb']}GB")

    # Paired significance between tiers (reuses #5's own test — the same "don't claim
    # a difference without it" discipline eval.py applies to model comparisons; a
    # precision sweep's F1 gaps are exactly as liable to be noise at n=60 as any other
    # comparison, so this isn't optional polish, it's what makes the F1 table trustworthy.
    tiers_present = [t for t in TIER_ORDER if t in preds_by_tier]
    if len(tiers_present) > 1:
        print("\n=== paired significance (micro-F1, 1000-resample bootstrap) ===")
        for i in range(len(tiers_present)):
            for j in range(i + 1, len(tiers_present)):
                a, b = tiers_present[i], tiers_present[j]
                test = ev.paired_bootstrap_test(gold, preds_by_tier[a], preds_by_tier[b])
                sig = "significant" if test["p_approx"] < 0.05 else "NOT significant"
                print(f"  {b} vs {a}: delta={test['mean_diff']:+.3f}  "
                      f"95% CI [{test['ci'][0]:+.3f}, {test['ci'][1]:+.3f}]  "
                      f"p~={test['p_approx']:.3f} ({sig})")
                metas.setdefault("_significance", {})[f"{b}_vs_{a}"] = test

    fp16_meta = metas.get("FP16")
    if fp16_meta is not None and not fp16_meta["fit"]:
        print("\nFINDING (per SKILL.md): FP16 does not fit standalone on this "
              "hardware — INT8 is the real ceiling for on-device serving, not a "
              "compression choice made for speed alone.")

    # --- required figure: precision x field, side by side, its own labeled plot ---
    if f1_table:
        fig, ax = plt.subplots(figsize=(10, 5.5))
        n_tiers = len(f1_table)
        bar_w = 0.8 / max(n_tiers, 1)
        colors = {"FP16": "#2a78d6", "INT8": "#eb6834", "INT4": "#1baf7a"}  # dataviz
        # skill's categorical slots 1-3 (blue/orange/aqua) — validated CVD-safe order
        x = range(len(field_order))
        for i, t in enumerate(TIER_ORDER):
            if t not in f1_table:
                continue
            offsets = [xi + (i - (n_tiers - 1) / 2) * bar_w for xi in x]
            ax.bar(offsets, [f1_table[t][f] for f in field_order], width=bar_w,
                  label=t, color=colors[t])
        ax.set_xticks(list(x))
        ax.set_xticklabels(field_order, rotation=30, ha="right")
        ax.set_ylabel("F1")
        ax.set_ylim(0, 1)
        ax.set_title(f"Quantization sweep: F1 by precision level x field (n={limit} receipts)")
        ax.legend(title="Precision")
        ax.set_facecolor("#fcfcfb")
        fig.patch.set_facecolor("#fcfcfb")
        ax.grid(axis="y", color="#e1e0d9", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        fig.tight_layout()
        fig_path = OUT_ROOT / "_quantize_report.png"
        fig.savefig(fig_path, dpi=150)
        print(f"\n-> {fig_path.name} (required precision x field figure)")

    summary_path = OUT_ROOT / "_quantize_summary.json"
    summary_path.write_text(json.dumps(
        {"limit": limit, "split": split, "tiers": metas, "f1_by_field": f1_table}, indent=2))
    print(f"-> {summary_path.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=[*PRECISIONS, "all"], default="all")
    ap.add_argument("--limit", type=int, default=60,
                    help="receipts per tier (a full 472 x 3 tiers is ~5 hours)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-tokens", type=int, default=1536)
    ap.add_argument("--image-resize", type=int, nargs=2, default=[768, 1024])
    ap.add_argument("--report-only", action="store_true",
                    help="rebuild the report/figure from existing quant_*_test.jsonl "
                         "predictions without re-running generation")
    args = ap.parse_args()

    if args.report_only:
        report(args.limit, args.split)
        return

    if args.tier == "all":
        for tier in TIER_ORDER:
            print(f"\n########## {tier}: launching in a fresh process "
                  f"(clean peak-memory reading) ##########", flush=True)
            subprocess.run([
                sys.executable, str(THIS_FILE), "--tier", tier,
                "--limit", str(args.limit), "--split", args.split,
                "--max-tokens", str(args.max_tokens),
                "--image-resize", str(args.image_resize[0]), str(args.image_resize[1]),
            ], check=False)
        report(args.limit, args.split)
        return

    image_ids = load_image_ids(args.split, args.limit)
    run_one_tier(args.tier, image_ids, tuple(args.image_resize), args.max_tokens)


if __name__ == "__main__":
    main()
