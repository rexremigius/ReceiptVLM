"""Hyperparameter sweep driver for #4 train.py (Person B's fine-tuning lane).

Coordinate-descent sweep over LoRA rank -> alpha -> lr (one dimension at a time, not
full 3x3x3 grid search) on a fast ~250-receipt subset, so ~7 short runs stand in for
what a full-grid sweep on the full dataset would cost. Each round fixes the winner(s)
from the previous round and only varies the fastest-changing values that weren't
already covered by an earlier run — this keeps total sweep runs to ~7 instead of 27.
Winning config is then confirmed with one full run on all 1140 train receipts.

All trial checkpoints go under checkpoints/sweep/<config>/ and the final confirmation
run under checkpoints/sweep_best/ — neither touches checkpoints/final (the existing
validated checkpoint already used for data/processed/finetuned_test.jsonl). Promoting
the sweep-confirmed checkpoint to checkpoints/final is a manual, separate decision.

Usage:
    python src/sweep.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SWEEP_ROOT = REPO_ROOT / "checkpoints" / "sweep"
CONFIRM_ROOT = REPO_ROOT / "checkpoints" / "sweep_best"

SUBSET_LIMIT = 250
SUBSET_EPOCHS = 1
DEFAULT_RANK, DEFAULT_ALPHA, DEFAULT_LR = 8, 1.0, 1e-4

RANK_GRID = [4, 8, 16]
ALPHA_GRID = [0.5, 1.0, 2.0]
LR_GRID = [5e-5, 1e-4, 2e-4]


def run_name(rank: float, alpha: float, lr: float) -> str:
    return f"r{rank}_a{alpha}_lr{lr:g}"


def run_trial(rank: float, alpha: float, lr: float) -> dict:
    """Run one subset trial (unless already run); return its training_log.json entries."""
    name = run_name(rank, alpha, lr)
    ckpt_dir = SWEEP_ROOT / name
    log_path = ckpt_dir / "final" / "training_log.json"
    if log_path.exists():
        print(f"[{name}] cached, skipping")
        return json.loads(log_path.read_text())

    print(f"[{name}] training...")
    t0 = time.time()
    subprocess.run(
        [
            sys.executable, "src/train.py",
            "--limit", str(SUBSET_LIMIT), "--epochs", str(SUBSET_EPOCHS),
            "--lora-rank", str(rank), "--lora-alpha", str(alpha), "--lr", str(lr),
            "--ckpt-root", str(ckpt_dir),
            "--eval-every", "25", "--save-every", "100000", "--print-every", "50",
        ],
        cwd=REPO_ROOT, check=True,
    )
    print(f"[{name}] done in {time.time() - t0:.0f}s")
    return json.loads(log_path.read_text())


def best_val_loss(history: list[dict]) -> float:
    vals = [e["val_loss"] for e in history if "val_loss" in e]
    return min(vals) if vals else float("nan")


def sweep_dimension(label: str, grid: list[float], fixed: dict) -> tuple[float, dict]:
    print(f"\n=== sweeping {label}: {grid} (fixed: {fixed}) ===")
    results = {}
    for value in grid:
        cfg = {**fixed, label: value}
        history = run_trial(cfg["rank"], cfg["alpha"], cfg["lr"])
        score = best_val_loss(history)
        results[value] = score
        print(f"  {label}={value}: best val_loss={score:.4f}")
    best_value = min(results, key=results.get)
    print(f"  -> best {label} = {best_value} (val_loss={results[best_value]:.4f})")
    return best_value, results


def main():
    all_results = {}

    best_rank, rank_results = sweep_dimension(
        "rank", RANK_GRID, {"rank": DEFAULT_RANK, "alpha": DEFAULT_ALPHA, "lr": DEFAULT_LR}
    )
    all_results["rank"] = rank_results

    best_alpha, alpha_results = sweep_dimension(
        "alpha", ALPHA_GRID, {"rank": best_rank, "alpha": DEFAULT_ALPHA, "lr": DEFAULT_LR}
    )
    all_results["alpha"] = alpha_results

    best_lr, lr_results = sweep_dimension(
        "lr", LR_GRID, {"rank": best_rank, "alpha": best_alpha, "lr": DEFAULT_LR}
    )
    all_results["lr"] = lr_results

    print(f"\n=== winning config: rank={best_rank} alpha={best_alpha} lr={best_lr} ===")

    summary = {
        "subset_limit": SUBSET_LIMIT, "subset_epochs": SUBSET_EPOCHS,
        "sweep_results": {k: {str(kk): vv for kk, vv in v.items()} for k, v in all_results.items()},
        "winning_config": {"rank": best_rank, "alpha": best_alpha, "lr": best_lr},
    }
    (SWEEP_ROOT / "_sweep_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n=== confirming winning config on full train.jsonl (2 epochs) ===")
    t0 = time.time()
    subprocess.run(
        [
            sys.executable, "src/train.py",
            "--epochs", "2",
            "--lora-rank", str(best_rank), "--lora-alpha", str(best_alpha), "--lr", str(best_lr),
            "--ckpt-root", str(CONFIRM_ROOT),
            "--eval-every", "100", "--save-every", "200", "--print-every", "25",
        ],
        cwd=REPO_ROOT, check=True,
    )
    print(f"confirmation run done in {time.time() - t0:.0f}s -> {CONFIRM_ROOT}/final")
    print("\nNOTE: this does NOT overwrite checkpoints/final — compare "
          f"{CONFIRM_ROOT}/final/training_log.json against the existing run before promoting it.")


if __name__ == "__main__":
    main()
