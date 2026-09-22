"""Checks the converted adapter applies the same update as the MLX original.

Pushes random activations through both update paths using the real trained weights:
MLX (alpha/rank) * ((x @ A) @ B) against peft (lora_alpha/r) * lora_B(lora_A(x)).
Proves the algebra, not the accuracy; validate_peft_adapter.py measures that.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open

MLX_TEXT_PREFIX = "language_model.model."


def load_all(path: Path) -> dict[str, np.ndarray]:
    with safe_open(str(path), framework="np") as f:
        return {k: f.get_tensor(k) for k in f.keys()}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--mlx-adapter", type=Path, default=Path("checkpoints/final"))
    ap.add_argument("--peft-adapter", type=Path, default=Path("checkpoints/final_peft"))
    ap.add_argument(
        "--batch", type=int, default=8, help="Random activation rows per module"
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    mlx_cfg = json.loads((args.mlx_adapter / "adapter_config.json").read_text())
    peft_cfg = json.loads((args.peft_adapter / "adapter_config.json").read_text())

    mlx_scale = mlx_cfg["alpha"] / mlx_cfg["rank"]
    peft_scale = peft_cfg["lora_alpha"] / peft_cfg["r"]
    print(f"MLX  scale = {mlx_cfg['alpha']} / {mlx_cfg['rank']} = {mlx_scale:g}")
    print(f"peft scale = {peft_cfg['lora_alpha']} / {peft_cfg['r']} = {peft_scale:g}")
    assert mlx_scale == peft_scale, "Scale mismatch: the update magnitude would differ."

    mlx = load_all(args.mlx_adapter / "adapters.safetensors")
    peft = load_all(args.peft_adapter / "adapter_model.safetensors")

    # Every MLX module must appear exactly once on the peft side, and vice versa.
    mlx_modules = {k[len(MLX_TEXT_PREFIX) :].rsplit(".", 1)[0] for k in mlx}
    peft_modules = {k.split(".layers.", 1)[1].rsplit(".lora_", 1)[0] for k in peft}
    peft_modules = {f"layers.{m}" for m in peft_modules}
    assert mlx_modules == peft_modules, (
        f"Module sets differ. Only in MLX: {sorted(mlx_modules - peft_modules)[:5]} | "
        f"only in peft: {sorted(peft_modules - mlx_modules)[:5]}"
    )
    print(f"Module sets match: {len(mlx_modules)} modules on both sides.")

    rng = np.random.default_rng(args.seed)
    worst_abs, worst_rel, worst_name = 0.0, 0.0, ""
    n_checked = 0

    for module in sorted(mlx_modules):
        A = mlx[f"{MLX_TEXT_PREFIX}{module}.A"]  # (in, r)
        B = mlx[f"{MLX_TEXT_PREFIX}{module}.B"]  # (r, out)
        layer_and_rest = module[len("layers.") :]
        # Match on suffix so this check stays independent of which layout was emitted.
        a_keys = [k for k in peft if k.endswith(f".{layer_and_rest}.lora_A.weight")]
        b_keys = [k for k in peft if k.endswith(f".{layer_and_rest}.lora_B.weight")]
        assert (
            len(a_keys) == 1 and len(b_keys) == 1
        ), f"Ambiguous peft keys for {module}"
        lora_A = peft[a_keys[0]]  # (r, in)
        lora_B = peft[b_keys[0]]  # (out, r)

        assert lora_A.shape == (
            A.shape[1],
            A.shape[0],
        ), f"{module}: lora_A is {lora_A.shape}, expected {(A.shape[1], A.shape[0])}"
        assert lora_B.shape == (
            B.shape[1],
            B.shape[0],
        ), f"{module}: lora_B is {lora_B.shape}, expected {(B.shape[1], B.shape[0])}"

        x = rng.standard_normal((args.batch, A.shape[0]), dtype=np.float32)
        delta_mlx = mlx_scale * ((x @ A) @ B)
        delta_peft = peft_scale * ((x @ lora_A.T) @ lora_B.T)

        abs_err = float(np.abs(delta_mlx - delta_peft).max())
        denom = float(np.abs(delta_mlx).max())
        rel_err = abs_err / denom if denom > 0 else 0.0
        if abs_err > worst_abs:
            worst_abs, worst_rel, worst_name = abs_err, rel_err, module
        n_checked += 1

    print(
        f"Compared update paths on {n_checked} modules "
        f"({args.batch} random rows each)."
    )
    print(f"Worst absolute difference: {worst_abs:.3e}  ({worst_name})")
    print(f"Worst relative difference: {worst_rel:.3e}")

    # B initializes to zeros and trains up, so an all-zero adapter would pass trivially.
    # Confirm the update is actually non-trivial.
    nonzero = sum(1 for k, v in mlx.items() if k.endswith(".B") and np.abs(v).max() > 0)
    total_b = sum(1 for k in mlx if k.endswith(".B"))
    print(
        f"Non-zero B matrices: {nonzero}/{total_b} "
        f"(a zeroed adapter would match trivially)"
    )
    assert (
        nonzero == total_b
    ), "Some B matrices are all zero - adapter may be untrained."

    assert worst_abs < 1e-4, f"Update paths diverge (max abs {worst_abs:.3e})"
    print("\nPASS: the converted adapter applies the same update as the MLX original.")


if __name__ == "__main__":
    main()
