"""Converts the MLX LoRA adapter from checkpoints/ into a PEFT-format adapter that
transformers can load on CUDA.

The on-device stack (mlx_vlm) and the hosted stack (transformers + peft) store the same
low-rank update in different conventions, so the port is a key rename plus a transpose of
every matrix. Both sides are pinned to the upstream definitions:

  mlx_vlm.trainer.lora.LoRaLayer:
      A: (in_dims, rank)   B: (rank, out_dims)   scale = alpha / rank
      y = original_layer(x) + scale * ((dropout(x) @ A) @ B)

  peft.tuners.lora.Linear:
      lora_A.weight: (r, in)   lora_B.weight: (out, r)   scaling = lora_alpha / r
      y = base(x) + scaling * lora_B(lora_A(dropout(x)))

Equating the two update paths gives lora_A.weight = A.T and lora_B.weight = B.T, with
lora_alpha/r reproducing MLX's alpha/rank exactly. Carrying alpha and rank over verbatim
(rather than renormalizing) keeps the scale identical instead of merely proportional.

Two things this deliberately does not do:

  * It does not touch the vision tower. The MLX adapter only ever contained
    language_model.* tensors, so target_modules is emitted as a path-anchored regex.
    A bare suffix list like ["q_proj", ...] would also match the vision tower's own
    attention and MLP projections and inject untrained adapters there.

  * It does not change precision. MLX stored these as float32 and they are written out
    as float32; peft casts to the base model's dtype at load time.

The adapter was trained by QLoRA on a 4-bit MLX base, so replaying it on an fp16 HF base
is not bit-identical by construction. That discrepancy is what validate_peft_adapter.py
measures; this script only guarantees the algebra is faithfully transcribed.

Usage:
    python scripts/convert_mlx_adapter_to_peft.py \
        --mlx-adapter checkpoints/final \
        --out checkpoints/final_peft
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

# Base checkpoint the hosted stack loads. The MLX run trained against the pre-quantized
# mlx-community 4-bit repack of this same model.
HF_BASE_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"

# Prefix every MLX text-decoder tensor carries.
MLX_TEXT_PREFIX = "language_model.model."

# Path to the text decoder inside Qwen2_5_VLForConditionalGeneration. transformers moved
# this when Qwen2.5-VL was restructured: the decoder used to hang directly off .model and
# now sits under .model.language_model. Selecting the wrong one produces an adapter whose
# keys silently match nothing, so validate_peft_adapter.py re-derives the layout from the
# instantiated model and reports a mismatch rather than loading a no-op adapter.
TEXT_PREFIXES = {
    "new": "model.language_model",  # transformers >= 4.52 (what the Space pins)
    "legacy": "model",              # transformers 4.49, the pin mlx_vlm needs
}

# The seven projections the MLX run adapted, via find_all_linear_names over the decoder.
TARGET_SUFFIXES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# language_model.model.layers.0.self_attn.q_proj.A -> ("0", "self_attn.q_proj", "A")
MLX_KEY_RE = re.compile(r"^layers\.(\d+)\.(.+)\.([AB])$")


def target_modules_regex(text_prefix: str) -> str:
    """Anchor target_modules to the text decoder so peft cannot match the vision tower."""

    return (rf"{re.escape(text_prefix)}\.layers\.\d+\."
            rf"(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)")


def convert_tensors(mlx_path: Path, text_prefix: str) -> tuple[dict[str, np.ndarray], dict]:
    """Remap and transpose every MLX LoRA tensor into its peft equivalent.

    Returns the peft state dict plus a stats dict for the caller to report on. Raises if
    any tensor fails to parse or if A/B do not pair up on a shared inner rank, since a
    silently dropped tensor is an adapter that is quietly weaker than the one evaluated
    in RESULTS.md.
    """

    out: dict[str, np.ndarray] = {}
    ranks: set[int] = set()
    layers: set[int] = set()
    suffixes: set[str] = set()
    pairs: dict[str, dict[str, tuple[int, ...]]] = {}

    with safe_open(str(mlx_path), framework="np") as f:
        keys = list(f.keys())
        for key in keys:
            if not key.startswith(MLX_TEXT_PREFIX):
                raise ValueError(
                    f"{key!r} is outside {MLX_TEXT_PREFIX!r}. This converter only handles "
                    "text-decoder LoRA; a vision-tower adapter would need its own mapping "
                    "and its own target_modules."
                )
            rest = key[len(MLX_TEXT_PREFIX):]
            m = MLX_KEY_RE.match(rest)
            if not m:
                raise ValueError(f"Unrecognized MLX LoRA key: {key!r}")
            layer, module, ab = m.group(1), m.group(2), m.group(3)

            suffix = module.rsplit(".", 1)[-1]
            if suffix not in TARGET_SUFFIXES:
                raise ValueError(
                    f"{key!r} adapts {suffix!r}, which is not in TARGET_SUFFIXES. The "
                    "emitted target_modules regex would not cover it."
                )

            tensor = f.get_tensor(key)
            if tensor.ndim != 2:
                raise ValueError(f"{key!r} has shape {tensor.shape}; expected a 2-D matrix.")

            # A: (in, r) -> lora_A.weight (r, in);  B: (r, out) -> lora_B.weight (out, r)
            ranks.add(tensor.shape[1] if ab == "A" else tensor.shape[0])
            layers.add(int(layer))
            suffixes.add(suffix)
            pairs.setdefault(f"{layer}.{module}", {})[ab] = tuple(tensor.shape)

            peft_key = f"base_model.model.{text_prefix}.layers.{layer}.{module}.lora_{ab}.weight"
            out[peft_key] = np.ascontiguousarray(tensor.T)

    if len(ranks) != 1:
        raise ValueError(f"Inconsistent LoRA rank across tensors: {sorted(ranks)}")

    for name, ab in sorted(pairs.items()):
        if set(ab) != {"A", "B"}:
            raise ValueError(f"{name} has {sorted(ab)}; every module needs both A and B.")
        if ab["A"][1] != ab["B"][0]:
            raise ValueError(
                f"{name}: A is {ab['A']} and B is {ab['B']}; inner rank does not match."
            )

    stats = {
        "n_tensors": len(out),
        "n_modules": len(pairs),
        "rank": ranks.pop(),
        "n_layers": len(layers),
        "layer_range": (min(layers), max(layers)),
        "suffixes": sorted(suffixes),
    }
    return out, stats


def build_peft_config(mlx_cfg: dict, text_prefix: str, rank: int) -> dict:
    """Build the peft adapter_config.json, carrying MLX's alpha and rank over unchanged."""

    alpha = mlx_cfg["alpha"]
    if mlx_cfg["rank"] != rank:
        raise ValueError(
            f"adapter_config.json says rank={mlx_cfg['rank']} but the tensors carry "
            f"rank={rank}."
        )
    return {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": HF_BASE_MODEL,
        "r": rank,
        # peft scales by lora_alpha / r; MLX scaled by alpha / rank. Same numbers in, same
        # scale out -- do not "normalize" alpha here.
        "lora_alpha": alpha,
        "lora_dropout": mlx_cfg.get("dropout", 0.0),
        "target_modules": target_modules_regex(text_prefix),
        "bias": "none",
        "fan_in_fan_out": False,
        "modules_to_save": None,
        "init_lora_weights": True,
        "inference_mode": True,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mlx-adapter", type=Path, default=Path("checkpoints/final"),
                    help="Directory holding adapters.safetensors + adapter_config.json")
    ap.add_argument("--out", type=Path, default=Path("checkpoints/final_peft"),
                    help="Destination directory for the peft-format adapter")
    ap.add_argument("--layout", choices=sorted(TEXT_PREFIXES), default="new",
                    help="Which transformers module layout to emit keys for "
                         "(new: >=4.52, legacy: 4.49). Default: new")
    args = ap.parse_args()

    src_tensors = args.mlx_adapter / "adapters.safetensors"
    src_config = args.mlx_adapter / "adapter_config.json"
    for p in (src_tensors, src_config):
        if not p.exists():
            raise SystemExit(f"Missing {p}")

    text_prefix = TEXT_PREFIXES[args.layout]
    mlx_cfg = json.loads(src_config.read_text())

    tensors, stats = convert_tensors(src_tensors, text_prefix)
    peft_cfg = build_peft_config(mlx_cfg, text_prefix, stats["rank"])

    args.out.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.out / "adapter_model.safetensors"))
    (args.out / "adapter_config.json").write_text(json.dumps(peft_cfg, indent=2) + "\n")

    lo, hi = stats["layer_range"]
    scale = peft_cfg["lora_alpha"] / peft_cfg["r"]
    print(f"Converted {stats['n_tensors']} tensors "
          f"({stats['n_modules']} modules x A/B) -> {args.out}")
    print(f"  layout        {args.layout} ({text_prefix})")
    print(f"  layers        {stats['n_layers']} (indices {lo}-{hi})")
    print(f"  projections   {', '.join(stats['suffixes'])}")
    print(f"  rank / alpha  {peft_cfg['r']} / {peft_cfg['lora_alpha']}  "
          f"-> scale {scale:g}")
    print("\nNext: python scripts/validate_peft_adapter.py --adapter "
          f"{args.out} --limit 30")


if __name__ == "__main__":
    main()
