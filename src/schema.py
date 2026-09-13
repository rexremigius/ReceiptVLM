"""The extraction schema and prompt, shared by every stack that touches the model.

These are the three constants that must not drift: train.py fine-tuned against this exact
prompt and key order, zeroshot.py evaluates against it, and any serving backend has to
send the identical string or the checkpoint is being asked a different question than it
was trained on.

They used to live in train.py, which imports mlx at module scope. That made them
unreachable off Apple Silicon -- importing the prompt pulled in the whole MLX training
stack -- so the CUDA backend behind the hosted Space could not reuse them. Constants only
here, no imports, so anything can read them.
"""
from __future__ import annotations

# Key order is part of the training target: target_json() serializes in exactly this
# order, so the model learned to emit it this way.
SCHEMA_KEYS = ["store", "date", "tax", "tip", "subtotal", "total", "line_items"]

SCALAR_KEYS = [k for k in SCHEMA_KEYS if k != "line_items"]

PROMPT = ("Extract the receipt fields as JSON with keys store, date, tax, tip, "
          "subtotal, total, line_items (each {name, price}). Use null for missing "
          "scalar fields and [] for no line items.")

# The MLX-side base checkpoint: a pre-quantized 4-bit repack, which is what the QLoRA run
# trained against. The CUDA backend uses the fp16 original instead (see backend_hf.py).
DEFAULT_MODEL = "mlx-community/Qwen2.5-VL-3B-Instruct-4bit"
