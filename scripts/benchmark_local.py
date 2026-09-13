"""Latency and peak-memory benchmark for the Apple-Silicon backends.

The companion to notebooks/kaggle_benchmark_cuda.ipynb, which covers the CUDA half. Same
receipt prefix, same warm-up handling and the same statistics, so the two produce one
comparable table instead of numbers measured four different ways.

Covers every backend reachable on this machine:

    mlx-int4 / mlx-int8 / mlx-fp16   mlx_vlm on Metal, using checkpoints/final
    mps                              transformers on Metal, using checkpoints/final_peft
    cpu                              transformers on CPU (fp32), using checkpoints/final_peft

Two things this deliberately does NOT pretend:

  * **The MLX and transformers rows use different adapters.** They have to -- an adapter
    only works against the base it was fit to, which is the whole subject of
    scripts/validate_peft_adapter.py. So a row-to-row gap is backend *and* adapter, not
    hardware alone.
  * **Peak memory is not one instrument.** MLX exposes mx.get_peak_memory() (an allocator
    tensor peak). torch on MPS has no equivalent max-allocated counter in every version,
    so current-allocated is sampled per receipt and the maximum kept. CPU has no allocator
    counter at all, so only process RSS is available. Every row therefore reports peak
    process RSS as the one metric measured identically everywhere, alongside whatever
    allocator figure the backend offers.

Why the sample is a prefix: zeroshot.load_image_ids takes records[:limit] from test.jsonl
in file order, so subsets nest. The first 30 are a strict subset of the first 60, which is
what makes these numbers comparable to both the 60-receipt quantize.py run and the
30-receipt accuracy gate.

Usage:
    python scripts/benchmark_local.py                          # mlx-int4 + mps, n=60
    python scripts/benchmark_local.py --backends all --limit 60
    python scripts/benchmark_local.py --backends mps,cpu --limit 30
"""
from __future__ import annotations

import argparse
import gc
import json
import resource
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

DATA_ROOT = REPO_ROOT / "data" / "wildreceipt"
PROC_ROOT = REPO_ROOT / "data" / "processed"
MLX_ADAPTER = REPO_ROOT / "checkpoints" / "final"
PEFT_ADAPTER = REPO_ROOT / "checkpoints" / "final_peft"

# Same tier -> repo map quantize.py used, so the MLX rows reproduce its measurements.
MLX_TIERS = {
    "mlx-fp16": "mlx-community/Qwen2.5-VL-3B-Instruct-bf16",
    "mlx-int8": "mlx-community/Qwen2.5-VL-3B-Instruct-8bit",
    "mlx-int4": "mlx-community/Qwen2.5-VL-3B-Instruct-4bit",
}
TORCH_BACKENDS = {"mps", "cpu"}
ALL_BACKENDS = list(MLX_TIERS) + sorted(TORCH_BACKENDS)

IMAGE_RESIZE = (768, 1024)
MAX_NEW_TOKENS = 1536

# Published reference points, printed alongside so the table is self-contained.
REFERENCE = [
    ("MLX INT4 (quantize.py)", 60, 9.5, 4.4),
    ("MLX INT8 (quantize.py)", 60, 12.0, 5.4),
    ("MLX FP16 (quantize.py)", 60, 17.8, 8.8),
    ("CUDA T4 NF4 (gate)", 30, 24.1, None),
]


# Rough resident footprint per backend, from measurements in this project. Used only to
# refuse a run that would page the model out to disk -- a swapped benchmark reports disk
# latency, not backend latency, and does it slowly enough to look like a hang.
EXPECTED_GB = {
    "mlx-int4": 4.4, "mlx-int8": 5.4, "mlx-fp16": 8.8,
    "mps": 10.0,     # fp16 weights + vision activations
    "cpu": 17.0,     # fp32 weights, the heaviest configuration by far
}


def available_gb() -> float | None:
    """Free + inactive + speculative memory, in GB. None if it can't be determined."""

    import re
    import subprocess

    try:
        out = subprocess.run(["vm_stat"], text=True, capture_output=True).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    page = 4096
    m = re.search(r"page size of (\d+) bytes", out)
    if m:
        page = int(m.group(1))
    counts = {}
    for key in ("Pages free", "Pages inactive", "Pages speculative"):
        m = re.search(rf"{key}:\s+(\d+)", out)
        if m:
            counts[key] = int(m.group(1))
    if not counts:
        return None
    return sum(counts.values()) * page / 1e9


def check_headroom(name: str, skip: bool) -> None:
    """Refuse a backend that will not fit in available memory."""

    need = EXPECTED_GB.get(name)
    have = available_gb()
    if need is None or have is None:
        return
    print(f"  memory: need ~{need:.1f} GB, ~{have:.1f} GB available")
    if have < need * 1.15:
        msg = (f"{name} needs ~{need:.1f} GB but only ~{have:.1f} GB is available. "
               "Close some applications, or pass --skip-memory-check to run anyway. "
               "Benchmarking while swapping measures disk, not the backend.")
        if skip:
            print(f"  WARNING: {msg}")
        else:
            raise MemoryError(msg)


def peak_rss_gb() -> float:
    """Peak resident set size. The one memory metric available on every backend.

    macOS reports ru_maxrss in bytes (Linux uses kilobytes); this script is macOS-only, so
    no unit switch is needed. On Apple Silicon's unified memory this also captures GPU-side
    allocations, which is exactly what makes it comparable across MLX, MPS and CPU.
    """

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9


def load_image_ids(split: str, limit: int | None) -> list[str]:
    """Prefix of the split, matching zeroshot.load_image_ids so subsets nest."""

    path = PROC_ROOT / f"{split}.jsonl"
    if not path.exists():
        raise SystemExit(f"Missing {path}; run src/prep.py first.")
    ids = [json.loads(line)["image_id"] for line in path.open() if line.strip()]
    return ids[:limit] if limit else ids


def aggregate(values: list[float]) -> dict:
    ordered = sorted(values)
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": ordered[0],
        "max": ordered[-1],
        "p95": ordered[int(0.95 * (len(ordered) - 1))],
    }


# --- backends ----------------------------------------------------------------------

def run_mlx(tier: str, image_ids: list[str]) -> dict:
    """Time the mlx_vlm stack on one precision tier."""

    import mlx.core as mx
    from mlx_vlm import generate, load
    from mlx_vlm.prompt_utils import apply_chat_template
    from schema import PROMPT

    repo_id = MLX_TIERS[tier]
    if not MLX_ADAPTER.exists():
        raise SystemExit(f"Missing {MLX_ADAPTER} (the MLX adapter) for {tier}.")

    t0 = time.time()
    model, processor = load(repo_id, adapter_path=str(MLX_ADAPTER),
                            processor_config={"trust_remote_code": True})
    prompt = apply_chat_template(processor, model.config.__dict__, PROMPT, num_images=1)
    load_s = time.time() - t0

    def one(image_id: str) -> str:
        return generate(model, processor, prompt,
                        image=str(DATA_ROOT / image_id), max_tokens=MAX_NEW_TOKENS,
                        temperature=0.0, resize_shape=IMAGE_RESIZE, verbose=False)

    # Warm-up excluded: the first call pays for lazy kernel compilation.
    t = time.time()
    one(image_ids[0])
    warm_s = time.time() - t

    peak_fn = getattr(mx, "get_peak_memory", None) or mx.metal.get_peak_memory
    if hasattr(mx, "reset_peak_memory"):
        mx.reset_peak_memory()

    latencies, chars = [], []
    for i, image_id in enumerate(image_ids, 1):
        t = time.time()
        raw = one(image_id)
        latencies.append(time.time() - t)
        chars.append(len(raw))
        if i % 10 == 0:
            print(f"    {i}/{len(image_ids)}  mean {statistics.mean(latencies):.1f}s",
                  flush=True)

    result = _summarise(tier, latencies, chars, load_s, warm_s)
    result["allocator_peak_gb"] = peak_fn() / 1e9
    result["allocator_metric"] = "mx.get_peak_memory"
    result["adapter"] = MLX_ADAPTER.name
    result["model"] = repo_id
    del model, processor
    gc.collect()
    return result


def run_torch(device: str, image_ids: list[str]) -> dict:
    """Time the transformers stack on mps or cpu."""

    import torch
    from PIL import Image

    import src.backend_hf as backend_hf

    dtype = torch.float16 if device == "mps" else torch.float32
    backend_hf._pick_device_dtype = lambda: (device, dtype)

    if not PEFT_ADAPTER.exists():
        raise SystemExit(f"Missing {PEFT_ADAPTER} (the transformers adapter).")

    t0 = time.time()
    model = backend_hf.ReceiptModel(adapter_path=PEFT_ADAPTER).load()
    load_s = time.time() - t0

    def one(image_id: str) -> str:
        with Image.open(DATA_ROOT / image_id) as img:
            raw, _ = model.generate_with_logprobs(img)
        return raw

    t = time.time()
    one(image_ids[0])
    warm_s = time.time() - t

    # torch exposes no max-allocated counter for mps in every version, so sample the
    # current-allocated figure per receipt and keep the maximum. cpu has no counter at all.
    sample = None
    if device == "mps" and hasattr(torch, "mps"):
        sample = getattr(torch.mps, "current_allocated_memory", None)

    peak_alloc = 0.0
    latencies, chars = [], []
    for i, image_id in enumerate(image_ids, 1):
        t = time.time()
        raw = one(image_id)
        latencies.append(time.time() - t)
        chars.append(len(raw))
        if sample is not None:
            peak_alloc = max(peak_alloc, sample() / 1e9)
        if i % 10 == 0:
            print(f"    {i}/{len(image_ids)}  mean {statistics.mean(latencies):.1f}s",
                  flush=True)

    result = _summarise(device, latencies, chars, load_s, warm_s)
    result["allocator_peak_gb"] = peak_alloc or None
    result["allocator_metric"] = ("torch.mps.current_allocated_memory (sampled max)"
                                 if sample is not None else "none available on cpu")
    result["adapter"] = PEFT_ADAPTER.name
    result["model"] = f"{backend_hf.HF_BASE_MODEL} ({dtype})".replace("torch.", "")
    del model
    gc.collect()
    return result


def _summarise(name: str, latencies: list[float], chars: list[int],
               load_s: float, warm_s: float) -> dict:
    return {
        "backend": name,
        "n": len(latencies),
        "load_s": load_s,
        "warmup_s": warm_s,
        "total_s": sum(latencies),
        "latency_all": aggregate(latencies),
        "latency_first30": aggregate(latencies[:30]) if len(latencies) >= 30 else None,
        "chars": aggregate([float(c) for c in chars]),
        "peak_rss_gb": peak_rss_gb(),
        "_lat": latencies,
        "_chars": chars,
    }


# --- reporting ---------------------------------------------------------------------

def report(results: list[dict], limit: int) -> None:
    print(f"\n{'=' * 78}")
    print(f"Local benchmark — first {limit} receipts of the test split")
    print("=" * 78)

    hdr = (f"{'backend':<10}{'n':>4}{'mean':>9}{'median':>9}{'p95':>9}{'max':>9}"
           f"{'alloc GB':>10}{'RSS GB':>9}")
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        a = r["latency_all"]
        alloc = f"{r['allocator_peak_gb']:.2f}" if r.get("allocator_peak_gb") else "—"
        print(f"{r['backend']:<10}{r['n']:>4}{a['mean']:>8.1f}s{a['median']:>8.1f}s"
              f"{a['p95']:>8.1f}s{a['max']:>8.1f}s{alloc:>10}{r['peak_rss_gb']:>9.2f}")

    if any(r["latency_first30"] for r in results):
        print("\nFirst 30 only (matches the accuracy gate and the CUDA runs):")
        for r in results:
            a = r["latency_first30"]
            if a:
                print(f"  {r['backend']:<10} mean {a['mean']:.1f}s  "
                      f"median {a['median']:.1f}s  max {a['max']:.1f}s")

    print("\nPer-backend detail:")
    for r in results:
        print(f"  {r['backend']:<10} load {r['load_s']:.1f}s | warm-up "
              f"{r['warmup_s']:.1f}s (excluded) | total {r['total_s']:.0f}s")
        print(f"  {'':<10} adapter {r['adapter']} | {r['model']}")
        print(f"  {'':<10} memory metric: {r['allocator_metric']}")

    print("\nPublished reference points:")
    for name, n, lat, mem in REFERENCE:
        mem_s = f"{mem} GB" if mem else "—"
        print(f"  {name:<26} n={n:<4} {lat:>5.1f}s   {mem_s}")

    print("\nCaveats: MLX rows use checkpoints/final, transformers rows use "
          "checkpoints/final_peft —\na row-to-row gap is backend *and* adapter. Peak RSS "
          "is the only metric measured\nidentically across all rows.")


def _run_in_subprocess(name: str, args) -> dict | None:
    """Run one backend in a fresh interpreter and read back its JSON."""

    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "result.json"
        cmd = [sys.executable, "-u", str(Path(__file__).resolve()),
               "--worker", name, "--worker-out", str(out),
               "--limit", str(args.limit), "--split", args.split]
        # Streamed rather than captured: these runs take tens of minutes, and buffering
        # the worker's per-10-receipt progress until it exits leaves no way to tell a slow
        # backend from a hung one.
        stderr_tail: list[str] = []
        proc = subprocess.Popen(cmd, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, bufsize=1)
        for line in proc.stdout:
            line = line.rstrip()
            if line and not any(s in line for s in
                                ("Loading checkpoint", "it/s]", "fast processor")):
                print("   ", line, flush=True)
        proc.wait()
        stderr_tail = (proc.stderr.read() or "").strip().splitlines()[-4:]

        if not out.exists():
            if any("No module named" in l for l in stderr_tail):
                raise ImportError(
                    next(l for l in stderr_tail if "No module named" in l).strip())
            raise RuntimeError("worker produced no result. " + " / ".join(stderr_tail))
        return json.loads(out.read_text())


def run_worker(name: str, image_ids: list[str], out_path: Path) -> None:
    """Benchmark one backend and write its JSON. Invoked as a subprocess by main()."""

    result = run_mlx(name, image_ids) if name in MLX_TIERS else run_torch(name, image_ids)
    out_path.write_text(json.dumps(result) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backends", default="mlx-int4,mps",
                    help=f"comma-separated, or 'all'. Choices: {', '.join(ALL_BACKENDS)}")
    ap.add_argument("--limit", type=int, default=60,
                    help="receipts to time; 60 matches quantize.py, 30 matches the gate")
    ap.add_argument("--split", choices=["train", "test"], default="test")
    ap.add_argument("--out", type=Path,
                    default=PROC_ROOT / "_benchmark_local.json")
    ap.add_argument("--skip-memory-check", action="store_true",
                    help="run even when available memory looks insufficient")
    ap.add_argument("--in-process", action="store_true",
                    help="run backends in this process instead of one subprocess each. "
                         "Faster to start, but peak-RSS figures then bleed between "
                         "backends and only the first is trustworthy.")
    # Internal: one backend, one process. Not for direct use.
    ap.add_argument("--worker", help=argparse.SUPPRESS)
    ap.add_argument("--worker-out", type=Path, help=argparse.SUPPRESS)
    args = ap.parse_args()

    names = (ALL_BACKENDS if args.backends == "all"
             else [b.strip() for b in args.backends.split(",") if b.strip()])
    unknown = [b for b in names if b not in MLX_TIERS and b not in TORCH_BACKENDS]
    if unknown:
        raise SystemExit(f"Unknown backend(s) {unknown}. Choices: {ALL_BACKENDS}")

    image_ids = load_image_ids(args.split, args.limit)
    missing = [i for i in image_ids if not (DATA_ROOT / i).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} images missing under {DATA_ROOT}, "
                         f"e.g. {missing[0]}")

    if args.worker:
        run_worker(args.worker, image_ids, args.worker_out)
        return

    print(f"{len(image_ids)} receipts from {args.split}; backends: {', '.join(names)}")

    results = []
    for name in names:
        print(f"\n=== {name} ===", flush=True)
        try:
            check_headroom(name, args.skip_memory_check)
            if args.in_process:
                r = (run_mlx(name, image_ids) if name in MLX_TIERS
                     else run_torch(name, image_ids))
            else:
                # Each backend gets its own process. resource.getrusage reports a
                # process-lifetime high-water mark that cannot be reset, so running two
                # backends in one process makes the second inherit the first's peak --
                # cpu at fp32 would silently make mps look like it used 16 GB. Separate
                # processes also guarantee the previous backend's weights are really gone.
                r = _run_in_subprocess(name, args)
        except ImportError as exc:
            print(f"  skipped: {exc}. Install requirements.txt for the MLX tiers, or "
                  "requirements-spaces.txt for mps/cpu.")
            continue
        except MemoryError as exc:
            print(f"  skipped: {exc}")
            continue
        except Exception as exc:  # a tier that will not fit is itself a finding
            print(f"  failed: {type(exc).__name__}: {exc}")
            continue
        if r is None:
            continue
        results.append(r)
        print(f"  -> {r['latency_all']['mean']:.1f}s/receipt, "
              f"peak RSS {r['peak_rss_gb']:.2f} GB")

    if not results:
        raise SystemExit("No backend produced results.")

    report(results, args.limit)
    payload = [{k: v for k, v in r.items() if not k.startswith("_")} for r in results]
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
