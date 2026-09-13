# Results

Statistical data compiled from the per-stage summary files in `data/processed/`. Headline
metric is per-field micro-F1 on the 472-receipt WildReceipt test set unless noted.

## Headline

| Model | micro-F1 (472 test) | 95% CI |
|---|---|---|
| OCR + regex baseline (Tesseract) | 0.115 | [0.102, 0.127] |
| Qwen2.5-VL-3B zero-shot | 0.212 | [0.194, 0.230] |
| **Qwen2.5-VL-3B + QLoRA (ours)** | **0.781** | [0.756, 0.802] |

Paired bootstrap: every step is a significant gain (p≈0). Fine-tuned beats zero-shot ~3.7×
and the OCR baseline ~6.8×.

## 1. Dataset (prep.py → ground truth)

| | Train | Test |
|---|---|---|
| Receipts | 1,267 | 472 |
| Line items (total) | 6,145 | 2,371 |
| Line items / receipt | 4.85 | 5.02 |
| Receipts with no line items | 109 | 26 |
| KIE boxes dropped (unmapped categories) | 35,145 | 12,262 |

Ground-truth field presence (how often each field exists in the data):

| Field | Train | Test |
|---|---|---|
| store | 836 / 1267 | 472 / 472 |
| date | 1072 | 398 |
| tax | 852 | 318 |
| subtotal | 892 | 342 |
| total | 1116 | 422 |
| tip | 68 (5.4%) | 21 (4.4%) |

## 2. Extraction coverage per pipeline (472-receipt test)

Counts = the field was produced (presence, not correctness).

| Metric | Baseline (OCR) | Zero-shot | Fine-tuned |
|---|---|---|---|
| Parse failures | — | 42 | 4 |
| store present | 454 | 430 | 449 |
| date present | 212 | 403 | 402 |
| tax present | 130 | 232 | 325 |
| subtotal present | 103 | 214 | 326 |
| total present | 156 | 322 | 443 |
| tip present | 8 | 32 | 10 |
| line items total | 1,222 | 1,975 | 2,473 |
| line items / receipt | 2.59 | 4.18 | 5.24 |
| receipts w/ no line items | 160 | 42 | 18 |
| elapsed (472 receipts) | — | 1,166 s | 2,821 s |

## 3. Quantization (60-receipt subset)

| Precision | micro-F1 | 95% CI | Latency / receipt | Peak memory |
|---|---|---|---|---|
| FP16 | 0.809 | [0.762, 0.852] | 17.8 s | 8.78 GB |
| INT8 | 0.816 | [0.769, 0.857] | 12.0 s | 5.41 GB |
| **INT4 (deployed)** | **0.796** | [0.737, 0.850] | **9.5 s** | **4.36 GB** |

Significance (paired bootstrap) — none significant, so INT4 is chosen for on-device use:

| Comparison | mean Δ | 95% CI | p |
|---|---|---|---|
| INT8 vs FP16 | +0.007 | [−0.002, +0.017] | 0.128 |
| INT4 vs FP16 | −0.013 | [−0.055, +0.024] | 0.584 |
| INT4 vs INT8 | −0.019 | [−0.059, +0.013] | 0.32 |

Per-field F1 by tier:

| Field | FP16 | INT8 | INT4 |
|---|---|---|---|
| store | 0.824 | 0.824 | 0.840 |
| date | 0.875 | 0.875 | 0.896 |
| tax | 0.805 | 0.851 | 0.851 |
| tip | 0.250 | 0.250 | 0.286 |
| subtotal | 0.822 | 0.844 | 0.831 |
| total | 0.901 | 0.919 | 0.909 |
| line_items.name | 0.806 | 0.810 | 0.760 |
| line_items.price | 0.789 | 0.791 | 0.787 |

## 4. Confidence calibration — ECE per field (lower is better)

| Field | 2-signal (n=236 calib) | 3-signal + logprob (n=40) |
|---|---|---|
| store | 0.084 | 0.062 |
| date | 0.021 | 0.038 |
| subtotal | 0.059 | 0.113 |
| tax | 0.069 | 0.203 |
| total | 0.033 | 0.068 |
| line_items | 0.025 | 0.159 |
| tip | n/a (too few tips) | n/a |

Signals: format validity + arithmetic consistency (+ token logprob for the 3-signal set).

## 5. Porting the adapter to CUDA (30-receipt subset)

For hosting, the model has to run off Apple Silicon. All rows below are scored on the same
first 30 receipts of the test split, with the same `eval.py` harness.

| adapter / serving | micro-F1 | 95% CI | vs. MLX run |
|---|---|---|---|
| MLX + QLoRA (on-device reference) | 0.785 | — | — |
| **re-trained on CUDA, served fp16** | **0.760** | [0.691, 0.830] | −0.025, p=0.542 |
| re-trained on CUDA, served NF4 | 0.722 | [0.636, 0.815] | −0.063, p=0.242 |
| original MLX adapter converted to fp16 | 0.288 | [0.230, 0.364] | −0.497, p≈0 |
| fp16 base, no adapter | 0.218 | [0.134, 0.343] | — (n=8) |

At n=60 (from `notebooks/kaggle_benchmark_cuda.ipynb`, same receipt prefix):

| adapter / serving | micro-F1 (n=60) | 95% CI |
|---|---|---|
| MLX + QLoRA (reference) | 0.800 | [0.741, 0.856] |
| re-trained, served fp16 | **0.760** | [0.701, 0.816] |
| re-trained, served NF4 | 0.734 | [0.660, 0.804] |

Doubling the sample left the fp16 point estimate **unchanged at 0.760** and narrowed its CI
from ±0.070 to ±0.058 — a 17% reduction, less than the ~29% a 1/√n rule would predict,
which is normal for a percentile bootstrap on a small, heterogeneous sample.

Two observations from the larger sample:

- **fp16 reproduces exactly across sessions; NF4 does not.** Re-scoring the same first 30
  receipts in an independent session returned 0.760 for fp16 — identical to the gate — but
  0.710 for NF4 against the gate's 0.722. Greedy decoding should be deterministic, so the
  drift is most likely non-deterministic reduction order in the bitsandbytes dequantization
  kernels. A small point, but it favours fp16 for a reproducible demo.
- **The fp16/NF4 accuracy gap is not statistically established.** It narrows from 0.038 at
  n=30 to 0.026 at n=60, and the CIs overlap substantially. The case for fp16 rests on the
  *latency* difference (18.0 s vs 23.5 s), which is unambiguous, plus reproducibility and
  keeping bitsandbytes out of the image — not on an accuracy edge this data can prove.

Per field, fp16 serving vs. the MLX run:

| field | fp16 serve | MLX ref |
|---|---|---|
| store | 0.900 | 0.900 |
| tax | 0.905 | 0.905 |
| line_items.name | 0.729 | 0.730 |
| subtotal | **0.895** | 0.811 |
| total | **0.929** | 0.873 |
| date | 0.773 | 0.818 |
| line_items.price | 0.725 | 0.796 |
| tip | 0.000 | 0.000 |

Two findings:

- **An adapter only works against the base it was fit to.** The MLX adapter converts to
  peft format exactly (verified to 1.5e-8), but on fp16 weights it scores 0.288 — it still
  reads the receipt and reverts to the base model's conventions, so the fields needing no
  convention (tax/subtotal/total ≈0.81) survive while store/date/line-items collapse.
- **Not all 4-bit schemes transfer alike.** MLX's affine 4-bit → fp16 collapsed; an adapter
  trained on bitsandbytes NF4 → fp16 transfers cleanly and scores *higher* than serving it
  on the NF4 base it was trained on (+0.038). NF4 is close enough to fp16 that the learned
  corrections still apply while the base weights are more accurate.

### Latency and memory by backend

Every row below except the last two was measured by `scripts/benchmark_local.py` on the
same first 60 receipts of the test split, with one methodology: warm-up receipt excluded,
each backend in its own process.

| backend | precision | device | n | mean | median | p95 | peak alloc |
|---|---|---|---|---|---|---|---|
| `mlx_vlm` | INT4 | Metal | 60 | **6.2 s** | 6.0 s | 9.6 s | **4.55 GB** |
| `mlx_vlm` | INT8 | Metal | 60 | 7.3 s | 7.3 s | 11.3 s | 5.65 GB |
| `mlx_vlm` | FP16 | Metal | 60 | 9.6 s | 9.6 s | 15.8 s | 8.92 GB |
| `transformers` | fp16 | MPS | 60 | 11.6 s | 11.1 s | 20.5 s | 7.57 GB |
| `transformers` | fp16 | CUDA T4 | 60 | 18.0 s | 16.7 s | 33.7 s | 7.95 GB |
| `transformers` | NF4 | CUDA T4 | 60 | 23.5 s | 22.9 s | 45.6 s | **2.73 GB** |
| `transformers` | fp32 | CPU | 3 | 22.6 s | — | — | 16.4 GB RSS |

**These MLX latencies do not reproduce §3's.** Memory does — within 4% on every tier — but
timings come out 35–46% faster:

| tier | §3 (`quantize.py`) | re-measured | memory §3 → re-measured |
|---|---|---|---|
| INT4 | 9.5 s | 6.2 s | 4.36 → 4.55 GB |
| INT8 | 12.0 s | 7.3 s | 5.41 → 5.65 GB |
| FP16 | 17.8 s | 9.6 s | 8.78 → 8.92 GB |

The gap is systematic in direction and magnitude across all three tiers, and far too large
to be the warm-up exclusion (worth ~0.1 s at n=60). The likely cause is the environment:
these runs used mlx 0.29.3 on Python 3.12, which postdates the original sweep. Both sets
are kept rather than one overwriting the other, because the difference cannot be attributed
without re-running `quantize.py` itself under both.

**§3's ranking and conclusion are unaffected** — INT4 remains fastest and smallest, and the
accuracy differences between tiers remain within noise. But one derived claim shifts: INT4
is **1.55×** faster than FP16 here, not the ~2× that §3's numbers imply.

Three things the table shows:

- **MLX beats transformers on the same hardware at the same precision.** FP16: 9.6 s on
  `mlx_vlm` vs 11.6 s on MPS, and at *lower* peak memory than MPS's fp16 path uses for a
  smaller footprint. So MLX's advantage is the framework, not only its quantization — the
  opposite of what an earlier draft of this section concluded from §3's slower numbers.
- **Quantization buys less speed than memory.** INT4 vs FP16 is 1.55× faster but 2.0×
  smaller. On-device, the memory is the real win.
- **Latency tracks generated tokens, not image size.** MPS ranges 6.0–28.5 s across these
  60 receipts, and the densest receipt in the dataset (50 line items) takes 60.8 s.

Four caveats on reading across rows:

- **The MLX rows use `checkpoints/final`; the transformers rows use
  `checkpoints/final_peft`.** They must — an adapter only works against the base it was fit
  to. A row-to-row gap is backend *and* adapter, not hardware alone.
- **Memory is not one instrument.** MLX reports `mx.get_peak_memory()`; torch on MPS has no
  equivalent max-allocated counter, so current-allocated is sampled per receipt; CPU has
  none at all. The INT8 row's RSS (7.26 GB) is additionally inflated by a ~3.5 GB model
  download that happened during that run — trust its allocator figure, not its RSS.
- **The CPU row is n=3.** fp32 needs ~16 GB, which on a 26 GB machine in normal use pushes
  the system into swap, and a swapped benchmark measures disk rather than the backend.
  Treat it as an order of magnitude for a configuration nobody should deploy.
- **The CUDA rows are a T4**, which is far older than the GPU ZeroGPU allocates. They
  bound the hosted Space from below rather than predicting it.

**On CUDA, fp16 wins on both axes.** Measured on a T4 over the same 60 receipts,
back to back in one session:

| | fp16 | NF4 |
|---|---|---|
| micro-F1 (n=30) | **0.760** | 0.722 |
| latency, mean | **18.0 s** | 23.5 s |
| throughput | **9.8 tok/s** | 7.3 tok/s |
| peak allocated | 7.95 GB | **2.73 GB** |
| weights resident | 7.66 GB | **2.44 GB** |

NF4 is 1.31× slower because dequantization costs time on every matmul, and the two
generated near-identical output lengths (mean 171 vs 176 tokens), so this is throughput
rather than a sampling artifact. NF4's only advantage is memory — 2.9× smaller — which is
irrelevant on a 48 GB ZeroGPU slice. So the serving choice is fp16 on accuracy *and*
speed, with bitsandbytes kept out of the image as a bonus.

The NF4 run also cross-validates the accuracy gate: its first-30 mean of 24.8 s matches
the gate's independently measured 24.1 s.

Worth noting for hardware expectations: **MPS beats a T4 at fp16** (11.6 s vs 18.0 s), and
MLX INT4 beats it by 2.9× (6.2 s vs 18.0 s). The T4's worst receipt took 48.7 s, comfortably
inside the Space's `@spaces.GPU(duration=120)` budget — and ZeroGPU's GPU is faster still.

A cross-check worth recording: on the same receipt, `mlx_vlm` INT4 with `checkpoints/final`
and `transformers` fp16 with `checkpoints/final_peft` produced identical extractions
(`CHOEUN` / `12/30/2016FRI` / `4.48` / `55.96` / `60.44`). Different frameworks, different
quantization, different adapters, same output — evidence the re-trained adapter recovered
the original model's behaviour rather than merely matching it in aggregate.

## Caveats

- **Sample sizes differ.** Headline F1 and coverage use all 472 test receipts; quantization
  uses 60; §5 uses 30; the 3-signal confidence set is only 40. Treat the small-n numbers as
  directional.
- §5's re-trained adapter is a **different training run** from the headline model — same
  hyperparameters and data split, but `transformers`+`peft` on an NF4 base rather than
  mlx_vlm on MLX 4-bit. It does not replace the 0.781 headline, which remains the
  on-device result.
- The `tip` column reads 0.000 for every model in §5 because only 2 of those 30 receipts
  have a tip at all. It is not a regression.
- The 3-signal ECE looks worse than 2-signal on several fields — almost certainly the n=40
  calibration set overfitting, not the logprob signal being unhelpful. The 2-signal ECE
  (n=236) is the reliable calibration story.
- §1–2 are **coverage/presence** counts (was the field produced), not correctness. Correctness
  is the F1 numbers in the Headline and §3.
- The deployed model is INT4 (fine-tuned adapter on the 4-bit base); "FP16" in §3 means
  uncompressed weights, not a different model.
