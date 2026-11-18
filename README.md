# ReceiptVLM - Receipt-to-JSON

Turn a photo of a receipt into structured data. Given an image, ReceiptVLM extracts the
merchant, date, tax, tip, subtotal, total, and the individual line items (name and price),
and returns them as JSON:

```json
{
  "store": "SAFEWAY",
  "date": "12/22/17",
  "tax": "4.24",
  "tip": null,
  "subtotal": "84.85",
  "total": "89.09",
  "line_items": [{"name": "BATHTOWEL", "price": "7.97"}, {"name": "OPPBT", "price": "4.00"}]
}
```

Extraction is done by a small vision-language model - Qwen2.5-VL-3B, fine-tuned with QLoRA
on the WildReceipt dataset - running on-device on Apple Silicon through MLX-VLM, so receipts
never leave the machine. A Streamlit app gives a spending dashboard on top.

Built as a CS6140 (Machine Learning) project at Northeastern University.

## Results

Per-field micro-F1 on the 472-receipt WildReceipt test set (95% bootstrap CI):

| Model | micro-F1 |
|---|---|
| OCR + regex baseline (Tesseract) | 0.115 |
| Qwen2.5-VL-3B zero-shot (no fine-tune) | 0.212 |
| **Qwen2.5-VL-3B + QLoRA (ours)** | **0.781** [0.756, 0.802] |

Fine-tuning beats the zero-shot model ~3.7× and the OCR baseline ~6.8×; the gap is
statistically significant (paired bootstrap, p≈0). The biggest single win came from raising
the training/inference image resolution from 448×448 to 768×1024 (0.525 → 0.724 before an
evaluation-harness fix, → 0.781 after): tall multi-item receipts were illegible when squashed.

**Quantization** - FP16 vs. INT8 vs. INT4 side by side (60-receipt subset):

| Precision | micro-F1 | Latency / receipt | Peak memory |
|---|---|---|---|
| FP16 | 0.809 | 17.8 s | 8.8 GB |
| INT8 | 0.816 | 12.0 s | 5.4 GB |
| **INT4** | **0.796** | **9.5 s** | **4.4 GB** |

The accuracy differences between tiers are within noise, so **INT4** is the on-device
choice: half the memory and ~2× faster than FP16 with no meaningful accuracy loss.

Each extracted field also gets a calibrated confidence score (token logprob + arithmetic
consistency + format validity), logged per receipt to `logs/confidence.jsonl`.

## Setup

Apple Silicon (tested on an M-series Mac). Python 3.9+.

```bash
pip install -r requirements.txt          # mlx-vlm, transformers==4.49, fastapi, streamlit, …
brew install tesseract                    # for the OCR baseline only
```

WildReceipt (images + annotations) downloads to `data/wildreceipt/`:

```bash
curl -L -o data/wildreceipt.tar https://download.openmmlab.com/mmocr/data/wildreceipt.tar
mkdir -p data/wildreceipt && tar -xf data/wildreceipt.tar -C data/wildreceipt --strip-components=1
```

## The app

A Streamlit front end (talking to the FastAPI backend) that turns the model into a
Rocket Money-style spending tracker. Two terminals, from the repo root:

```bash
uvicorn src.serve:app --port 8000         # FastAPI backend (loads the model + adapter)
streamlit run app/streamlit_app.py        # UI at http://localhost:8501
```

**Receipts tab - analyze a receipt.** Drag & drop a receipt photo (or pick one from the
WildReceipt sample set). The image is run through the fine-tuned model
live - HEIC/iPhone photos included - and the extracted fields (store, date, tax, tip,
subtotal, total) and line items are shown in a clean statement card, headed by the store
name and a merchant-category chip (grocery / dining / fuel / …). Confidence is computed per
field but hidden from the display and logged to `logs/confidence.jsonl` instead.

**Overview tab - your spending dashboard.** Built live from the receipts you've analyzed
this session:
- a large **total spent** figure, plus receipts-analyzed and average-per-receipt tiles;
- **spend by month** (bar chart) and **spend by category** (donut, colorblind-safe palette);
- a **recent transactions** list, each row tagged with its colored merchant-category chip.

A **Dark / Light theme toggle** sits in the top-right, and the layout is responsive down to
phone width. Category buckets are inferred heuristically from the store name and items, so
they're directional rather than authoritative.

### Two inference backends

Live inference runs on either of two stacks, picked automatically at startup. `/health`
reports which one is active.

| backend | adapter | runs on | latency / receipt | memory |
|---|---|---|---|---|
| `mlx_vlm` (preferred when present) | `checkpoints/final` | Apple Silicon | 6.2 s (INT4) | 4.6 GB |
| `transformers` + `peft` (fallback) | `checkpoints/final_peft` | CUDA, MPS or CPU | 11.6 s (MPS fp16) | 7.6 GB |

Both measured over the same 60 test receipts with `scripts/benchmark_local.py`; see
RESULTS.md §5 for percentiles, the per-instrument memory caveats, and why these MLX
timings run faster than §3's. `src/serve.py` prefers MLX because it is 1.9× faster and uses
about 60% of the memory, and falls back to `src/backend_hf.py` anywhere else - so the
Streamlit stack now works on a Linux GPU box, not only on Apple Silicon. If neither is
available the API still serves cached predictions and `/infer` returns 503.

**The two adapters are not interchangeable.** Each was fit to a different base, and
loading the MLX adapter through `transformers` is the 0.288 micro-F1 failure documented
below. Which is why they are separate files rather than one shared checkpoint.

Both stacks share `src/pipeline.py` (confidence, spend aggregation, categories) and
`src/render.py` (markup and charts), so the Streamlit and Gradio UIs cannot drift apart on
badges or totals.

## Pipeline

Each stage is one script under `src/`, all reading/writing `data/processed/*.jsonl`:

```
prep.py        WildReceipt boxes → per-receipt JSON (ground truth)
baseline.py    OCR + regex extraction (non-ML floor)
zeroshot.py    run the base/fine-tuned model → predictions (also the inference path)
train.py       QLoRA fine-tune → checkpoints/final/adapters.safetensors
sweep.py       hyperparameter sweep driver for train.py
eval.py        per-field micro-F1 + bootstrap CI + paired significance test
taxonomy.py    failure taxonomy from eval logs
quantize.py    FP16/INT8/INT4 sweep (F1 + latency + peak memory)
confidence.py  calibrated per-field confidence + risk-coverage
repair.py      JSON repair layer for malformed model output
categorize.py  heuristic merchant-type buckets (grocery / dining / fuel / …)
display_format.py  splits concatenated line-item names for display (e.g. "GrossesWasser" -> "Grosses Wasser")
serve.py       FastAPI serving layer (+ app/streamlit_app.py front end)
```

## Hosting it publicly (Hugging Face Space)

The MLX stack above is Apple-Silicon-only, so it cannot be hosted anywhere but a Mac. For
a public always-on demo there is a second path: a Gradio app on **ZeroGPU**, which is free
(the GPU quota is charged to the visitor, not the owner) but CUDA- and Gradio-only.

```bash
pip install -r requirements-spaces.txt              # separate from requirements.txt on purpose

python scripts/convert_mlx_adapter_to_peft.py       # MLX LoRA -> peft format
python scripts/verify_adapter_equivalence.py        # proves the conversion is exact
python scripts/validate_peft_adapter.py --limit 30  # measures fp16 accuracy vs. the MLX run

python scripts/build_space.py --out build/space     # stage the deployable Space
cd build/space && python app.py                     # sanity-check, then push to the Space remote
```

`requirements-spaces.txt` is deliberately a separate file: `requirements.txt` pins
`transformers==4.49.0` because newer releases break mlx_vlm's preprocessing, while the
Space needs a newer transformers for the restructured Qwen2.5-VL the converted adapter
targets.

### Status: live inference works, served in fp16

Re-training the adapter on CUDA closed the gap. Scored on the same 30 test receipts with
the project's own `eval.py`:

| adapter / serving | micro-F1 | vs. MLX run |
|---|---|---|
| MLX + QLoRA (on-device reference) | 0.785 | - |
| **re-trained, served fp16** | **0.760** [0.691, 0.830] | −0.025, p=0.542 |
| re-trained, served NF4 | 0.722 [0.636, 0.815] | −0.063, p=0.242 |
| original adapter converted to fp16 | 0.288 | −0.497, p≈0 |

Per field, fp16 serving matches the MLX run on `store` (0.900), `tax` (0.905) and
`line_items.name` (0.729 vs 0.730), and **exceeds** it on `subtotal` (0.895 vs 0.811) and
`total` (0.929 vs 0.873). Only `date` (−0.045) and `line_items.price` (−0.071) trail.

Two results worth keeping:

**Serve it in fp16, not NF4** - even though it was trained with QLoRA on an NF4 base.
fp16 scores +0.038 higher, because NF4 is a close enough approximation that the adapter's
learned corrections still apply while the base weights themselves are more accurate. This
also removes bitsandbytes from the deployment entirely, which matters because HF ZeroGPU
does not reliably support it.

**Not all 4-bit schemes transfer alike.** MLX's affine 4-bit → fp16 collapsed (0.785 →
0.288); bitsandbytes NF4 → fp16 transfers cleanly and improves. The first failure was
about MLX's quantizer being far from fp16, not about quantized→unquantized transfer being
impossible in general.

`RECEIPTVLM_SAMPLES_ONLY=1` remains available on the Space to serve only the cached
sample predictions with upload disabled, but it is no longer needed.

<details>
<summary>Why the first attempt failed (kept for the record)</summary>

The port itself was not the problem; four measurements ruled that out:

| check | result |
|---|---|
| conversion algebra (`verify_adapter_equivalence.py`) | exact to 1.5e-8 |
| base model through the same harness (`--no-adapter`) | 0.218, reproducing the published zero-shot 0.212 |
| LoRA scale | 0.125 both sides - mlx_vlm reads `alpha` from the saved config |
| update magnitude `‖ΔW‖/‖W‖` | median 1.09e-2, normal for a working LoRA |

The cause is the base model. The adapter was fit by QLoRA to correct
`mlx-community/Qwen2.5-VL-3B-Instruct-4bit`; against fp16 weights its ~1% perturbation
points elsewhere. The predictions show it plainly - the model still *reads* the receipt
but reverts to the base model's conventions instead of the fine-tune's
(`CHO EUN KOREAN RESTAUR` where the WildReceipt target is `CHOEUN KOREANRESTAURAN`,
`24,65` for `24.65`, unit price where the target is the extended price). Large numerals
survive (tax/subtotal/total ≈0.81 F1) because they need no convention; store, date and
line items collapse (0.30 / 0.53 / 0.19) because they do.

</details>

### Re-training the adapter

[`notebooks/kaggle_retrain_qlora.ipynb`](notebooks/kaggle_retrain_qlora.ipynb) re-fits the
adapter with `transformers` + `peft`, mirroring `src/train.py`'s run exactly - rank 4,
alpha 0.5 (scale 0.125), lr 1e-4, Adam (not AdamW), element-wise gradient clipping at 1.0,
NaN/Inf steps skipped, batch size 1, the same `Random(0)` split (1140 train / 127 val →
2280 steps), 768×1024 aspect-preserving sizing, and completion-only loss. The only
deliberate changes are the framework and the base.

It trains in **NF4** - not for memory but because a 16 GB T4 cannot hold the fp16 base
plus activations. Serving, though, is best done in **fp16**: measured at +0.038 micro-F1
over NF4 serving (0.760 vs 0.722), and it keeps bitsandbytes out of the deployment. Free on Kaggle
(~30 GPU-hours/week, phone verification required for both GPU and internet); the run is
~2-4 hours, so use **Save & Run All (Commit)** so it survives closing the tab.

Pick **GPU T4 x2**, not the P100. Kaggle's P100 is Pascal (`sm_60`) and current PyTorch
wheels and bitsandbytes builds ship no kernels for it - the model load fails with
`CUDA error: no kernel image is available for execution on the device` after downloading
7 GB. The notebook's first cell checks the device's compute capability against
`torch.cuda.get_arch_list()` and stops immediately if it won't work. It also pins torch to
whatever Kaggle preinstalled, via a constraints file, so installing the other packages
can't pull in a torch wheel built for a narrower set of architectures. The notebook downloads WildReceipt itself and ends with an in-session accuracy gate
(micro-F1 + bootstrap CI + paired test against the MLX run) using the project's own
`eval.py`, because `--load-4bit` cannot run on Apple Silicon.

Attach one small Kaggle Dataset (~1 MB) containing:

```
data/processed/train.jsonl           # required, to train
data/processed/test.jsonl            # for the gate
data/processed/finetuned_test.jsonl  # MLX reference, for the paired test
src/eval.py src/repair.py src/schema.py src/zeroshot.py
```

Folder layout inside the dataset doesn't matter - the notebook searches any depth - with
one constraint: the four `.py` files must sit **in the same folder as each other**, since
they import by bare name. If the scoring files are absent the gate skips with
instructions rather than failing a good training run.

Drop the resulting `final_peft/` into `checkpoints/final_peft` and serve it as-is - fp16
is the default, so no environment variable is needed:

```bash
python app.py                            # fp16, the measured-best serving path
```

[`notebooks/kaggle_eval_adapter.ipynb`](notebooks/kaggle_eval_adapter.ipynb) scores any
saved adapter without training (~15 min), with a `LOAD_4BIT` switch to compare the two
serving precisions. That is how the fp16-vs-NF4 result above was measured. `--load-4bit`
on `scripts/validate_peft_adapter.py` does the same locally, but needs CUDA - bitsandbytes
has no MPS backend, so an NF4 comparison cannot run on Apple Silicon.

The alternative - **dequantizing the MLX 4-bit checkpoint into HF format** so the original
adapter meets the weights it was fit to - would avoid retraining, but needs an MLX→HF key
remap plus per-group dequantization and is unproven here.

Two behaviour differences from the local app, both required by a public multi-user URL:
the uploaded-receipt list is per browser session rather than a module global, and the
server-side image-URL fetch is not exposed.

## Repo layout

```
data/          wildreceipt/ (gitignored) + processed JSON predictions
src/           the pipeline scripts above
               + pipeline.py / render.py / samples.py  (framework-free core, shared)
               + backend_hf.py                          (transformers/CUDA inference)
               + schema.py                              (prompt + schema constants)
app/           streamlit_app.py - the on-device UI
app.py         Gradio entrypoint for the hosted Space
scripts/       adapter conversion, validation, Space packaging
notebooks/     kaggle_retrain_qlora.ipynb - re-fit the adapter for CUDA
checkpoints/   trained LoRA adapters (gitignored)
logs/          per-session confidence logs (reset on each backend start)
```

## Stack

Qwen2.5-VL-3B · QLoRA · Plotly · Tesseract (baseline).
On-device: MLX-VLM · FastAPI · Streamlit. Hosted: transformers · peft · Gradio · ZeroGPU.

## Collaborators

Shweta Perumal: MS DS

Rex Remigius Stephen Jothi: MS CS

Dheeraaj Pinjala: MS CS
