# ReceiptVLM — Receipt-to-JSON

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

Extraction is done by a small vision-language model — Qwen2.5-VL-3B, fine-tuned with QLoRA
on the WildReceipt dataset — running on-device on Apple Silicon through MLX-VLM, so receipts
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

**Quantization** — FP16 vs. INT8 vs. INT4 side by side (60-receipt subset):

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

**Receipts tab — analyze a receipt.** Drag & drop a receipt photo (or paste an image URL,
or pick one from the WildReceipt sample set). The image is run through the fine-tuned model
live — HEIC/iPhone photos included — and the extracted fields (store, date, tax, tip,
subtotal, total) and line items are shown in a clean statement card, headed by the store
name and a merchant-category chip (grocery / dining / fuel / …). Confidence is computed per
field but hidden from the display and logged to `logs/confidence.jsonl` instead.

**Overview tab — your spending dashboard.** Built live from the receipts you've analyzed
this session:
- a large **total spent** figure, plus receipts-analyzed and average-per-receipt tiles;
- **spend by month** (bar chart) and **spend by category** (donut, colorblind-safe palette);
- a **recent transactions** list, each row tagged with its colored merchant-category chip.

A **Dark / Light theme toggle** sits in the top-right, and the layout is responsive down to
phone width. Category buckets are inferred heuristically from the store name and items, so
they're directional rather than authoritative.

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
serve.py       FastAPI serving layer (+ app/streamlit_app.py front end)
```

## Repo layout

```
data/          wildreceipt/ (gitignored) + processed JSON predictions
src/           the pipeline scripts above
app/           streamlit_app.py — the UI
checkpoints/   trained LoRA adapters (gitignored)
logs/          per-session confidence logs (reset on each backend start)
```

## Stack

Qwen2.5-VL-3B · MLX-VLM · QLoRA · FastAPI · Streamlit · Plotly · Tesseract (baseline).
