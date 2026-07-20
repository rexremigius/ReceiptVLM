Deliverable 1 (src/prep.py — WildReceipt boxes → JSON) is complete.

Update PROGRESS.md:
1. Tick off item 1 in the Status checklist.
2. Add a new entry at the top of the Session Log with today's date, following the
   template: what was built, what's next, and a 3-5 line "Learned" explanation covering
   the key design decision made in prep.py, the alternative(s) considered, and why
   this approach was picked (per CLAUDE.md's Learning mode section).

Base the explanation on the actual code you wrote, not a generic description.# Progress Log


## Status

Team split (~5% / 75-80% / 15% compute): **Person A** = data + eval (CPU-only) ·
**Person B** = fine-tuning + generalization (critical path, this machine) · **Person C** =
quantization + serving (inference-bound). Owner assignments for #8/#9/#11 are this
session's best-fit call, not yet confirmed by the team — flagged below.

| # | Deliverable | Status | Owner |
|---|---|---|---|
| — | MLX-VLM validated on M5/16GB · WildReceipt loaded/mapped · zero-shot baseline run | [x] done | Person B (this machine) |
| 1 | `src/prep.py` — WildReceipt boxes → JSON (1267 train / 472 test → `data/processed/*.jsonl`) | [x] done | Person A |
| 2 | `src/baseline.py` — OCR+regex extraction | [x] done | Person B |
| 3 | Zero-shot VLM baseline (Qwen2.5-VL-3B-Instruct-4bit, no adapter) — `data/processed/zeroshot_test.jsonl` | [x] done | Person B |
| 4 | `src/train.py` — QLoRA fine-tune (sweep done, winning config rank=4/alpha=0.5/lr=5e-5 promoted to `checkpoints/final/adapters.safetensors`; generalization/augmentation retraining still open) | [x] initial run / [x] sweep | Person B |
| 5 | `src/eval.py` — per-field eval harness | [ ] | Person A |
| 6 | `src/taxonomy.py` — failure taxonomy | [ ] | Person A |
| 7 | `src/quantize.py` — FP16/INT8/INT4 sweep | [ ] | Person C |
| 8 | `src/confidence.py` — calibrated confidence | [ ] | Person A *(unconfirmed)* |
| 9 | JSON repair layer | [ ] | Person C *(unconfirmed)* |
| 10 | `src/serve.py` + `app/` — FastAPI + Streamlit | [ ] | Person C |
| 11 | OOD dataset (20-40 labeled photos) | [ ] | Person A *(unconfirmed)* |

## Session Log
(most recent first — one entry per session: what was built, what's next, what was learned)

<!--
Template for each entry:

### YYYY-MM-DD
**Built:** 
**Next:** 
**Learned:** (key design decision + alternative considered, per learning mode in CLAUDE.md)
-->

### 2026-07-19
**Built:** #3 `src/zeroshot.py` (zero-shot VLM baseline, Track B). The "zero-shot baseline
run" claim already in this file/CLAUDE.md predated this environment and had no backing
artifact anywhere in the repo (same story as WildReceipt images and the mlx-vlm install —
this checked-in state doesn't match a fresh checkout) — this actually runs it. Reuses
`train.py`'s exact PROMPT/SCHEMA_KEYS so the zero-shot and fine-tuned runs are asking the
model the same question, which matters once eval.py compares them. Generation wraps JSON in
a ```json fence even when asked not to; strips that, falls back to the first `{...}` span,
and gives up (all-null record, not a guess) on anything still unparseable — a real JSON
repair layer is #9's job, not this script's. Ran full 472-receipt test split: 42/472 (8.9%)
parse failures, ~2.5s/receipt. Field coverage far exceeds the OCR+regex baseline everywhere
(e.g. date 403/472 vs. baseline's 212/472, total 322/472 vs. 156/472) — store name presence
is close (430 vs. 454) since baseline's OCR-first-line heuristic "finds" a store name even
when it's wrong, so presence alone isn't the real comparison; per-field correctness (F1
against #1's ground truth) needs #5's eval harness, not this run.
**Next:** #5 `src/eval.py` — with baseline, zero-shot, and (eventually) fine-tuned
predictions all sitting in the same `data/processed/{name}_test.jsonl` schema, this is the
first point real per-field accuracy numbers exist instead of coverage/presence proxies.
**Learned:** Kept the same image resize (448x448) and prompt as train.py rather than giving
zero-shot a more elaborate/hand-tuned prompt (which would likely improve its numbers in
isolation). The comparison that matters is zero-shot vs. fine-tuned on identical inputs —
tuning the zero-shot prompt specifically would confound that later, so left it as the exact
prompt the fine-tune trains toward.

**Built:** #4 `src/train.py` (QLoRA fine-tune, Track B). Environment was missing mlx-vlm/
Qwen weights entirely (fresh machine state, like WildReceipt images last session) — installed
mlx-vlm 0.1.15, downloaded `mlx-community/Qwen2.5-VL-3B-Instruct-4bit` (4-bit MLX checkpoint,
confirmed its language_model layers are `nn.QuantizedLinear`, i.e. true QLoRA once LoRA
adapters wrap them, not LoRA-then-downgrade). Had to pin `transformers==4.49.0` (newer
versions made Qwen's image processor "fast"-only, which hard-requires PyTorch tensors —
incompatible with mlx_vlm's own backend) and patch a key-name mismatch in mlx_vlm's own
trainer (`image_token_index` vs. Qwen's `image_token_id`) — see `requirements.txt` and
inline comments. train.py: receipt-level train/val split from `data/processed/train.jsonl`,
completion-only loss masking (assistant-turn token resolved from the tokenizer, verified
present in a real tokenized example rather than assumed), checkpointing every N steps with
only the last 2 kept, tiny-subset (`--limit`) mode for validation runs.
Ran the spec-required ~20-receipt validation run first: it caught a real bug before any
full-scale run — mlx_vlm's LoRA layer applies `alpha` as a raw multiplier on the update
(not the conventional alpha/rank scaling used elsewhere), so my first attempt at alpha=16
blew the loss up to NaN within 15 steps. Fixed by matching mlx_vlm's own convention
(alpha≈1, their CLI defaults to 0.1) plus gradient clipping as a safety net; re-ran and loss
dropped cleanly 0.42→~0.03 with val_loss stable around 0.08-0.09, no NaNs. Peak memory ~3.7GB
for a single step — the 3B model has plenty of headroom on 16GB, so no need for the
SmolVLM2-2.2B fallback.
**Next:** kick off a full QLoRA run on all 1267 train receipts (Track B), and `src/eval.py`
(Track A) so the fine-tuned checkpoint has a real per-field F1 to report once trained.

**Bug found + fixed (same day), full run redone:** while generating `finetuned_test.jsonl`
for the handoff to teammates, raw fine-tuned generations came back garbled — degenerate
repetition loops and single/double-quote-mixed pseudo-JSON (e.g. `[{'name": "SUPERSTORE"...`)
— clearly worse than the zero-shot base model, not better, which was the tell that something
was actually broken rather than just "needs more training." Root cause: `to_example()` built
each message's `content` as an already-structured list
(`[{"type": "image"}, {"type": "text", "text": ...}]`), but `apply_chat_template(...,
return_messages=True)` expects a **plain string** for `content` and does that list-wrapping
itself. Passing a pre-wrapped list caused it to wrap again, so the assistant target the model
was actually trained on was a corrupted Python-repr string around the real JSON, not the
clean JSON — the loss curve looked completely normal throughout (that's why the earlier tiny-
subset validation didn't catch it: it checked convergence, not actual generated output).
Fixed by making `content` a plain string in both messages; verified the rendered prompt has
the assistant-turn token appearing exactly once with clean JSON in the target position. Wiped
the corrupted checkpoint, reran the tiny 20-receipt validation, and this time actually
inspected generated JSON (not just the loss number): 0/5 parse failures, clean well-formed
output. Redid the full 2280-step run on the corrected data.
**Learned:** A converging loss curve is necessary but not sufficient evidence a fine-tune
worked — it only proves the model is fitting *whatever target it was actually shown*, not
that the target was the one intended. The tiny-subset validation step (mandated by CLAUDE.md
for exactly this kind of failure) needs to check the model's actual output, not just that
the loss goes down, or it can rubber-stamp a corrupted training run.

**Update (same day, full run):** ran the full QLoRA fine-tune — 1140 train / 127 val receipts
(90/10 receipt-level split off `train.jsonl`), 2 epochs = 2280 steps, ~102 minutes, no NaNs
(the alpha/gradient-clipping fix from the tiny-subset run held up at scale). Train loss
0.27 → mostly 0.01-0.07 by the end. Val loss converged fast and then plateaued: 0.095 (step
100) → 0.077 (step 400) → flat around 0.072-0.077 for the remaining ~1.9 epochs — most of
the real learning happened in well under one epoch; the rest mainly drove train loss down
further without moving val loss, i.e. mild memorization rather than continued
generalization. Worth revisiting epoch count (or adding early stopping) once eval.py can
turn this into a real accuracy number instead of a loss number. Final adapter:
`checkpoints/final/adapters.safetensors`.
**Learned:** Chose per-example training steps (effective batch size 1, one optimizer update
per receipt) over batched/grad-accumulated steps. Tried mlx_vlm's own batch-slicing
(`dataset[i:i+batch]`) first, since the reference CLI uses it — it crashes for us because HF
`datasets`' batched-slice access returns each example's `images` list nested inside another
list, and mlx_vlm's image preprocessing isn't written to unwrap that. Batch size 1 with Adam
sidesteps the bug entirely and matches "don't over-engineer" — building a custom manual
gradient-accumulation loop to get batching would be real complexity for a memory-constrained
setup that doesn't need the throughput.

**Update (hyperparameter sweep + promotion):** built `src/sweep.py` — a coordinate-descent
sweep (rank → alpha → lr, one dimension at a time with the others held fixed) instead of a
full 3×3×3 grid, since a full grid on full data would be ~13-25 machine-hours. Each of 7
trials ran on a 250-receipt subset (1 epoch, ~8.5 min each, ~1hr total), picking the best
value per dimension by minimum val_loss, then one full-data (1140 receipts, 2 epochs)
confirmation run with the winning config. Added `--ckpt-root` to `train.py` so sweep trials
write to `checkpoints/sweep/<config>/` instead of clobbering the production
`checkpoints/final/`. Result: **rank=4, alpha=0.5, lr=5e-5** beat the original
(rank=8, alpha=1.0, lr=1e-4) on full-data val_loss (0.0719 vs 0.0775) — smaller LoRA rank
(half the trainable params, ~7.5M vs ~15M) generalized better than the larger one, not just
comparably.
Built `src/quick_accuracy.py` — a rough exact-match per-field accuracy check against #1's
ground truth (NOT #5 eval.py, which Person A still owns: no bootstrap CI, no significance
test, no proper line-item alignment) — to compare configs on real field accuracy rather than
val_loss alone before promoting. Sweep-confirmed model won or tied on 7/8 metrics (store
47.0%→53.0%, line-item price precision 74.2%→76.8%, recall 72.1%→73.6%, tax/subtotal/total/tip
flat-to-slightly-up) with only date dipping slightly (75.8%→74.4%, likely noise at n=472).
Promoted: `checkpoints/sweep_best/final` → `checkpoints/final`, and its predictions →
`data/processed/finetuned_test.jsonl` (previous rank=8 predictions kept as
`_finetuned_test.old_rank8.jsonl` for reference, not deleted).
**Learned:** val_loss alone wasn't trusted as the promotion criterion even though it's the
metric the sweep optimized — checked real per-field accuracy first, since a lower loss
doesn't guarantee better extraction accuracy on fields that matter (e.g. store name, which
is exact-string-graded and could move independently of aggregate token-level loss). Also:
tip's ~96-97% "accuracy" across every model (baseline/zero-shot/both fine-tunes) is a
class-imbalance artifact (~95% of receipts have no tip, so predicting null is almost always
"correct") — flagged as a trap for #5 eval.py to handle properly (precision/recall
conditioned on tip being present), not a real signal that any model handles tips well.

### 2026-07-18
**Built:** #2 `src/baseline.py` (OCR+regex, Track A). Re-downloaded WildReceipt images
(gitignored, weren't present locally) into `data/wildreceipt/`. Tesseract OCR (grayscale +
2x upscale) over each receipt, then keyword-anchored regex per field: subtotal/tax/tip/total
take the *last* keyword-line match (bottom-most occurrence is the authoritative one when a
field is printed more than once); date is a US numeric/month regex; merchant is the first
OCR line; line_items use a line+trailing-price heuristic (rightmost money match on a line =
price, text before it = name), skipping lines already claimed by a scalar-field keyword.
Output: `data/processed/baseline_{split}.jsonl` (same schema as #1) + summary JSON.
Validated on the 20-receipt subset against #1's ground truth: Walmart/CarWash/OldCastle Pub
receipts got tax/total/date right, subtotal/merchant sometimes missed to OCR typos
("SUBTCTAL", skewed store name) — realistic classical-baseline noise, not code bugs.
Coverage on subset20: store 20/20, date 8/20, tax 6/20, subtotal 4/20, total 7/20, tip 0/20
(confirms the sparse-tip hypothesis from #1 — plain regex has nothing to find anyway).
Kicked off a full 472-receipt test-split run in the background for #5 (eval.py) to consume.
**Next:** #5 `src/eval.py` (per-field micro-F1 + bootstrap CI, Track A) now that both #1's
ground truth and #2's predictions exist on the test split. #4 `train.py` can proceed in
parallel (Track B).
**Learned:** Tried forcing Tesseract's `--psm 6` (uniform block of text) on top of grayscale+
2x upscale, since it fixed one receipt's totally-missing totals. But on a second receipt with
a sparser layout it wrecked OCR that was legible under Tesseract's default automatic page
segmentation (psm 3) — turned real text into noise. Picking psm per-receipt to maximize
coverage on a 20-image sample would be overfitting the config to that sample, not building a
generically-reasonable baseline, so kept automatic psm and accepted the resulting misses
(e.g. "SUBTOTAL" OCR'd as "SUBTCTAL" doesn't match the keyword regex) as genuine baseline
weakness for #6's taxonomy to categorize later, rather than a bug to engineer around.

### 2026-07-15
**Built:** #1 `src/prep.py`. Downloaded + extracted WildReceipt to `data/wildreceipt/`
(1267 train / 472 test). Maps 8 of 25 KIE categories → our 6-field + line_items schema;
drops+counts the other 17 (keys/Others/addr/tel/time/quantity — 35k train boxes dropped,
none schema-relevant). Output: `data/processed/{train,test}.jsonl` + `_prep_summary_*.json`.
Validated Chipotle + Walmart records against source images (all fields + line-item prices
correct); tip present in only 5.4%/4.4% of receipts, confirming the sparse-tip hypothesis.
**Next:** #2 `src/baseline.py` (OCR+regex, Track A) and #5 `src/eval.py` — both consume
`data/processed/*.jsonl`. #4 train.py can start in parallel (Track B).
**Learned:** Line-item name↔price pairing was the crux. First tried same-row clustering
(band = 0.6× median box height); it dropped most prices because WildReceipt puts the price
box on a systematically offset baseline (~35-50px below the name, sometimes above), never
the same y. Switched to nearest-unused-price matching with the band scaled to the median
*inter-item gap* — that gap (~140px) dwarfs the name→price offset, so it bridges the offset
without ever crossing into the next item's row. Alternative (zip Nth-item↔Nth-price by y)
is simpler but silently misaligns everything after any single missing/extra box; nearest-
match degrades locally instead. Also: `clean_money` regex must allow leading-dot amounts
(`.49`) or 49¢ reads as $49.
