# Receipt-to-JSON

This project turns a photo of a receipt into structured data. Given an image, it
extracts the merchant, date, tax, tip, subtotal, total, and the individual line
items (name and price), and returns them as JSON.

The extraction is done by a small vision-language model, Qwen2.5-VL-3B, fine-tuned
with QLoRA on the WildReceipt dataset. The whole thing runs on-device on Apple
Silicon through MLX-VLM, so receipts never leave the machine.

The goal is to be more useful than plain OCR. We compare the fine-tuned model
against a rules-based OCR-plus-regex baseline, and we report accuracy per field
rather than as a single blended number, since getting the total right matters
differently than getting a line item right. Each extracted value also comes with a
confidence score, so a user can see which fields are reliable and which are worth
double-checking.

We also measure how much accuracy is lost when the model is quantized (compressed to
FP16, INT8, and INT4) to run faster and smaller on-device.

Stack: Qwen2.5-VL, MLX-VLM, FastAPI for serving, and a Streamlit interface.

Built as a CS6140 (Machine Learning) project at Northeastern University.
