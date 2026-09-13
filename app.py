"""Gradio entrypoint for the hosted Space (ZeroGPU).

Lives at the repo root because that is where Spaces looks for it. The Streamlit app in
app/streamlit_app.py stays the on-device front end talking to the MLX FastAPI backend;
this one runs the transformers/CUDA backend in-process, because a Space is a single
process with no uvicorn to call and ZeroGPU only supports the Gradio SDK.

Structural differences from the Streamlit app, all forced by the execution model:

  * No reruns. Streamlit re-executed the whole script on any interaction, so a render
    function could just read current widget values. Here every dependency is an explicit
    event edge, which also means the upload handler fires exactly once per file -- the
    md5 de-duplication the Streamlit app needed to stop /infer being re-POSTed on every
    rerun is simply gone.

  * Per-session state. serve.py kept the uploaded-receipt list at module scope, which is
    right for a single-user localhost demo and wrong for a public URL: every visitor
    would pool into one dashboard and see each other's receipts. The list lives in
    gr.State, so it is per browser session.

  * No URL-fetch input. Both apps expose upload and the bundled samples only. Streamlit
    used to accept an image URL and fetch it server-side; on a public Space that is a
    server-side request forgery vector against anything the container can reach, so it
    was dropped from both. Nothing stops it being added back behind a
    scheme/size/redirect allowlist.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import gradio as gr

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from src import render, samples
from src.categorize import infer_category
from src.display_format import format_item_name
from src.pipeline import (
    ConfidenceScorer, PREDICTION_FIELDS, categories_payload, dashboard_payload,
    load_jsonl,
)

# spaces provides the ZeroGPU decorator. It is a no-op off-Spaces by design, but is not
# installed in the MLX requirements set, so running locally must not require it.
try:
    import spaces

    _gpu = spaces.GPU
except ImportError:  # pragma: no cover - depends on the deployment target
    def _gpu(*dargs, **dkwargs):
        def wrap(fn):
            return fn
        return wrap(dargs[0]) if dargs and callable(dargs[0]) else wrap

PROC_ROOT = REPO_ROOT / "data" / "processed"
IMG_ROOT = REPO_ROOT / "data" / "wildreceipt"

# Cached predictions for the bundled sample receipts, so a visitor can see the model's
# output on a known receipt without spending their GPU quota.
PREDICTIONS = load_jsonl(PROC_ROOT / "finetuned_test.jsonl")
GROUND_TRUTH = load_jsonl(PROC_ROOT / "test.jsonl")

# Selection and labelling live in src/samples.py so build_space.py ships images for
# exactly the receipts this dropdown offers.
SAMPLE_ORDER = samples.pick(PREDICTIONS, IMG_ROOT)
SAMPLE_LABELS = samples.label_map(PREDICTIONS, SAMPLE_ORDER)
SAMPLE_CHOICES = list(SAMPLE_LABELS)

MAX_UPLOAD_BYTES = 12 * 1024 * 1024  # public endpoint: cap before the GPU is involved

SCORER = ConfidenceScorer()

# --- model ------------------------------------------------------------------------
# ZeroGPU wants the model placed on cuda during startup rather than lazily inside the
# @spaces.GPU function, so this runs at import. A failure here is reported in the UI
# instead of crashing the Space, so the sample receipts still work.
MODEL = None
MODEL_ERROR: str | None = None
if os.environ.get("RECEIPTVLM_SAMPLES_ONLY"):
    # Set this on any deployment whose adapter has not passed
    # scripts/validate_peft_adapter.py. The converted MLX adapter currently scores 0.288
    # micro-F1 on the fp16 base against the MLX run's 0.785 (see that script's header),
    # and serving predictions that weak under a README advertising 0.781 would
    # misrepresent the model. Samples still show real fine-tuned output, because they are
    # the MLX run's cached predictions.
    MODEL_ERROR = ("live inference is disabled on this deployment until the adapter "
                   "passes scripts/validate_peft_adapter.py")
elif os.environ.get("RECEIPTVLM_SKIP_MODEL"):
    # For UI work: loading the 3B base costs a multi-GB download and minutes of startup,
    # which is pure overhead when iterating on layout or CSS. The sample receipts serve
    # cached predictions and exercise the same render path.
    MODEL_ERROR = "skipped (RECEIPTVLM_SKIP_MODEL is set)"
else:
    try:
        from src.backend_hf import ReceiptModel, analyze

        MODEL = ReceiptModel().load()
    except Exception as exc:  # pragma: no cover - depends on weights/hardware
        MODEL_ERROR = f"{type(exc).__name__}: {exc}"


@_gpu(duration=120)
def run_inference(image, filename: str) -> dict:
    """Extract one receipt on the GPU.

    120s ceiling: the fine-tune emits up to 1536 tokens and a dense multi-item receipt is
    the slow case. Shorter durations get better ZeroGPU queue priority, so this is not set
    higher than needed.
    """

    return analyze(MODEL, SCORER, image, filename)


# --- handlers ---------------------------------------------------------------------

def _status(msg: str, theme: str) -> str:
    return render.wrap(f'<div class="empty-note">{msg}</div>', theme)


def on_upload(file_path, theme: str, session: list):
    """Run a freshly uploaded photo through the model.

    A generator, not a plain function, so the photo appears the moment it is uploaded
    instead of after inference. Returning both outputs together meant the panel *and* the
    image waited on a ~11s model call, which reads as the upload being slow. Streamlit got
    this for free from its top-down rerun; Gradio needs the two yields.
    """

    if not file_path:
        yield (_status("Upload a receipt photo to begin.", theme), "", session)
        return

    # Checked before touching the file: on a samples-only deployment this is the common
    # path, and there is no reason to read an upload we will not run.
    if MODEL is None:
        raise gr.Error(
            "Live inference is unavailable on this instance "
            f"({MODEL_ERROR}). The sample receipts still work."
        )

    path = Path(file_path)
    size = path.stat().st_size
    if size > MAX_UPLOAD_BYTES:
        raise gr.Error(
            f"That image is {size / 1e6:.1f} MB; the limit is "
            f"{MAX_UPLOAD_BYTES / 1e6:.0f} MB. Try a smaller photo."
        )

    image_bytes = path.read_bytes()
    try:
        from PIL import Image

        image = Image.open(path)
        image.load()
    except Exception as exc:
        raise gr.Error(f"Couldn't read that file as an image: {exc}")

    # First yield: show the photo and a working note straight away. Session is passed
    # through unchanged so an interrupted run cannot leave a half-built entry behind.
    frame = render.wrap(render.receipt_frame(image_bytes), theme)
    yield frame, render.wrap(render.analyzing_panel(theme), theme), session

    result = run_inference(image, path.name)
    session = list(session) + [{
        "timestamp": result["timestamp"],
        "filename": result["filename"],
        "prediction": result["prediction"],
    }]
    panel = render.wrap(render.detail_panel(
        result["prediction"], confidence=result["confidence"],
        category=infer_category(result["prediction"]), theme=theme,
        format_item_name=format_item_name), theme)
    yield frame, panel, session


def on_upload_cleared(theme: str):
    """Reset both panels when the file is removed.

    Needed because the inference is bound to .upload() rather than .change(): .upload()
    fires only when a file actually arrives, so clearing the input has to be handled
    separately.
    """

    return (_status("Upload a receipt photo to begin.", theme),
            _status("Extracted details will appear here.", theme))


def on_sample(label: str, compare: bool, theme: str):
    """Show a bundled receipt's cached prediction, optionally beside the reference."""

    image_id = SAMPLE_LABELS.get(label)
    if image_id is None:
        return (_status("No sample selected.", theme), "")

    rec = PREDICTIONS[image_id]
    prediction = {k: rec.get(k) for k in PREDICTION_FIELDS}
    frame = render.wrap(
        render.receipt_frame((IMG_ROOT / image_id).read_bytes()), theme)
    panel = render.wrap(render.detail_panel(
        prediction, confidence=SCORER.field_confidence(rec),
        ground_truth=GROUND_TRUTH.get(image_id) if compare else None,
        category=infer_category(prediction), theme=theme,
        format_item_name=format_item_name), theme)
    return frame, panel


def render_overview(session: list, theme: str):
    """Rebuild the dashboard from this session's uploads only."""

    dash = dashboard_payload(session)
    cats = categories_payload(session)
    # recent/caveat are stripped here and emitted separately below, so the charts sit
    # between the stat cards and the transaction list -- the order streamlit_app.py uses.
    # transactions_panel exists precisely so a caller can place it after the plots.
    blocks = render.wrap(
        render.overview_blocks({**dash, "recent": [], "caveat": None}, cats,
                               theme=theme, infer_category=infer_category), theme)
    month_fig = (render.month_figure(dash["by_month"], theme)
                 if dash.get("by_month") else None)
    cat_fig = render.category_figure(cats["categories"], theme)

    recent = dash.get("recent") or []
    if recent:
        tail = '<div class="section-title">Recent transactions</div>'
        tail += render.transactions_panel(recent, theme=theme,
                                          infer_category=infer_category)
        if dash.get("caveat"):
            tail += ('<div class="section-sub" style="margin-top:0.8rem;">'
                     f'{dash["caveat"]}</div>')
        transactions = render.wrap(tail, theme)
    else:
        transactions = ""

    return (blocks,
            gr.update(value=month_fig, visible=month_fig is not None),
            gr.update(value=cat_fig, visible=cat_fig is not None),
            transactions)


def on_source_change(source: str):
    """Show the input controls for the chosen source, hide the others."""

    is_upload = source == "Upload"
    return (gr.update(visible=is_upload),        # upload
            gr.update(visible=not is_upload),    # sample dropdown
            gr.update(visible=not is_upload))    # compare checkbox


def on_source_panel(source: str, label: str, compare: bool, theme: str):
    """Render the panel for the newly chosen source.

    Without this, switching to "Sample receipts" would show a dropdown with a
    preselected value but an empty panel until the visitor changed the selection --
    .change() fires on change, not on reveal.
    """

    if source == "Upload":
        return (_status("Upload a receipt photo to begin.", theme), "")
    return on_sample(label, compare, theme)


def brand_header(theme: str) -> str:
    return render.wrap(
        '<p class="brand-mark">Receipt<span class="dot">VLM</span></p>'
        '<p class="brand-tag">Track every dollar, straight from your receipts.</p>',
        theme)


# --- layout -----------------------------------------------------------------------

SHELL_CSS = render.full_css() + render.upload_css(MAX_UPLOAD_BYTES) + """
/* Full-bleed: Gradio caps its container at a fixed width by default, which leaves dead
   space on a wide monitor. Let it take the whole viewport and rely on padding for the
   gutters instead. The receipt image keeps its own max-width in render.py so a photo
   does not scale up with the window. */
.gradio-container {
  max-width: 100% !important;
  width: 100% !important;
  margin: 0 !important;
  padding-left: clamp(0.75rem, 2.5vw, 3rem) !important;
  padding-right: clamp(0.75rem, 2.5vw, 3rem) !important;
}
"""

# Repainting Gradio's own widgets on a theme change is done client-side: the palette is
# declared at container level, so flipping one class restyles every widget at once. Going
# through Python would mean listing every component as an output of the theme radio.
THEME_JS = f"""
(theme) => {{
  const root = document.querySelector('.gradio-container');
  if (root) root.classList.toggle('{render.ROOT_LIGHT_CLASS}', theme === 'Light');
  return theme;
}}
"""

with gr.Blocks(css=SHELL_CSS, title="ReceiptVLM",
               analytics_enabled=False) as demo:
    session_state = gr.State([])

    with gr.Row():
        with gr.Column(scale=3):
            header = gr.HTML(brand_header("Dark"))
        with gr.Column(scale=1, min_width=160):
            # elem_classes gives the stylesheet a stable hook. Gradio's own markup is
            # Svelte with per-build hashed class names (svelte-1nguped and friends), so
            # styling it directly would break on any gradio upgrade.
            theme_radio = gr.Radio(["Dark", "Light"], value="Dark", label="Theme",
                                   show_label=False, container=False,
                                   elem_classes=["rv-seg"])

    if MODEL is None:
        # theme=None: this banner is built once at startup and never re-rendered, so it
        # has to follow the container palette rather than be pinned to the initial theme.
        gr.HTML(render.wrap(
            '<div class="panel"><div class="empty-note">Live inference is unavailable '
            f'on this instance ({MODEL_ERROR}). The sample receipts below still '
            'work.</div></div>', None))

    with gr.Tabs():
        with gr.Tab("Receipts"):
            source = gr.Radio(["Upload", "Sample receipts"], value="Upload",
                              label="Source", container=False,
                              elem_classes=["rv-seg"])

            # Visibility is toggled on these leaf components directly rather than on a
            # gr.Group wrapper around them. Verified on gradio 5.49.1: a gr.Group created
            # with visible=False never renders its children into the DOM at all, and a
            # later gr.update(visible=True) on the group does not materialize them --
            # whereas per-component visibility updates apply correctly.
            # gr.File, not gr.Image: gr.Image decodes and re-encodes, which discards the
            # original bytes -- and HEIC needs pillow-heif to open them at all.
            gr.HTML(render.wrap(
                '<div class="upload-caption">Drag &amp; drop a receipt photo here, or '
                'browse your files.</div>', None), elem_classes=["rv-caption"])
            # show_label=False: the floating "Receipt photo" chip overlaps the Upload
            # pill inside the dropzone, and Streamlit's uploader is unlabelled too.
            upload = gr.File(label="Receipt photo", show_label=False,
                             file_types=[".png", ".jpg", ".jpeg", ".webp",
                                         ".heic", ".heif"],
                             type="filepath", visible=True,
                             elem_classes=["rv-file"])
            sample = gr.Dropdown(SAMPLE_CHOICES or ["(no samples bundled)"],
                                 value=(SAMPLE_CHOICES[0]
                                        if SAMPLE_CHOICES else None),
                                 label="Sample receipt", visible=False,
                                 elem_classes=["rv-select"])
            compare = gr.Checkbox(False, label="Compare to reference", visible=False,
                                  elem_classes=["rv-check"])

            with gr.Row():
                # 5:7 reproduces the Streamlit app's [1, 1.4] split; gr.Column scale must
                # be an integer.
                with gr.Column(scale=5):
                    image_html = gr.HTML(_status("Upload a receipt photo to begin.",
                                                 "Dark"))
                with gr.Column(scale=7):
                    detail_html = gr.HTML(
                        _status("Extracted details will appear here.", "Dark"))

        with gr.Tab("Overview") as overview_tab:
            overview_html = gr.HTML(_status(
                "Analyze a receipt to build your spending overview.", "Dark"))
            month_plot = gr.Plot(label="Spending by month", visible=False,
                                 elem_classes=["rv-plot"])
            cat_plot = gr.Plot(label="Spending by category", visible=False,
                               elem_classes=["rv-plot"])
            # After the charts, matching streamlit_app.py's dashboard order.
            transactions_html = gr.HTML("")

    # --- events -------------------------------------------------------------------
    source.change(on_source_change, inputs=source,
                  outputs=[upload, sample, compare]).then(
        on_source_panel, inputs=[source, sample, compare, theme_radio],
        outputs=[image_html, detail_html])

    # .upload() rather than .change(): gr.File fires change twice for a single selection
    # (once locally, once when the resolved server-side path lands), which ran inference
    # twice and appended the same receipt to the session list twice -- one upload showed
    # as two rows in Overview. .upload() fires once, when a file actually arrives.
    upload.upload(on_upload, inputs=[upload, theme_radio, session_state],
                  outputs=[image_html, detail_html, session_state],
                  # Gradio's built-in indicator draws an orange progress bar and a
                  # "queue: 1/1" overlay across the output blocks. render.analyzing_panel
                  # already communicates the wait, so the default chrome is redundant.
                  show_progress="hidden")
    upload.clear(on_upload_cleared, inputs=theme_radio,
                 outputs=[image_html, detail_html])

    for control in (sample, compare):
        control.change(on_sample, inputs=[sample, compare, theme_radio],
                       outputs=[image_html, detail_html], show_progress="hidden")

    # Overview is rebuilt on tab open rather than on every upload, so a visitor
    # analyzing several receipts does not pay for a chart redraw each time.
    overview_tab.select(render_overview, inputs=[session_state, theme_radio],
                        outputs=[overview_html, month_plot, cat_plot,
                                 transactions_html])

    # Theme: the client-side handler restyles Gradio's widgets by toggling the root
    # class, then the server re-emits the .rv blocks with the other wrapper class and
    # redraws the charts (plotly cannot read CSS variables).
    theme_radio.change(None, inputs=theme_radio, outputs=None, js=THEME_JS)
    theme_radio.change(brand_header, inputs=theme_radio, outputs=header).then(
        on_source_panel, inputs=[source, sample, compare, theme_radio],
        outputs=[image_html, detail_html]).then(
        render_overview, inputs=[session_state, theme_radio],
        outputs=[overview_html, month_plot, cat_plot, transactions_html])

if __name__ == "__main__":
    demo.queue().launch()
