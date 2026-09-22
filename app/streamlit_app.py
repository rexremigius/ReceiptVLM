"""Streamlit front end for the on-device demo, talking to serve.py over HTTP.

Shares src/render.py and src/pipeline.py with the Gradio Space, so both UIs render
identical badges and totals. RECEIPTVLM_API_BASE points it at a non-local backend.
"""

from __future__ import annotations

import hashlib
import os
import sys

import requests
import streamlit as st
import pillow_heif

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import render, samples
from src.categorize import infer_category
from src.display_format import format_item_name

pillow_heif.register_heif_opener()  # stock Pillow can't decode iPhone HEIC

# Overridable so the UI can point at a backend that is not on this machine (e.g. a
# tunnelled Mac), without editing the file.
API_BASE = os.environ.get("RECEIPTVLM_API_BASE", "http://127.0.0.1:8000")

st.set_page_config(page_title="ReceiptVLM", layout="wide")

if "theme" not in st.session_state:
    st.session_state.theme = "Dark"


# Streamlit's own chrome. These selectors target Streamlit internals and have no Gradio
# equivalent, which is why they are not in render.py's shared stylesheet.
STREAMLIT_CSS = """
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"],
[data-testid="stMainBlockContainer"], [data-testid="stBottomBlockContainer"] { background: var(--bg) !important; }
[data-testid="stHeader"] { display: none; }
/* Full-bleed, matching app.py. Streamlit caps its block container even under
   layout="wide", which leaves dead space on a wide monitor; the clamped padding
   keeps gutters sane from phone to ultrawide. */
[data-testid="stMainBlockContainer"] {
  padding-top: 1.4rem;
  max-width: 100%;
  padding-left: clamp(0.75rem, 2.5vw, 3rem);
  padding-right: clamp(0.75rem, 2.5vw, 3rem);
}
html, body, [data-testid="stAppViewContainer"] { color: var(--text-primary); font-family: var(--font-sans); }
[data-testid="stAppViewContainer"] *:not([data-testid="stIconMaterial"]) { font-family: inherit; }
/* dark config base sets white text on markdown containers; force theme color so light mode is readable */
[data-testid="stMarkdownContainer"], [data-testid="stMarkdown"],
[data-testid="stText"], [data-testid="stHeading"] { color: var(--text-primary) !important; }
[data-testid="stSelectbox"] div, [data-baseweb="select"] div, [data-testid="stSpinner"] div { color: var(--text-primary); }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color: var(--text-muted) !important; }

div[data-testid="stAlertContainer"] { background: var(--surface) !important; border: 1px solid var(--border) !important; border-left: 3px solid var(--green) !important; border-radius: 14px; }
div[data-testid="stAlertContainer"] p { color: var(--text-secondary) !important; font-size: 0.9rem; }
div[data-testid="stAlertContainer"] svg { fill: var(--green) !important; }

/* tab spacing via flex gap - reliable whether the tab is a div or a button */
[data-testid="stTabs"] [role="tablist"] { border-bottom: 1px solid var(--border); gap: 2.75rem !important; }
div[data-testid="stTab"], button[data-baseweb="tab"] { padding: 0.35rem 0.2rem 0.7rem !important; margin-right: 0 !important; }
div[data-testid="stTab"] p { font-size: 0.95rem !important; font-weight: 600; color: var(--text-muted); }
div[data-testid="stTab"][aria-selected="true"] p { color: var(--text-primary); }
div[data-testid="stTab"] .react-aria-SelectionIndicator,
div[data-testid="stTab"][data-selected="true"]::after { background: var(--green) !important; }

div[data-testid="stRadioGroup"] { display: inline-flex; gap: 3px; background: var(--surface); border: 1px solid var(--border); border-radius: 999px; padding: 4px; }
label[data-testid="stRadioOption"] { margin: 0 !important; min-height: 0 !important; }
label[data-testid="stRadioOption"] > div > div > div:not([data-testid="stMarkdownContainer"]) { display: none; }
label[data-testid="stRadioOption"] div[data-testid="stMarkdownContainer"] p { padding: 0.32rem 0.95rem; border-radius: 999px; margin: 0; font-size: 0.83rem; font-weight: 600; color: var(--text-secondary); white-space: nowrap; }
label[data-testid="stRadioOption"][data-selected="true"] div[data-testid="stMarkdownContainer"] p { background: var(--green); color: var(--on-green); }

[data-testid="stWidgetLabel"] p { font-size: 0.82rem; color: var(--text-secondary); }
[data-testid="stSelectbox"] > div > div { font-size: 0.9rem; background: var(--surface); border-radius: 12px; border-color: var(--border); }
[data-testid="stCheckbox"] p { color: var(--text-secondary); font-size: 0.85rem; }
[data-baseweb="popover"] [role="listbox"], [data-baseweb="menu"], ul[role="listbox"] { background: var(--surface) !important; }
[data-baseweb="popover"] li, ul[role="listbox"] li { color: var(--text-primary) !important; }
[data-baseweb="popover"] li:hover { background: var(--surface-2) !important; }

[data-testid="stFileUploaderDropzone"] { padding: 1.1rem !important; border-radius: 16px !important; border: 1.5px dashed var(--border) !important; background: var(--surface) !important; }
[data-testid="stFileUploaderDropzone"] svg { fill: var(--green) !important; }
[data-testid="stFileUploaderDropzoneInstructions"] span { font-size: 0.82rem; color: var(--text-secondary); }
[data-testid="stBaseButton-secondary"] { background: var(--green) !important; color: var(--on-green) !important; border: none !important; font-weight: 700 !important; border-radius: 999px !important; }
/* -webkit-text-fill-color beats a plain color on inputs, else typed text stays white in light mode */
[data-testid="stTextInput"] input, [data-baseweb="input"] input, [data-baseweb="base-input"] input {
  background: var(--surface) !important; border-radius: 12px !important; border-color: var(--border) !important;
  color: var(--text-primary) !important; -webkit-text-fill-color: var(--text-primary) !important;
}
[data-testid="stTextInput"] input::placeholder { color: var(--text-muted) !important; -webkit-text-fill-color: var(--text-muted) !important; }

@media (max-width: 680px) {
  [data-testid="stTabs"] [role="tablist"] { gap: 1.5rem !important; }
}
"""


def inject_css(theme: str) -> None:
    """Palette + shared block styles + Streamlit chrome, as one stylesheet.

    Variables go on :root rather than a wrapper class: Streamlit reruns repaint the whole
    page, so there is no need for the per-block scoping the Gradio app uses.
    """

    st.markdown(
        "<style>"
        + render.vars_block(":root", theme)
        + render.SHARED_CSS
        + STREAMLIT_CSS
        + "</style>",
        unsafe_allow_html=True,
    )


@st.cache_data(ttl=60)
def fetch_receipts():
    r = requests.get(f"{API_BASE}/receipts", timeout=10)
    r.raise_for_status()
    return r.json()


def fetch_receipt(image_id: str, include_gt: bool):
    r = requests.get(
        f"{API_BASE}/receipts/{image_id}", params={"include_gt": include_gt}, timeout=10
    )
    r.raise_for_status()
    return r.json()


def fetch_infer(image_bytes: bytes, filename: str):
    # NOT cached: /infer appends to the backend's Overview list, and a cache hit would skip
    # that append. Rerun de-dup is handled by the caller via a per-file hash in session_state.
    r = requests.post(
        f"{API_BASE}/infer", files={"file": (filename, image_bytes)}, timeout=120
    )
    r.raise_for_status()
    return r.json()


def fetch_dashboard():
    r = requests.get(f"{API_BASE}/dashboard", timeout=10)
    r.raise_for_status()
    return r.json()


def fetch_categories():
    r = requests.get(f"{API_BASE}/categories", timeout=10)
    r.raise_for_status()
    return r.json()


def render_receipt_tab(theme: str):
    source = st.radio(
        "Source",
        ["Upload", "Sample receipts"],
        horizontal=True,
        key="imgsrc",
        label_visibility="collapsed",
    )

    image_bytes, image_name, image_id = None, "upload.jpg", None
    if source == "Upload":
        st.caption("Drag & drop a receipt photo here, or browse your files.")
        uploaded = st.file_uploader(
            "Choose a photo",
            type=["png", "jpg", "jpeg", "webp", "heic", "heif"],
            key="upload_widget",
            label_visibility="collapsed",
        )
        if uploaded is not None:
            image_bytes, image_name = uploaded.getvalue(), uploaded.name
    else:
        try:
            receipts = fetch_receipts()
        except requests.RequestException as e:
            st.error(f"Can't reach the service ({e}).")
            return
        if not receipts:
            st.warning("No sample receipts available.")
            return
        options = {samples.format_receipt_label(r): r["image_id"] for r in receipts}
        label = st.selectbox("Receipt", list(options.keys()), key="receipt_select")
        image_id = options[label]
        img_resp = requests.get(f"{API_BASE}/receipts/{image_id}/image", timeout=10)
        if img_resp.ok:
            image_bytes = img_resp.content

    col_img, col_table = st.columns([1, 1.4], gap="large")
    with col_img:
        if image_bytes:
            try:
                frame = render.receipt_frame(image_bytes)
            except Exception as e:
                st.error(f"Couldn't read that file as an image: {e}")
                frame = None
            if frame:
                st.markdown(frame, unsafe_allow_html=True)
        elif source == "Upload":
            st.info("Upload a receipt photo to begin.")
        else:
            st.info("No image available for this receipt.")

    detail = None
    with col_table:
        if source == "Sample receipts" and image_id is not None:
            include_gt = st.checkbox("Compare to reference", value=False)
            detail = fetch_receipt(image_id, include_gt)
        elif image_bytes:
            # analyze once per unique file: reruns reuse the stored result, no duplicate append
            h = hashlib.md5(image_bytes).hexdigest()
            if st.session_state.get("infer_hash") == h and st.session_state.get(
                "infer_result"
            ):
                detail = st.session_state["infer_result"]
            else:
                with st.spinner("Analyzing receipt…"):
                    try:
                        detail = fetch_infer(image_bytes, image_name)
                        st.session_state["infer_hash"] = h
                        st.session_state["infer_result"] = detail
                    except requests.RequestException as e:
                        st.error(f"Analysis failed: {e}")

        if detail is None:
            st.caption("Extracted details will appear here.")
        else:
            prediction = detail["prediction"]
            st.markdown(
                render.detail_panel(
                    prediction,
                    confidence=detail.get("confidence"),
                    ground_truth=detail.get("ground_truth"),
                    category=infer_category(
                        {
                            "store": prediction.get("store") or "",
                            "line_items": prediction.get("line_items"),
                        }
                    ),
                    theme=theme,
                    format_item_name=format_item_name,
                ),
                unsafe_allow_html=True,
            )


def render_dashboard_tab(theme: str):
    try:
        dash = fetch_dashboard()
    except requests.RequestException as e:
        st.error(f"Can't reach the service ({e}).")
        return

    if dash["n_receipts"] == 0:
        st.info("Analyze a few receipts to build your spending overview.")
        return

    cats = None
    try:
        cats = fetch_categories()
    except requests.RequestException:
        pass

    # Hero + stat cards, shared with the Gradio app. Recent transactions are rendered
    # separately below so the charts can sit between them, as before.
    st.markdown(
        render.overview_blocks(
            {**dash, "recent": [], "caveat": None},
            cats,
            theme=theme,
            infer_category=infer_category,
        ),
        unsafe_allow_html=True,
    )

    if dash.get("by_month"):
        st.markdown(
            '<div class="section-title">Spending by month</div>', unsafe_allow_html=True
        )
        st.plotly_chart(
            render.month_figure(dash["by_month"], theme), use_container_width=True
        )

    if cats:
        fig = render.category_figure(cats["categories"], theme)
        if fig is not None:
            st.markdown(
                '<div class="section-title">Spending by category</div>',
                unsafe_allow_html=True,
            )
            st.plotly_chart(fig, use_container_width=True)

    if dash.get("recent"):
        st.markdown(
            '<div class="section-title">Recent transactions</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            render.transactions_panel(
                dash["recent"], theme=theme, infer_category=infer_category
            ),
            unsafe_allow_html=True,
        )
        if dash.get("caveat"):
            st.markdown(
                f'<div class="section-sub" style="margin-top:0.8rem;">'
                f'{dash["caveat"]}</div>',
                unsafe_allow_html=True,
            )


c_brand, c_theme = st.columns([3, 1.1], gap="small", vertical_alignment="center")
with c_theme:
    mode = st.radio(
        "Theme",
        ["Dark", "Light"],
        horizontal=True,
        key="theme",
        label_visibility="collapsed",
    )
inject_css(mode)
with c_brand:
    st.markdown(
        '<p class="brand-mark">Receipt<span class="dot">VLM</span></p>'
        '<p class="brand-tag">Track every dollar, straight from your receipts.</p>',
        unsafe_allow_html=True,
    )

tab1, tab2 = st.tabs(["Receipts", "Overview"])
with tab1:
    render_receipt_tab(mode)
with tab2:
    render_dashboard_tab(mode)
