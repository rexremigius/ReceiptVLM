"""Deliverable #10 (Streamlit half) — talks to `src/serve.py` over HTTP.

Run (from repo root, two terminals):
    uvicorn src.serve:app --port 8000
    streamlit run app/streamlit_app.py

UPDATED 2026-07-24: UI redesign pass (CSS/layout/copy only — no change to API calls or
data wiring). This is a reviewer tool for scanning many receipts quickly, not a
landing page: compact header, dense two-column body, per-row line-item confidence,
badge-consistent status pills throughout. See PROGRESS.md for the before/after
screenshots and self-critique.

Uploading a photo or pasting an image URL runs *live* inference via serve.py's
`/infer` (real model + #9 repair + #8's full 3-signal confidence). The "Dataset
(cached)" option reads the fast pre-computed predictions (2-signal confidence,
matching what #5/#6 report against).
"""
from __future__ import annotations

import base64
import io

import plotly.graph_objects as go
import requests
import streamlit as st
from PIL import Image
import pillow_heif

# Stock Pillow has no HEIC/HEIF decoder at all (confirmed empirically, not assumed —
# Image.open() on a real HEIC file raises without this). iPhone photos default to
# HEIC, so this is the realistic case for someone actually photographing a receipt,
# not an edge case. mlx_vlm's own image loader also goes through PIL.Image.open, so
# serve.py needs this same registration independently (a separate process).
pillow_heif.register_heif_opener()

API_BASE = "http://127.0.0.1:8000"

FIELD_LABELS = {"store": "Store", "date": "Date", "tax": "Tax", "tip": "Tip",
                "subtotal": "Subtotal", "total": "Total"}
# Copy-only display labels for serve.py's repair_status values — the raw strings
# (e.g. "handled_upstream_at_generation") are accurate but too long/technical for a
# compact badge; this maps to short labels without touching the underlying data.
REPAIR_LABELS = {
    "clean": "clean", "handled_upstream_at_generation": "n/a (cached)",
    "repaired_trailing_comma": "repaired", "repaired_python_literal": "repaired",
    "repaired_truncation": "repaired", "hard_failure": "failed",
}

# --- design tokens: receipt-paper grounded, not default Streamlit ------------------
BG = "#FAF7F0"           # warm paper, not pure white
SURFACE = "#FFFFFF"
BORDER = "#E8E2D4"       # subtle paper-edge feel
TEXT_PRIMARY = "#1C1B18"  # warm near-black, not pure #000
TEXT_SECONDARY = "#6B6659"
ACCENT = "#2B4C4A"        # deep teal — replaces Streamlit's default blue everywhere
STATUS = {
    "green":   {"fg": "#3A6B4A", "bg": "#E8F0E5"},
    "amber":   {"fg": "#A9752E", "bg": "#FBF0DE"},
    "red":     {"fg": "#A33D3D", "bg": "#F7E6E6"},
    # "na" = legitimately no value (nothing to be confident about); "missing" = a
    # null field that's almost always a real gap (currently just `store` — see
    # PROGRESS.md 2026-07-25). "missing" reuses red's fg/bg since it's still a real
    # concern, just labeled "missing" instead of a fabricated numeric score.
    "na":      {"fg": "#8A8578", "bg": "#EFEDE7"},
    "missing": {"fg": "#A33D3D", "bg": "#F7E6E6"},
    # "unscored" = line item, only: a value IS present and looks well-formed, but no
    # consistency signal exists to judge it against (see src/serve.py's
    # _unscored_badge). Same neutral gray as "na" — both mean "no numeric score",
    # just for a different reason — deliberately not a color that reads as good/bad.
    "unscored": {"fg": "#8A8578", "bg": "#EFEDE7"},
}
STATUS_LABELS = {"na": "not applicable", "missing": "missing", "unscored": "no signal"}
GRID_COLOR = BORDER
CHART_SURFACE = SURFACE
SERIES_BLUE = "#2a78d6"   # chart series color — a separate dataviz decision, left as
                          # the existing validated categorical hue (not the UI accent)
FONT_SANS = ("'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif")
FONT_MONO = ("'JetBrains Mono', 'IBM Plex Mono', ui-monospace, "
            "'SFMono-Regular', Menlo, Consolas, monospace")

st.set_page_config(page_title="ReceiptVLM", layout="wide")

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {{
  --bg: {BG}; --surface: {SURFACE}; --border: {BORDER};
  --text-primary: {TEXT_PRIMARY}; --text-secondary: {TEXT_SECONDARY}; --accent: {ACCENT};
  --font-sans: {FONT_SANS};
  --font-mono: {FONT_MONO};
}}

/* page canvas — Streamlit's own header toolbar (Deploy button, hamburger menu) is
   `position: absolute`, 60px tall, and was physically covering our compact wordmark
   when top padding was reduced below its height. It's irrelevant chrome for an
   internal reviewer tool, so hide it outright rather than just clear it. Theme
   (dark/light) is pinned explicitly in .streamlit/config.toml — this CSS assumes
   that theme, it doesn't fight a possibly-different one. */
[data-testid="stAppViewContainer"], [data-testid="stMain"] {{ background: var(--bg); }}
[data-testid="stHeader"] {{ display: none; }}
[data-testid="stMainBlockContainer"] {{
  padding-top: 1rem; max-width: 1320px;
}}
html, body, [data-testid="stAppViewContainer"] {{
  color: var(--text-primary); font-family: var(--font-sans);
}}
/* NOT a blanket `*` — Streamlit's icon buttons (upload icon, etc.) render their glyph
   as literal ligature text in a dedicated icon font (e.g. the upload button's actual
   DOM text is the word "upload", turned into a glyph only by that font). Forcing
   Inter onto it made the ligature render as literal readable text next to the real
   "Upload" label — a real visible bug, caught by inspecting the button's rendered
   HTML after screenshotting, not by reading the CSS alone. Excluding icon elements
   from the inheritance fixes it without giving up the font override everywhere else. */
[data-testid="stAppViewContainer"] *:not([data-testid="stIconMaterial"]) {{ font-family: inherit; }}

/* alerts (st.info/warning/error) — restyled to the paper palette; Streamlit's
   default blue/orange/red alert backgrounds otherwise clash hard with the warm
   background and would be the one remaining "default Streamlit" look on the page.
   The visible background actually lives on stAlertContainer (the outer wrapper),
   not stAlertContentInfo/Warning/Error (an inner inset panel) — styling only the
   inner one left Streamlit's default blue outer tint showing around a white inset,
   a layered look caught by checking computed background-color on both elements
   rather than assuming the first testid found was the right one. `:has()` lets the
   container pick up the right treatment based on which content type it wraps. */
div[data-testid="stAlertContainer"]:has(div[data-testid="stAlertContentInfo"]) {{
  background: var(--surface) !important; border: 1px solid var(--border) !important;
  border-left: 3px solid var(--accent) !important; border-radius: 6px;
}}
div[data-testid="stAlertContainer"]:has(div[data-testid="stAlertContentWarning"]) {{
  background: {STATUS["amber"]["bg"]} !important; border: 1px solid {STATUS["amber"]["fg"]}40 !important;
  border-left: 3px solid {STATUS["amber"]["fg"]} !important; border-radius: 6px;
}}
div[data-testid="stAlertContainer"]:has(div[data-testid="stAlertContentError"]) {{
  background: {STATUS["red"]["bg"]} !important; border: 1px solid {STATUS["red"]["fg"]}40 !important;
  border-left: 3px solid {STATUS["red"]["fg"]} !important; border-radius: 6px;
}}
div[data-testid="stAlertContentInfo"], div[data-testid="stAlertContentWarning"],
div[data-testid="stAlertContentError"] {{ background: transparent !important; }}
div[data-testid="stAlertContentInfo"] p, div[data-testid="stAlertContentWarning"] p,
div[data-testid="stAlertContentError"] p {{ color: var(--text-primary) !important; font-size: 0.85rem; }}
div[data-testid="stAlertContentInfo"] svg, div[data-testid="stAlertContentWarning"] svg,
div[data-testid="stAlertContentError"] svg {{ fill: var(--accent) !important; }}

/* wordmark: clearly the largest text on the page, with real breathing room before
   the tab row beneath it — not just marginally bigger than the tab labels.
   `!important` needed: it's rendered as `[data-testid="stMarkdownContainer"] p`,
   and that attribute+element selector otherwise out-specifies a plain `.app-wordmark`
   class selector, silently resetting font-size back to Streamlit's default 16px —
   confirmed via computed style, not assumed, after a size bump visually did nothing. */
.app-wordmark {{
  font-size: 2.5rem !important; font-weight: 700 !important; letter-spacing: 0.01em;
  color: var(--text-primary) !important; margin: 0 0 0.9rem 0 !important; line-height: 1.2 !important;
}}

/* tabs: small, underline-style, no oversized default font/padding */
[data-testid="stTabs"] {{ margin-top: 0; }}
div[data-testid="stTab"] {{
  padding: 0.25rem 0.1rem 0.5rem 0.1rem !important;
  margin-right: 1.1rem !important;
  font-size: 0.85rem !important;
}}
div[data-testid="stTab"] p {{ font-size: 0.85rem !important; font-weight: 500; color: var(--text-secondary); }}
div[data-testid="stTab"][aria-selected="true"] p {{ color: var(--text-primary); font-weight: 700; }}
div[data-testid="stTab"] .react-aria-SelectionIndicator,
div[data-testid="stTab"][data-selected="true"]::after {{ background: var(--accent) !important; }}
[data-testid="stTabs"] [role="tablist"] {{ border-bottom: 1px solid var(--border); gap: 0; }}

/* section dividers use the hairline border, not Streamlit's heavier default */
hr {{ border-color: var(--border) !important; margin: 0.6rem 0 !important; }}

/* segmented pill toggle (built from st.radio) — accent is the deep teal, never the
   Streamlit-default blue/red */
div[data-testid="stRadioGroup"] {{
  display: inline-flex; gap: 2px; background: var(--bg);
  border: 1px solid var(--border); border-radius: 8px; padding: 2px;
}}
label[data-testid="stRadioOption"] {{ margin: 0 !important; min-height: 0 !important; }}
label[data-testid="stRadioOption"] > div > div > div:not([data-testid="stMarkdownContainer"]) {{
  display: none;  /* hide the default radio-circle indicator */
}}
label[data-testid="stRadioOption"] div[data-testid="stMarkdownContainer"] p {{
  padding: 0.25rem 0.7rem; border-radius: 6px; margin: 0;
  font-size: 0.8rem; color: var(--text-secondary); white-space: nowrap;
}}
label[data-testid="stRadioOption"][data-selected="true"] div[data-testid="stMarkdownContainer"] p {{
  background: var(--accent); color: #ffffff; font-weight: 600;
}}

/* compact widget labels */
[data-testid="stWidgetLabel"] p {{ font-size: 0.78rem; color: var(--text-secondary); margin-bottom: 0.15rem; }}
[data-testid="stSelectbox"] > div > div {{ font-size: 0.85rem; }}

/* status pill badges — ONE consistent shape/padding/type treatment for every badge
   on the page (confidence, repair layer, n/a, caveat): mono face (this is itself a
   piece of "data read off the receipt or the model", same logic as field values),
   same size, same padding, same pill radius, regardless of which status it shows. */
.status-pill {{
  display: inline-block; padding: 0.1rem 0.55rem; border-radius: 999px;
  font-family: var(--font-mono); font-size: 0.72rem; font-weight: 600;
  line-height: 1.5; white-space: nowrap;
}}

/* field-table rows: tighten and align. ONE header style (uppercase, tracked) shared
   by both the scalar-fields table and the line-items table — no casing mismatch. */
.field-row {{
  display: grid; grid-template-columns: 130px 1fr 130px; align-items: center;
  padding: 0.32rem 0; border-bottom: 1px solid var(--border);
  font-size: 0.85rem;
}}
.field-row.with-gt {{ grid-template-columns: 130px 1fr 130px 1fr; }}
.field-row.header {{
  font-family: var(--font-sans); font-size: 0.72rem; text-transform: uppercase;
  letter-spacing: 0.04em; font-weight: 600;
  color: var(--text-secondary); border-bottom: 1px solid var(--text-secondary);
  padding-bottom: 0.4rem;
}}
.field-name {{ color: var(--text-secondary); font-family: var(--font-sans); }}
.field-value {{ font-family: var(--font-mono); font-variant-numeric: tabular-nums; }}
.field-gt {{ color: var(--text-secondary); font-size: 0.8rem; font-family: var(--font-mono); }}
/* line-item name/price columns are extracted receipt data too (not UI labels like
   .field-name is for the scalar table), so they get the mono treatment as well */
.li-name {{ font-family: var(--font-mono); }}
.li-price {{ font-family: var(--font-mono); font-variant-numeric: tabular-nums; }}

/* metadata strip at the bottom of the record */
.meta-strip {{
  display: flex; gap: 0.9rem; align-items: center; flex-wrap: wrap;
  margin-top: 0.6rem; padding-top: 0.5rem; border-top: 1px solid var(--border);
  font-size: 0.78rem; color: var(--text-secondary);
}}

/* receipt image: signature element — a subtle drop shadow + slight rotation, like a
   physical receipt photographed on a desk, not a plain bounded rectangle. Kept
   restrained: one shadow, ~1.5deg, a small paper-white mat border around the photo
   itself (echoing a printed photo's white edge) rather than any further texture. */
.receipt-image-frame {{
  display: inline-block; transform: rotate(-1.4deg);
  background: var(--surface); padding: 8px 8px 14px 8px; border-radius: 3px;
  box-shadow: 0 10px 22px rgba(28,27,24,0.16), 0 2px 6px rgba(28,27,24,0.10);
  margin: 6px 0 14px 4px;
}}
.receipt-image-frame img {{
  display: block; max-width: 404px; width: 100%;
  border: 1px solid var(--border); border-radius: 2px;
}}

/* file uploader: tighter, clean dashed border in the new palette, visible icon —
   the default rendering went solid-black under a dark-mode browser (fixed for real
   via .streamlit/config.toml pinning a light theme); this styles the light-theme
   version deliberately rather than leaving it to Streamlit's own defaults. */
[data-testid="stFileUploaderDropzone"] {{
  padding: 0.9rem !important; border-radius: 8px !important;
  border: 1.5px dashed var(--border) !important; background: var(--surface) !important;
}}
[data-testid="stFileUploaderDropzone"] svg {{ fill: var(--accent) !important; opacity: 0.8; }}
[data-testid="stFileUploaderDropzoneInstructions"] span {{ font-size: 0.78rem; color: var(--text-secondary); }}
[data-testid="stBaseButton-secondary"] {{
  background: var(--surface) !important; color: var(--accent) !important;
  border: 1px solid var(--accent) !important;
}}

/* section captions */
.section-caption {{ font-size: 0.76rem; color: var(--text-secondary); margin: 0 0 0.4rem 0; }}
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=60)
def fetch_receipts():
    r = requests.get(f"{API_BASE}/receipts", timeout=10)
    r.raise_for_status()
    return r.json()


def fetch_receipt(image_id: str, include_gt: bool):
    r = requests.get(f"{API_BASE}/receipts/{image_id}",
                      params={"include_gt": include_gt}, timeout=10)
    r.raise_for_status()
    return r.json()


def fetch_infer(image_bytes: bytes, filename: str):
    r = requests.post(f"{API_BASE}/infer",
                      files={"file": (filename, image_bytes)}, timeout=120)
    r.raise_for_status()
    return r.json()


def fetch_dashboard():
    # No @st.cache_data here on purpose — this is a real-time view over receipts
    # actually uploaded and analyzed this session (see serve.py's LIVE_RECEIPTS), so
    # a cache would show stale data right after the receipt you just analyzed.
    # Aggregating an in-memory list this small is cheap; there's nothing to cache.
    r = requests.get(f"{API_BASE}/dashboard", timeout=10)
    r.raise_for_status()
    return r.json()


def pill(text: str, level: str) -> str:
    s = STATUS.get(level, STATUS["red"])
    return (f'<span class="status-pill" style="background:{s["bg"]};color:{s["fg"]};'
            f'border:1px solid {s["fg"]}40">{text}</span>')


def confidence_pill(level: str, score: float | None) -> str:
    if score is None:
        return pill(STATUS_LABELS.get(level, level), level)
    return pill(f"{level} · {score:.2f}", level)


def image_data_uri(image_bytes: bytes) -> str:
    """Embed the receipt photo as a single <img> tag (base64 data URI) rather than
    st.image — the signature rotated/drop-shadow "receipt-image-frame" treatment
    needs the image and its wrapper in ONE HTML fragment. Three separate st.markdown/
    st.image calls (open tag / image / close tag) do NOT nest in Streamlit's DOM —
    each becomes its own sibling element — so the wrapper CSS silently never reached
    the image at all (confirmed via computed styles: transform was "none").

    No blind format fallback: the format is always identified from the real file
    content (Pillow's own sniffing), never assumed to be JPEG. HEIC/HEIF is a real
    case now (iPhone photos default to it), and it needs its own handling regardless
    of decoding: Chrome/Firefox don't render HEIC via <img> at all (only Safari has
    partial support), so even a correctly-decoded HEIC image would show as broken in
    most browsers. It's re-encoded to real JPEG bytes here — a genuine conversion, not
    a relabel — so the preview actually renders; the *model* gets the original bytes
    untouched via serve.py's /infer, since mlx_vlm decodes HEIC natively once the same
    opener is registered there.
    """
    img = Image.open(io.BytesIO(image_bytes))
    fmt = img.format
    if fmt in ("HEIF", "HEIC"):
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG")
        image_bytes = buf.getvalue()
        fmt = "JPEG"
    b64 = base64.b64encode(image_bytes).decode("ascii")
    return f"data:image/{fmt.lower()};base64,{b64}"


def format_receipt_label(r: dict) -> str:
    """Store · formatted date · formatted total — explicit separators, nothing cut off
    mid-word (the old f-string truncated arbitrarily at the selectbox's fixed width,
    e.g. 'CHOEUN · 12/30/2016FRI · 60.44 — image_files/Image_12/1...' clipping to
    '...ir'). Money is prefixed with $ and the raw ground-truth date artifacts (a
    trailing weekday like 'FRI' with no separator) are left as-is but the id suffix
    that caused the clipping is dropped entirely — the id is never useful to a human
    reviewer picking a receipt by eye.
    """
    store = (r["store"] or "(no store)").strip()
    date = (r["date"] or "no date").strip()
    total = f"${r['total']}" if r["total"] else "no total"
    return f"{store} · {date} · {total}"


def style_bar_fig(fig: go.Figure, title: str) -> go.Figure:
    fig.update_layout(title=title, plot_bgcolor=CHART_SURFACE, paper_bgcolor=CHART_SURFACE,
                      margin=dict(t=40, b=20), font_color=TEXT_PRIMARY)
    fig.update_xaxes(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR)
    fig.update_yaxes(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR)
    return fig


def render_receipt_tab():
    # Plain page on load — no default receipt, no pre-selected data. Upload is the
    # primary path (a real photo, real inference), so it leads; the cached WildReceipt
    # dataset is a QA/reference tool, not the first thing anyone should see.
    source = st.radio("Image source", ["Upload", "From URL", "Dataset (cached)"],
                      horizontal=True, key="imgsrc", label_visibility="visible")

    image_bytes, image_name, image_id = None, "upload.jpg", None
    if source == "Upload":
        uploaded = st.file_uploader("Choose a photo", type=["png", "jpg", "jpeg", "webp", "heic", "heif"],
                                    key="upload_widget", label_visibility="collapsed")
        if uploaded is not None:
            image_bytes, image_name = uploaded.getvalue(), uploaded.name
    elif source == "From URL":
        url = st.text_input("Image URL", key="url_widget", label_visibility="collapsed",
                            placeholder="https://…")
        if url:
            try:
                resp = requests.get(url, timeout=10)
                resp.raise_for_status()
                if not resp.headers.get("content-type", "").startswith("image/"):
                    st.error("That URL didn't return an image.")
                else:
                    image_bytes, image_name = resp.content, url.rsplit("/", 1)[-1] or "url.jpg"
            except requests.RequestException as e:
                st.error(f"Couldn't fetch that URL: {e}")
    else:
        try:
            receipts = fetch_receipts()
        except requests.RequestException as e:
            st.error(f"Can't reach the API at {API_BASE} — is `uvicorn src.serve:app` "
                     f"running? ({e})")
            return
        if not receipts:
            st.warning("No predictions loaded (data/processed/finetuned_test.jsonl is "
                        "empty or missing).")
            return
        options = {format_receipt_label(r): r["image_id"] for r in receipts}
        label = st.selectbox("Receipt", list(options.keys()), key="receipt_select")
        image_id = options[label]
        img_resp = requests.get(f"{API_BASE}/receipts/{image_id}/image", timeout=10)
        if img_resp.ok:
            image_bytes = img_resp.content

    col_img, col_table = st.columns([1, 1.5], gap="large")
    with col_img:
        if image_bytes:
            try:
                data_uri = image_data_uri(image_bytes)
            except Exception as e:
                # No silent format guess here either — if Pillow genuinely can't
                # identify the file, say so plainly rather than rendering it as a
                # (wrong) assumed type.
                st.error(f"Couldn't read that file as an image: {e}")
                data_uri = None
            if data_uri:
                st.markdown(
                    f'<div class="receipt-image-frame"><img src="{data_uri}" /></div>',
                    unsafe_allow_html=True,
                )
        elif source == "Upload":
            st.info("Choose a photo above to run live inference.")
        elif source == "From URL":
            st.info("Paste an image URL above to run live inference.")
        else:
            st.info("No cached image for this receipt on this server.")

    include_gt = False
    detail = None
    conf_signal_note = ""
    with col_table:
        if source == "Dataset (cached)" and image_id is not None:
            include_gt = st.checkbox("Show ground truth (QA)", value=False)
            detail = fetch_receipt(image_id, include_gt)
            conf_signal_note = "2-signal (format validity + arithmetic consistency)"
        elif image_bytes:
            with st.spinner("Running live inference (model generation, ~10-60s)…"):
                try:
                    detail = fetch_infer(image_bytes, image_name)
                except requests.RequestException as e:
                    st.error(f"Inference failed: {e}")
            conf_signal_note = "3-signal (+ token logprob) — live generation"

        if detail is None:
            st.caption("Provide an image (above) to see extracted fields.")
        else:
            pred = detail["prediction"]
            conf = detail["confidence"]
            gt = detail.get("ground_truth")

            row_class = "field-row with-gt" if gt else "field-row"
            header_cells = ["Field", "Predicted", "Confidence"] + (["Ground truth"] if gt else [])
            st.markdown(
                f'<div class="{row_class} header">' +
                "".join(f"<div>{c}</div>" for c in header_cells) + "</div>",
                unsafe_allow_html=True,
            )
            for field in ["store", "date", "tax", "tip", "subtotal", "total"]:
                c = conf[field]
                val = pred.get(field) if pred.get(field) is not None else "—"
                cells = [
                    f'<div class="field-name">{FIELD_LABELS[field]}</div>',
                    f'<div class="field-value">{val}</div>',
                    f'<div>{confidence_pill(c["level"], c["score"])}</div>',
                ]
                if gt:
                    gt_val = gt.get(field) if gt else None
                    cells.append(f'<div class="field-gt">{gt_val if gt_val is not None else "—"}</div>')
                st.markdown(f'<div class="{row_class}">' + "".join(cells) + "</div>",
                           unsafe_allow_html=True)

            li_conf = conf["line_items"]
            items = pred.get("line_items") or []
            st.markdown(
                f'<div style="margin-top:0.9rem;display:flex;align-items:center;gap:0.5rem;">'
                f'<span style="font-size:0.85rem;font-weight:600;">Line items</span>'
                f'<span style="font-size:0.76rem;color:{TEXT_SECONDARY}">(aggregate)</span>'
                f'{confidence_pill(li_conf["aggregate"]["level"], li_conf["aggregate"]["score"])}</div>',
                unsafe_allow_html=True,
            )
            if items:
                # real per-item confidence (format validity + subtotal-consistency,
                # see src/confidence.py) — each row's badge can genuinely differ now,
                # not a copy of one shared aggregate value.
                rows = "".join(
                    f'<div class="field-row" style="grid-template-columns:1fr 90px 130px;">'
                    f'<div class="li-name">{it.get("name", "—")}</div>'
                    f'<div class="li-price">{it.get("price") if it.get("price") is not None else "—"}</div>'
                    f'<div>{confidence_pill(badge["level"], badge["score"])}</div>'
                    f"</div>"
                    for it, badge in zip(items, li_conf["items"])
                )
                st.markdown(
                    '<div class="field-row header" style="grid-template-columns:1fr 90px 130px;">'
                    "<div>Name</div><div>Price</div><div>Confidence</div></div>" + rows,
                    unsafe_allow_html=True,
                )
            else:
                st.caption("No line items extracted.")

            repair_status = detail["repair_status"]
            repair_label = REPAIR_LABELS.get(repair_status, repair_status)
            repair_level = "red" if repair_status == "hard_failure" \
                else "amber" if repair_status.startswith("repaired") \
                else "green"
            st.markdown(
                '<div class="meta-strip">'
                f'<span>Repair layer: {pill(repair_label, repair_level)}</span>'
                f'<span>Confidence signals: {conf_signal_note}</span>'
                f'<span>Source: {source}</span>'
                "</div>",
                unsafe_allow_html=True,
            )


def render_dashboard_tab():
    try:
        dash = fetch_dashboard()
    except requests.RequestException as e:
        st.error(f"Can't reach the API at {API_BASE} ({e})")
        return

    if dash["n_receipts"] == 0:
        st.info("No receipts analyzed yet this session. Go to **Receipt viewer**, "
                "pick **Upload** or **From URL**, and analyze a real receipt — this "
                "dashboard builds from that, live, not from the WildReceipt eval set.")
        return

    st.caption(f"Real-time — {dash['n_receipts']} receipt(s) uploaded and analyzed "
               f"via live inference this session (not the WildReceipt eval set).")
    st.markdown(
        f'<div class="meta-strip" style="border-top:none;padding-top:0;margin-top:0;">'
        f'{pill("caveat", "amber")} <span>{dash["caveat"]}</span></div>',
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    c1.metric("Sum of predicted `total` field", f"{dash['total_spend']:,.2f}",
              help=f"Summed across {dash['n_priced']} of {dash['n_receipts']} "
                   f"analyzed receipts with a parseable total.")
    c2.metric("Receipts analyzed", f"{dash['n_receipts']}")

    recent = dash.get("recent") or []
    if recent:
        st.markdown('<div style="margin-top:0.8rem;font-weight:600;font-size:0.85rem;">'
                    "Recently analyzed</div>", unsafe_allow_html=True)

        def _recent_row(r: dict) -> str:
            total_display = f"${r['total']}" if r["total"] else "—"
            return (
                '<div class="field-row" style="grid-template-columns:1fr 1fr 1fr 110px;">'
                f'<div class="li-name">{r["store"] or "—"}</div>'
                f'<div class="li-name">{r["date"] or "—"}</div>'
                f'<div class="li-price">{total_display}</div>'
                f'<div class="field-gt">{r["timestamp"][11:16]} UTC</div>'
                "</div>"
            )

        rows = "".join(_recent_row(r) for r in recent)
        st.markdown(
            '<div class="field-row header" style="grid-template-columns:1fr 1fr 1fr 110px;">'
            "<div>Store</div><div>Date</div><div>Total</div><div>Analyzed</div></div>" + rows,
            unsafe_allow_html=True,
        )

    by_month = dash["by_month"]
    if by_month:
        fig = go.Figure(go.Bar(x=[m["month"] for m in by_month],
                               y=[m["spend"] for m in by_month],
                               marker_color=SERIES_BLUE))
        # Force a categorical x-axis explicitly. With few receipts (the live-only
        # dashboard's normal case now, vs. the old 472-receipt eval set), a single
        # "YYYY-MM" string reads as a date to Plotly's auto-inference, and it
        # silently switches to a continuous date axis with nonsense sub-second tick
        # labels ("23:59:59.9996") instead of one clean categorical bar.
        fig.update_xaxes(type="category")
        fig.update_yaxes(title="Sum of predicted total")
        st.plotly_chart(style_bar_fig(fig, "Spend by month"), use_container_width=True)
        with st.expander("Table view"):
            st.dataframe(by_month, use_container_width=True)

    by_store = dash["by_store"]
    if by_store:
        fig2 = go.Figure(go.Bar(x=[s["spend"] for s in by_store],
                                y=[s["store"] for s in by_store],
                                orientation="h", marker_color=SERIES_BLUE))
        fig2.update_yaxes(autorange="reversed", type="category")
        fig2.update_xaxes(title="Sum of predicted total")
        st.plotly_chart(style_bar_fig(fig2, "Top 10 stores by spend"), use_container_width=True)
        with st.expander("Table view"):
            st.dataframe(by_store, use_container_width=True)


st.markdown('<p class="app-wordmark">ReceiptVLM</p>', unsafe_allow_html=True)
tab1, tab2 = st.tabs(["Receipt viewer", "Spending dashboard"])
with tab1:
    render_receipt_tab()
with tab2:
    render_dashboard_tab()
