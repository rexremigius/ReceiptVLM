from __future__ import annotations

import base64
import hashlib
import io
import os
import sys

import plotly.graph_objects as go
import requests
import streamlit as st
from PIL import Image
import pillow_heif

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.display_format import format_item_name
from src.categorize import infer_category

pillow_heif.register_heif_opener()  # stock Pillow can't decode iPhone HEIC

API_BASE = "http://127.0.0.1:8000"

FIELD_LABELS = {"store": "Store", "date": "Date", "tax": "Tax", "tip": "Tip",
                "subtotal": "Subtotal", "total": "Total"}

# Category colors validated CVD-safe against each mode's surface, assigned per category.
THEMES = {
    "Dark": {
        "bg": "#14161B", "surface": "#1E212B", "surface2": "#262A35", "border": "#2C313D",
        "text": "#F4F6FA", "text2": "#9BA3B2", "text_muted": "#6B7280",
        "green": "#22DD8A", "on_green": "#0B1F16",
        "hero_grad": "linear-gradient(145deg, #1F2A26 0%, #1E212B 55%)",
        "cats": {"dining": "#60A5FA", "grocery": "#34D399", "fuel": "#FB923C", "retail": "#C084FC",
                 "transport": "#FACC15", "misc": "#F472B6", "other": "#94A3B8"},
    },
    "Light": {
        "bg": "#F4F6F9", "surface": "#FFFFFF", "surface2": "#EEF1F6", "border": "#E3E8EF",
        "text": "#14213B", "text2": "#566175", "text_muted": "#8A94A6",
        "green": "#12B76A", "on_green": "#FFFFFF",
        "hero_grad": "linear-gradient(145deg, #E8F7EF 0%, #FFFFFF 55%)",
        "cats": {"dining": "#2563EB", "grocery": "#059669", "fuel": "#EA580C", "retail": "#7C3AED",
                 "transport": "#A16207", "misc": "#DB2777", "other": "#64748B"},
    },
}
FONT_SANS = "'Manrope', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif"
FONT_MONO = "'JetBrains Mono', ui-monospace, 'SFMono-Regular', Menlo, Consolas, monospace"

st.set_page_config(page_title="ReceiptVLM", layout="wide")

if "theme" not in st.session_state:
    st.session_state.theme = "Dark"


def inject_css(T: dict) -> None:
    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
:root {{
  --bg:{T['bg']}; --surface:{T['surface']}; --surface-2:{T['surface2']}; --border:{T['border']};
  --text-primary:{T['text']}; --text-secondary:{T['text2']}; --text-muted:{T['text_muted']};
  --green:{T['green']}; --on-green:{T['on_green']}; --hero-grad:{T['hero_grad']};
  --font-sans:{FONT_SANS}; --font-mono:{FONT_MONO};
}}
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"],
[data-testid="stMainBlockContainer"], [data-testid="stBottomBlockContainer"] {{ background: var(--bg) !important; }}
[data-testid="stHeader"] {{ display: none; }}
[data-testid="stMainBlockContainer"] {{ padding-top: 1.4rem; max-width: 1180px; }}
html, body, [data-testid="stAppViewContainer"] {{ color: var(--text-primary); font-family: var(--font-sans); }}
[data-testid="stAppViewContainer"] *:not([data-testid="stIconMaterial"]) {{ font-family: inherit; }}
/* dark config base sets white text on markdown containers; force theme color so light mode is readable */
[data-testid="stMarkdownContainer"], [data-testid="stMarkdown"],
[data-testid="stText"], [data-testid="stHeading"] {{ color: var(--text-primary) !important; }}
[data-testid="stSelectbox"] div, [data-baseweb="select"] div, [data-testid="stSpinner"] div {{ color: var(--text-primary); }}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {{ color: var(--text-muted) !important; }}

.brand-mark {{ font-size: clamp(2rem, 5.5vw, 2.7rem); font-weight: 800; letter-spacing: -0.02em; color: var(--text-primary) !important; margin: 0 !important; line-height: 1.1; }}
.brand-mark .dot {{ color: var(--green); }}
.brand-tag {{ font-size: 0.9rem; color: var(--text-secondary); margin: 0.1rem 0 0 0 !important; }}

div[data-testid="stAlertContainer"] {{ background: var(--surface) !important; border: 1px solid var(--border) !important; border-left: 3px solid var(--green) !important; border-radius: 14px; }}
div[data-testid="stAlertContainer"] p {{ color: var(--text-secondary) !important; font-size: 0.9rem; }}
div[data-testid="stAlertContainer"] svg {{ fill: var(--green) !important; }}

/* tab spacing via flex gap — reliable whether the tab is a div or a button */
[data-testid="stTabs"] [role="tablist"] {{ border-bottom: 1px solid var(--border); gap: 2.75rem !important; }}
div[data-testid="stTab"], button[data-baseweb="tab"] {{ padding: 0.35rem 0.2rem 0.7rem !important; margin-right: 0 !important; }}
div[data-testid="stTab"] p {{ font-size: 0.95rem !important; font-weight: 600; color: var(--text-muted); }}
div[data-testid="stTab"][aria-selected="true"] p {{ color: var(--text-primary); }}
div[data-testid="stTab"] .react-aria-SelectionIndicator,
div[data-testid="stTab"][data-selected="true"]::after {{ background: var(--green) !important; }}

div[data-testid="stRadioGroup"] {{ display: inline-flex; gap: 3px; background: var(--surface); border: 1px solid var(--border); border-radius: 999px; padding: 4px; }}
label[data-testid="stRadioOption"] {{ margin: 0 !important; min-height: 0 !important; }}
label[data-testid="stRadioOption"] > div > div > div:not([data-testid="stMarkdownContainer"]) {{ display: none; }}
label[data-testid="stRadioOption"] div[data-testid="stMarkdownContainer"] p {{ padding: 0.32rem 0.95rem; border-radius: 999px; margin: 0; font-size: 0.83rem; font-weight: 600; color: var(--text-secondary); white-space: nowrap; }}
label[data-testid="stRadioOption"][data-selected="true"] div[data-testid="stMarkdownContainer"] p {{ background: var(--green); color: var(--on-green); }}

[data-testid="stWidgetLabel"] p {{ font-size: 0.82rem; color: var(--text-secondary); }}
[data-testid="stSelectbox"] > div > div {{ font-size: 0.9rem; background: var(--surface); border-radius: 12px; border-color: var(--border); }}
[data-testid="stCheckbox"] p {{ color: var(--text-secondary); font-size: 0.85rem; }}
[data-baseweb="popover"] [role="listbox"], [data-baseweb="menu"], ul[role="listbox"] {{ background: var(--surface) !important; }}
[data-baseweb="popover"] li, ul[role="listbox"] li {{ color: var(--text-primary) !important; }}
[data-baseweb="popover"] li:hover {{ background: var(--surface-2) !important; }}

.panel {{ background: var(--surface); border: 1px solid var(--border); border-radius: 20px; padding: 1.3rem 1.5rem; margin-top: 0.3rem; }}
.panel .field-row:last-child {{ border-bottom: none; }}

.hero {{ background: var(--hero-grad); border: 1px solid var(--border); border-radius: 22px; padding: 1.6rem 1.8rem; margin-bottom: 1rem; }}
.hero-label {{ font-size: 0.8rem; font-weight: 600; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.05em; }}
.hero-value {{ font-size: clamp(2.1rem, 7vw, 3rem); font-weight: 800; color: var(--text-primary); line-height: 1.05; margin-top: 0.35rem; font-variant-numeric: tabular-nums; }}
.hero-value .cur {{ color: var(--green); font-size: 0.6em; font-weight: 700; vertical-align: 0.5em; margin-right: 0.12rem; }}
.hero-sub {{ font-size: 0.85rem; color: var(--text-muted); margin-top: 0.4rem; }}

.stat-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1.4rem; }}
.stat-card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 18px; padding: 1.1rem 1.3rem; }}
.stat-label {{ font-size: 0.75rem; font-weight: 600; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em; }}
.stat-value {{ font-size: clamp(1.4rem, 5vw, 1.7rem); font-weight: 800; color: var(--text-primary); margin-top: 0.3rem; font-variant-numeric: tabular-nums; }}

.section-title {{ font-size: 1.15rem; font-weight: 700; color: var(--text-primary); margin: 1.5rem 0 0.6rem; }}
.section-sub {{ font-size: 0.82rem; color: var(--text-muted); margin: -0.35rem 0 0.7rem; }}

.chip {{ display: inline-block; padding: 0.12rem 0.6rem; border-radius: 999px; font-size: 0.72rem; font-weight: 700; }}

.field-row {{ display: grid; align-items: center; padding: 0.5rem 0; border-bottom: 1px solid var(--border); font-size: 0.9rem; grid-template-columns: 130px 1fr; gap: 0.6rem; }}
.field-row.with-gt {{ grid-template-columns: 120px 1fr 1fr; }}
.field-row.header {{ font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 700; color: var(--text-muted); border-bottom: 1px solid var(--border); padding-bottom: 0.5rem; }}
.field-name {{ color: var(--text-secondary); }}
.field-value {{ font-family: var(--font-mono); font-variant-numeric: tabular-nums; color: var(--text-primary); word-break: break-word; }}
.field-gt {{ color: var(--text-muted); font-size: 0.82rem; font-family: var(--font-mono); word-break: break-word; }}
.li-name {{ color: var(--text-primary); font-weight: 500; word-break: break-word; }}
.li-price {{ font-family: var(--font-mono); font-variant-numeric: tabular-nums; color: var(--text-primary); font-weight: 600; }}
.li-head {{ margin: 1.3rem 0 0.4rem; font-size: 0.95rem; font-weight: 700; color: var(--text-primary); }}
.txn {{ display: grid; grid-template-columns: minmax(0,1.4fr) auto minmax(0,1fr) auto; gap: 0.8rem; align-items: center; padding: 0.7rem 0; border-bottom: 1px solid var(--border); }}
.txn:last-child {{ border-bottom: none; }}
.txn-store {{ font-weight: 700; color: var(--text-primary); font-size: 0.92rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
.txn-date {{ color: var(--text-muted); font-size: 0.82rem; }}
.txn-amt {{ text-align: right; font-weight: 700; font-family: var(--font-mono); font-variant-numeric: tabular-nums; }}

.receipt-frame {{ display: inline-block; background: var(--surface); padding: 8px; border-radius: 18px; border: 1px solid var(--border); margin: 4px 0 12px; max-width: 100%; }}
.receipt-frame img {{ display: block; max-width: 380px; width: 100%; border-radius: 12px; }}

[data-testid="stFileUploaderDropzone"] {{ padding: 1.1rem !important; border-radius: 16px !important; border: 1.5px dashed var(--border) !important; background: var(--surface) !important; }}
[data-testid="stFileUploaderDropzone"] svg {{ fill: var(--green) !important; }}
[data-testid="stFileUploaderDropzoneInstructions"] span {{ font-size: 0.82rem; color: var(--text-secondary); }}
[data-testid="stBaseButton-secondary"] {{ background: var(--green) !important; color: var(--on-green) !important; border: none !important; font-weight: 700 !important; border-radius: 999px !important; }}
/* -webkit-text-fill-color beats a plain color on inputs, else typed text stays white in light mode */
[data-testid="stTextInput"] input, [data-baseweb="input"] input, [data-baseweb="base-input"] input {{
  background: var(--surface) !important; border-radius: 12px !important; border-color: var(--border) !important;
  color: var(--text-primary) !important; -webkit-text-fill-color: var(--text-primary) !important;
}}
[data-testid="stTextInput"] input::placeholder {{ color: var(--text-muted) !important; -webkit-text-fill-color: var(--text-muted) !important; }}

@media (max-width: 680px) {{
  [data-testid="stMainBlockContainer"] {{ padding-left: 0.6rem; padding-right: 0.6rem; }}
  .stat-grid {{ grid-template-columns: 1fr; }}
  .panel, .hero {{ padding: 1.1rem 1.1rem; }}
  [data-testid="stTabs"] [role="tablist"] {{ gap: 1.5rem !important; }}
  .field-row {{ grid-template-columns: 96px 1fr !important; }}
  .field-row.with-gt {{ grid-template-columns: 90px 1fr 1fr !important; }}
  .txn {{ grid-template-columns: 1fr auto; gap: 0.4rem 0.6rem; }}
  .txn-date {{ display: none; }}
}}
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=60)
def fetch_receipts():
    r = requests.get(f"{API_BASE}/receipts", timeout=10)
    r.raise_for_status()
    return r.json()


def fetch_receipt(image_id: str, include_gt: bool):
    r = requests.get(f"{API_BASE}/receipts/{image_id}", params={"include_gt": include_gt}, timeout=10)
    r.raise_for_status()
    return r.json()


def fetch_infer(image_bytes: bytes, filename: str):
    # NOT cached: /infer appends to the backend's Overview list, and a cache hit would skip
    # that append. Rerun de-dup is handled by the caller via a per-file hash in session_state.
    r = requests.post(f"{API_BASE}/infer", files={"file": (filename, image_bytes)}, timeout=120)
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


def category_chip(cat: str, T: dict) -> str:
    color = T["cats"].get(cat, T["cats"]["other"])
    return f'<span class="chip" style="background:{color}22;color:{color};">{cat.title()}</span>'


def stat_card(label: str, value: str) -> str:
    return f'<div class="stat-card"><div class="stat-label">{label}</div><div class="stat-value">{value}</div></div>'


def image_data_uri(image_bytes: bytes) -> str:
    """One base64 <img> so the framed wrapper and image live in a single HTML fragment.
    HEIC is re-encoded to JPEG (browsers can't render it via <img>); the model gets the
    original bytes through serve.py."""
    img = Image.open(io.BytesIO(image_bytes))
    fmt = img.format
    if fmt in ("HEIF", "HEIC"):
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG")
        image_bytes, fmt = buf.getvalue(), "JPEG"
    b64 = base64.b64encode(image_bytes).decode("ascii")
    return f"data:image/{fmt.lower()};base64,{b64}"


def format_receipt_label(r: dict) -> str:
    store = (r["store"] or "(no store)").strip()
    date = (r["date"] or "no date").strip()
    total = f"${r['total']}" if r["total"] else "no total"
    return f"{store} · {date} · {total}"


def style_fig(fig: go.Figure, T: dict) -> go.Figure:
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                      margin=dict(t=8, b=8, l=8, r=8), font_color=T["text2"], font_family="Manrope")
    fig.update_xaxes(gridcolor=T["border"], zerolinecolor=T["border"], tickfont_color=T["text_muted"])
    fig.update_yaxes(gridcolor=T["border"], zerolinecolor=T["border"], tickfont_color=T["text_muted"])
    return fig


def render_receipt_tab(T: dict):
    source = st.radio("Source", ["Upload", "From URL", "Sample receipts"],
                      horizontal=True, key="imgsrc", label_visibility="collapsed")

    image_bytes, image_name, image_id = None, "upload.jpg", None
    if source == "Upload":
        st.caption("Drag & drop a receipt photo here, or browse your files.")
        uploaded = st.file_uploader("Choose a photo", type=["png", "jpg", "jpeg", "webp", "heic", "heif"],
                                    key="upload_widget", label_visibility="collapsed")
        if uploaded is not None:
            image_bytes, image_name = uploaded.getvalue(), uploaded.name
    elif source == "From URL":
        url = st.text_input("Image URL", key="url_widget", label_visibility="collapsed",
                            placeholder="Paste an image link…")
        if url:
            try:
                resp = requests.get(url, timeout=10)
                resp.raise_for_status()
                if not resp.headers.get("content-type", "").startswith("image/"):
                    st.error("That link didn't return an image.")
                else:
                    image_bytes, image_name = resp.content, url.rsplit("/", 1)[-1] or "url.jpg"
            except requests.RequestException as e:
                st.error(f"Couldn't fetch that link: {e}")
    else:
        try:
            receipts = fetch_receipts()
        except requests.RequestException as e:
            st.error(f"Can't reach the service ({e}).")
            return
        if not receipts:
            st.warning("No sample receipts available.")
            return
        options = {format_receipt_label(r): r["image_id"] for r in receipts}
        label = st.selectbox("Receipt", list(options.keys()), key="receipt_select")
        image_id = options[label]
        img_resp = requests.get(f"{API_BASE}/receipts/{image_id}/image", timeout=10)
        if img_resp.ok:
            image_bytes = img_resp.content

    col_img, col_table = st.columns([1, 1.4], gap="large")
    with col_img:
        if image_bytes:
            try:
                data_uri = image_data_uri(image_bytes)
            except Exception as e:
                st.error(f"Couldn't read that file as an image: {e}")
                data_uri = None
            if data_uri:
                st.markdown(f'<div class="receipt-frame"><img src="{data_uri}" /></div>', unsafe_allow_html=True)
        elif source == "Upload":
            st.info("Upload a receipt photo to begin.")
        elif source == "From URL":
            st.info("Paste an image link to begin.")
        else:
            st.info("No image available for this receipt.")

    include_gt = False
    detail = None
    with col_table:
        if source == "Sample receipts" and image_id is not None:
            include_gt = st.checkbox("Compare to reference", value=False)
            detail = fetch_receipt(image_id, include_gt)
        elif image_bytes:
            # analyze once per unique file: reruns reuse the stored result, no duplicate append
            h = hashlib.md5(image_bytes).hexdigest()
            if st.session_state.get("infer_hash") == h and st.session_state.get("infer_result"):
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
            pred = detail["prediction"]
            gt = detail.get("ground_truth")
            store = pred.get("store") or ""
            chip = category_chip(infer_category({"store": store, "line_items": pred.get("line_items")}), T)

            row_class = "field-row with-gt" if gt else "field-row"
            cols = "120px 1fr 1fr" if gt else "130px 1fr"
            header_cells = ["Field", "Value"] + (["Reference"] if gt else [])
            html = [f'<div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.7rem;flex-wrap:wrap;">'
                    f'<span style="font-size:1.1rem;font-weight:800;">{store or "Receipt"}</span>{chip}</div>']
            html.append(f'<div class="{row_class} header" style="grid-template-columns:{cols};">' +
                        "".join(f"<div>{c}</div>" for c in header_cells) + "</div>")
            for field in ["store", "date", "tax", "tip", "subtotal", "total"]:
                val = pred.get(field) if pred.get(field) is not None else "—"
                cells = [f'<div class="field-name">{FIELD_LABELS[field]}</div>',
                         f'<div class="field-value">{val}</div>']
                if gt:
                    gt_val = gt.get(field) if gt else None
                    cells.append(f'<div class="field-gt">{gt_val if gt_val is not None else "—"}</div>')
                html.append(f'<div class="{row_class}" style="grid-template-columns:{cols};">' + "".join(cells) + "</div>")

            items = pred.get("line_items") or []
            html.append('<div class="li-head">Line items</div>')
            if items:
                html.append('<div class="field-row header" style="grid-template-columns:1fr 110px;">'
                            "<div>Name</div><div>Price</div></div>")
                for it in items:
                    name = format_item_name(it.get("name")) or "—"
                    price = it.get("price") if it.get("price") is not None else "—"
                    html.append('<div class="field-row" style="grid-template-columns:1fr 110px;">'
                                f'<div class="li-name">{name}</div><div class="li-price">{price}</div></div>')
            else:
                html.append('<div class="section-sub" style="margin-top:0.5rem;">No line items found.</div>')

            st.markdown(f'<div class="panel">{"".join(html)}</div>', unsafe_allow_html=True)


def render_dashboard_tab(T: dict):
    try:
        dash = fetch_dashboard()
    except requests.RequestException as e:
        st.error(f"Can't reach the service ({e}).")
        return

    if dash["n_receipts"] == 0:
        st.info("Analyze a few receipts to build your spending overview.")
        return

    n = dash["n_receipts"]
    avg = dash["total_spend"] / dash["n_priced"] if dash.get("n_priced") else 0
    st.markdown(
        f'<div class="hero"><div class="hero-label">Total spent</div>'
        f'<div class="hero-value"><span class="cur">$</span>{dash["total_spend"]:,.2f}</div>'
        f'<div class="hero-sub">across {dash["n_priced"]} of {n} analyzed receipts</div></div>',
        unsafe_allow_html=True)
    st.markdown('<div class="stat-grid">' + stat_card("Receipts", f"{n}")
                + stat_card("Avg / receipt", f"${avg:,.2f}") + "</div>", unsafe_allow_html=True)

    by_month = dash.get("by_month") or []
    if by_month:
        st.markdown('<div class="section-title">Spending by month</div>', unsafe_allow_html=True)
        figm = go.Figure(go.Bar(x=[m["month"] for m in by_month], y=[m["spend"] for m in by_month],
                                marker_color=T["green"]))
        figm.update_xaxes(type="category")
        figm.update_traces(hovertemplate="%{x}: $%{y:,.2f}<extra></extra>")
        st.plotly_chart(style_fig(figm, T), use_container_width=True)

    cats = None
    try:
        cats = fetch_categories()
    except requests.RequestException:
        pass
    if cats:
        priced = [c for c in cats["categories"] if (c.get("total_spend") or 0) > 0]
        if priced:
            st.markdown('<div class="section-title">Spending by category</div>', unsafe_allow_html=True)
            figc = go.Figure(go.Pie(
                labels=[c["category"].title() for c in priced],
                values=[c["total_spend"] for c in priced],
                marker=dict(colors=[T["cats"].get(c["category"], T["cats"]["other"]) for c in priced],
                            line=dict(color=T["bg"], width=3)),
                hole=0.62, sort=False))
            figc.update_traces(textinfo="percent", textposition="outside",
                               textfont_family="Manrope", textfont_color=T["text2"],
                               hovertemplate="%{label}: $%{value:.2f} (%{percent})<extra></extra>")
            figc.update_layout(showlegend=True, legend=dict(orientation="h", y=-0.05,
                               font=dict(color=T["text2"], family="Manrope")))
            st.plotly_chart(style_fig(figc, T), use_container_width=True)

    recent = dash.get("recent") or []
    if recent:
        st.markdown('<div class="section-title">Recent transactions</div>', unsafe_allow_html=True)

        def _txn(r: dict) -> str:
            total = f'${r["total"]}' if r["total"] else "—"
            cat = r.get("category") or infer_category({"store": r.get("store") or ""})
            chip = category_chip(cat, T)
            return ('<div class="txn">'
                    f'<div class="txn-store">{r["store"] or "—"}</div>{chip}'
                    f'<div class="txn-date">{r["date"] or "—"}</div>'
                    f'<div class="txn-amt">{total}</div></div>')

        st.markdown('<div class="panel">' + "".join(_txn(r) for r in recent) + "</div>", unsafe_allow_html=True)


c_brand, c_theme = st.columns([3, 1.1], gap="small", vertical_alignment="center")
with c_theme:
    mode = st.radio("Theme", ["Dark", "Light"], horizontal=True, key="theme", label_visibility="collapsed")
T = THEMES[mode]
inject_css(T)
with c_brand:
    st.markdown('<p class="brand-mark">Receipt<span class="dot">VLM</span></p>'
                '<p class="brand-tag">Track every dollar, straight from your receipts.</p>',
                unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Receipts", "Overview"])
with tab1:
    render_receipt_tab(T)
with tab2:
    render_dashboard_tab(T)
