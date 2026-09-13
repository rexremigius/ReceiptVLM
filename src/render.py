"""Presentation layer for the Gradio Space.

The markup here is derived from app/streamlit_app.py, which already built its visuals as
HTML strings handed to st.markdown(unsafe_allow_html=True) -- the same strings work
verbatim in gr.HTML, so only the widget shell differed.

NOT yet shared with the Streamlit app. Its builders are interleaved with st.* calls
inside render_receipt_tab/render_dashboard_tab and its stylesheet is ~50 lines of
Streamlit-internal selectors (data-testid="stTabs", data-baseweb=..., and so on) that have
no meaning in Gradio, so folding it onto this module is a separate, invasive change that
cannot be validated without running the Streamlit + FastAPI pair. Until then the two UIs
do hold parallel copies of the palette and the row markup; changing colours or the field
table means editing both. app/streamlit_app.py remains the on-device MLX front end.

Theming works differently here. In Streamlit a radio flipped st.session_state and the
script rerun repainted everything. Gradio has no rerun, so threading a theme dict through
every component would make the theme control an input to every element on the page.
Instead both palettes are emitted as CSS variables under `.rv-dark` / `.rv-light` and each
block self-wraps via wrap(), so switching themes is just re-emitting the same HTML with a
different wrapper class -- no Python-side palette plumbing.
"""
from __future__ import annotations

import base64
import io

from PIL import Image

FIELD_LABELS = {"store": "Store", "date": "Date", "tax": "Tax", "tip": "Tip",
                "subtotal": "Subtotal", "total": "Total"}

SCALAR_ORDER = ["store", "date", "tax", "tip", "subtotal", "total"]

# Category colours validated CVD-safe against each mode's surface.
THEMES = {
    "Dark": {
        "bg": "#14161B", "surface": "#1E212B", "surface2": "#262A35", "border": "#2C313D",
        "text": "#F4F6FA", "text2": "#9BA3B2", "text_muted": "#6B7280",
        "green": "#22DD8A", "on_green": "#0B1F16",
        "hero_grad": "linear-gradient(145deg, #1F2A26 0%, #1E212B 55%)",
        "cats": {"dining": "#60A5FA", "grocery": "#34D399", "fuel": "#FB923C",
                 "retail": "#C084FC", "transport": "#FACC15", "misc": "#F472B6",
                 "other": "#94A3B8"},
    },
    "Light": {
        "bg": "#F4F6F9", "surface": "#FFFFFF", "surface2": "#EEF1F6", "border": "#E3E8EF",
        "text": "#14213B", "text2": "#566175", "text_muted": "#8A94A6",
        "green": "#12B76A", "on_green": "#FFFFFF",
        "hero_grad": "linear-gradient(145deg, #E8F7EF 0%, #FFFFFF 55%)",
        "cats": {"dining": "#2563EB", "grocery": "#059669", "fuel": "#EA580C",
                 "retail": "#7C3AED", "transport": "#A16207", "misc": "#DB2777",
                 "other": "#64748B"},
    },
}

# Confidence levels produced by pipeline.ConfidenceScorer. "na" and "unscored" are
# deliberately neutral: a null tip on a receipt with no tip is not a low-confidence
# extraction, and a red badge there would misreport the model.
CONFIDENCE_COLORS = {
    "green": ("var(--green)", "High confidence"),
    "amber": ("#F59E0B", "Medium confidence"),
    "red": ("#EF4444", "Low confidence"),
    "missing": ("#EF4444", "Expected but not found"),
    "na": ("var(--text-muted)", "Not present on receipt"),
    "unscored": ("var(--text-muted)", "No confidence signal"),
}

FONT_SANS = ("'Manrope', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, "
             "sans-serif")
FONT_MONO = "'JetBrains Mono', ui-monospace, 'SFMono-Regular', Menlo, Consolas, monospace"

THEME_CLASS = {"Dark": "rv-dark", "Light": "rv-light"}


def vars_block(selector: str, theme: str = "Dark") -> str:
    """Emit one palette as CSS variables under `selector`.

    For hosts that scope variables differently from the Gradio app -- the Streamlit app
    puts them on :root, since its rerun model repaints the whole page anyway.
    """

    return _vars_block(selector, THEMES.get(theme, THEMES["Dark"]))


def _vars_block(selector: str, T: dict) -> str:
    return f"""{selector} {{
  --bg:{T['bg']}; --surface:{T['surface']}; --surface-2:{T['surface2']};
  --border:{T['border']}; --text-primary:{T['text']}; --text-secondary:{T['text2']};
  --text-muted:{T['text_muted']}; --green:{T['green']}; --on-green:{T['on_green']};
  --hero-grad:{T['hero_grad']};
  --font-sans:{FONT_SANS}; --font-mono:{FONT_MONO};
}}"""


# Only the palette-independent structure lives here; each block gets its variables from
# the .rv-dark / .rv-light wrapper that wrap() applies.
SHARED_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

.rv { font-family: var(--font-sans); color: var(--text-primary); }
.rv * { font-family: inherit; box-sizing: border-box; }

/* Compact header. Streamlit's markdown rules force its paragraphs to 1rem regardless of
   what this sheet asks for, so the size is pinned here to keep both front ends identical
   rather than letting one silently win. */
.brand-mark { font-size: 1rem; font-weight: 800; letter-spacing: -0.01em; color: var(--text-primary); margin: 0; line-height: 1.3; }
.brand-mark .dot { color: var(--green); }
.brand-tag { font-size: 1rem; color: var(--text-secondary); margin: 0.1rem 0 0 0; }

.panel { background: var(--surface); border: 1px solid var(--border); border-radius: 20px; padding: 1.3rem 1.5rem; margin-top: 0.3rem; }
.panel .field-row:last-child { border-bottom: none; }

.hero { background: var(--hero-grad); border: 1px solid var(--border); border-radius: 22px; padding: 1.6rem 1.8rem; margin-bottom: 1rem; }
.hero-label { font-size: 0.8rem; font-weight: 600; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.05em; }
.hero-value { font-size: clamp(2.1rem, 7vw, 3rem); font-weight: 800; color: var(--text-primary); line-height: 1.05; margin-top: 0.35rem; font-variant-numeric: tabular-nums; }
.hero-value .cur { color: var(--green); font-size: 0.6em; font-weight: 700; vertical-align: 0.5em; margin-right: 0.12rem; }
.hero-sub { font-size: 0.85rem; color: var(--text-muted); margin-top: 0.4rem; }

.stat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1.4rem; }
.stat-card { background: var(--surface); border: 1px solid var(--border); border-radius: 18px; padding: 1.1rem 1.3rem; }
.stat-label { font-size: 0.75rem; font-weight: 600; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em; }
.stat-value { font-size: clamp(1.4rem, 5vw, 1.7rem); font-weight: 800; color: var(--text-primary); margin-top: 0.3rem; font-variant-numeric: tabular-nums; }

.section-title { font-size: 1.15rem; font-weight: 700; color: var(--text-primary); margin: 1.5rem 0 0.6rem; }
.section-sub { font-size: 0.82rem; color: var(--text-muted); margin: -0.35rem 0 0.7rem; }

.chip { display: inline-block; padding: 0.12rem 0.6rem; border-radius: 999px; font-size: 0.72rem; font-weight: 700; }

.field-row { display: grid; align-items: center; padding: 0.5rem 0; border-bottom: 1px solid var(--border); font-size: 0.9rem; grid-template-columns: 130px 1fr; gap: 0.6rem; }
.field-row.with-gt { grid-template-columns: 120px 1fr 1fr; }
.field-row.header { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 700; color: var(--text-muted); padding-bottom: 0.5rem; }
.field-name { color: var(--text-secondary); }
.field-value { font-family: var(--font-mono); font-variant-numeric: tabular-nums; color: var(--text-primary); word-break: break-word; }
.field-gt { color: var(--text-muted); font-size: 0.82rem; font-family: var(--font-mono); word-break: break-word; }
.li-name { color: var(--text-primary); font-weight: 500; word-break: break-word; }
.li-price { font-family: var(--font-mono); font-variant-numeric: tabular-nums; color: var(--text-primary); font-weight: 600; }
.li-head { margin: 1.3rem 0 0.4rem; font-size: 0.95rem; font-weight: 700; color: var(--text-primary); }

.dot-conf { position: relative; display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-left: 0.45rem; vertical-align: 0.05em; cursor: help; }
/* An 8px dot is a poor hover target, so a transparent pseudo-element widens it to 22px
   without affecting layout. */
.dot-conf[data-tip]::before { content: ""; position: absolute; inset: -7px; border-radius: 50%; }
.dot-conf[data-tip]::after {
  content: attr(data-tip);
  position: absolute; bottom: calc(100% + 7px); left: 50%;
  transform: translateX(-50%) translateY(3px);
  background: var(--surface-2); color: var(--text-primary);
  border: 1px solid var(--border); border-radius: 8px;
  padding: 0.3rem 0.55rem; font-size: 0.72rem; font-weight: 600;
  font-family: var(--font-sans); letter-spacing: 0;
  white-space: nowrap; opacity: 0; pointer-events: none; z-index: 50;
  box-shadow: 0 6px 18px rgba(0,0,0,0.35);
  transition: opacity 0.12s ease, transform 0.12s ease;
}
.dot-conf[data-tip]:hover::after { opacity: 1; transform: translateX(-50%) translateY(0); }

/* Skeleton used while a receipt is being analysed. */
@keyframes rv-shimmer { 0% { background-position: -420px 0; } 100% { background-position: 420px 0; } }
.skel { background: linear-gradient(90deg, var(--surface-2) 25%, var(--border) 37%, var(--surface-2) 63%); background-size: 840px 100%; animation: rv-shimmer 1.3s linear infinite; border-radius: 6px; height: 0.72rem; }
.skel-row { display: grid; grid-template-columns: 120px 1fr; gap: 0.8rem; align-items: center; padding: 0.62rem 0; border-bottom: 1px solid var(--border); }
.skel-row:last-child { border-bottom: none; }
.analyzing-head { display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.9rem; font-size: 0.9rem; font-weight: 700; color: var(--text-secondary); }
.analyzing-head .pulse { width: 9px; height: 9px; border-radius: 50%; background: var(--green); animation: rv-pulse 1s ease-in-out infinite; }
@keyframes rv-pulse { 0%,100% { opacity: 1; transform: scale(1); } 50% { opacity: 0.35; transform: scale(0.75); } }
.analyzing-head .ell::after { content: ""; animation: rv-ell 1.4s steps(4,end) infinite; }
@keyframes rv-ell { 0% { content: ""; } 25% { content: "."; } 50% { content: ".."; } 75% { content: "..."; } }
.conf-legend { display: flex; flex-wrap: wrap; gap: 0.9rem; margin-top: 1rem; font-size: 0.74rem; color: var(--text-muted); }
.conf-legend span { display: inline-flex; align-items: center; gap: 0.3rem; }

.txn { display: grid; grid-template-columns: minmax(0,1.4fr) auto minmax(0,1fr) auto; gap: 0.8rem; align-items: center; padding: 0.7rem 0; border-bottom: 1px solid var(--border); }
.txn:last-child { border-bottom: none; }
.txn-store { font-weight: 700; color: var(--text-primary); font-size: 0.92rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.txn-date { color: var(--text-muted); font-size: 0.82rem; }
.txn-amt { text-align: right; font-weight: 700; font-family: var(--font-mono); font-variant-numeric: tabular-nums; }

.receipt-frame { display: inline-block; background: var(--surface); padding: 8px; border-radius: 18px; border: 1px solid var(--border); margin: 4px 0 12px; max-width: 100%; }
.receipt-frame img { display: block; max-width: 380px; width: 100%; border-radius: 12px; }

.empty-note { color: var(--text-muted); font-size: 0.85rem; padding: 0.6rem 0; }

@media (max-width: 680px) {
  .stat-grid { grid-template-columns: 1fr; }
  .panel, .hero { padding: 1.1rem 1.1rem; }
  .field-row { grid-template-columns: 96px 1fr !important; }
  .field-row.with-gt { grid-template-columns: 90px 1fr 1fr !important; }
  .txn { grid-template-columns: 1fr auto; gap: 0.4rem 0.6rem; }
  .txn-date { display: none; }
}
"""


# Gradio's own chrome, restyled to match the .rv blocks.
#
# Only Gradio's semantic class names are targeted (`selected`, `wrap`, `wrap-inner`,
# `secondary-wrap`, `container`, `float`, `icon-wrap`, `or`) plus element tags and the
# elem_classes hooks app.py attaches. The per-build hashed names in the same markup
# (svelte-1bx8sav and friends) are never referenced, since they change on every gradio
# release.
WIDGET_CSS = """
.gradio-container { background: var(--bg) !important; }
.gradio-container .block,
.gradio-container fieldset.block { background: transparent; border: none; box-shadow: none; }

/* widget labels. label.float is gradio's floating chip over a dropzone and carries its
   own background, so that needs setting too or it stays dark in light mode. */
.gradio-container span[data-testid="block-info"],
.gradio-container .block > .container > span,
.gradio-container label.float {
  font-size: 0.82rem !important; color: var(--text-secondary) !important;
  font-family: var(--font-sans);
}
.gradio-container label.float {
  background: var(--surface) !important; border: 1px solid var(--border) !important;
  border-radius: 8px !important;
}

/* tabs: green indicator, themed text, generous spacing like the Streamlit build */
.gradio-container .tab-wrapper,
.gradio-container .tab-nav { border-bottom: 1px solid var(--border) !important; }
.gradio-container .tab-container { gap: 2.75rem !important; border: none !important; }
.gradio-container .tabs button {
  color: var(--text-muted) !important; font-weight: 600 !important;
  font-size: 0.95rem !important; font-family: var(--font-sans);
  border: none !important; background: transparent !important;
  padding: 0.35rem 0.2rem 0.7rem !important;
}
.gradio-container .tabs button.selected { color: var(--text-primary) !important; }
/* The active-tab underline is a ::after pseudo-element carrying gradio's own accent
   (orange), painted over any border-bottom -- so it has to be recoloured directly. */
.gradio-container .tabs button::after,
.gradio-container .tabs button.selected::after {
  background: var(--green) !important;
}

/* radios as a green segmented pill control */
.rv-seg > .wrap:last-of-type,
.rv-seg .wrap:has(> label) {
  display: inline-flex !important; gap: 3px; background: var(--surface);
  border: 1px solid var(--border); border-radius: 999px; padding: 4px;
  width: auto !important;
}
.rv-seg label {
  margin: 0 !important; min-height: 0 !important; border: none !important;
  background: transparent !important; box-shadow: none !important;
  padding: 0.32rem 0.95rem !important; border-radius: 999px !important;
  font-size: 0.83rem !important; font-weight: 600 !important;
  color: var(--text-secondary) !important; font-family: var(--font-sans);
  white-space: nowrap; cursor: pointer;
}
.rv-seg label.selected {
  background: var(--green) !important; color: var(--on-green) !important;
}
.rv-seg label input { display: none !important; }

/* file dropzone */
.rv-file button {
  border: 1.5px dashed var(--border) !important; border-radius: 16px !important;
  background: var(--surface) !important; color: var(--text-secondary) !important;
  font-family: var(--font-sans);
}
.rv-file button:hover { border-color: var(--green) !important; }
.rv-file .icon-wrap svg, .rv-file label.float svg { fill: var(--green) !important; color: var(--green) !important; }
/* The dropzone's own spans set a lighter colour that is unreadable on the light surface,
   so the prompt text is coloured on the descendants rather than just the button. */
.rv-file button, .rv-file button span, .rv-file .wrap {
  color: var(--text-secondary) !important;
}
.rv-file .or { color: var(--text-muted) !important; }

/* Compact the dropzone to match Streamlit's single ~78px row. min-height alone is not
   enough: Gradio stacks icon / "Drop File Here" / "- or -" / "Click to Upload"
   vertically, so the content drives the height. Laying that stack out horizontally and
   dropping the separator is what actually shrinks it. Selectors stay on semantic class
   names under .rv-file -- Gradio's hashed svelte-* classes change between releases. */
.rv-file button.center {
  min-height: 78px !important;
  height: 78px !important;
  padding: 0 1.1rem !important;
  /* The button is flex-direction: column, so justify-content controls the *vertical*
     axis here -- align-items is what moves content left. */
  align-items: flex-start !important;
  justify-content: center !important;
}
.rv-file button .wrap {
  flex-direction: row !important;
  align-items: center !important;
  justify-content: flex-start !important;
  gap: 0.55rem !important;
  min-height: 0 !important;
  padding: 0 !important;
}
.rv-file .icon-wrap, .rv-file .icon-wrap svg {
  width: 18px !important; height: 18px !important; margin: 0 !important;
}
/* Relabel the dropzone to match Streamlit's "[Upload] 12MB per file - PNG, JPG, ..." row.
   Gradio's own wording ("Drop File Here", "- or -", "Click to Upload") lives in bare text
   nodes that no selector can reach, so the whole label is blanked with font-size: 0 and
   rebuilt from two pseudo-elements. upload_css() supplies the hint text so the size limit
   tracks MAX_UPLOAD_BYTES instead of being duplicated here. */
.rv-file button .wrap { font-size: 0 !important; }
.rv-file .icon-wrap { display: none !important; }
.rv-file .or { display: none !important; }
.rv-file button .wrap::before {
  content: "⬆  Upload";
  font-size: 0.85rem;
  font-weight: 700;
  color: var(--on-green);
  background: var(--green);
  border-radius: 999px;
  padding: 0.4rem 0.95rem;
  white-space: nowrap;
}
.rv-caption .upload-caption {
  font-size: 0.82rem; color: var(--text-muted); margin: 0.2rem 0 0.35rem;
}

/* dropdown */
.rv-select .wrap-inner, .rv-select .secondary-wrap, .rv-select input {
  background: var(--surface) !important; border-radius: 12px !important;
  border-color: var(--border) !important; color: var(--text-primary) !important;
  -webkit-text-fill-color: var(--text-primary) !important;
  font-family: var(--font-sans) !important; font-size: 0.9rem !important;
}
.rv-select ul, .rv-select [role="listbox"] { background: var(--surface) !important; }
.rv-select li { color: var(--text-primary) !important; font-size: 0.88rem !important; }
.rv-select li:hover, .rv-select li.selected { background: var(--surface-2) !important; }

/* checkbox */
.rv-check label { color: var(--text-secondary) !important; font-size: 0.85rem !important; font-family: var(--font-sans); }
.rv-check input[type="checkbox"] { accent-color: var(--green); }

/* plots sit on the page background, not in a Gradio card */
.rv-plot, .rv-plot > div { background: transparent !important; border: none !important; }

footer { display: none !important; }
"""


# Gradio's own widgets are siblings of the .rv blocks, not descendants, so they cannot
# inherit palette variables from a .rv wrapper. The palette is therefore also declared at
# container level, switched by toggling ROOT_LIGHT_CLASS (see app.py's client-side theme
# handler) rather than by a server round-trip.
ROOT_LIGHT_CLASS = "rv-light-root"


def upload_css(max_upload_bytes: int) -> str:
    """The dropzone hint, generated so the size limit is not duplicated in the stylesheet.

    Paired with the .wrap::before pill in SHARED_CSS; together they replace Gradio's
    built-in dropzone wording, which is unreachable bare text.
    """

    return (".rv-file button .wrap::after {"
            f' content: "{max_upload_bytes // (1024 * 1024)}MB per file '
            '\\2022  PNG, JPG, WEBP, HEIC, HEIF";'
            " font-size: 0.8rem; color: var(--text-muted);"
            " margin-left: 0.7rem; white-space: nowrap; }")


def full_css() -> str:
    """Both palettes (container-level and per-block) plus the shared structure.

    Specificity is deliberate. The container rules come first at two classes
    (`.gradio-container.rv-light-root`), and the per-block rules follow, also at two
    classes (`.gradio-container .rv-dark`). Equal specificity means source order decides,
    so a block's own wrapper always wins over the container default -- which is what lets
    wrap() theme an individual block regardless of the root class.
    """

    return "\n".join([
        _vars_block(".gradio-container", THEMES["Dark"]),
        _vars_block(f".gradio-container.{ROOT_LIGHT_CLASS}", THEMES["Light"]),
        _vars_block(".gradio-container .rv-dark", THEMES["Dark"]),
        _vars_block(".gradio-container .rv-light", THEMES["Light"]),
        SHARED_CSS,
        WIDGET_CSS,
    ])


def wrap(html: str, theme: str | None = "Dark") -> str:
    """Wrap a block so it picks up a palette's CSS variables.

    Pass theme=None for a block that is emitted once and never re-rendered on a theme
    change: it then inherits the container-level palette, which the client-side toggle
    switches, instead of being pinned to whatever theme was active when it was built.
    """

    if theme is None:
        return f'<div class="rv">{html}</div>'
    return f'<div class="rv {THEME_CLASS.get(theme, "rv-dark")}">{html}</div>'


# --- small pieces ------------------------------------------------------------------

def category_chip(cat: str, theme: str = "Dark") -> str:
    cats = THEMES.get(theme, THEMES["Dark"])["cats"]
    color = cats.get(cat, cats["other"])
    return f'<span class="chip" style="background:{color}22;color:{color};">{cat.title()}</span>'


def stat_card(label: str, value: str) -> str:
    return (f'<div class="stat-card"><div class="stat-label">{label}</div>'
            f'<div class="stat-value">{value}</div></div>')


def confidence_dot(badge: dict | None) -> str:
    """A coloured dot carrying the calibrated confidence level, with the score on hover.

    The Streamlit app logged confidence but never showed it. Surfacing it here is the
    point of the demo -- the calibrated score is the project's headline result -- and a
    dot keeps it from competing with the value itself.
    """

    if not badge:
        return ""
    level = badge.get("level")
    if level not in CONFIDENCE_COLORS:
        return ""
    color, description = CONFIDENCE_COLORS[level]
    score = badge.get("score")
    tip = description + (f" · {score:.2f}" if isinstance(score, (int, float)) else "")
    # data-tip drives the CSS tooltip in SHARED_CSS rather than the native title
    # attribute: a browser tooltip needs ~1s of hover on an 8px target, which in practice
    # meant the calibrated score was unreachable. aria-label keeps it available to
    # screen readers, which the styled tooltip alone would not be.
    return (f'<span class="dot-conf" style="background:{color};" '
            f'data-tip="{tip}" aria-label="{tip}"></span>')


def confidence_legend() -> str:
    """Legend for the dots.

    "unscored" is included because it is common and shares grey with "na": receipts whose
    line-item prices give confidence.py no arithmetic signal come back unscored for every
    item, and an unexplained grey dot reads as a bug rather than as an honest abstention.
    """

    seen, parts = set(), []
    for level in ("green", "amber", "red", "na", "unscored"):
        color, description = CONFIDENCE_COLORS[level]
        if description in seen:
            continue
        seen.add(description)
        parts.append(f'<span><i class="dot-conf" style="background:{color};margin-left:0;'
                     f'"></i>{description}</span>')
    return ('<div class="conf-legend">' + "".join(parts)
            + '<span>hover a dot for the calibrated score</span></div>')


def image_data_uri(image_bytes: bytes) -> str:
    """One base64 <img> so the framed wrapper and image are a single HTML fragment.

    HEIC is re-encoded to JPEG because browsers cannot render it via <img>; the model
    still receives the original bytes.
    """

    img = Image.open(io.BytesIO(image_bytes))
    fmt = img.format
    if fmt in ("HEIF", "HEIC"):
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG")
        image_bytes, fmt = buf.getvalue(), "JPEG"
    b64 = base64.b64encode(image_bytes).decode("ascii")
    return f"data:image/{(fmt or 'jpeg').lower()};base64,{b64}"


def receipt_frame(image_bytes: bytes) -> str:
    return f'<div class="receipt-frame"><img src="{image_data_uri(image_bytes)}" /></div>'


# Sample-receipt labelling lives in src/samples.py, which owns both the label format and
# the pick order the Space bundles images for.


# --- composite blocks --------------------------------------------------------------

def detail_panel(prediction: dict, confidence: dict | None = None,
                 ground_truth: dict | None = None, category: str | None = None,
                 theme: str = "Dark", format_item_name=None) -> str:
    """The extracted-fields table, optionally beside a reference column."""

    gt = ground_truth
    store = prediction.get("store") or ""
    chip = category_chip(category, theme) if category else ""

    row_class = "field-row with-gt" if gt else "field-row"
    cols = "120px 1fr 1fr" if gt else "130px 1fr"
    header_cells = ["Field", "Value"] + (["Reference"] if gt else [])

    html = [f'<div style="display:flex;align-items:center;gap:0.6rem;'
            f'margin-bottom:0.7rem;flex-wrap:wrap;">'
            f'<span style="font-size:1.1rem;font-weight:800;">'
            f'{store or "Receipt"}</span>{chip}</div>']
    html.append(f'<div class="{row_class} header" style="grid-template-columns:{cols};">'
                + "".join(f"<div>{c}</div>" for c in header_cells) + "</div>")

    for field in SCALAR_ORDER:
        val = prediction.get(field)
        val = val if val is not None else "—"
        dot = confidence_dot((confidence or {}).get(field))
        cells = [f'<div class="field-name">{FIELD_LABELS[field]}</div>',
                 f'<div class="field-value">{val}{dot}</div>']
        if gt:
            gt_val = gt.get(field)
            cells.append(f'<div class="field-gt">'
                         f'{gt_val if gt_val is not None else "—"}</div>')
        html.append(f'<div class="{row_class}" style="grid-template-columns:{cols};">'
                    + "".join(cells) + "</div>")

    items = prediction.get("line_items") or []
    li_conf = (confidence or {}).get("line_items") or {}
    agg_dot = confidence_dot(li_conf.get("aggregate"))
    html.append(f'<div class="li-head">Line items{agg_dot}</div>')
    if items:
        item_badges = li_conf.get("items") or []
        html.append('<div class="field-row header" '
                    'style="grid-template-columns:1fr 110px;">'
                    "<div>Name</div><div>Price</div></div>")
        for idx, it in enumerate(items):
            raw_name = it.get("name")
            name = (format_item_name(raw_name) if format_item_name else raw_name) or "—"
            price = it.get("price") if it.get("price") is not None else "—"
            dot = confidence_dot(item_badges[idx] if idx < len(item_badges) else None)
            html.append('<div class="field-row" style="grid-template-columns:1fr 110px;">'
                        f'<div class="li-name">{name}</div>'
                        f'<div class="li-price">{price}{dot}</div></div>')
    else:
        html.append('<div class="section-sub" style="margin-top:0.5rem;">'
                    "No line items found.</div>")

    if confidence:
        html.append(confidence_legend())

    return f'<div class="panel">{"".join(html)}</div>'


def analyzing_panel(theme: str = "Dark") -> str:
    """Placeholder shown while the model runs.

    Mirrors the shape of detail_panel -- a header line plus six field rows -- so the real
    result replaces it in place rather than the layout jumping. Widths vary per row so it
    reads as content loading rather than a progress bar.
    """

    widths = ["58%", "34%", "26%", "22%", "30%", "30%"]
    rows = "".join(
        f'<div class="skel-row"><div class="skel" style="width:62%"></div>'
        f'<div class="skel" style="width:{w}"></div></div>'
        for w in widths
    )
    return (
        '<div class="panel">'
        '<div class="analyzing-head"><span class="pulse"></span>'
        'Analyzing receipt<span class="ell"></span></div>'
        f'<div class="skel" style="width:38%;height:1.05rem;margin-bottom:1rem"></div>'
        f'{rows}'
        '<div class="li-head" style="opacity:0.65">Line items</div>'
        '<div class="skel-row"><div class="skel" style="width:70%"></div>'
        '<div class="skel" style="width:28%"></div></div>'
        '<div class="skel-row"><div class="skel" style="width:52%"></div>'
        '<div class="skel" style="width:28%"></div></div>'
        '</div>'
    )


def overview_blocks(dash: dict, cats: dict | None, theme: str = "Dark",
                    infer_category=None) -> str:
    """Hero + stat cards + recent transactions. Charts are separate gr.Plot components."""

    n = dash.get("n_receipts", 0)
    if not n:
        return ('<div class="panel"><div class="empty-note">'
                "Analyze a receipt to build your spending overview.</div></div>")

    n_priced = dash.get("n_priced") or 0
    avg = dash["total_spend"] / n_priced if n_priced else 0
    out = [
        '<div class="hero"><div class="hero-label">Total spent</div>'
        f'<div class="hero-value"><span class="cur">$</span>'
        f'{dash["total_spend"]:,.2f}</div>'
        f'<div class="hero-sub">across {n_priced} of {n} analyzed receipts</div></div>',
        '<div class="stat-grid">' + stat_card("Receipts", f"{n}")
        + stat_card("Avg / receipt", f"${avg:,.2f}") + "</div>",
    ]

    recent = dash.get("recent") or []
    if recent:
        out.append('<div class="section-title">Recent transactions</div>')
        out.append(transactions_panel(recent, theme=theme,
                                      infer_category=infer_category))

    if dash.get("caveat"):
        out.append(f'<div class="section-sub" style="margin-top:0.8rem;">'
                   f'{dash["caveat"]}</div>')
    return "".join(out)


def transactions_panel(recent: list[dict], theme: str = "Dark",
                       infer_category=None) -> str:
    """The recent-transactions list.

    Separate from overview_blocks so a caller can place it after the charts (the
    Streamlit app's order) rather than immediately after the stat cards.
    """

    rows = []
    for r in recent:
        total = f'${r["total"]}' if r.get("total") else "—"
        cat = r.get("category")
        if cat is None and infer_category is not None:
            cat = infer_category({"store": r.get("store") or ""})
        chip = category_chip(cat, theme) if cat else ""
        rows.append('<div class="txn">'
                    f'<div class="txn-store">{r.get("store") or "—"}</div>{chip}'
                    f'<div class="txn-date">{r.get("date") or "—"}</div>'
                    f'<div class="txn-amt">{total}</div></div>')
    return '<div class="panel">' + "".join(rows) + "</div>"


def style_fig(fig, theme: str = "Dark"):
    """Apply the palette to a plotly figure (charts cannot read CSS variables)."""

    T = THEMES.get(theme, THEMES["Dark"])
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                      margin=dict(t=8, b=8, l=8, r=8), font_color=T["text2"],
                      font_family="Manrope",
                      # plotly defaults to 450px, which dominates the page next to the
                      # stat cards; this matches the dashboard card proportions.
                      height=300, autosize=True)
    fig.update_xaxes(gridcolor=T["border"], zerolinecolor=T["border"],
                     tickfont_color=T["text_muted"])
    fig.update_yaxes(gridcolor=T["border"], zerolinecolor=T["border"],
                     tickfont_color=T["text_muted"])
    return fig


def month_figure(by_month: list[dict], theme: str = "Dark"):
    import plotly.graph_objects as go

    T = THEMES.get(theme, THEMES["Dark"])
    fig = go.Figure(go.Bar(x=[m["month"] for m in by_month],
                           y=[m["spend"] for m in by_month],
                           marker_color=T["green"]))
    fig.update_xaxes(type="category")
    # Plotly sizes a bar as a fraction of its category slot, and with a single month that
    # slot is the entire plot -- the chart renders as one solid block. Cap the width so a
    # one- or two-month dashboard still reads as a bar chart.
    fig.update_traces(width=min(0.5, 0.18 * max(len(by_month), 1)),
                      hovertemplate="%{x}: $%{y:,.2f}<extra></extra>")
    return style_fig(fig, theme)


def category_figure(categories: list[dict], theme: str = "Dark"):
    import plotly.graph_objects as go

    T = THEMES.get(theme, THEMES["Dark"])
    priced = [c for c in categories if (c.get("total_spend") or 0) > 0]
    if not priced:
        return None
    fig = go.Figure(go.Pie(
        labels=[c["category"].title() for c in priced],
        values=[c["total_spend"] for c in priced],
        marker=dict(colors=[T["cats"].get(c["category"], T["cats"]["other"])
                            for c in priced],
                    line=dict(color=T["bg"], width=3)),
        hole=0.62, sort=False))
    fig.update_traces(textinfo="percent", textposition="outside",
                      textfont_family="Manrope", textfont_color=T["text2"],
                      hovertemplate="%{label}: $%{value:.2f} (%{percent})<extra></extra>")
    fig.update_layout(showlegend=True,
                      legend=dict(orientation="h", y=-0.05,
                                  font=dict(color=T["text2"], family="Manrope")))
    return style_fig(fig, theme)
