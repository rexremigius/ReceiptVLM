"""Picks and labels the bundled demo receipts.

Shared by app.py (which offers them) and scripts/build_space.py (which copies their
images into the Space). If the two disagreed, the Space would ship images for receipts the
dropdown never lists, or list receipts whose images are missing - so the ranking lives
here and both import it.
"""

from __future__ import annotations

from pathlib import Path

# A demo picker, not the evaluation set: a 472-entry dropdown is unusable, and a plain
# alphabetical sort leads with every storeless receipt - the model at its least
# impressive.
DEFAULT_MAX_SAMPLES = 40


def format_receipt_label(rec: dict) -> str:
    store = (rec.get("store") or "(no store)").strip()
    date = (rec.get("date") or "no date").strip()
    total = f"${rec['total']}" if rec.get("total") else "no total"
    return f"{store} · {date} · {total}"


def _rank(rec: dict) -> tuple:
    """Sort key: complete receipts first, then the ones with the most line items."""

    has_store = bool((rec.get("store") or "").strip())
    has_total = bool(rec.get("total"))
    n_items = len(rec.get("line_items") or [])
    # Negated so "more complete" and "more items" sort first.
    return (not has_store, not has_total, -n_items, format_receipt_label(rec))


def available_ids(predictions: dict[str, dict], image_root: Path) -> list[str]:
    """Prediction ids whose image is actually present on disk."""

    return [i for i in predictions if (image_root / i).exists()]


def pick(
    predictions: dict[str, dict],
    image_root: Path,
    max_samples: int = DEFAULT_MAX_SAMPLES,
) -> list[str]:
    """The ordered image ids to offer as samples."""

    ids = available_ids(predictions, image_root)
    return sorted(ids, key=lambda i: _rank(predictions[i]))[:max_samples]


def label_map(predictions: dict[str, dict], ids: list[str]) -> dict[str, str]:
    """{label: image_id}, with collisions disambiguated.

    store/date/total is not unique - two receipts from the same shop on the same day
    collide - and keying a dict on a colliding label would silently drop samples.
    """

    out: dict[str, str] = {}
    for image_id in ids:
        label = format_receipt_label(predictions[image_id])
        if label in out:
            n = 2
            while f"{label} ({n})" in out:
                n += 1
            label = f"{label} ({n})"
        out[label] = image_id
    return out
