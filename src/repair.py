"""Deliverable #9 — JSON repair layer (used by #10; see CLAUDE.md / SKILL.md #9).

Constrained/grammar decoding (SKILL.md's "first line of defense") is an MLX-VLM
generation-time concern that needs a loaded model — this analysis machine has neither,
so that half of #9 is an M5 task, not this file's. What's buildable and testable here
is the "second line": a fixer for the JSON a model *did* generate but got slightly
wrong — trailing commas, a stray Python-repr dialect, or truncation from a max_tokens
cutoff mid-generation. Anything still unparseable after every fixup is a genuine hard
failure, reported as such (never silently dropped, never guessed at) so the eval
denominator stays honest.

`zeroshot.py` (#3, and #4's fine-tuned inference path via --adapter-path) is this
module's first real caller — its old inline `extract_json` explicitly deferred to
"#9's job" in a comment; this replaces that.

Usage:
    python src/repair.py            # runs the synthetic smoke test (representative
                                     # broken-JSON cases, no model needed)
"""
from __future__ import annotations

import ast
import json
import re

CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")


def _strip_fence(text: str) -> str:
    m = CODE_FENCE_RE.search(text)
    return m.group(1) if m else text


def _outer_braces(text: str) -> str | None:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    return text[start:end + 1]


def _fix_trailing_commas(text: str) -> str:
    return TRAILING_COMMA_RE.sub(r"\1", text)


def _fix_python_literal(text: str) -> dict | None:
    """Model output has, at least once (see PROGRESS.md 2026-07-20, the corrupted
    training-target bug), degenerated into Python-repr-style pseudo-JSON — single
    quotes, `None`/`True`/`False` — instead of real JSON. `ast.literal_eval` parses
    that dialect directly; regex-swapping single quotes for double quotes was
    considered and rejected, since it breaks on any string value that itself
    contains an apostrophe (e.g. store name "Wendy's").
    """
    try:
        val = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return None
    return val if isinstance(val, dict) else None


def _close_truncated(text: str) -> str:
    """Best-effort close-out for JSON cut off mid-generation by a max_tokens limit.

    Walks the text tracking bracket depth (skipping the inside of strings so a stray
    brace in a value doesn't miscount) and the position right after the last complete
    comma at any nesting depth. If the text ends *inside* an unterminated string (the
    common truncation case — cut off mid key or mid value), trims back to that last
    safe boundary before closing; if it ends cleanly outside a string (e.g. right
    after a complete nested `}`), closes as-is without trimming, since trimming there
    would discard an already-complete trailing element.
    """
    stack: list[str] = []
    in_string = False
    escape = False
    last_safe = 0
    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if stack:
                stack.pop()
        elif ch == "," and stack:
            last_safe = i + 1

    if in_string and last_safe:
        text = text[:last_safe]
        stack, in_string, escape = [], False, False
        for ch in text:
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch in "{[":
                stack.append(ch)
            elif ch in "}]":
                if stack:
                    stack.pop()

    text = text.rstrip()
    if text.endswith(","):
        text = text[:-1]
    closers = {"{": "}", "[": "]"}
    return text + "".join(closers[c] for c in reversed(stack))


def repair_json(raw: str) -> tuple[dict | None, str]:
    """Extract and repair a JSON object from a model's raw completion.

    Returns (parsed, status). `parsed` is the dict, or None on hard failure — callers
    must still emit a record (all-null fields) for that receipt, per SKILL.md #5/#9:
    an unparseable receipt is a scored failure, not a dropped one. `status` is one of
    "clean" / "repaired_trailing_comma" / "repaired_python_literal" /
    "repaired_truncation" / "hard_failure", for the caller to tally.
    """
    candidate = _outer_braces(_strip_fence(raw))
    if candidate is None:
        return None, "hard_failure"

    try:
        return json.loads(candidate), "clean"
    except json.JSONDecodeError:
        pass

    fixed = _fix_trailing_commas(candidate)
    try:
        return json.loads(fixed), "repaired_trailing_comma"
    except json.JSONDecodeError:
        pass

    literal = _fix_python_literal(candidate)
    if literal is not None:
        return literal, "repaired_python_literal"

    closed = _close_truncated(fixed)
    try:
        return json.loads(closed), "repaired_truncation"
    except json.JSONDecodeError:
        pass

    return None, "hard_failure"


def _smoke():
    cases = [
        ("clean", '{"store": "CVS", "total": "5.40", "line_items": []}'),
        ("clean (fenced)",
         '```json\n{"store": "CVS", "total": "5.40", "line_items": []}\n```'),
        ("repaired_trailing_comma",
         '{"store": "CVS", "total": "5.40", "line_items": [{"name": "Advil", "price": "5.00"},]}'),
        ("repaired_python_literal",
         "{'store': \"Wendy's\", 'total': '5.40', 'tip': None, 'line_items': []}"),
        ("repaired_truncation (mid-string)",
         '{"store": "CVS", "line_items": [{"name": "Advil", "price": "5.00"}, '
         '{"name": "Cough Sy'),
        ("repaired_truncation (mid-structure)",
         '{"store": "CVS", "line_items": [{"name": "Advil", "price": "5.00"}'),
        ("hard_failure (no braces at all)", "the model rambled and never produced json"),
        ("hard_failure (irrecoverable garbage)", '{"store": "CVS", "line_items": [{{{'),
    ]
    print(f"{'expected':<34}{'got':<28}{'match':<7}parsed")
    all_ok = True
    for expected, raw in cases:
        parsed, status = repair_json(raw)
        ok = status == expected.split(" ")[0]
        all_ok &= ok
        print(f"{expected:<34}{status:<28}{'OK' if ok else 'MISMATCH':<7}{parsed}")
    print("\nALL PASS" if all_ok else "\nSOME MISMATCHES ABOVE")


if __name__ == "__main__":
    _smoke()
