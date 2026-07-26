from __future__ import annotations

import ast
import json
import re

CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")


def _strip_fence(text: str) -> str:
    """If the text is fenced in a ```json block, return the inner text; otherwise return the text."""

    m = CODE_FENCE_RE.search(text)
    return m.group(1) if m else text


def _outer_braces(text: str) -> str | None:
    """Return the substring of text that is enclosed in the outermost braces, or None if none found."""

    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    return text[start:end + 1]


def _fix_trailing_commas(text: str) -> str:
    """Remove trailing commas before closing braces/brackets, which are invalid in JSON."""
    
    return TRAILING_COMMA_RE.sub(r"\1", text)


def _fix_python_literal(text: str) -> dict | None:
    """Attempt to parse a Python literal dict (single quotes, None, etc.) and return it as a dict.
    Returns None if the text is not a valid Python literal dict.
    """

    try:
        val = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return None
    return val if isinstance(val, dict) else None


def _close_truncated(text: str) -> str:
    """Attempt to close a truncated JSON string by adding closing braces/brackets. Returns the
    closed string, which may still be invalid JSON if the truncation was mid-string or mid-structure.
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
    """Attempt to repair a raw string that should be JSON, returning a tuple of (parsed dict or None,
    status string). Status is one of:
    - "clean": valid JSON, no repair needed
    - "repaired_trailing_comma": valid JSON after removing trailing commas
    - "repaired_python_literal": valid JSON after converting from Python literal syntax
    - "repaired_truncation": valid JSON after closing truncated structure
    - "hard_failure": could not parse as JSON
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
    """Run a smoke test of the repair_json function on various cases, printing results."""

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
