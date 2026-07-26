from __future__ import annotations

import re

_LOWER_TO_UPPER = re.compile(r"(?<=[a-z])(?=[A-Z])")          # GrossesWasser - Grosses Wasser
_ACRONYM_TO_WORD = re.compile(r"(?<=[A-Z])(?=[A-Z][a-z])")    # HALFDietCake - HALF DietCake
# opt-in only: mangles brands that fuse letters+digits (7UP, V8), so off by default
_LETTER_DIGIT = re.compile(r"(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])")


def format_item_name(name, split_digits: bool = False) -> str:
    """Split camel-case and acronyms into words, optionally splitting letters from digits."""
    
    if not isinstance(name, str) or not name:
        return name
    s = _ACRONYM_TO_WORD.sub(" ", name)
    s = _LOWER_TO_UPPER.sub(" ", s)
    if split_digits:
        s = _LETTER_DIGIT.sub(" ", s)
    return re.sub(r"\s{2,}", " ", s).strip()


if __name__ == "__main__":
    cases = [
        ("GrossesWasser", "Grosses Wasser"),
        ("LöwenbräuOriginal", "Löwenbräu Original"),
        ("OakSmokedSalmon", "Oak Smoked Salmon"),
        ("HALFDietCake", "HALF Diet Cake"),
        ("CajunChixWRAP", "Cajun Chix WRAP"),
        ("SWIRLPOPS", "SWIRLPOPS"),
        ("LACTAIDFF", "LACTAIDFF"),
        ("7UP", "7UP"),
        (None, None),
        ("", ""),
    ]
    for raw, want in cases:
        got = format_item_name(raw)
        status = "ok" if got == want else "FAIL"
        print(f"[{status}] {raw!r:24} -> {got!r:26} (want {want!r})")
    print("--- split_digits=True ---")
    for raw in ["PUDDING36CT", "GRMN6PK", "7UP"]:
        print(f"  {raw!r:16} -> {format_item_name(raw, split_digits=True)!r}")
