"""Controls whose widget is Nuke's knob type while the document keeps the text it always held.

Panel sweep S3 (2026-11-04). Three knobs stored as text show as the controls Nuke's guide gives them, and write back the same text:

* `point_text` (DirBlur's Target): an x, y pair of number fields over the text "x,y".
* `inverse_text` (CornerPin2D's Invert): a check box over the direction menu's "forward" and "inverse".
* `factor_text` (DirBlur's Quality): a whole-number scaling factor over the menu's "high", "normal" and "low" (1, 2 and 4). Any other
  whole number is stored as its digits, so a factor of 3 or 8 can be typed; the three names keep writing as they did, and a document
  that holds one of them loads and renders as before.

Pure Python, so the panel, the kernel's reading and the tests share one conversion.
"""
from __future__ import annotations

# The widget kind each stored-as-text control is laid out as (knobs.KnobGroup.kind).
STORED_KINDS = {"point_text": "xy", "inverse_text": "bool", "factor_text": "float_slider"}

FACTOR_NAMES = {"high": 1, "normal": 2, "low": 4}
FACTOR_LIMITS = (1, 64)
DIRECTIONS = ("forward", "inverse")


def point_of(text):
    """(x, y) read from "x,y"; (0.0, 0.0) when the text holds no pair."""
    try:
        values = [float(part) for part in str(text).replace(";", ",").split(",")]
    except (TypeError, ValueError):
        return (0.0, 0.0)
    return (values[0], values[1]) if len(values) == 2 else (0.0, 0.0)


def point_text(x, y):
    return f"{float(x):g},{float(y):g}"


def inverse_of(value):
    return str(value) == DIRECTIONS[1]


def direction_text(checked):
    return DIRECTIONS[1 if checked else 0]


def factor_of(value):
    """The whole-number scaling factor a Quality value stands for: the three names, or digits. Anything else reads as 1 (the highest quality)."""
    text = str(value)
    if text in FACTOR_NAMES:
        return FACTOR_NAMES[text]
    try:
        factor = int(round(float(text)))
    except (TypeError, ValueError, OverflowError):
        return 1
    return min(max(factor, FACTOR_LIMITS[0]), FACTOR_LIMITS[1])


def factor_text(number):
    """The text a typed scaling factor is stored as: the name for 1, 2 and 4, else its digits."""
    factor = min(max(int(round(float(number))), FACTOR_LIMITS[0]), FACTOR_LIMITS[1])
    for name, size in FACTOR_NAMES.items():
        if size == factor:
            return name
    return str(factor)


def is_factor_digits(value):
    """Is `value` the digits of a whole scaling factor above 1 that no name stands for (DirBlur's Quality takes these besides its three names)?"""
    return isinstance(value, str) and value.isascii() and value.isdigit() and FACTOR_LIMITS[0] < int(value) <= FACTOR_LIMITS[1] \
        and value == str(int(value)) and int(value) not in FACTOR_NAMES.values()
