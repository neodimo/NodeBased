"""Nuke's frame range text, shared by Profile's `frame_range` and the Viewer's.

`first-last`, `first-lastxstep` (every step-th frame; `x`, `X` and `*` all work), a single frame, or several of these separated by
commas or spaces: `1-10`, `1-100x5`, `3`, `1-4, 8, 12-20x2`. A reversed range (`10-1`) counts down. Empty text means "the default",
which the caller supplies (the project's range), and text that cannot be read is an error with a message the panel can show.
"""
from __future__ import annotations

import re

_PART = re.compile(r"^(-?\d+)(?:\s*-\s*(-?\d+))?(?:\s*[xX*]\s*(\d+))?$")
MAX_FRAMES = 1_000_000


def parse(text, default=None):
    """The frames `text` names, in the order written, duplicates dropped. Empty text gives `default` (a list) or [].

    Raises ValueError for text that is not a range."""
    text = str(text or "").strip()
    if not text:
        return list(default or [])
    frames = []
    for part in re.split(r"[,;]\s*|\s+(?![xX*]|-\d)", text):
        part = part.strip()
        if not part:
            continue
        match = _PART.match(part)
        if match is None:
            raise ValueError(f"'{part}' is not a frame range; write 1-10, 1-100x5 or 3")
        first = int(match.group(1))
        last = int(match.group(2)) if match.group(2) is not None else first
        step = int(match.group(3)) if match.group(3) is not None else 1
        if step < 1:
            raise ValueError("A frame range's step must be 1 or more")
        sign = 1 if last >= first else -1
        if abs(last - first) // step + 1 > MAX_FRAMES:
            raise ValueError("That frame range names too many frames")
        frames.extend(range(first, last + sign, sign * step))
    return list(dict.fromkeys(frames))


def is_valid(text):
    try:
        parse(text)
    except ValueError:
        return False
    return True


def describe(frames):
    """A short text for `frames`: `1-10`, `1-100x5` or a comma list."""
    frames = list(frames)
    if not frames:
        return ""
    if len(frames) == 1:
        return str(frames[0])
    step = frames[1] - frames[0]
    if step and all(b - a == step for a, b in zip(frames, frames[1:])):
        return f"{frames[0]}-{frames[-1]}" + (f"x{abs(step)}" if abs(step) != 1 else "")
    return ", ".join(str(frame) for frame in frames)
