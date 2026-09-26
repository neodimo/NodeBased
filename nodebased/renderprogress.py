"""Render progress for the desktop app: per-thread routing and the words shown to the user.

`Evaluator.progress` is one attribute on an evaluator that several threads share (the preview
worker, and the GUI thread for exports and sequence writes). `ThreadProgress` is installed there
once and hands each event to whatever handler the calling thread registered, so two renders never
see each other's callback. Its presence is also what marks the app's renders as interactive: the
CPU splat budget does not refuse them (see docs/3D_FOUNDATION.md, "Progress and interactive
renders"). A thread without a handler renders silently.
"""
from __future__ import annotations

import threading
from contextlib import contextmanager

# Before this much of the frame is done the ETA is the reference-rate guess, which measured up to
# 1.7x too long on a real capture; say "roughly" until the extrapolated figure takes over.
ETA_TRUST_FRACTION = 0.05


class ThreadProgress:
    def __init__(self):
        self._local = threading.local()

    def __call__(self, stage, fraction, info):
        handler = getattr(self._local, "handler", None)
        if handler is not None:
            handler(stage, fraction, info)

    @contextmanager
    def handler(self, callback):
        """Route this thread's progress events to `callback` for the duration of the block."""
        previous = getattr(self._local, "handler", None)
        self._local.handler = callback
        try:
            yield
        finally:
            self._local.handler = previous


def format_duration(seconds):
    seconds = max(0.0, float(seconds))
    if seconds < 10:
        return f"{max(1, int(seconds + .5))} s"
    if seconds < 90:
        return f"{5 * int(seconds / 5 + .5)} s"
    minutes, rest = divmod(10 * int(seconds / 10 + .5), 60)
    return f"{minutes} min {rest:02d} s" if rest else f"{minutes} min"


def progress_text(stage, fraction, info):
    """Status-bar text for one `scene3d.render` progress event, or None when there is nothing to show."""
    if stage == "prepare":
        return "Preparing splats…"
    if stage == "delight":
        return f"De-lighting splats  ·  {int(100 * min(1.0, max(0.0, fraction)))}%"
    if stage != "splats":
        return None
    text = f"Rendering splats  ·  {int(100 * min(1.0, max(0.0, fraction)))}%"
    eta = info.get("eta_seconds")
    if eta is not None and fraction >= ETA_TRUST_FRACTION:
        return f"{text}  ·  about {format_duration(eta)} left"
    estimate = info.get("estimate_seconds")
    if estimate is not None and estimate >= 2:
        return f"{text}  ·  roughly {format_duration(estimate)} in total"
    return text
