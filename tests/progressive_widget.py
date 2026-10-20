"""Helpers for tests of the viewport's Render mode, which traces on a worker thread (`progressivejob`): a paint submits the
view and returns at once, so a test paints, waits for the worker to publish what it asks about, and paints again."""


def settled(state, phase):
    return phase == "settled"


def first_picture(state, phase):
    return state is not None


def paint_until(widget, predicate=settled, timeout=180.0):
    """Paint `widget` (which submits its view), wait until `predicate(state, phase)` holds for the worker's newest picture of
    that view, paint again so the widget draws it. Returns the wait's result; a test asserts it."""
    widget.grab()
    reached = widget._progressive_job.wait(predicate, timeout)
    widget.grab()
    return reached
