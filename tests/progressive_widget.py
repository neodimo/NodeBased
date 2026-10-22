"""Helpers for tests of the viewport's Render mode, which traces on a worker thread (`progressivejob`): a paint submits the
view and returns at once, so a test paints, waits for the worker to publish what it asks about, and paints again."""


import os

# tools/run_tests.py --jobs N sets NODEBASED_TEST_TIMEOUT_SCALE=N: with N interpreters sharing the cores a
# trace gets about 1/N of the CPU, and CI run 39786527823 showed the 180 s settle wait expiring in the
# path-tracer convergence tests ("the viewport did not settle") that settle in time when run alone.
TIMEOUT_SCALE = float(os.environ.get("NODEBASED_TEST_TIMEOUT_SCALE") or 1)


def settled(state, phase):
    return phase == "settled"


def first_picture(state, phase):
    return state is not None


def paint_until(widget, predicate=settled, timeout=180.0):
    """Paint `widget` (which submits its view), wait until `predicate(state, phase)` holds for the worker's newest picture of
    that view, paint again so the widget draws it. Returns the wait's result; a test asserts it."""
    widget.grab()
    reached = widget._progressive_job.wait(predicate, timeout * TIMEOUT_SCALE)
    widget.grab()
    return reached
