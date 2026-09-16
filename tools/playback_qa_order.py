"""Pure frame-order checks shared by the native-display playback QA harness and tests."""


def is_forward_frame_order(frames, first=1, last=100, wrap_window=10):
    """Return whether ``frames`` advances forward, allowing an end-to-start loop wrap.

    Playback may complete the configured range during a timed QA run, so ``last -> first`` is
    a valid forward transition. A generic backward jump remains a failure: the boundary window
    prevents an out-of-order render such as ``62 -> 14`` from being mistaken for a loop.
    """
    low = int(first)
    high = int(last)
    window = max(0, int(wrap_window))
    for before, after in zip(frames, frames[1:]):
        if after >= before:
            continue
        if before >= high - window and after <= low + window:
            continue
        return False
    return True
