"""Small deterministic queue tasks shared by local and subprocess worker tests."""
import threading
import time


def make(options, inputs, progress, cancelled):
    if options.get("sleep"):
        until = time.monotonic() + options["sleep"]
        while time.monotonic() < until:
            if cancelled.is_set(): raise RuntimeError("cancelled by fixture")
            time.sleep(.01)
    if options.get("fail"):
        raise ValueError(options.get("reason", "fixture failure"))
    progress(.5, "halfway")
    return (options.get("label", "artifact") + ":" + ",".join(inputs)).encode()

