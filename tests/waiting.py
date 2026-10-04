"""Non-blocking Qt test waits that let worker threads run."""

import time

# Every Qt window test imports this module; settings isolation rides along (see tests/isolation.py).
import tests.isolation  # noqa: F401


def wait_until(condition, timeout=30.0):
    """Process UI events until condition succeeds or timeout seconds elapse."""
    from PySide6.QtWidgets import QApplication

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app = QApplication.instance()
        if app is not None:
            app.processEvents()
        if condition():
            return True
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(min(0.01, remaining))
    return False


def pause(ms):
    """Wait a fixed number of milliseconds while continuing to service Qt events."""
    from PySide6.QtWidgets import QApplication

    deadline = time.monotonic() + max(0.0, ms / 1000.0)
    while True:
        app = QApplication.instance()
        if app is not None:
            app.processEvents()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(0.01, remaining))
