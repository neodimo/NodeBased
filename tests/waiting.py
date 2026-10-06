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


def settle_layout(window, widget, timeout=5.0, turns=5):
    """Wait until a widget's size holds still inside a window's panel layout.

    A widget in the window's panel layout is resized again once the default split is applied (a
    zero-delay timer after show). On a slow runner that lands after screen points computed right
    after show, so a click or drag aims at the old geometry (10/3 picking, 10/6 gizmo and target
    handles). This processes events until the window reports no default split pending and the
    widget's size has been identical for `turns` event-loop turns. Read screen positions from the
    widget's current size after calling it, at the point of use. Returns True once settled."""
    from PySide6.QtWidgets import QApplication

    deadline = time.monotonic() + timeout
    last, still = None, 0
    while time.monotonic() < deadline:
        app = QApplication.instance()
        if app is not None:
            app.processEvents()
        size = (widget.width(), widget.height())
        pending = bool(getattr(window, "_default_split_pending", False))
        still = still + 1 if size == last and not pending else 0
        last = size
        if still >= turns:
            return True
        time.sleep(0.02)
    return False
