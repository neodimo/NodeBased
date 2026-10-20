"""The viewport's progressive Render mode, off the interface thread (lane 4, plan "Progress", step H2).

`progressiverender.step_interactive` is one blocking call; the first look takes about half a second and a full-size doubling step
can take several seconds on a heavy scene. Called from the paint handler it froze the window for as long as the pass lasted and
could not be abandoned when the artist moved the camera. `ProgressiveJob` runs the same steps on one worker thread:

* ``submit(request)`` starts the work for a view (the request's ``identity``). A request with the identity already being worked on
  or already settled is ignored; a different identity supersedes the running one at once: its cancel event is set, the trace
  stops at the next GPU pass boundary (`cancellation.Cancelled`), and the worker takes the newest request. Requests that arrive
  within ``settle`` seconds of each other (an orbit or a slider drag sends one per mouse move) collapse into the last, so only the
  view the artist stopped on gets a first look.
* The worker publishes every finished state (the denoised first look, the second preview stage, then the doubling steps) and calls
  ``on_update``; the interface thread asks ``view(identity)`` for the newest state of the view it is showing. A state of another
  identity is never returned, so a superseded picture of an old camera or scene cannot be painted.
* ``phase`` says what the worker is doing for that identity: ``waiting`` (nothing published yet, first look pending), ``tracing``
  (a state is published and the next pass is running), ``settled`` (converged), ``failed`` (``error`` holds the exception) or
  ``idle`` (nobody asked).

Scenes and cameras are immutable values, so the worker reads the request's objects while the interface builds the next ones.
"""
from __future__ import annotations

import functools
import threading
import time
from dataclasses import dataclass, field, replace

from . import progressiverender
from .cancellation import Cancelled

SETTLE_SECONDS = 0.04      # requests this close together collapse into the last one


@dataclass(frozen=True)
class Request:
    """Everything one progressive view needs. ``identity`` distinguishes views: a new one restarts with a fresh first look."""
    identity: object
    scene: object
    camera: object
    width: int
    height: int
    background: tuple
    ambient: float
    moments: object = None
    backend: str = "auto"
    max_bounces: int = progressiverender.MAX_BOUNCES


@dataclass(frozen=True)
class Frame:
    """One published state with when it appeared (seconds since the request was submitted, the display conversion included)."""
    state: object
    since_submit: float
    serial: int = 0
    rgba: object = None           # the 8-bit sRGB RGBA the viewport paints (`display_rgba8`), converted on the worker thread


class ProgressiveJob:
    def __init__(self, on_update=None, settle=SETTLE_SECONDS, step=None, denoise_passes=True, convert=True, keep_images=False):
        self.on_update = on_update
        self.settle = float(settle)
        self.keep_images = keep_images   # False: `frames` keeps the numbers of older pictures, not their 1080p arrays
        self.convert = convert           # convert each picture for display on the worker, keeping the interface thread's paint cheap
        # the full-size doubling passes are filtered too, so a refinement never looks noisier than the preview before it
        self._step = step or functools.partial(progressiverender.step_interactive, denoise_passes=denoise_passes)
        self._cond = threading.Condition()
        self._pending = None            # (request, submitted_at) not yet taken by the worker
        self._current = None            # the request being traced
        self._cancel = threading.Event()
        self._frame = None              # newest Frame, of `_frame_identity`
        self._frame_identity = None
        self._identity = None           # what the interface last asked for
        self._phase = "idle"
        self._error = None
        self._closed = False
        self._busy = False
        self.frames = []                # every Frame published for the identity asked for last (the benchmark and tests read it)
        self._serial = 0
        self._thread = None
        self._pass_started = time.perf_counter()
        self._cancel_asked = None       # when the running pass was told to stop
        self.cancel_latencies = []      # seconds from a superseding request to the end of the pass it cancelled (benchmarks read it)

    # --- the interface thread -----------------------------------------------------------------------------------------

    def submit(self, request):
        """Ask for `request`. Returns True when it superseded earlier work (or started new work), False when its identity is
        already the one being worked on, settled or failed (a failure is not retried for the same view)."""
        with self._cond:
            if self._closed:
                return False
            if request.identity == self._identity:
                return False
            self._identity = request.identity
            self._phase = "waiting"
            self._error = None
            self._frame = None
            self._frame_identity = None
            self.frames = []
            self._pass_started = time.perf_counter()
            self._pending = (request, self._pass_started)
            if self._busy and not self._cancel.is_set():
                self._cancel_asked = time.perf_counter()
            self._cancel.set()             # obsolete work stops at its next pass boundary
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name="nodebased-progressive", daemon=True)
                self._thread.start()
            self._cond.notify_all()
            return True

    def view(self, identity):
        """(state, phase) for `identity`: the newest published state (None before the first look) and what the worker is
        doing. A request nobody made, or one superseded since, reads (None, "idle")."""
        with self._cond:
            if identity != self._identity:
                return None, "idle"
            state = self._frame.state if self._frame is not None and self._frame_identity == identity else None
            return state, self._phase

    def display(self, identity, state):
        """The converted picture of `state` (as returned by `view` for `identity`), or None when not converted / not current."""
        with self._cond:
            frame = self._frame
            return frame.rgba if frame is not None and frame.state is state and self._frame_identity == identity else None

    @property
    def error(self):
        with self._cond:
            return self._error

    def working_seconds(self):
        """Seconds the pass in flight has been running (since the last published picture, or the start of the first look)."""
        with self._cond:
            return max(0.0, time.perf_counter() - self._pass_started) if self._busy or self._pending is not None else 0.0

    @property
    def busy(self):
        with self._cond:
            return self._pending is not None or self._busy

    def cancel(self):
        """Stop the work in hand and forget the identity: the next `submit` of any view starts again."""
        with self._cond:
            self._pending = None
            self._identity = None
            self._phase = "idle"
            self._frame = None
            self._frame_identity = None
            self._cancel.set()
            self._cond.notify_all()

    def wait(self, predicate=None, timeout=60.0):
        """Block until the worker is idle (nothing pending and no pass running) and, when given, `predicate(state, phase)`
        holds for the current view. For tests and tools; the interface never waits. Returns True on success."""
        end = time.perf_counter() + timeout
        with self._cond:
            while True:
                state = self._frame.state if self._frame is not None else None
                settled = self._pending is None and not self._busy
                if predicate is None:
                    if settled:
                        return True
                elif predicate(state, self._phase):
                    return True
                elif settled and self._phase in ("settled", "failed", "idle"):
                    return False
                left = end - time.perf_counter()
                if left <= 0:
                    return False
                self._cond.wait(min(left, 0.05))

    def close(self):
        """Cancel everything and end the worker thread."""
        with self._cond:
            self._closed = True
            self._pending = None
            self._cancel.set()
            self._cond.notify_all()
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=10.0)

    # --- the worker thread --------------------------------------------------------------------------------------------

    def _take(self):
        """The newest pending request once none newer has arrived for `settle` seconds, or None when closed."""
        with self._cond:
            while True:
                if self._closed:
                    return None
                if self._pending is None:
                    self._cond.wait(0.25)
                    continue
                taken = self._pending
                if self.settle > 0:
                    self._cond.wait(self.settle)
                    if self._closed:
                        return None
                    if self._pending is not taken:     # a newer request arrived while settling
                        continue
                self._pending = None
                self._cancel = threading.Event()
                self._busy = True
                self._pass_started = taken[1]       # the first look is timed from the request, settle included
                return taken

    def _run(self):
        while True:
            taken = self._take()
            if taken is None:
                return
            request, submitted = taken
            try:
                self._work(request, submitted)
            finally:
                with self._cond:
                    if self._cancel_asked is not None:
                        self.cancel_latencies.append(time.perf_counter() - self._cancel_asked)
                        self._cancel_asked = None
                    self._busy = False
                    self._cond.notify_all()

    def _work(self, request, submitted):
        cancel = self._cancel
        state = None
        while not cancel.is_set():
            try:
                state = self._step(state, request.scene, request.camera, request.width, request.height, request.background,
                                   request.ambient, request.identity, request.backend, request.max_bounces, request.moments,
                                   cancel)
            except Cancelled:
                return
            except Exception as error:     # a scene the path tracer refuses: the interface falls back and says why
                with self._cond:
                    if self._identity == request.identity:
                        self._phase, self._error = "failed", error
                    self._cond.notify_all()
                self._notify()
                return
            done = progressiverender.converged(state, request.width, request.height)
            try:
                rgba = (progressiverender.display_rgba8(state.image, cancel)
                        if self.convert and state.image is not None else None)
            except Cancelled:
                return
            with self._cond:
                if cancel.is_set() or self._identity != request.identity:
                    return          # superseded while the pass finished: its picture belongs to a view nobody shows now
                self._serial += 1
                frame = Frame(state, time.perf_counter() - submitted, self._serial, rgba)
                self._frame, self._frame_identity = frame, request.identity
                if self.frames and not self.keep_images:
                    last = self.frames[-1]
                    self.frames[-1] = replace(last, state=replace(last.state, image=None, guides=None), rgba=None)
                self.frames.append(frame)
                self._pass_started = time.perf_counter()
                self._phase = "settled" if done else "tracing"
                self._cond.notify_all()
            self._notify()
            if done:
                return

    def _notify(self):
        if self.on_update is not None:
            try:
                self.on_update()
            except Exception:       # a closing widget must not kill the worker
                pass
