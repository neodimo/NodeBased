"""Step-by-step previews of a few-step SDXL edit (spike A, docs/SPIKE_A_FEWSTEP.md). Qt-free; the widget is `steppreviewview`.

A `PreviewSession` keeps one model process warm (`sdxl_controlnet.py serve`, the SDXL-Lightning 4-step path) and takes
one edit at a time: a text brief, a seed and the frames to make. Each denoising step comes back as a provisional picture
(the scheduler's clean-image estimate, decoded by a separate copy of the VAE), the finished frame as the final.

* One active sampling task, one latest pending request: an edit that arrives while another samples stops it after its
  current step, and a waiting edit is replaced by a newer one (counted as coalesced, never run).
* Results go through a `LatestSlot`, a one-place channel where the newest picture replaces an unread older one, so a slow
  viewer never builds a queue.
* Every picture carries its edit id, revision, frame, stage and step. A picture of a revision older than the newest
  submitted one is dropped at the door (`stale`), and the viewer checks again before it paints.
* All times are `time.monotonic()` seconds (the model process shares the clock on Linux), so submit, model, receive and
  paint times subtract directly.
"""
from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class PreviewEvent:
    kind: str                 # "preview" (provisional) or "final"
    edit: int                 # the id of the request
    revision: int
    frame: int
    stage: str                # "text2img" (all steps from noise), "fill" or "guided" (image-to-image tail)
    step: int                 # 1-based; a final carries the last step
    of: int
    array: np.ndarray         # uint8 (height, width, 3), display sRGB
    done: float               # model side: picture written and announced
    received: float           # this side: picture read back
    decode_seconds: float = 0.0
    width: int = 0
    height: int = 0

    @property
    def provisional(self):
        return self.kind != "final"


class LatestSlot:
    """A bounded newest-result channel: one place, the newest item replaces an unread older one."""

    def __init__(self):
        self._lock = threading.Lock()
        self._item = None
        self.replaced = 0   # unread items overwritten by a newer one
        self.peak_depth = 0

    def put(self, item):
        with self._lock:
            if self._item is not None:
                self.replaced += 1
            self._item = item
            self.peak_depth = max(self.peak_depth, 1)

    def take(self):
        with self._lock:
            item, self._item = self._item, None
            return item

    def clear(self):
        with self._lock:
            self._item = None


@dataclass
class SessionLog:
    submitted: dict = field(default_factory=dict)   # revision -> {"t": ..., "frame": ...}
    events: list = field(default_factory=list)       # one dict per picture received
    stale: int = 0
    coalesced: int = 0
    cancelled: list = field(default_factory=list)
    done: list = field(default_factory=list)
    pending_peak: int = 0


class PreviewSession:
    """One warm model process and its edit protocol (see `sdxl_controlnet._serve_main`)."""

    def __init__(self, controls, settings, workdir, *, python=None, on_available=None, env=None, command=None):
        from .providers import sdxl_controlnet as provider
        self.provider = provider
        self.settings = provider.resolve_settings(settings)
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.rows, self.width, self.height = provider.write_control_files(controls, self.workdir)
        self.preview_dir = self.workdir / "previews"
        self.preview_dir.mkdir(exist_ok=True)
        self.slot = LatestSlot()
        self.log = SessionLog()
        self.on_available = on_available     # called on the reader thread after a picture is in the slot
        self.python = python or provider.provider_python()
        self.command = command   # tests: another program that speaks the same line protocol
        self.process_env = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=":4096:8", HF_HUB_OFFLINE="1",
                                TRANSFORMERS_OFFLINE="1", PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false",
                                **(env or {}))
        self.revision = 0
        self._last_step = {}
        self.ready_info = None
        self._ready = threading.Event()
        self._lock = threading.Lock()
        self._finished = threading.Event()
        self.process = None
        self.final_hook = None   # optional callable(event) run on the reader thread after a final reached the slot

    def start(self, timeout=600):
        self.started = time.monotonic()
        self.process = subprocess.Popen(
            self.command or [self.python, "-I", str(Path(self.provider.__file__).resolve()), "serve", json.dumps(self.settings)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=self.process_env)
        self._tail = []
        threading.Thread(target=self._read, daemon=True).start()
        if not self._ready.wait(timeout):
            self.close()
            raise RuntimeError("the preview process did not become ready: " + " | ".join(self._tail[-6:]))
        self.ready_info["cold_start_seconds"] = round(time.monotonic() - self.started, 2)
        return self.ready_info

    def submit(self, text, *, seed=7, frames=None, frame_numbers=None):
        """Ask for an edit; returns its revision. `frames` are indexes into the controls, or `frame_numbers`."""
        wanted = list(frames if frames is not None else range(len(self.rows)))
        if frame_numbers is not None:
            numbers = {row["frame"]: i for i, row in enumerate(self.rows)}
            wanted = [numbers[n] for n in frame_numbers]
        with self._lock:
            self.revision += 1
            revision = self.revision
            self.log.submitted[revision] = {"t": time.monotonic(), "frames": [self.rows[i]["frame"] for i in wanted],
                                            "text": text}
            rows = [dict(self.rows[i], result=str(self.workdir / f"r{revision:04d}_{self.rows[i]['frame']:04d}.result.npy"))
                    for i in wanted]
            job = {"frames": rows, "width": self.width, "height": self.height, "prompt": text, "seed": int(seed),
                   "settings": self.settings, "preview_dir": str(self.preview_dir)}
            path = self.workdir / f"edit_{revision:04d}.json"
            path.write_text(json.dumps(job), encoding="utf-8")
            self.process.stdin.write(json.dumps({"op": "edit", "id": revision, "revision": revision, "job": str(path)}) + "\n")
            self.process.stdin.flush()
            self.log.submitted[revision]["sent"] = time.monotonic()
        return revision

    def wait_idle(self, revision, timeout=300):
        """Wait until the edit `revision` finished, was cancelled or coalesced away."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if (revision in {d["revision"] for d in self.log.done}
                        or revision in {c["revision"] for c in self.log.cancelled}):
                    return True
            time.sleep(0.005)
        return False

    def close(self):
        if self.process is None:
            return
        try:
            self.process.stdin.write(json.dumps({"op": "quit"}) + "\n")
            self.process.stdin.flush()
            self.process.stdin.close()
        except (BrokenPipeError, ValueError):
            pass
        try:
            self.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        self.process = None

    # -- reader thread ---------------------------------------------------------------------------------------
    def _read(self):
        for line in self.process.stdout:
            line = line.rstrip()
            received = time.monotonic()
            word, _, rest = line.partition(" ")
            try:
                payload = json.loads(rest) if rest.startswith("{") else None
            except ValueError:
                payload = None
            if word == "ready" and payload is not None:
                self.ready_info = payload
                self._ready.set()
            elif word in ("preview", "final") and payload is not None:
                self._picture(word, payload, received)
            elif word == "coalesced" and payload is not None:
                with self._lock:
                    self.log.coalesced += 1
                    self.log.cancelled.append({"revision": payload["revision"], "coalesced": True})
            elif word == "cancelled" and payload is not None:
                with self._lock:
                    self.log.cancelled.append(payload)
            elif word == "done" and payload is not None:
                with self._lock:
                    self.log.done.append(payload)
            elif word in ("started",):
                pass
            else:
                self._tail = (self._tail + [line])[-40:]

    def _picture(self, kind, payload, received):
        path = payload["path"]
        array = np.load(path)
        if kind == "preview":
            self._last_step[(payload["revision"], payload["frame"])] = (payload["stage"], payload["step"], payload["of"])
        else:   # a final completes the schedule its last preview announced
            payload = {**payload, **dict(zip(("stage", "step", "of"), self._last_step.get((payload["revision"], payload["frame"]), ("final", 0, 0))))}
        event = PreviewEvent(
            kind=kind, edit=payload["id"], revision=payload["revision"], frame=payload["frame"],
            stage=payload.get("stage", "final"), step=payload.get("step", payload.get("of", 0)),
            of=payload.get("of", 0), array=array, done=payload["done"], received=received,
            decode_seconds=payload.get("decode_seconds", 0.0), width=int(array.shape[1]), height=int(array.shape[0]))
        with self._lock:
            newest = self.revision
            self.log.events.append({"kind": kind, "revision": event.revision, "frame": event.frame, "stage": event.stage,
                                    "step": event.step, "of": event.of, "model_done": event.done,
                                    "received": event.received, "decode_seconds": event.decode_seconds,
                                    "write_seconds": payload.get("write_seconds"), "width": event.width,
                                    "height": event.height, "stale": event.revision < newest})
            if event.revision < newest:
                self.log.stale += 1
                return
        self.slot.put(event)
        if self.on_available:
            self.on_available()
        if kind == "final" and self.final_hook:
            self.final_hook(event)
