"""Playback settings kept per machine: the GPU nodes and the number of source and decode workers.

Each setting has one default, here, and flipping a default is a one-line change to this file
(`docs/PERFORMANCE.md`, "Lane 8 step notes", holds the measurements behind each choice):

* `DEFAULT_GPU_NODES` -- whether Grade, Transform, Blur and Merge run on the graphics card (the resident
  island of `gpu2d.py`) instead of the CPU kernels. Off: every node on the CPU kernels, as before.
* `DEFAULT_DECODE_WORKERS` -- threads that decode the Read frames ahead of the playhead
  (`decodepool.DecodeAheadPool`). `None` is automatic: one thread, none at four CPUs or fewer.
* `DEFAULT_SOURCE_WORKERS` -- threads that compute the independent sources of one frame side by side
  (`sourcepool.SourcePool`). `None` is automatic: none at three CPUs or fewer, one at four, up to three beyond.

The environment overrides the benchmarks use win over the saved choice (`NODEBASED_GPU_NODES=0|1`,
`NODEBASED_DECODE_AHEAD_WORKERS=N`, `NODEBASED_SOURCE_WORKERS=N`), so a measurement never depends on what
a Settings dialog saved on the machine.
"""
from __future__ import annotations

import os

DEFAULT_GPU_NODES = False
DEFAULT_DECODE_WORKERS = None
DEFAULT_SOURCE_WORKERS = None

MAX_DECODE_WORKERS = 4
MAX_SOURCE_WORKERS = 3
GPU_NODES_ENV = "NODEBASED_GPU_NODES"
DECODE_WORKERS_ENV = "NODEBASED_DECODE_AHEAD_WORKERS"
SOURCE_WORKERS_ENV = "NODEBASED_SOURCE_WORKERS"

_TRUE = ("1", "true", "yes", "on")
_FALSE = ("0", "false", "no", "off")


def parse_flag(value, default=False):
    """A saved boolean (QSettings hands back strings from some backends)."""
    if value is None:
        return bool(default)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in _TRUE:
            return True
        if text in _FALSE:
            return False
        return bool(default)
    return bool(value)


def parse_workers(value, maximum):
    """A saved worker count: ``None`` (automatic) or an integer from 0 to ``maximum``."""
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("", "auto", "automatic", "none"):
            return None
        try:
            value = int(text)
        except ValueError:
            return None
    try:
        return max(0, min(int(maximum), int(value)))
    except (TypeError, ValueError):
        return None


def env_flag(name, environ=None):
    """``True``/``False`` when the variable says so, ``None`` when it is unset or unreadable."""
    raw = (os.environ if environ is None else environ).get(name)
    if raw is None:
        return None
    text = raw.strip().lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    return None


def env_workers(name, maximum, environ=None):
    raw = (os.environ if environ is None else environ).get(name)
    if not raw:
        return None
    try:
        return max(0, min(int(maximum), int(raw)))
    except ValueError:
        return None


def gpu_nodes_wanted(preference=None, environ=None):
    """Whether the island should run: the environment when it says, else the saved choice, else the default."""
    forced = env_flag(GPU_NODES_ENV, environ)
    if forced is not None:
        return forced
    return DEFAULT_GPU_NODES if preference is None else bool(preference)


def gpu_nodes_forced(environ=None):
    return env_flag(GPU_NODES_ENV, environ) is not None


def automatic_decode_workers(cpus):
    """Today's rule: one decode thread, none at four CPUs or fewer (the stand-in lost throughput to it)."""
    from . import decodepool
    return 0 if cpus <= 4 else decodepool.DEFAULT_WORKERS


def automatic_source_workers(cpus):
    from . import sourcepool
    return sourcepool.default_workers(cpus)


def decode_workers(setting=None, cpus=None, environ=None):
    """The decode-ahead thread count to run: the environment, else the saved choice, else the default, else automatic."""
    forced = env_workers(DECODE_WORKERS_ENV, MAX_DECODE_WORKERS, environ)
    if forced is not None:
        return forced
    chosen = parse_workers(setting, MAX_DECODE_WORKERS)
    if chosen is None:
        chosen = parse_workers(DEFAULT_DECODE_WORKERS, MAX_DECODE_WORKERS)
    if chosen is not None:
        return chosen
    from . import sourcepool
    return automatic_decode_workers(sourcepool.available_cpus() if cpus is None else int(cpus))


def source_workers(setting=None, cpus=None, environ=None):
    """The source-pool thread count to run, resolved like `decode_workers`."""
    forced = env_workers(SOURCE_WORKERS_ENV, MAX_SOURCE_WORKERS, environ)
    if forced is not None:
        return forced
    chosen = parse_workers(setting, MAX_SOURCE_WORKERS)
    if chosen is None:
        chosen = parse_workers(DEFAULT_SOURCE_WORKERS, MAX_SOURCE_WORKERS)
    if chosen is not None:
        return chosen
    from . import sourcepool
    return automatic_source_workers(sourcepool.available_cpus() if cpus is None else int(cpus))


def workers_forced(name, environ=None):
    """True when the environment fixes this count, so the Settings choice has no effect."""
    return (os.environ if environ is None else environ).get(name) not in (None, "")
