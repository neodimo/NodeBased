"""Which graphics card the 2D island runs on when Settings says Automatic: the one that measures fastest, not the one that ranks first.

`gpu3d.rank_adapters` puts a discrete card ahead of an integrated one, which is right for 3D work and wrong for the island on a machine
whose discrete card sits behind a slow link. The island uploads about 75 MB of sources and downloads 33 MB for every frame, so what it
pays is the card's transfer rate: the RTX 3080 Ti of the development machine, in a USB4 enclosure, moves 1 GB/s up and 3 GB/s down, and
the island on it runs a quarter slower than the CPU nodes; the integrated Radeon, which shares memory with the processor, moves 20 GB/s
and doubles them (docs/PERFORMANCE.md, "Lane 8 step notes").

The probe is one island frame in the pattern of the realistic comp, 1920 x 1080 in float: a plate, a background and a matte uploaded,
Grade then Merge run, the result downloaded, repeated five times after a warming run, taking the median of the whole run. The same two
nodes on the CPU kernels are timed once beside it. A card must beat the CPU nodes on that chain to be chosen; of the cards that do,
the fastest wins. When none does, the CPU nodes stay in use and Settings says so.

* The result is kept per adapter identity (vendor, device, driver version, backend) in the machine's preferences, with one CPU
  figure per processor count; a new card, a new driver or a new probe version measures again, and nothing else does.
* Nothing is measured while a clip plays, or while another job holds `/tmp/nb-gpu.lock`: the island waits (`Deferred`) and the window
  asks again when playback stops and every 20 seconds after a busy lock.
* An adapter named in Settings, or by `NODEBASED_ADAPTER`, is used as named and is never measured or overridden.
* `NODEBASED_ISLAND_PROBE=0` skips the measurement and keeps the old ranking (discrete first), for a benchmark that wants it.

`resolve` is deterministic given the stored measurements: `decide` sorts by measured time, then by kind, then by name.
"""
from __future__ import annotations

import contextlib
import json
import os
import platform
import statistics
import threading
import time
from dataclasses import dataclass, field

PROBE_VERSION = 1
PROBE_SIZE = (1920, 1080)
PROBE_RUNS = 5
CPU_RUNS = 5
SETTING = "playback/island_probe"
PROBE_ENV = "NODEBASED_ISLAND_PROBE"
PASS_MARGIN = 1.0               # a card is chosen only when its chain takes less than this fraction of the CPU nodes' time
CHOICES = ("discrete", "integrated")        # the adapter kinds `gpu3d._state` can open besides the software one
KIND_RANK = {"discrete": 0, "integrated": 1}
RETRY_SECONDS = 20


class Deferred(Exception):
    """The measurement has to wait (a clip is playing, or another job holds the card); the CPU nodes stay in use meanwhile."""

    def __init__(self, reason, why):
        super().__init__(reason)
        self.why = why              # "playback" or "lock"


class NoCard(Exception):
    """No graphics card is available, or none was faster than the CPU nodes."""


@dataclass
class Candidate:
    choice: str                     # what `gpu3d._state` opens: "discrete" or "integrated"
    kind: str
    name: str
    identity: str


@dataclass
class Choice:
    choice: str                     # what `GPU2D.open` opens: "discrete", "integrated" or "default" (an adapter named in Settings)
    name: str
    kind: str
    reason: str
    source: str                     # "measured", "stored" or "settings"
    scores: list = field(default_factory=list)      # per candidate: name, kind, ms (or the error)
    cpu_ms: float = 0.0


# -- identities and the stored measurements ---------------------------------------------------------


def adapter_identity(info):
    """What makes two measurements comparable: the same card on the same driver through the same backend."""
    return "|".join(str(info.get(key, "")).strip() for key in
                    ("vendor", "device", "description", "driver", "driver_info", "adapter_type", "backend_type"))


def cpu_identity(cpus=None):
    from . import sourcepool
    count = sourcepool.available_cpus() if cpus is None else int(cpus)
    return f"cpu|{count}|{platform.machine()}|{platform.processor() or platform.system()}"


FILE_ENV = "NODEBASED_ISLAND_PROBE_FILE"


class SettingsStore:
    """The measurements, as one JSON text under `playback/island_probe` in the machine's preferences, or in the file
    `NODEBASED_ISLAND_PROBE_FILE` names (a benchmark that must not touch the preferences)."""

    def load(self):
        path = os.environ.get(FILE_ENV)
        if path:
            try:
                with open(path) as handle:
                    data = json.load(handle)
            except (OSError, ValueError):
                return {}
            return data if isinstance(data, dict) else {}
        try:
            from PySide6.QtCore import QSettings
            text = QSettings("NodeBased", "NodeBased").value(SETTING, "")
            data = json.loads(str(text)) if text else {}
        except Exception:
            return {}
        return data if isinstance(data, dict) else {}

    def save(self, records):
        path = os.environ.get(FILE_ENV)
        if path:
            try:
                with open(path, "w") as handle:
                    json.dump(records, handle, sort_keys=True, indent=1)
            except OSError:
                pass
            return
        try:
            from PySide6.QtCore import QSettings
            store = QSettings("NodeBased", "NodeBased")
            store.setValue(SETTING, json.dumps(records, sort_keys=True))
            store.sync()
        except Exception:
            pass


class MemoryStore:
    """The same, in memory (tests, benchmarks that must not touch the machine's preferences)."""

    def __init__(self, records=None):
        self.records = json.loads(json.dumps(records or {}))

    def load(self):
        return json.loads(json.dumps(self.records))

    def save(self, records):
        self.records = json.loads(json.dumps(records))


def record_valid(record):
    return (isinstance(record, dict) and record.get("version") == PROBE_VERSION
            and list(record.get("size") or ()) == list(PROBE_SIZE)
            and isinstance(record.get("ms"), (int, float)) and record["ms"] > 0)


# -- the candidates ---------------------------------------------------------------------------------


def candidates(raw=None):
    """The cards Automatic may pick from: the first adapter of each kind, the one `gpu3d._state(kind)` opens, discrete first."""
    from . import gpu3d
    if raw is None:
        raw = gpu3d._raw_adapters()
    found = []
    for choice in CHOICES:
        wanted = {"discrete": "discretegpu", "integrated": "integratedgpu"}[choice]
        adapter = next((a for a in raw if gpu3d._kind_of(a.info) == wanted), None)
        if adapter is not None:
            info = dict(adapter.info)
            found.append(Candidate(choice, choice, gpu3d._name_of(info), adapter_identity(info)))
    return found


# -- the probe --------------------------------------------------------------------------------------


def _probe_inputs():
    import numpy as np
    from .raster import Raster
    from .tiers import Region
    width, height = PROBE_SIZE
    display = Region(0, 0, width, height)
    rng = np.random.default_rng(3)

    def picture():
        pixels = rng.random((height, width, 4), dtype=np.float32)
        pixels[..., 3] = 1.0
        return Raster(pixels, Region(0, 0, width, height), display)
    matte = np.zeros((height, width, 4), np.float32)
    matte[..., 3] = rng.random((height, width), dtype=np.float32)
    return picture(), picture(), Raster(matte, Region(0, 0, width, height), display)


def _probe_params():
    from .core import SPECS
    grade, merge = dict(SPECS["Grade"]["params"]), dict(SPECS["Merge"]["params"])
    grade["multiply"] = 1.1
    merge["operation"] = "over"
    return grade, merge


def measure_card(candidate, budget_bytes, clock=time.perf_counter):
    """One island frame on the card, `PROBE_RUNS` times after a warming run: the record that is kept."""
    from . import gpu2d
    backend = gpu2d.GPU2D.open(candidate.choice, budget_bytes)
    if backend is None:
        raise RuntimeError(gpu2d.GPU2D.open_error or "the card did not open")
    try:
        plate, background, matte = _probe_inputs()
        grade, merge = _probe_params()
        backend.warm()

        def frame():
            node = backend.stage("Merge", merge, [backend.stage("Grade", grade, [plate]), background, matte])
            backend.run(node)
        frame()
        before = backend.counters.snapshot()
        times = []
        for _ in range(PROBE_RUNS):
            started = clock()
            frame()
            times.append((clock() - started) * 1000.0)
        spent = gpu2d.Counters.delta(backend.counters.snapshot(), before)
    finally:
        try:
            backend.release_all()
        except Exception:
            pass
    return {"version": PROBE_VERSION, "size": list(PROBE_SIZE), "name": candidate.name, "kind": candidate.kind,
            "ms": round(statistics.median(times), 2), "runs_ms": [round(t, 2) for t in times],
            "upload_ms": round(spent["upload_ms"] / PROBE_RUNS, 2), "kernel_ms": round(spent["gpu_ms"] / PROBE_RUNS, 2),
            "download_ms": round((spent["readback_map_ms"] + spent["readback_host_ms"]) / PROBE_RUNS, 2),
            "taken": time.strftime("%Y-%m-%d %H:%M:%S")}


def measure_cpu(clock=time.perf_counter):
    """The same two nodes on the CPU kernels."""
    from .imaging import Evaluator
    plate, background, matte = _probe_inputs()
    grade, merge = _probe_params()

    def frame():
        graded = Evaluator._windowed_kernel("Grade", grade, [plate], 1, None)
        Evaluator._windowed_kernel("Merge", merge, [graded, background, matte], 1, None)
    frame()
    times = []
    for _ in range(CPU_RUNS):
        started = clock()
        frame()
        times.append((clock() - started) * 1000.0)
    return {"version": PROBE_VERSION, "size": list(PROBE_SIZE), "ms": round(statistics.median(times), 2),
            "runs_ms": [round(t, 2) for t in times], "taken": time.strftime("%Y-%m-%d %H:%M:%S")}


# -- the decision -----------------------------------------------------------------------------------


def decide(found, records, cpu_id=None):
    """The card Automatic picks given the stored measurements, or `NoCard`. Deterministic: the fastest chain among the cards that
    beat the CPU nodes, ties broken by kind (discrete first) and name."""
    cpu_id = cpu_id or cpu_identity()
    cpu = records.get(cpu_id)
    if not record_valid(cpu):
        raise NoCard("the CPU nodes have not been timed")
    cpu_ms = float(cpu["ms"])
    scores = []
    for candidate in found:
        record = records.get(candidate.identity)
        if not record_valid(record):
            scores.append({"name": candidate.name, "kind": candidate.kind, "choice": candidate.choice, "ms": None,
                           "error": (record or {}).get("error", "not measured")})
        else:
            scores.append({"name": candidate.name, "kind": candidate.kind, "choice": candidate.choice, "ms": float(record["ms"]),
                           "upload_ms": record.get("upload_ms"), "download_ms": record.get("download_ms")})
    passing = [s for s in scores if s["ms"] is not None and s["ms"] < cpu_ms * PASS_MARGIN]
    if not passing:
        if not found:
            raise NoCard("no graphics card is available, so the CPU nodes stay in use")
        figures = ", ".join(f"{s['name']} {s['ms']:.0f} ms" if s["ms"] is not None else f"{s['name']} {s['error']}" for s in scores)
        raise NoCard(f"no graphics card was faster than the CPU nodes in the measurement ({figures}; CPU nodes {cpu_ms:.0f} ms)")
    best = min(passing, key=lambda s: (s["ms"], KIND_RANK.get(s["kind"], 9), s["name"]))
    others = [s for s in scores if s is not best]
    reason = (f"Automatic: measured {best['ms']:.0f} ms for a frame's upload, kernels and download, against {cpu_ms:.0f} ms on the CPU nodes"
              + "".join(f"; {s['name']} took {s['ms']:.0f} ms" if s["ms"] is not None else f"; {s['name']} could not be measured"
                        for s in others))
    return Choice(best["choice"], best["name"], best["kind"], reason, "stored", scores, cpu_ms)


# -- playback and lock gates ------------------------------------------------------------------------

_playback_probe = lambda: False
_lock = threading.Lock()


def set_playback_probe(function):
    """The window tells this module whether a clip is playing (called from the preview worker: a plain attribute read)."""
    global _playback_probe
    _playback_probe = function or (lambda: False)


def probing_enabled(environ=None):
    return (os.environ if environ is None else environ).get(PROBE_ENV, "1").strip().lower() not in ("0", "false", "no", "off")


def resolve(budget_bytes, *, preference=None, store=None, raw=None, measure=None, measure_cpu_fn=None, playing=None, hold_lock=None):
    """The adapter the island opens on: a `Choice`. Raises `Deferred` while a measurement cannot run and `NoCard` when the CPU nodes
    should stay in use. Every dependency can be replaced, which is how the tests drive it without a card."""
    from . import gpu3d, gpulock
    preference = gpu3d.adapter_preference() if preference is None else preference
    wanted = (preference or "").strip().lower()
    if wanted and wanted != "automatic":
        info = gpu3d.adapter_choice(preference=preference)
        return Choice("default", info["name"], info["type"], info["reason"], "settings")
    found = candidates(raw)
    if not found:
        raise NoCard("no graphics card is available, so the CPU nodes stay in use")
    if not probing_enabled():
        first = found[0]
        return Choice("default", first.name, first.kind,
                      f"Automatic: the discrete card is preferred ({PROBE_ENV}=0 skips the measurement)", "settings")
    store = store if store is not None else SettingsStore()
    measure = measure or measure_card
    measure_cpu_fn = measure_cpu_fn or measure_cpu
    cpu_id = cpu_identity()
    failures = {}
    with _lock:
        records = store.load()
        missing = [c for c in found if not record_valid(records.get(c.identity))]
        cpu_missing = not record_valid(records.get(cpu_id))
        if missing or cpu_missing:
            if (playing or _playback_probe)():
                raise Deferred("Automatic measures the graphics cards when playback stops", "playback")
            with contextlib.ExitStack() as stack:
                try:
                    stack.enter_context(hold_lock() if hold_lock else gpulock.gpu_job(timeout=0))
                except TimeoutError:
                    raise Deferred("Automatic measures the graphics cards when the other graphics job has finished", "lock") from None
                if cpu_missing:
                    records[cpu_id] = measure_cpu_fn()
                for candidate in missing:
                    try:
                        records[candidate.identity] = measure(candidate, budget_bytes)
                    except Exception as exc:     # a card that cannot run the probe is out of this decision and is tried again next time
                        failures[candidate.identity] = f"{type(exc).__name__}: {exc}"
                store.save(records)
    view = dict(records)
    for identity, message in failures.items():
        view[identity] = {"error": message}
    chosen = decide(found, view, cpu_id)
    chosen.source = "measured" if (missing or cpu_missing) else "stored"
    return chosen
