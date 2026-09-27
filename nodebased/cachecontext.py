"""Resolves a live FluidCache3D/ParticleCache3D node to its `simcache.SimCache` and run key, and a
FluidSolver3D/FluidCache3D node to its evaluated `scene3d.Volume`, for the artist tools panels.

Kept separate from `imaging.py`'s own dispatch of these node kinds so the panels can ask "what is
this node's cache/volume right now" without duplicating the evaluator's run-identity logic; both
sides read the same `.stream` carried on the node's evaluated value.
"""
from __future__ import annotations

VOLUME_KINDS = ("FluidSolver3D", "FluidCache3D")
CACHE_KINDS = ("FluidCache3D", "ParticleCache3D")


def volume_for_node(evaluator, document, key, frame=None):
    """The `scene3d.Volume` this fluid node evaluates to right now, or `None` when the node is
    missing, not a volume-producing kind, disabled, or evaluation fails (a broken upstream must
    not take a panel down)."""
    from .scene3d import Volume

    node = document.get("nodes", {}).get(key)
    if node is None or node["type"] not in VOLUME_KINDS or node["disabled"]:
        return None
    try:
        value = evaluator.evaluate_raster(document, key, frame=frame, typed=True)
    except Exception:
        return None
    return value if isinstance(value, Volume) else None


def cache_context(evaluator, document, key, frame=None):
    """(SimCache, run, start_frame, substeps) for the FluidCache3D/ParticleCache3D node `key`, or
    `None` when it is missing, disabled, not a cache node, or its upstream is not (yet) a solving
    run. `frame` only affects which frame's upstream digest is sampled; the run key it produces is
    the run's identity, the same for every frame."""
    from . import simcache

    node = document.get("nodes", {}).get(key)
    if node is None or node["type"] not in CACHE_KINDS or node["disabled"]:
        return None
    params = node["params"]
    if node["type"] == "FluidCache3D":
        source = node["inputs"].get("volume")
        if source is None:
            return None
        try:
            upstream = evaluator.evaluate_raster(document, source, frame=frame, typed=True)
        except Exception:
            return None
        stream = getattr(upstream, "stream", None)
        if stream is None:
            return None
        sparse = getattr(stream, "backend", None) == "resident_sparse"
        run = simcache.run_key(stream.run, {"out": [params["cache_precision"], params["cache_channels"]]
                                            + (["sparse"] if sparse else [])})
        store = evaluator.sim_store(params["cache_memory_mb"], params["cache_disk_mb"])
        return store, run, stream.start_frame, stream.substeps
    source = node["inputs"].get("particles")
    if source is None:
        return None
    try:
        upstream = evaluator.evaluate_raster(document, source, frame=frame, typed=True)
    except Exception:
        return None
    stream = getattr(upstream, "stream", None)
    if stream is None:
        return None
    store = evaluator.sim_store(params["cache_memory_mb"], params["cache_disk_mb"])
    return store, stream.run, stream.start_frame, stream.substeps
