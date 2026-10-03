"""M1 gate: cache correctness (docs/VISION.md).

Walks every 2D node kind in `SPECS`, changes each parameter in turn and asserts:

1. the full-frame `Evaluator` cache key (digest) changes and the render is a fresh cache
   miss, never a stale hit;
2. the tile-path content digest (`tileexec._compute_node_digests`) changes and
   `TileExecutor.compose` records a fresh miss, never a stale hit;
3. an unchanged graph is served from cache (a hit, not a recompute) on both paths.

`_alt_value` picks a second valid value generically from the param's own default (bool
negation, a different CHOICES entry, a perturbed colour-curve JSON string, a clamped
numeric step, or an appended string), so a newly added node or param is covered without
edits here. `STRUCTURAL_EXCLUSIONS` and `PARAM_SKIPS` are the stated, precedented
exceptions this project already uses for nodes needing state a generic single-node graph
cannot fabricate (a real file, a linked Tracker, a multichannel layer) -- the same pattern
as Transform/Crop/Mirror's tile exclusion in tileexec.py.
"""
import copy
import json
import unittest

from nodebased.core import CHOICES, Dispatcher, LIMITS, OUTPUT_TYPES, SPECS
from nodebased.imaging import Evaluator
from nodebased.tileexec import TileExecutor, _compute_node_digests

# Kinds a single-node graph with plain Constant sources cannot exercise: they need a real
# file, a linked node of another kind, a multichannel layer, or are graph-structural rather
# than a per-pixel kernel. Each reason was confirmed empirically (constructing the minimal
# graph raises the named condition), not assumed.
STRUCTURAL_EXCLUSIONS = {
    "AppendClip": "multi-clip timeline container, no single-image identity",
    "Cryptomatte": "needs a multichannel EXR source with crypto_object layers",
    "Group": "graph container, no per-pixel kernel of its own",
    "IDistort": "needs a wired uv input or an existing uv_layer on the source",
    "Inpaint": "needs a connected matte input beyond the required image",
    "Input": "only valid inside a Group",
    "MatchGrade": "needs a connected reference or a baked Analyze pass",
    "Precomp": "needs a referenced project document on disk",
    "Profile": "has no params; a pass-through probe, nothing to key on",
    "LightMixer": "needs light.* layers from a Render3D lights pass or an EXR; tests/test_lightmixer.py covers its pixels",
    "Read": "file-backed identity, covered by test_hostile_media_2d.py and the Read suite",
    "ReadBundle": "file-backed identity (.bundle.json), covered by the bundle suite",
    "Relight": "needs a Render3D 'Relight passes' upstream, out of L2's 2D scope",
    "Render3D": "3D scene/camera inputs, owned by L4",
    "Roto": "needs roto shape payload beyond params to determine bounds",
    "STMap": "needs a wired uv input or an existing uv_layer on the source",
    "VectorBlur": "needs a wired uv input or an existing uv_layer on the source",
    "VectorCornerPin": "needs an upstream SmartVector vector layer",
    "VectorDistort": "needs an upstream SmartVector vector layer",
    "VectorToMotion": "needs an upstream SmartVector vector layer",
    "Vectorfield": "file-backed identity (.cube/.3dl)",
    "ZDefocus": "needs a depth layer from a multichannel Read or Render3D",
    "ZMerge": "needs a depth layer from a multichannel Read or Render3D",
    "ZSlice": "needs a depth layer from a multichannel Read or Render3D",
}

# Individual params, on an otherwise-testable kind, that need state beyond a plain
# single-node graph (a real linked node, a real OCIO config/look, a real layer name) or are
# constrained jointly with another param already covered by that other param's own check.
PARAM_SKIPS = {
    ("Flare", "tracker_id"): "must name a real Tracker/Stabilize node",
    ("Flare", "track_index"): "coupled to tracker_id",
    ("GridWarpTracker", "tracker_id"): "must name a real Tracker/Stabilize node",
    ("OCIOColorspace", "config"): "must name a real OCIO config file/URI",
    ("OCIODisplay", "config"): "must name a real OCIO config file/URI",
    ("OCIODisplay", "look"): "must name a look defined in the active config "
                             "(regression-tested directly in test_ocio_nodes.py)",
    ("OCIOFileTransform", "path"): "file-backed identity, covered by the OCIO file-transform suite",
    ("OCIOFileTransform", "config"): "must name a real OCIO config file/URI",
    ("OCIOLogConvert", "config"): "must name a real OCIO config file/URI",
    ("OCIOLookTransform", "config"): "must name a real OCIO config file/URI",
    ("OCIOLookTransform", "look"): "must name a look defined in the active config",
    ("Shuffle", "layer"): "needs a real named layer on a multichannel source",
    ("ShuffleCopy", "layer1"): "needs a real named layer on a multichannel source",
    ("ShuffleCopy", "layer2"): "needs a real named layer on a multichannel source",
    ("SmartVector", "frame_start"): "constrained jointly with frame_end/reference_frame",
    ("SmartVector", "reference_frame"): "constrained by the document's frame range at construction time",
    ("TimeDissolve", "which"): "only read under ease='animation curve' with a curve set on it "
                               "(imaging.py: 't = float(params[\"which\"]) if \"which\" in curves "
                               "else t'); under every other ease it is computed from in/out/frame "
                               "and the stored value is correctly inert, confirmed by reading the "
                               "digest formula rather than assumed",
}

# `Evaluator.evaluate_raster`'s own docstring (nodebased/imaging.py, `cache_fractional`): every
# caller of a fractional sub-frame except TimeBlur's own nested call keeps an "ephemeral"
# contract on purpose -- a scrub position is rarely revisited, so those sub-samples bypass the
# memory cache on every call, by design. Their param-to-digest-change checks above are unaffected
# (the outer node's own digest still caches correctly); only the repeat-call recompute count does
# not apply to them.
FRACTIONAL_SAMPLING_KINDS = {"Kronos", "OFlow", "MotionBlur", "MotionBlur2D", "MotionBlur3D",
                             "VectorGenerator", "TimeWarp"}

# Overrides for params whose valid value space a generic type-based rule cannot guess
# (grammar-checked strings, cross-referenced ids, discrete non-CHOICES sets).
VALUE_OVERRIDES = {
    ("AddTimeCode", "timecode"): "00:00:01:00",
    ("Assert", "condition"): "2",
    ("GenerateLUT", "lut_size"): lambda default: 17 if default != 17 else 33,
    ("GridWarpTracker", "track_indices"): "0,1",
    ("LevelSet", "channel"): "rgba.red",
    ("LevelSet", "output"): "rgba.red",
    ("ModifyMetaData", "edits"): "set foo bar",
    ("OCIOColorspace", "src"): "ACES2065-1",
    ("OCIOColorspace", "dst"): "ACES2065-1",
    ("OCIOLookTransform", "src"): "ACES2065-1",
    ("OCIOLookTransform", "dst"): "ACES2065-1",
    ("OCIODisplay", "display"): "Gamma 2.2 Rec.709 - Display",  # still lists the default view
    ("OCIODisplay", "view"): "Raw",
    ("Expression", "expr_r"): "r*0.5",
    ("Expression", "expr_g"): "g*0.5",
    ("Expression", "expr_b"): "b*0.5",
    ("Expression", "expr_a"): "a*0.5",
}


def _alt_curve(decoded):
    """A curve JSON dict (`nodebased/colorcurves.py`'s wire format) with one extra point."""
    points = sorted(decoded["points"])
    lo_x, hi_x = points[0][0], points[-1][0]
    mid_x = (lo_x + hi_x) / 2
    if any(abs(p[0] - mid_x) < 1e-9 for p in points):
        mid_x = lo_x + (hi_x - lo_x) * 0.25
    mutated = copy.deepcopy(decoded)
    mutated["points"] = sorted(points + [[mid_x, points[0][1]]])
    return mutated


def _alt_value(kind, param, default):
    """A second valid value for `param`, different from `default`, chosen generically from
    its own type and the project's LIMITS/CHOICES registries where declared."""
    override = VALUE_OVERRIDES.get((kind, param))
    if override is not None:
        return override(default) if callable(override) else override
    if isinstance(default, str):
        try:
            decoded = json.loads(default)
        except (TypeError, ValueError):
            decoded = None
        if isinstance(decoded, dict) and "points" in decoded and "interpolation" in decoded:
            return json.dumps(_alt_curve(decoded), separators=(",", ":"))
    choices = CHOICES.get(param)
    if choices:
        return next((c for c in choices if c != default), choices[0])
    if isinstance(default, bool):
        return not default
    if isinstance(default, int):
        lo, hi = LIMITS.get(param, (None, None))
        value = default + 1
        if hi is not None and value > hi:
            value = default - 1
        return value
    if isinstance(default, float):
        lo, hi = LIMITS.get(param, (None, None))
        value = default + 0.5
        if hi is not None and value > hi:
            value = default - 0.5
        if lo is not None and value < lo:
            value = default + 0.5
        return value
    if isinstance(default, str):
        return default + "_alt"
    return default


def _build_graph(kind):
    """A Dispatcher with one node of `kind`, its required inputs wired to distinct Constants."""
    d = Dispatcher()
    d.execute({"op": "create", "type": kind, "id": "n", "params": {}})
    for index, slot in enumerate(SPECS[kind]["inputs"]):
        source_id = f"src{index}"
        d.execute({"op": "create", "type": "Constant", "id": source_id,
                   "params": {"width": 64, "height": 64, "red": 0.5, "green": 0.3,
                              "blue": 0.2, "alpha": 1.0}})
        d.execute({"op": "connect", "id": "n", "input": slot, "source": source_id})
    return d


def _testable_kinds():
    return sorted(k for k in SPECS
                  if OUTPUT_TYPES.get(k, "image") == "image" and k not in STRUCTURAL_EXCLUSIONS)


class CacheCorrectnessTests(unittest.TestCase):
    """One shared Evaluator + TileExecutor per kind, so cache hit/miss counters are real
    telemetry from a live cache rather than a fresh (trivially-miss) instance each time."""

    def test_every_2d_kind_keys_and_serves_its_cache_correctly(self):
        checked_kinds = 0
        checked_params = 0
        for kind in _testable_kinds():
            with self.subTest(kind=kind):
                d = _build_graph(kind)
                ev = Evaluator()
                exe = TileExecutor()

                # Baseline render: first look is always a miss on both paths. (`ev.misses`
                # counts every node in the graph, so a two-input kind like Merge starts
                # above 1 from evaluating its Constant sources; what matters is that it
                # does not grow again below on a repeat, unchanged look.)
                base_raster, base_digest = ev.evaluate_raster(
                    d.document, "n", tier=1, typed=True, return_digest=True)
                misses_after_base = ev.misses
                self.assertGreater(misses_after_base, 0)
                # A kind without a tile-native kernel (a stated, precedented exclusion like
                # Transform/Crop/Mirror -- see docs/BENCHMARKS-v0.9-4k.md) renders through
                # the executor's own internal full-frame Evaluator instead of its tile
                # cache, so the combined counter is what "recomputed" means on this path.
                def tile_misses():
                    return exe.cache.misses + exe.evaluator.misses

                base_tile_digest = _compute_node_digests(d.document, tier=1, frame=1)["n"]
                exe.compose(d.document, "n", frame=1, tier=1)
                tile_misses_after_base = tile_misses()
                self.assertGreater(tile_misses_after_base, 0)

                # Unchanged graph: a repeat look must be a hit, not a recompute -- except the
                # kinds that deliberately keep fractional-subframe sampling ephemeral.
                _, repeat_digest = ev.evaluate_raster(
                    d.document, "n", tier=1, typed=True, return_digest=True)
                self.assertEqual(repeat_digest, base_digest)
                if kind not in FRACTIONAL_SAMPLING_KINDS:
                    self.assertEqual(ev.misses, misses_after_base,
                                     "an unchanged full-frame graph must not recompute")
                    self.assertGreaterEqual(ev.hits, 1)
                    exe.compose(d.document, "n", frame=1, tier=1)
                    self.assertEqual(tile_misses(), tile_misses_after_base,
                                     "an unchanged tiled graph must not recompute")

                checked_kinds += 1
                for param, default in SPECS[kind]["params"].items():
                    if (kind, param) in PARAM_SKIPS:
                        continue
                    with self.subTest(kind=kind, param=param):
                        alt = _alt_value(kind, param, default)

                        misses_before = ev.misses
                        tile_misses_before = tile_misses()
                        d.execute({"op": "set", "id": "n", "param": param, "value": alt})

                        _, changed_digest = ev.evaluate_raster(
                            d.document, "n", tier=1, typed=True, return_digest=True)
                        self.assertNotEqual(changed_digest, base_digest,
                                            f"{kind}.{param}: cache key missing this param")
                        self.assertGreater(ev.misses, misses_before,
                                           f"{kind}.{param}: edit did not force a recompute")

                        changed_tile_digest = _compute_node_digests(d.document, tier=1, frame=1)["n"]
                        self.assertNotEqual(changed_tile_digest, base_tile_digest,
                                            f"{kind}.{param}: tile cache key missing this param")
                        exe.compose(d.document, "n", frame=1, tier=1)
                        self.assertGreater(tile_misses(), tile_misses_before,
                                           f"{kind}.{param}: tiled edit did not force a recompute")

                        # Revert: the original digest and cache entry must still be valid.
                        d.execute({"op": "set", "id": "n", "param": param, "value": default})
                        _, reverted_digest = ev.evaluate_raster(
                            d.document, "n", tier=1, typed=True, return_digest=True)
                        self.assertEqual(reverted_digest, base_digest)
                        checked_params += 1

        self.assertGreater(checked_kinds, 100, "the 2D node set should not shrink this far")
        self.assertGreater(checked_params, 500, "most kinds should have testable params")


if __name__ == "__main__":
    unittest.main()
