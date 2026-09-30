"""Shared graph builders for the M1 golden-image gate (docs/M1_GATE.md, tests/test_golden_2d.py).

Not a test module itself (no `Test` classes): both `test_golden_2d.py` and
`tests/data/golden/regenerate.py` import `CASES` from here so the graphs the suite checks and the
graphs the regenerator renders can never drift apart.

Canvases are small (24x16 or less) so every reference `.npy` under tests/data/golden/ stays well
under the 256 KB budget the lane brief sets.
"""
from pathlib import Path
import tempfile

import numpy as np

from nodebased.core import Dispatcher
from nodebased.imaging import Evaluator
from nodebased.media import write_exr
from nodebased.tileexec import SUPPORTED_TILED_KINDS, TileExecutor

W, H = 24, 16
GOLDEN_DIR = Path(__file__).parent / "data" / "golden"


class Graph:
    def __init__(self):
        self.d = Dispatcher()

    def add(self, key, kind, params=None, **inputs):
        self.d.execute(dict(op="create", id=key, type=kind, params=params or {}))
        for slot, source in inputs.items():
            self.d.execute(dict(op="connect", id=key, input=slot, source=source))
        return key

    @property
    def doc(self):
        return self.d.document


def evaluator_raster(document, target):
    return Evaluator().evaluate_raster(dict(document, view=target))


def tile_pixels(document, target):
    executor = TileExecutor(evaluator=Evaluator())
    document = dict(document, view=target)
    region = executor.canvas_region(document, target, frame=1, tier=1)
    return executor.compose_region(document, target, region, frame=1, tier=1).pixels


def _checker(g, key="plate", width=W, height=H, size=4):
    return g.add(key, "Checker", dict(width=width, height=height, size=size))


def _write_layered_exr(directory, name, beauty, **layers):
    """A tiny multichannel EXR fixture, generated fresh in `directory` every run so the golden
    reference only ever has to hold the small comparison arrays, never source imagery."""
    path = Path(directory) / f"{name}.exr"
    write_exr(path, beauty, bits="float", layers=layers or None)
    return str(path)


def _beauty(fill=(0.1, 0.2, 0.3, 1.0), width=W, height=H):
    frame = np.empty((height, width, 4), np.float32)
    frame[:] = fill
    return frame


def _ramp_layer(a_gain, b_gain, width=W, height=H):
    yy, xx = np.mgrid[0:height, 0:width]
    layer = np.zeros((height, width, 4), np.float32)
    layer[..., 0] = (xx / width) * a_gain
    layer[..., 1] = (yy / height) * b_gain
    layer[..., 3] = 1.0
    return layer


class GoldenCase:
    """One golden graph. `build(tmpdir)` returns (document, target); `tiled` says whether the
    node under test is on `SUPPORTED_TILED_KINDS` (Transform/Crop/Reformat are not -- the
    stated, precedented tile-path exclusion COMMON.md and docs/PARITY_2D.md both document)."""

    def __init__(self, name, build, tiled, atol, extra=None):
        self.name = name
        self.build = build
        self.tiled = tiled
        self.atol = atol
        self.extra = extra or (lambda raster: {})


def _grade_hdr_above_one(tmpdir):
    g = Graph()
    g.add("src", "Constant", dict(width=W, height=H, red=2.5, green=1.8, blue=3.2, alpha=1.0))
    g.add("node", "Grade", dict(multiply=1.4, offset=0.1), image="src")
    return g.doc, "node"


def _grade_negative_values(tmpdir):
    g = Graph()
    g.add("src", "Constant", dict(width=W, height=H, red=-0.6, green=-0.2, blue=0.9, alpha=1.0))
    g.add("node", "Grade", dict(exposure=0.3, offset=-0.15), image="src")
    return g.doc, "node"


def _merge(operation):
    def build(tmpdir):
        g = Graph()
        _checker(g, "a", size=4)
        g.add("b", "Constant", dict(width=W, height=H, red=0.4, green=0.7, blue=0.2, alpha=0.5))
        g.add("node", "Merge", dict(operation=operation), A="a", B="b")
        return g.doc, "node"
    return build


def _premult(tmpdir):
    g = Graph()
    g.add("src", "Constant", dict(width=W, height=H, red=0.8, green=0.5, blue=0.3, alpha=0.4))
    g.add("node", "Premult", {}, image="src")
    return g.doc, "node"


def _unpremult(tmpdir):
    g = Graph()
    g.add("src", "Constant", dict(width=W, height=H, red=0.32, green=0.2, blue=0.12, alpha=0.4))
    g.add("node", "Unpremult", {}, image="src")
    return g.doc, "node"


def _shuffle_named_layer(tmpdir):
    g = Graph()
    beauty = _beauty((0.1, 0.2, 0.3, 1.0))
    normals = _ramp_layer(1.0, -0.5)
    path = _write_layered_exr(tmpdir, "shuffle_src", beauty, normals=normals)
    g.add("src", "Read", dict(path=path))
    g.add("node", "Shuffle", dict(layer="normals"), image="src")
    return g.doc, "node"


def _shufflecopy_named_layers(tmpdir):
    g = Graph()
    beauty1 = _beauty((0.1, 0.2, 0.3, 1.0))
    beauty2 = _beauty((0.5, 0.4, 0.9, 1.0))
    depth = _ramp_layer(0.6, 0.9)
    motion = _ramp_layer(-1.0, 2.0)
    path1 = _write_layered_exr(tmpdir, "shufflecopy_src1", beauty1, depth=depth)
    path2 = _write_layered_exr(tmpdir, "shufflecopy_src2", beauty2, motion=motion)
    g.add("in1", "Read", dict(path=path1))
    g.add("in2", "Read", dict(path=path2))
    g.add("node", "ShuffleCopy",
          dict(layer1="depth", layer2="motion",
               out1_r="in1.r", out1_g="in1.g", out1_b="in1.b", out1_a="in1.a",
               out2_r="in2.r", out2_g="in2.g", out2_b="in2.b", out2_a="in2.a"),
          in1="in1", in2="in2")
    return g.doc, "node"


def _shufflecopy_out2(raster):
    layer = (raster.layers or {}).get("out2")
    return {"out2": layer.pixels} if layer is not None else {}


def _transform(filter_name):
    def build(tmpdir):
        g = Graph()
        _checker(g, "plate", size=4)
        g.add("node", "Transform",
              dict(translate_x=2.0, translate_y=-1.0, rotate=12.0, scale=1.15, filter=filter_name),
              image="plate")
        return g.doc, "node"
    return build


def _crop(x, y, width, height, name_suffix):
    def build(tmpdir):
        g = Graph()
        _checker(g, "plate", size=4)
        g.add("node", "Crop", dict(x=x, y=y, width=width, height=height), image="plate")
        return g.doc, "node"
    return build


def _reformat(width, height):
    def build(tmpdir):
        g = Graph()
        _checker(g, "plate", size=4)
        g.add("node", "Reformat",
              dict(reformat_type="to_box", width=width, height=height, resize_type="fit"),
              image="plate")
        return g.doc, "node"
    return build


def _exr_multichannel_roundtrip(tmpdir):
    """Read a multichannel EXR, write it straight back out, and read the copy: the round trip a
    Read->Write chain performs on every render, with a named layer riding along."""
    g = Graph()
    beauty = _beauty((0.25, 0.5, 1.6, 1.0))  # includes an HDR (> 1) channel
    depth = _ramp_layer(3.0, -2.0)
    path = _write_layered_exr(tmpdir, "roundtrip_src", beauty, depth=depth)
    g.add("src", "Read", dict(path=path))
    raster = evaluator_raster(g.doc, "src")
    from nodebased.media import raster_layer_arrays
    copy_path = Path(tmpdir) / "roundtrip_copy.exr"
    write_exr(copy_path, raster.to_display(), bits="float", layers=raster_layer_arrays(raster))
    g2 = Graph()
    g2.add("copy", "Read", dict(path=str(copy_path)))
    return g2.doc, "copy"


def _exr_roundtrip_depth_layer(raster):
    layer = (raster.layers or {}).get("depth")
    return {"depth": layer.pixels} if layer is not None else {}


CASES = [
    GoldenCase("grade_hdr_above_one", _grade_hdr_above_one, True, 1e-6),
    GoldenCase("grade_negative_values", _grade_negative_values, True, 1e-6),
    GoldenCase("merge_over", _merge("over"), True, 1e-6),
    GoldenCase("merge_plus", _merge("plus"), True, 1e-6),
    GoldenCase("merge_multiply", _merge("multiply"), True, 1e-6),
    GoldenCase("premult", _premult, True, 1e-6),
    GoldenCase("unpremult", _unpremult, True, 1e-6),
    GoldenCase("shuffle_named_layer", _shuffle_named_layer, True, 1e-6),
    GoldenCase("shufflecopy_named_layers", _shufflecopy_named_layers, True, 1e-6, _shufflecopy_out2),
    GoldenCase("transform_nearest", _transform("nearest"), False, 1e-6),
    GoldenCase("transform_bilinear", _transform("bilinear"), False, 1e-5),
    GoldenCase("transform_cubic", _transform("cubic"), False, 1e-5),
    GoldenCase("crop_larger_than_format", _crop(-4, -3, W + 8, H + 6, "larger"), False, 1e-6),
    GoldenCase("crop_smaller_than_format", _crop(4, 3, W - 8, H - 6, "smaller"), False, 1e-6),
    GoldenCase("reformat_larger_than_format", _reformat(W + 8, H + 6), False, 1e-5),
    GoldenCase("reformat_smaller_than_format", _reformat(W - 8, H - 6), False, 1e-5),
    GoldenCase("exr_multichannel_roundtrip", _exr_multichannel_roundtrip, True, 1e-6,
               _exr_roundtrip_depth_layer),
]


def worst_pixel(got, expected):
    """(index, |diff|, got value, expected value) of the largest per-element mismatch."""
    got, expected = np.asarray(got, np.float64), np.asarray(expected, np.float64)
    diff = np.abs(got - expected)
    index = np.unravel_index(int(np.argmax(diff)), diff.shape)
    return index, float(diff[index]), float(got[index]), float(expected[index])


def assert_close(test, got, expected, atol, label):
    got, expected = np.asarray(got), np.asarray(expected)
    if got.shape != expected.shape or not np.allclose(got, expected, atol=atol, rtol=0):
        index, worst, got_value, expected_value = worst_pixel(
            got if got.shape == expected.shape else np.zeros_like(expected), expected)
        test.fail(f"{label}: shapes {got.shape} vs {expected.shape}; worst pixel at {index}: "
                  f"got {got_value!r} expected {expected_value!r} (|diff|={worst!r} > atol={atol!r})")


def run_case(case, tmpdir):
    """(pixels, extra, tile_pixels_or_None) for `case`, evaluated fresh in `tmpdir`."""
    document, target = case.build(tmpdir)
    raster = evaluator_raster(document, target)
    pixels = raster.to_display()
    extra = case.extra(raster)
    tiled = None
    if case.tiled:
        assert document["nodes"][target]["type"] in SUPPORTED_TILED_KINDS, target
        tiled = tile_pixels(document, target)
    return pixels, extra, tiled


def reference_path(name):
    return GOLDEN_DIR / f"{name}.npy"


def extra_reference_path(name, key):
    return GOLDEN_DIR / f"{name}.{key}.npy"
