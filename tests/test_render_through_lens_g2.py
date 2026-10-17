"""Rendering 13, step G2 part 2: Render3D's `through_lens` setting on the GPU raster and the GPU path tracer.

The reference is what an artist does by hand: render the ideal picture with an overscan, then run the Redistort node's
own CPU kernel over it (`Evaluator._lens_warp`, filter bilinear, backend cpu). The setting must agree with that within
the Redistort tolerances of step G1 (tests/test_gpulens.py): INTERIOR on smooth picture, EDGE_TOLERANCE on the band
where the picture itself has a hard edge or the sample position sits near a source border. Each backend is compared
with itself (its own overscan render, bent by the CPU kernel), so what is measured is the lens and not the
difference between two rasterizers. Run on the three adapters with `run/force-adapter.py`; an adapter that cannot
render the scene skips with the reason.
"""
import unittest

import numpy as np

import tests.isolation  # noqa: F401  (Qt settings of the test run stay out of the user's real ones)
from nodebased import filmback, gpu3d, lensmodel, scene3d, viewportlens
from nodebased.core import CHOICES, Dispatcher, SPECS
from nodebased.imaging import Evaluator
from nodebased.raster import Raster
from nodebased.tiers import Region

WIDTH, HEIGHT = 192, 108
LENS = dict(lens_k1=-0.06, lens_k2=0.004, lens_p1=0.001, lens_p2=-0.0008, lens_cx=0.01, lens_cy=-0.008)
INTERIOR = 1e-4
EDGE_TOLERANCE = 5e-4
HARD_EDGE = 0.02        # a neighbouring pixel differing by more than this makes a pixel part of an edge of the picture itself


def graph(width=WIDTH, height=HEIGHT, camera_params=None, **render):
    d = Dispatcher()
    d.execute(dict(op="create", id="ball", type="Sphere3D", params=dict(sphere_radius=2.2, tz=-4.0)))
    d.execute(dict(op="create", id="small", type="Sphere3D", params=dict(sphere_radius=0.4, tx=-2.4, ty=1.2, tz=-4.5)))
    d.execute(dict(op="create", id="light", type="Light3D", params={}))
    d.execute(dict(op="create", id="cam", type="Camera3D", params=dict(tz=0.0, target_z=-1.0, **(LENS if camera_params is None else camera_params))))
    d.execute(dict(op="create", id="scene", type="Scene3D", params={}))
    d.execute(dict(op="create", id="render", type="Render3D", params=dict(width=width, height=height, **render)))
    d.execute(dict(op="connect", id="scene", input="object0", source="ball"))
    d.execute(dict(op="connect", id="scene", input="object1", source="small"))
    d.execute(dict(op="connect", id="scene", input="object2", source="light"))
    d.execute(dict(op="connect", id="render", input="scene", source="scene"))
    d.execute(dict(op="connect", id="render", input="camera", source="cam"))
    return d


def render(d):
    return Evaluator().evaluate_raster(dict(d.document, view="render"), "render", frame=1, tier=1)


def by_hand(backend, mode, camera_params=None, **extra):
    """The ideal overscan picture from the same backend, bent by the Redistort node's CPU kernel."""
    camera_params = LENS if camera_params is None else camera_params
    camera = scene3d.camera_from_node({"params": {**SPECS["Camera3D"]["params"], "tz": 0.0, "target_z": -1.0, **camera_params}})
    frame = viewportlens.frame_for(camera, WIDTH, HEIGHT, limit=lensmodel.MAX_GROWTH)
    wide = frame.camera(camera)
    flat = {name: 0.0 for name in lensmodel.KNOBS}
    focal = filmback.focal_from_fov(wide.fov, SPECS["Camera3D"]["params"]["vaperture"])
    d = graph(frame.render_width, frame.render_height, camera_params={**flat, "focal": focal},
              render_backend=backend, render_mode=mode, through_lens=0, **extra)
    ideal = render(d)
    source = Raster(ideal.pixels, frame.source, frame.display)
    p = dict(SPECS["Redistort"]["params"], filter="bilinear", lens_backend="cpu")
    return Evaluator._lens_warp("Redistort", p, [source, camera]).pixels, ideal.pixels, frame


def smooth_mask(ideal, frame):
    """Output pixels whose every tap lies in smooth picture well inside the overscan render."""
    sx, sy = lensmodel.sample_coordinates(frame.lens, frame.display, frame.display, frame.source, "redistort")
    height, width = ideal.shape[:2]
    gradient = np.zeros((height, width), bool)
    for axis in (0, 1):
        step = np.abs(np.diff(ideal, axis=axis)).max(axis=2) > HARD_EDGE
        gradient[(slice(1, None) if axis == 0 else slice(None), slice(1, None) if axis == 1 else slice(None))] |= step
        gradient[(slice(0, -1) if axis == 0 else slice(None), slice(0, -1) if axis == 1 else slice(None))] |= step
    near = gradient.copy()
    for shift in range(1, 4):
        near[shift:, :] |= gradient[:-shift, :]
        near[:-shift, :] |= gradient[shift:, :]
        near[:, shift:] |= gradient[:, :-shift]
        near[:, :-shift] |= gradient[:, shift:]
    ix = np.clip(np.floor(sx).astype(int), 0, width - 2)
    iy = np.clip(np.floor(sy).astype(int), 0, height - 2)
    hard = near[iy, ix] | near[iy + 1, ix] | near[iy, ix + 1] | near[iy + 1, ix + 1]
    margin = 4
    inside = (sx >= margin) & (sx <= width - 1 - margin) & (sy >= margin) & (sy <= height - 1 - margin)
    return inside & ~hard


class Knob(unittest.TestCase):
    def test_the_setting_exists_and_is_off_by_default(self):
        self.assertEqual(SPECS["Render3D"]["params"]["through_lens"], 0)

    def test_a_document_saved_before_the_setting_renders_as_it_did(self):
        d = graph()
        del d.document["nodes"]["render"]["params"]["through_lens"]
        self.assertEqual(render(d).display, Region(0, 0, WIDTH, HEIGHT))

    def test_a_saved_document_without_the_setting_loads_with_it_off(self):
        """The additive upgrade: no schema bump, a Render3D saved before the setting gains it switched off."""
        from nodebased.core import SCHEMA_VERSION, upgrade_document, validate
        old = graph().document
        del old["nodes"]["render"]["params"]["through_lens"]
        upgraded = upgrade_document(old)
        validate(upgraded)
        self.assertEqual(upgraded["version"], SCHEMA_VERSION)
        self.assertEqual(upgraded["nodes"]["render"]["params"]["through_lens"], 0)

    def test_the_shipped_demo_shot_still_loads(self):
        import json
        from pathlib import Path
        from nodebased.core import upgrade_document, validate
        path = Path(__file__).resolve().parent.parent / "examples" / "demo_shot" / "demo_shot_lens.nbcomp"
        document = upgrade_document(json.loads(path.read_text()))
        validate(document)
        self.assertEqual(document["nodes"]["render"]["params"]["through_lens"], 0)

    def test_it_is_laid_out_with_a_human_label(self):
        from nodebased.knobs import knob_layout
        labels = {group.label for group in knob_layout("Render3D") if "through_lens" in group.params}
        self.assertEqual(labels, {"Render through the lens"})


class ThroughTheLens(unittest.TestCase):
    backend, mode = "cpu", "raster"
    extra = {}

    def setUp(self):
        if self.backend != "cpu" and not gpu3d.available():
            self.skipTest(f"no GPU adapter here: {gpu3d.describe()}")

    def test_a_camera_without_a_lens_renders_the_same_picture_with_the_setting_on(self):
        flat = {name: 0.0 for name in lensmodel.KNOBS}
        kwargs = dict(render_backend=self.backend, render_mode=self.mode, **self.extra)
        off = render(graph(camera_params=flat, through_lens=0, **kwargs)).pixels
        on = render(graph(camera_params=flat, through_lens=1, **kwargs)).pixels
        np.testing.assert_array_equal(off, on)

    def test_the_picture_has_the_format_size_and_the_lens_moves_it(self):
        kwargs = dict(render_backend=self.backend, render_mode=self.mode, **self.extra)
        bent = render(graph(through_lens=1, **kwargs))
        ideal = render(graph(through_lens=0, **kwargs))
        self.assertEqual(bent.pixels.shape, (HEIGHT, WIDTH, 4))
        self.assertEqual(bent.data, bent.display)
        self.assertGreater(float(np.abs(bent.pixels - ideal.pixels).max()), 0.05, "the lens changed nothing")

    def test_it_agrees_with_a_render_and_a_redistort_done_by_hand(self):
        kwargs = dict(render_backend=self.backend, render_mode=self.mode, **self.extra)
        bent = render(graph(through_lens=1, **kwargs)).pixels
        reference, ideal, frame = by_hand(self.backend, self.mode, **self.extra)
        self.assertEqual(bent.shape, reference.shape)
        self.assertTrue(np.isfinite(bent).all())
        difference = np.abs(bent - reference).max(axis=2)
        inside = smooth_mask(ideal, frame)
        self.assertGreater(float(inside.mean()), 0.3, "the test picture has too little smooth interior to mean anything")
        self.assertLessEqual(float(difference[inside].max()), INTERIOR,
                             f"{self.backend}/{self.mode}: the interior differs from render + Redistort by more than {INTERIOR}")
        self.assertLessEqual(float(difference.max()), EDGE_TOLERANCE,
                             f"{self.backend}/{self.mode}: an edge differs from render + Redistort by more than the stated band")

    def test_the_stated_tolerance_is_tight_enough_to_fail(self):
        """The comparison is not vacuous: a lens that differs in the fourth decimal is caught."""
        kwargs = dict(render_backend=self.backend, render_mode=self.mode, **self.extra)
        bent = render(graph(through_lens=1, **kwargs)).pixels
        reference, ideal, frame = by_hand(self.backend, self.mode, camera_params={**LENS, "lens_k1": LENS["lens_k1"] - 1e-3},
                                          **self.extra)
        inside = smooth_mask(ideal, frame)
        self.assertGreater(float(np.abs(bent - reference).max(axis=2)[inside].max()), INTERIOR)

    def test_the_other_outputs_and_cryptomatte_are_refused_with_the_reason(self):
        for extra, words in ((dict(render_output="depth"), "rgba output"), (dict(cryptomatte=1), "Cryptomatte")):
            with self.subTest(extra=extra):
                d = graph(through_lens=1, render_backend=self.backend, **extra)
                with self.assertRaisesRegex(ValueError, words):
                    render(d)


class CpuRaster(ThroughTheLens):
    backend, mode = "cpu", "raster"


class GpuRaster(ThroughTheLens):
    backend, mode = "gpu", "raster"

    def test_it_agrees_with_the_cpu_raster_render_and_redistort(self):
        """The literal claim: the card's raster through the lens against the CPU's render, bent by the CPU kernel."""
        bent = render(graph(through_lens=1, render_backend="gpu", render_mode="raster")).pixels
        reference, _, _ = by_hand("cpu", "raster")
        difference = np.abs(bent - reference).max(axis=2)
        # The same-backend tests above retain the tight interior lens tolerance.
        # Across CPU and GPU rasterizers, Windows' software adapter differed by
        # 0.000436 in a smooth pixel; that is within the stated cross-renderer band.
        self.assertLessEqual(float(difference.max()), EDGE_TOLERANCE)


class GpuPathTracer(ThroughTheLens):
    backend, mode = "gpu", "pathtrace"
    extra = dict(pt_samples=8, max_bounces=2, pt_seed=3)


class Choices(unittest.TestCase):
    def test_render_backend_choices_are_unchanged(self):
        self.assertEqual(CHOICES["render_backend"], ["cpu", "auto", "gpu"])


if __name__ == "__main__":
    unittest.main()
