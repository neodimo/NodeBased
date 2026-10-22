"""Lane 4 step H2: the converged picture of the viewport's Render mode agrees with Render3D.

The viewport traces on a worker thread with its own sample schedule (a denoised first look, then doubling adaptive steps to
`SAMPLE_CAP`); Render3D traces once at `pt_samples`. Both estimate the same image, so the check is the one the path tracer's own
tests use: against a 1024-sample Render3D reference, the viewport's converged picture must be about as close as Render3D's own
64-sample render is (noise, not bias), with the mean brightness within a few percent, for a lit mesh room, with depth of
field, through a distorting lens, and for a mixed mesh and splat scene. The denoised picture the artist sees must be no farther
from the reference than the raw pass it was filtered from. Render3D's outputs are untouched: its picture is identical before
and after a viewport pass, and the viewport's guide passes are the ones `pathtrace.guide_aovs` gives the final denoiser.

Every case runs the real tracer on whatever adapter the process has (`run/force-adapter.py` runs the module on the Radeon and on
llvmpipe); a machine with no working path tracer skips.
"""
import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtWidgets import QApplication

import tests.isolation  # noqa: F401  (Qt settings of the test run stay out of the user's real ones)
from nodebased import gpu3d, pathtrace, progressiverender as P, scene3d, viewportlens
from nodebased.core import Dispatcher
from nodebased.imaging import Evaluator
from nodebased.progressivejob import ProgressiveJob, Request
from nodebased.viewport3d import Viewport3D
from tests.progressive_widget import paint_until, settled
from tests.test_3d_progressive_render import _SKIP

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

APP = QApplication.instance() or QApplication([])

WIDTH, HEIGHT = 320, 220         # the widget's own minimum size
REFERENCE_SAMPLES = 1024           # 256 without a graphics card, set below `_software_adapter`
BOUNCES = P.MAX_BOUNCES
AMBIENT = 0.15
# The viewport's converged picture against the reference, as a multiple of Render3D's own 64-sample error plus a floor (the
# two estimators differ in seeds and in the viewport's adaptive stop, so equal is not expected), and the mean brightness bias.
NOISE_FACTOR = 1.6
NOISE_FLOOR = 0.01
BIAS = 0.03                       # raw picture
DENOISED_BIAS = 0.05              # the filter softens bright speckle and thin splats, which moves the mean a little
DISPLAY_LEVELS = 4.0              # mean 8-bit levels between the viewport's lens picture and Render3D's, see `test_lens`


def _software_adapter():
    """True when the process's graphics adapter is a CPU rasteriser (llvmpipe), or there is no adapter at all
    (the Linux CI runner: the path tracer falls back to numpy and the mixed-scene job ran out its 300 s wait)."""
    try:
        return gpu3d._kind_of(gpu3d._state()["info"]) == "cpu"
    except Exception:
        return True


# DiMo 10/21 9:16 AM: on a CPU tracer (GitHub's runners have no graphics card) the 1024-sample references
# took about two of the conformance run's three and a half hours. 256 samples there; the viewport's own
# 64-sample target stays. The full 1024 runs wherever there is a graphics card, before every tag.
if _software_adapter():
    REFERENCE_SAMPLES = 256


def room(camera=None, **render):
    """A lit room: rough sphere and cube on a floor under a rectangular area light (indirect light makes it noisy)."""
    d = Dispatcher()
    d.execute(dict(op="create", id="floor", type="Card3D", params=dict(
        card_width=14.0, card_height=14.0, rx=-90.0, ty=-1.2, tz=-4.0, material="pbr", pbr_roughness=0.7, red=0.75, green=0.7, blue=0.6)))
    d.execute(dict(op="create", id="ball", type="Sphere3D", params=dict(
        sphere_radius=1.0, ty=-0.2, tz=-4.0, material="pbr", pbr_roughness=0.35, red=0.8, green=0.25, blue=0.2)))
    d.execute(dict(op="create", id="cube", type="Cube3D", params=dict(
        cube_size=1.0, tx=-2.0, ty=-0.7, tz=-3.0, ry=30.0, material="pbr", pbr_roughness=0.9, red=0.3, green=0.5, blue=0.8)))
    d.execute(dict(op="create", id="light", type="Light3D", params=dict(
        light_type="Rect", tx=0.5, ty=3.0, tz=-3.5, target_x=0.0, target_y=0.0, target_z=-4.0, intensity=8.0,
        area_width=2.0, area_height=2.0, light_samples=1)))
    d.execute(dict(op="create", id="cam", type="Camera3D", params=dict(tz=0.0, target_z=-1.0, **(camera or {}))))
    d.execute(dict(op="create", id="scene", type="Scene3D", params={}))
    d.execute(dict(op="create", id="render", type="Render3D", params=dict(
        width=WIDTH, height=HEIGHT, render_mode="pathtrace", render_backend="auto", max_bounces=BOUNCES, ambient=AMBIENT,
        red=0.0, green=0.0, blue=0.0, alpha=1.0, **render)))
    for i, key in enumerate(("floor", "ball", "cube", "light")):
        d.execute(dict(op="connect", id="scene", input=f"object{i}", source=key))
    d.execute(dict(op="connect", id="render", input="scene", source="scene"))
    d.execute(dict(op="connect", id="render", input="camera", source="cam"))
    d.execute({"op": "view", "id": "render"})
    return d


def render3d(d, samples, **more):
    for param, value in (("pt_samples", samples), *more.items()):
        d.execute(dict(op="set", id="render", param=param, value=value))
    return Evaluator().evaluate_raster(dict(d.document, view="render"), "render", frame=1, tier=1).pixels


def viewport_state(d, denoise):
    """The converged state of the real widget (look-through, Render mode) for `d`, traced with passes denoised or raw."""
    widget = Viewport3D()
    widget.resize(WIDTH, HEIGHT)
    widget.set_document(d.document)
    widget.look_through = True
    widget._progressive_job = ProgressiveJob(on_update=widget.progressive_ready.emit, denoise_passes=denoise)
    widget.render_mode = True
    try:
        if not paint_until(widget, settled):
            raise AssertionError("the viewport did not settle")
        return widget._progressive_state, widget
    finally:
        widget._progressive_job.close()


def errors(image, reference):
    """(mean absolute rgb error / reference mean, |mean difference| / reference mean)."""
    mean = float(reference[..., :3].mean())
    return (float(np.abs(image - reference)[..., :3].mean()) / mean,
            abs(float(image[..., :3].mean()) - mean) / mean)


@unittest.skipIf(_SKIP, f"no working path tracer backend: {_SKIP}")
class ConvergenceTests(unittest.TestCase):
    def agree(self, d, label=""):
        reference = render3d(d, REFERENCE_SAMPLES)
        own, own_bias = errors(render3d(d, P.SAMPLE_CAP), reference)
        raw, _ = viewport_state(d, False)
        shown, _ = viewport_state(d, True)
        self.assertEqual(raw.samples, P.SAMPLE_CAP, label)
        self.assertEqual(raw.image.shape, reference.shape, label)
        self.assertFalse(raw.denoised)
        self.assertTrue(shown.denoised and not shown.preview)
        raw_error, raw_bias = errors(raw.image, reference)
        shown_error, shown_bias = errors(shown.image, reference)
        self.assertGreater(own, 0.002, f"{label}: the scene is too quiet to tell noise from bias")
        self.assertLess(raw_error, NOISE_FACTOR * own + NOISE_FLOOR, f"{label}: raw viewport error {raw_error:.4f} vs Render3D {own:.4f}")
        self.assertLess(raw_bias, BIAS, label)
        self.assertLess(shown_error, raw_error * 1.05 + 1e-3, f"{label}: the denoised picture is farther from the reference than the raw one")
        self.assertLess(shown_bias, DENOISED_BIAS, label)
        return reference, raw, shown

    def test_a_lit_mesh_room_agrees_with_render3d(self):
        self.agree(room(), "room")

    def test_depth_of_field_agrees_and_is_really_there(self):
        blur = dict(fstop=0.2, focus_distance=2.0)
        reference, raw, shown = self.agree(room(blur), "depth of field")
        pinhole = render3d(room(), REFERENCE_SAMPLES)
        self.assertGreater(errors(pinhole, reference)[0], 0.02)        # the thin lens visibly changes this scene
        # the viewport's picture is nearer the blurred reference than the pinhole one, raw and denoised
        for name, image in (("raw", raw.image), ("denoised", shown.image)):
            blurred, sharp = errors(image, reference)[0], errors(image, pinhole)[0]
            self.assertLess(blurred * 1.1, sharp, f"{name}: {blurred:.4f} from the blurred reference, {sharp:.4f} from the pinhole one")

    def test_the_viewport_picture_through_a_lens_agrees_with_render3d_through_lens(self):
        lens = dict(lens_k1=-0.06, lens_k2=0.004)
        d = room(lens)
        reference = P.display_rgba8(render3d(d, REFERENCE_SAMPLES, through_lens=1))[..., :3].astype(np.float64)
        own = P.display_rgba8(render3d(d, P.SAMPLE_CAP, through_lens=1))[..., :3].astype(np.float64)
        state, widget = viewport_state(d, True)
        camera = scene3d.camera_from_node(d.document["nodes"]["cam"])
        frame = viewportlens.frame_for(camera, WIDTH, HEIGHT)
        self.assertEqual(state.image.shape[:2], (frame.render_height, frame.render_width))      # the overscan picture is traced
        rgba = viewportlens.redistort(frame, viewportlens.fit(frame, P.display_rgba8(state.image)))
        bent = np.asarray(rgba)[..., :3].astype(np.float64)
        self.assertEqual(bent.shape, reference.shape)
        shown = float(np.abs(bent - reference).mean())
        noise = float(np.abs(own - reference).mean())
        self.assertLess(shown, noise + DISPLAY_LEVELS, f"{shown:.2f} levels against Render3D's own {noise:.2f}")
        ideal = P.display_rgba8(render3d(room(), REFERENCE_SAMPLES))[..., :3].astype(np.float64)
        self.assertGreater(float(np.abs(ideal - reference).mean()), 2 * shown)      # the bend is a real difference, and the viewport has it

    @unittest.skipIf(_software_adapter(), "the 1024-sample mixed-scene reference does not settle in minutes on a software adapter")
    def test_a_mixed_mesh_and_splat_scene_agrees_with_the_path_tracer(self):
        from benchmark_pathtrace import furnished_scene, scene3d_instance, splat_shell
        room_scene = furnished_scene()
        matrix = np.eye(4)
        matrix[:3, 3] = (-0.1, 0.6, -1.0)
        scene = scene3d.Scene(room_scene.geometries, lights=room_scene.lights, environments=room_scene.environments,
                              instances=room_scene.instances, splats=(scene3d_instance(splat_shell(6000), matrix),))
        camera = scene3d.Camera(scene3d.Transform3D(scene3d.Vec3(0.0, 1.2, 4.5)), scene3d.Vec3(0.0, 0.5, 0.0), 45.0)
        background = (0.0, 0.0, 0.0, 1.0)
        width, height = 192, 108

        def reference(samples):
            return pathtrace.render(scene, camera, width, height, background, AMBIENT, "rgba",
                                    pathtrace.PathSettings(samples=samples, max_bounces=BOUNCES), backend="auto")
        truth = reference(REFERENCE_SAMPLES)
        own, _ = errors(reference(P.SAMPLE_CAP), truth)

        def viewport(denoise):
            job = ProgressiveJob(denoise_passes=denoise, settle=0.0)
            self.addCleanup(job.close)
            identity = ("mixed", denoise)
            job.submit(Request(identity, scene, camera, width, height, background, AMBIENT))
            self.assertTrue(job.wait(settled, 300))
            return job.view(identity)[0]
        raw, shown = viewport(False), viewport(True)
        self.assertEqual((raw.samples, shown.samples), (P.SAMPLE_CAP, P.SAMPLE_CAP))
        self.assertGreater(own, 0.002)
        raw_error, raw_bias = errors(raw.image, truth)
        shown_error, shown_bias = errors(shown.image, truth)
        self.assertLess(raw_error, NOISE_FACTOR * own + NOISE_FLOOR, f"raw {raw_error:.4f} vs Render3D {own:.4f}")
        self.assertLess(raw_bias, BIAS)
        self.assertLess(shown_error, raw_error * 1.05 + 1e-3)
        self.assertLess(shown_bias, DENOISED_BIAS)
        self.assertGreater(float(np.abs(truth - reference(P.SAMPLE_CAP)).max()), 0.0)      # the splat shell is part of what is compared


@unittest.skipIf(_SKIP, f"no working path tracer backend: {_SKIP}")
class OutputsUntouchedTests(unittest.TestCase):
    def test_render3d_gives_the_same_picture_before_and_after_a_viewport_pass(self):
        d = room()
        before = render3d(d, 16)
        viewport_state(d, True)
        np.testing.assert_array_equal(render3d(d, 16), before)

    def test_the_viewports_guides_are_the_ones_the_final_denoiser_reads(self):
        d = room()
        state, _ = viewport_state(d, True)
        scene = Evaluator().evaluate_raster(dict(d.document, view="render"), "scene", frame=1, tier=1, typed=True)
        camera = scene3d.camera_from_node(d.document["nodes"]["cam"])
        guides = pathtrace.guide_aovs(scene, camera, WIDTH, HEIGHT, pathtrace.PathSettings(samples=P.GUIDE_SAMPLES, max_bounces=BOUNCES),
                                      backend="auto", ambient=0.15)
        albedo, normals, depth = state.guides
        np.testing.assert_allclose(normals, guides["normals"][..., :3], atol=1e-5)
        np.testing.assert_allclose(depth, guides["depth"][..., 0], atol=1e-4)
        self.assertEqual(albedo.shape, guides["albedo"].shape)

    def test_the_raw_pass_is_unchanged_when_denoising_is_off(self):
        d = room()
        scene = Evaluator().evaluate_raster(dict(d.document, view="render"), "scene", frame=1, tier=1, typed=True)
        camera = scene3d.camera_from_node(d.document["nodes"]["cam"])
        state = None
        for _ in range(3):
            state = P.step_interactive(state, scene, camera, WIDTH, HEIGHT, (0, 0, 0, 1), AMBIENT, "k", stages=())
        plain = None
        for _ in range(3):
            plain = P.step(plain, scene, camera, WIDTH, HEIGHT, (0, 0, 0, 1), AMBIENT, "k")
        np.testing.assert_array_equal(state.image, plain.image)
        self.assertFalse(state.denoised)


if __name__ == "__main__":
    unittest.main()
