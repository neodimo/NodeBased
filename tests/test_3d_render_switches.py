"""Per-object render switches (lane L4, plan "Rendering 7", step R1 of 3).

A mesh has three switches, all on by default: `cast_shadows` (off: no shadow ray is blocked by it), `receive_shadows` (off: it
is lit as if nothing shadowed it) and `visible_to_camera` (off: primary rays and every data pass pass through it, while shadow,
reflection, refraction and bounce rays still meet it). The tests state those three properties on the CPU raster and
ray-traced modes, the CPU path tracer, the GPU raster and ray-traced modes and the GPU path tracer, on whichever adapter the
run forces (`run/force-adapter.py default|integrated|cpu`)."""
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from nodebased import core, cryptomatte3d, gpu3d, gpuinstance, pathtrace as pt, scene3d as s, viewportgpu
from nodebased.core import Dispatcher
from nodebased.imaging import Evaluator
from tests import golden_render_switches as golden
from tests import gpu_precision
from tests.test_3d_light_linking import CAMERA, SIZE, SUN, LAMP, ball, core_of, coverage, ground
from tests.test_3d_light_linking_gpu import KEY as SPLAT_KEY
from tests.test_3d_pathtrace import box
from tests.test_3d_pathtrace_splats import SPLAT_CAMERA, instance, plane_cloud

AMBIENT = 0.1
BLACK = (0.0, 0.0, 0.0, 0.0)
DIRECT = pt.PathSettings(samples=24, max_bounces=1, diffuse_bounces=1)     # direct light only: exact against "the object absent"
GOLDEN = Path(__file__).parent / "data" / "golden"
SPLAT_TOLERANCE = gpu_precision.tolerance(2e-3)


def scene(*, hit=None, floor=None, lights=(SUN, LAMP), x=0.0):
    """A ball over a floor; `hit` and `floor` are fields to replace on the ball and the floor."""
    return s.Scene((replace(ball(x), **(hit or {})), replace(ground(), **(floor or {}))), lights)


def without_the_ball(lights=(SUN, LAMP)):
    return s.Scene((ground(),), lights)


class Renderer:
    """One final renderer: `(scene, output) -> picture` and how close two pictures from it must be."""

    def __init__(self, name, draw, atol, gpu=False):
        self.name, self.draw, self.atol, self.gpu = name, draw, atol, gpu

    def __call__(self, scene, output="rgba"):
        return self.draw(scene, output)


def _cpu(mode):
    return lambda scene, output: s.render(scene, CAMERA, *SIZE, ambient=AMBIENT, mode=mode, output=output)


def _gpu(mode):
    return lambda scene, output: gpu3d.render(scene, CAMERA, *SIZE, ambient=AMBIENT, mode=mode, output=output)


def _path(backend):
    return lambda scene, output: pt.render(scene, CAMERA, *SIZE, BLACK, 0.0, output, DIRECT, backend=backend)


CPU = (Renderer("cpu raster", _cpu("raster"), 1e-6), Renderer("cpu ray traced", _cpu("raytrace"), 1e-6),
       Renderer("cpu path tracer", _path("cpu"), 2e-5))
GPU = (Renderer("gpu raster", _gpu("raster"), 5e-4, True), Renderer("gpu ray traced", _gpu("raytrace"), 5e-4, True),
       Renderer("gpu path tracer", _path("gpu"), 5e-4, True))


class NodeTests(unittest.TestCase):
    KINDS = ("Card3D", "Cube3D", "Sphere3D", "Cylinder3D", "ReadGeo3D", "ReadGLTF3D")

    def test_every_mesh_node_has_the_three_switches_on(self):
        for kind in self.KINDS:
            for name in ("cast_shadows", "receive_shadows", "visible_to_camera"):
                self.assertEqual(core.SPECS[kind]["params"][name], "on", (kind, name))
        for name in ("cast_shadows", "receive_shadows", "visible_to_camera"):
            self.assertEqual(core.CHOICES[name], ["off", "on"])

    def test_the_knobs_are_laid_out_with_their_labels(self):
        from nodebased import knobs
        for kind in self.KINDS:
            labels = {g.label for g in knobs.KNOB_LAYOUT[kind]}
            self.assertTrue({"Cast shadows", "Receive shadows", "Visible to camera"} <= labels, kind)

    def test_the_params_become_geometry_fields(self):
        self.assertEqual(s.render_switches_from_params({}), dict(cast_shadows=True, receive_shadows=True, visible_to_camera=True))
        got = s.render_switches_from_params(dict(cast_shadows="off", receive_shadows="on", visible_to_camera="off"))
        self.assertEqual(got, dict(cast_shadows=False, receive_shadows=True, visible_to_camera=False))
        self.assertFalse(s.has_render_switches(s.Scene((ball(0),))))
        self.assertTrue(s.has_render_switches(s.Scene((ball(0, cast_shadows=False),))))

    def graph(self, **params):
        d = Dispatcher()
        for key, kind, p in (("ball", "Sphere3D", dict(sphere_radius=.8, ty=-.2, **params)), ("floor", "Card3D", dict(
                card_width=14, card_height=14, ty=-1, rx=-90, red=.6, green=.6, blue=.6)),
                ("sun", "Light3D", dict(shadows="on", light_type="Directional", tx=-4, ty=5, tz=2)),
                ("camera", "Camera3D", dict(ty=5, tz=8, target_y=-.5)),
                ("scene", "Scene3D", {}), ("render", "Render3D", dict(width=64, height=40, samples=1))):
            d.execute(dict(op="create", id=key, type=kind, params=p))
        for slot, source in (("object0", "ball"), ("object1", "floor"), ("object2", "sun")):
            d.execute(dict(op="connect", id="scene", input=slot, source=source))
        for slot in ("scene", "camera"):
            d.execute(dict(op="connect", id="render", input=slot, source=slot))
        return d

    def test_the_nodes_carry_the_switches_into_the_render(self):
        d = self.graph(cast_shadows="off", visible_to_camera="off")
        scene3 = Evaluator().evaluate_raster(d.document, "scene", typed=True)
        ball_geometry = scene3.geometries[0]
        self.assertEqual((ball_geometry.cast_shadows, ball_geometry.receive_shadows, ball_geometry.visible_to_camera),
                         (False, True, False))
        shown = Evaluator().evaluate_raster(self.graph().document, "render").pixels
        hidden = Evaluator().evaluate_raster(d.document, "render").pixels
        self.assertGreater(float(np.abs(shown - hidden).max()), .1)

    def test_a_document_without_the_switches_renders_as_one_with_their_defaults(self):
        d = self.graph()
        new = Evaluator().evaluate_raster(d.document, "render").pixels
        for name in ("cast_shadows", "receive_shadows", "visible_to_camera"):
            d.document["nodes"]["ball"]["params"].pop(name)
        old = Evaluator().evaluate_raster(core.upgrade_document(d.document) if hasattr(core, "upgrade_document") else d.document,
                                          "render").pixels
        np.testing.assert_array_equal(old, new)
        self.assertGreater(float(new.max()), .1)

    def test_an_instance_copy_takes_its_sources_switches(self):
        source = ball(0, cast_shadows=False, receive_shadows=False, visible_to_camera=False)
        matrices = np.tile(np.eye(4), (2, 1, 1))
        matrices[:, 0, 3] = (-2, 2)
        copies = s.expand_instances(s.InstanceSet((source,), matrices, np.zeros(2, np.int32)))
        for copy in copies:
            self.assertEqual((copy.cast_shadows, copy.receive_shadows, copy.visible_to_camera), (False, False, False))


class GoldenTests(unittest.TestCase):
    """With every switch at its default a scene renders as it did before the switches existed."""

    @classmethod
    def setUpClass(cls):
        cls.now = golden.render_all()

    def test_each_renderer_matches_its_pre_switch_picture(self):
        for name, picture in self.now.items():
            before = np.load(GOLDEN / f"render_switches_{name}.npy")
            if name.startswith("cpu"):
                np.testing.assert_array_equal(picture, before, err_msg=name)
            elif name == "gpu_pathtrace":     # another adapter rounds differently, and a path that diverges changes a pixel
                self.assertLess(float(np.abs(picture - before).mean()), .01 * float(before.mean()) + .002, name)
            else:
                np.testing.assert_allclose(picture, before, atol=2e-3, rtol=0, err_msg=name)
        self.assertTrue(set(self.now) >= {"cpu_raster", "cpu_raytrace", "cpu_pathtrace"})


class SwitchTests:
    """The three properties, for each renderer of `RENDERERS` (set by the subclasses below)."""
    RENDERERS = ()

    def each(self):
        for renderer in self.RENDERERS:
            with self.subTest(renderer=renderer.name):
                yield renderer

    def test_a_sphere_that_casts_no_shadow_leaves_the_floor_unshadowed(self):
        mine = coverage(ball(0))
        for r in self.each():
            base, off, gone = r(scene()), r(scene(hit=dict(cast_shadows=False))), r(without_the_ball())
            np.testing.assert_allclose(off[~mine], gone[~mine], atol=r.atol)         # the floor as if the sphere were not there
            self.assertGreater(float(np.abs(base[~mine] - gone[~mine]).max()), .05)  # which it was not, before
            inside = core_of(ball(0))
            np.testing.assert_allclose(off[inside], base[inside], atol=max(r.atol, 5e-3))   # the sphere itself is still drawn and lit

    def test_a_floor_that_takes_no_shadow_shows_none_and_the_sphere_is_unchanged(self):
        mine = coverage(ball(0))
        for r in self.each():
            base = r(scene())
            floor_off = r(scene(floor=dict(receive_shadows=False)))
            unshadowed = r(scene(hit=dict(cast_shadows=False)))
            np.testing.assert_allclose(floor_off[~mine], unshadowed[~mine], atol=r.atol)
            self.assertGreater(float(np.abs(base[~mine] - floor_off[~mine]).max()), .05)
            inside = core_of(ball(0))
            np.testing.assert_allclose(floor_off[inside], base[inside], atol=max(r.atol, 5e-3))

    def test_a_sphere_hidden_from_the_camera_vanishes_but_keeps_its_shadow(self):
        mine = coverage(ball(0))
        inside = core_of(ball(0))
        for r in self.each():
            base, hidden = r(scene()), r(scene(hit=dict(visible_to_camera=False)))
            np.testing.assert_allclose(hidden[~mine], base[~mine], atol=r.atol)       # the shadow is still on the floor
            shadow = np.abs(base[~mine] - r(without_the_ball())[~mine]).max(axis=-1) > .05
            self.assertGreater(int(shadow.sum()), 20)
            self.assertGreater(float(np.abs(hidden[inside] - base[inside]).max()), .1)   # the sphere itself is gone
            self.assertTrue(np.all(hidden[inside][..., 3] > .99))                       # and the floor shows through

    def test_a_hidden_sphere_is_absent_from_depth_and_the_id_pass(self):
        inside = core_of(ball(0))
        for r in self.each():
            gone_depth = r(without_the_ball(), "depth")
            depth = r(scene(hit=dict(visible_to_camera=False)), "depth")
            np.testing.assert_allclose(depth[inside][..., 0], gone_depth[inside][..., 0], rtol=2e-3)
            ids = r(scene(hit=dict(visible_to_camera=False)), "object_id")
            self.assertEqual(int((np.rint(ids[..., 0]) == 1).sum()), 0)              # the sphere is object 1
            self.assertGreater(int((np.rint(r(scene(), "object_id")[..., 0]) == 1).sum()), 20)
            self.assertGreater(float(np.abs(r(scene(), "depth")[inside] - depth[inside]).max()), .1)


class CpuSwitchTests(SwitchTests, unittest.TestCase):
    RENDERERS = CPU


@unittest.skipUnless(gpu3d.available(), "no GPU adapter")
class GpuSwitchTests(SwitchTests, unittest.TestCase):
    RENDERERS = GPU


class CryptomatteTests(unittest.TestCase):
    @staticmethod
    def covered(layers, metadata):
        """The names in the CryptoObject manifest whose id has any coverage in the object set's rank layers."""
        seen = set()
        for key, layer in layers.items():
            if key.startswith("CryptoObject"):
                for id_channel, coverage_channel in ((0, 1), (2, 3)):
                    ids = layer[..., id_channel].view(np.uint32)[layer[..., coverage_channel] > 0]
                    seen |= {format(int(i), "08x") for i in np.unique(ids)}
        return {name for name, bits in metadata["CryptoObject"]["manifest"].items() if bits in seen}

    def test_a_hidden_object_has_no_coverage_in_cryptomatte(self):
        for mode in ("raster", "raytrace"):
            for hidden in (False, True):
                with self.subTest(mode=mode, hidden=hidden):
                    shown = s.Scene((ball(0, name="Ball", visible_to_camera=not hidden), replace(ground(), name="Floor")), (SUN,))
                    layers, metadata = cryptomatte3d.render_cryptomatte(shown, CAMERA, *SIZE, samples=2, mode=mode)
                    covered = self.covered(layers, metadata)
                    self.assertEqual("Ball" in covered, not hidden, covered)
                    self.assertIn("Floor", covered)


class PathTracerLightTests(unittest.TestCase):
    """A hidden mesh is still seen in reflections and still bounces light in the path tracer."""
    MIRROR = dict(material="pbr", metallic=1.0, pbr_roughness=0.0, pbr_specular=1.0)
    PATH = pt.PathSettings(samples=48, max_bounces=4, diffuse_bounces=3, specular_bounces=3)

    def scene(self, hidden=False, present=True):
        red = replace(ball(-.6, visible_to_camera=not hidden), color=(.9, .1, .1, 1))
        floor = replace(ground(), **self.MIRROR)
        wall = replace(s._card(10, 6, (.8, .8, .8, 1), s.Transform3D(s.Vec3(0, 1.5, -3.5))))
        return s.Scene(((red,) if present else ()) + (floor, wall), (SUN,))

    def draw(self, scene, backend):
        return pt.render(scene, CAMERA, *SIZE, BLACK, 0.0, "rgba", self.PATH, backend=backend)

    def check(self, backend):
        shown, hidden, removed = (self.draw(self.scene(), backend), self.draw(self.scene(hidden=True), backend),
                                  self.draw(self.scene(present=False), backend))
        mine = coverage(ball(-.6))
        # everywhere the camera does not see the sphere directly, the hidden one is still there: its reflection in the mirror
        # floor and the light it bounces onto the wall make the picture the one with the sphere, not the one without it
        outside = np.abs(removed - shown)[..., :3].max(axis=-1)
        outside[mine] = 0
        agree = np.abs(hidden - shown)[..., :3].max(axis=-1)
        agree[mine] = 0
        missing, kept = int((outside > .05).sum()), int((agree > .05).sum())
        self.assertGreater(missing, 20, backend)
        self.assertLessEqual(kept, missing // 5, (backend, kept, missing))
        self.assertGreater(float(np.abs(hidden[core_of(ball(-.6))] - shown[core_of(ball(-.6))]).max()), .1)

    def test_cpu(self):
        self.check("cpu")

    @unittest.skipUnless(gpu3d.available(), "no GPU adapter")
    def test_gpu(self):
        self.check("gpu")


@unittest.skipUnless(gpu3d.available(), "no GPU adapter")
class GpuPathTracerAgreesWithTheCpuTests(unittest.TestCase):
    PATH = pt.PathSettings(samples=48, max_bounces=3, diffuse_bounces=2)

    def test_each_switch(self):
        cases = {"cast off": scene(hit=dict(cast_shadows=False)),
                 "receive off": scene(floor=dict(receive_shadows=False)),
                 "hidden": scene(hit=dict(visible_to_camera=False)),
                 "all three": scene(hit=dict(cast_shadows=False, visible_to_camera=False), floor=dict(receive_shadows=False)),
                 "on": scene()}
        for label, case in cases.items():
            with self.subTest(case=label):
                cpu = pt.render(case, CAMERA, *SIZE, BLACK, 0.0, "rgba", self.PATH, backend="cpu")
                gpu = pt.render(case, CAMERA, *SIZE, BLACK, 0.0, "rgba", self.PATH, backend="gpu")
                self.assertLess(float(np.abs(cpu - gpu).mean()), .01 * float(cpu.mean()) + .002)
                for output in ("depth", "object_id"):
                    np.testing.assert_allclose(
                        pt.render(case, CAMERA, *SIZE, BLACK, 0.0, output, self.PATH, backend="gpu"),
                        pt.render(case, CAMERA, *SIZE, BLACK, 0.0, output, self.PATH, backend="cpu"), atol=1e-3, rtol=1e-3)


class SplatTests(unittest.TestCase):
    """A mesh in front of a relit splat sheet: it shadows the sheet unless it casts none, and it is not drawn when hidden."""
    SPLAT_SIZE = (24, 24)

    @staticmethod
    def blocker(**fields):
        return replace(box((.4, .4, .2), (.5, .5, .5, 1), (-.4, .3, 1.0)), **fields)

    def scene(self, blocker=True, **fields):
        sheet = instance(plane_cloud(), relight=1.0)
        return s.Scene((self.blocker(**fields),) if blocker else (), (SPLAT_KEY,), splats=(sheet,))

    def draw(self, scene, mode, gpu):
        return (gpu3d.render if gpu else s.render)(scene, SPLAT_CAMERA, *self.SPLAT_SIZE, ambient=.1, mode=mode)

    def test_the_switches_on_the_cpu_modes(self):
        mesh = s.render(s.Scene((self.blocker(),)), SPLAT_CAMERA, *self.SPLAT_SIZE)[..., 3] > 0
        for mode in ("raster", "raytrace"):
            with self.subTest(mode=mode):
                shown, hidden = self.draw(self.scene(), mode, False), self.draw(self.scene(visible_to_camera=False), mode, False)
                no_cast, empty = self.draw(self.scene(cast_shadows=False), mode, False), self.draw(self.scene(False), mode, False)
                np.testing.assert_allclose(no_cast[~mesh], empty[~mesh], atol=1e-5)       # no shadow on the sheet
                self.assertGreater(float(np.abs(shown[~mesh] - empty[~mesh]).max()), .05)  # which it cast before
                np.testing.assert_allclose(hidden[~mesh], shown[~mesh], atol=1e-5)         # a hidden mesh still shadows it
                self.assertGreater(float(np.abs(hidden[mesh] - shown[mesh]).max()), .05)   # and is not drawn

    @unittest.skipUnless(gpu3d.available(), "no GPU adapter")
    def test_the_gpu_modes_agree_with_the_cpu(self):
        for mode in ("raster", "raytrace"):
            for fields in (dict(cast_shadows=False), dict(visible_to_camera=False), dict(receive_shadows=False)):
                with self.subTest(mode=mode, fields=fields):
                    scene = self.scene(**fields)
                    np.testing.assert_allclose(self.draw(scene, mode, True)[..., :3], self.draw(scene, mode, False)[..., :3],
                                               atol=SPLAT_TOLERANCE, rtol=0)


class InstanceTests(unittest.TestCase):
    def copies(self, **fields):
        matrices = np.tile(np.eye(4), (2, 1, 1))
        matrices[:, 0, 3] = (-1.6, 1.6)
        return s.InstanceSet((ball(0, **fields),), matrices, np.zeros(2, np.int32))

    def test_hidden_instances_leave_the_camera_but_keep_their_shadows_in_every_cpu_renderer(self):
        mine = coverage(ball(-1.6)) | coverage(ball(1.6))
        make = lambda **f: s.Scene((ground(),), (SUN, LAMP), instances=(self.copies(**f),))
        for r in (CPU[0], CPU[1], CPU[2]):
            with self.subTest(renderer=r.name):
                base, hidden = r(make()), r(make(visible_to_camera=False))
                np.testing.assert_allclose(hidden[~mine], base[~mine], atol=r.atol)
                self.assertGreater(float(np.abs(hidden[mine] - base[mine]).max()), .1)
                cast_off = r(make(cast_shadows=False))
                self.assertGreater(float(np.abs(cast_off[~mine] - base[~mine]).max()), .05)

    @unittest.skipUnless(gpu3d.available(), "no GPU adapter")
    def test_the_gpu_instance_renderer_refuses_a_switch_so_auto_falls_back(self):
        scene3 = s.Scene((), (SUN,), instances=(self.copies(cast_shadows=False),))
        with self.assertRaises(gpu3d.Unsupported):
            gpu3d.render(scene3, CAMERA, *SIZE, mode="raytrace")
        plain = s.Scene((), (SUN,), instances=(self.copies(),))
        self.assertEqual(gpu3d.render(plain, CAMERA, *SIZE, mode="raytrace").shape, (SIZE[1], SIZE[0], 4))


@unittest.skipUnless(gpu3d.available(), "no GPU adapter")
class ViewportTests(unittest.TestCase):
    BACKGROUND = (0.02, 0.02, 0.03, 1.0)
    VIEW = s.Camera(s.Transform3D(s.Vec3(0, 4, 9)), s.Vec3(0, 0, 0), 45.0, 0.1, 100.0)

    def setUp(self):
        self.gpu = viewportgpu.renderer()
        self.assertIsNotNone(self.gpu, viewportgpu.failure())

    def draw(self, shadows=True, **fields):
        sun = s.Light("Directional", (1, 1, 1), 1.5, s.Vec3(4, 6, 2), s.Vec3(), shadows=shadows, name="sun")
        cube = replace(s._cube(1.5, (.8, .2, .2, 1), s.Transform3D(s.Vec3(0, .75, 0))), material="pbr", pbr_roughness=.5, **fields)
        floor = replace(s._card(8, 8, (.8, .8, .8, 1), s.Transform3D(s.Vec3(0, -1, 0), s.Vec3(-90, 0, 0))),
                        material="pbr", pbr_roughness=.9)
        return self.gpu.render(s.Scene((floor, cube), (sun,)), self.VIEW, 160, 120, self.BACKGROUND,
                               headlight=False, ambient=.1).astype(int)

    def test_a_cube_that_casts_no_shadow_leaves_the_viewport_shadow_map(self):
        plain, off = self.draw(), self.draw(cast_shadows=False)
        np.testing.assert_array_equal(off, self.draw(shadows=False))               # the shadow map holds no cube
        self.assertGreater(int(np.abs(plain - off).max()), 20)                      # it did cast one

    def test_a_cube_hidden_from_the_camera_is_still_drawn_in_the_viewport(self):
        np.testing.assert_array_equal(self.draw(visible_to_camera=False), self.draw())


class CpuViewportFallbackTests(unittest.TestCase):
    def test_the_cpu_fallback_draws_a_hidden_mesh(self):
        from nodebased import viewport3d
        import inspect
        self.assertIn("visible_to_camera=True", inspect.getsource(viewport3d))


if __name__ == "__main__":
    unittest.main()
