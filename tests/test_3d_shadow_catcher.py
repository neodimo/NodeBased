"""Shadow catcher floors (lane L4, plan "Rendering 7", step R3 of 3).

A mesh with **Shadow catcher** on renders no colour of its own: its alpha is how much of the scene's light the other
objects take away from it (one minus shadowed over unshadowed light), and its colour is black premultiplied by that alpha, so
a Merge over the plate darkens the plate exactly where the shadow falls. The tests state that on the CPU raster and
ray-traced modes, the CPU path tracer, the GPU raster and ray-traced modes and the GPU path tracer, on whichever adapter the
run forces (`run/force-adapter.py default|integrated|cpu`).

The reference every renderer is held to is the artist's own method, automated: render the floor as an ordinary surface with the
object hidden from the camera but still casting (the shadowed floor) and with the object absent (the unshadowed floor); the
alpha is one minus their ratio."""
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from nodebased import core, cryptomatte3d, gpu3d, knobs, pathtrace as pt, scene3d as s, shadowcatch
from nodebased.core import Dispatcher
from nodebased.imaging import Evaluator
from tests.test_3d_light_groups import environment
from tests.test_3d_pathtrace_splats import instance, plane_cloud
from tests.test_volume_scene import box as smoke_box

SIZE = (120, 72)
CAMERA = s.Camera(s.Transform3D(position=s.Vec3(0, 5.0, 8.0)), s.Vec3(0, -0.5, 0), 40.0)
BLACK = (0.0, 0.0, 0.0, 0.0)
SUN = s.Light("Directional", (1, 1, 1), 1.0, s.Vec3(-4, 5, 2), s.Vec3(), shadows=True)
LAMP = s.Light("Point", (1, 1, 1), 40.0, s.Vec3(-3, 4, 2), s.Vec3(), shadows=True)
PANEL = s.Light("Disc", (1, 1, 1), 40.0, s.Vec3(-4, 5, 2), s.Vec3(), area_radius=1.5, light_samples=32, shadows=True)
DIRECT = pt.PathSettings(samples=24, max_bounces=1, diffuse_bounces=1)     # direct light only: exact against the reference
LUMINANCE = np.array((0.2126, 0.7152, 0.0722))
ROOT = Path(__file__).resolve().parents[1]


def floor(y=-1.0, **fields):
    return s.Geometry(np.array([[-8, y, 6], [8, y, 6], [8, y, -8], [-8, y, -8]], "f4"),
                      np.array([[0, 1, 2], [0, 2, 3]], "i4"), fields.pop("color", (.8, .8, .8, 1)), **fields)


def ball(x=0.0, **fields):
    return replace(s._sphere(.8, 24, (.75, .45, .3, 1), s.Transform3D(position=s.Vec3(x, -.2, 0))), **fields)


def scene(lights=(SUN,), catcher=True, **more):
    return s.Scene((ball(), floor(shadow_catcher=catcher)), lights, **more)


class Renderer:
    """One final renderer: `(scene, output) -> picture` and how close two pictures from it must be."""

    def __init__(self, name, draw, gpu=False):
        self.name, self.draw, self.gpu = name, draw, gpu

    def __call__(self, scene, output="rgba", ambient=0.0):
        return self.draw(scene, output, ambient)


def _cpu(mode):
    return lambda scene, output, ambient: s.render(scene, CAMERA, *SIZE, ambient=ambient, mode=mode, output=output, samples=2)


def _gpu(mode):
    return lambda scene, output, ambient: gpu3d.render(scene, CAMERA, *SIZE, ambient=ambient, mode=mode, output=output, samples=2)


def _path(backend):
    return lambda scene, output, ambient: pt.render(scene, CAMERA, *SIZE, BLACK, ambient, output, DIRECT, backend=backend)


CPU = (Renderer("cpu raster", _cpu("raster")), Renderer("cpu ray traced", _cpu("raytrace")),
       Renderer("cpu path tracer", _path("cpu")))
GPU = (Renderer("gpu raster", _gpu("raster"), True), Renderer("gpu ray traced", _gpu("raytrace"), True),
       Renderer("gpu path tracer", _path("gpu"), True))


def luminance(picture):
    return picture[..., :3] @ LUMINANCE


def reference(draw, lights=(SUN,), ambient=0.0, extra=()):
    """(alpha, floor mask) the artist's way: the shadowed floor and the unshadowed floor as ordinary surfaces. The object is
    hidden from the camera in the first (it still casts), absent in the second; `extra` meshes are in both."""
    shadowed = draw(s.Scene((ball(visible_to_camera=False), floor(), *extra), lights), "rgba", ambient)
    free = draw(s.Scene((floor(), *extra), lights), "rgba", ambient)
    with np.errstate(divide="ignore", invalid="ignore"):
        alpha = np.where(luminance(free) > 1e-9, 1.0 - luminance(shadowed) / np.maximum(luminance(free), 1e-9), 0.0)
    return np.clip(alpha, 0, 1), free[..., 3] > .99


def ball_cover(x=0.0):
    alone = s.render(s.Scene((ball(x),)), CAMERA, *SIZE, mode="raster")[..., 3] > 0
    grown = alone.copy()
    for axis in (0, 1):
        for step in (-1, 1):
            grown |= np.roll(alone, step, axis)
    return grown


def inner(mask):
    shrunk = mask.copy()
    for axis in (0, 1):
        for step in (-1, 1):
            shrunk &= np.roll(mask, step, axis)
    return shrunk


class Fields(unittest.TestCase):
    KINDS = ("Card3D", "Cube3D", "Sphere3D", "Cylinder3D", "ReadGeo3D", "ReadGLTF3D")

    def test_every_mesh_node_has_the_switch_off_by_default(self):
        for kind in self.KINDS:
            self.assertEqual(core.SPECS[kind]["params"]["shadow_catcher"], "off", kind)
        self.assertEqual(core.CHOICES["shadow_catcher"], ["off", "on"])

    def test_the_knob_is_laid_out_beside_the_other_switches(self):
        for kind in self.KINDS:
            labels = [g.label for g in knobs.KNOB_LAYOUT[kind]]
            self.assertIn("Shadow catcher", labels, kind)
            self.assertEqual(labels.index("Shadow catcher"), labels.index("Holdout") + 1, kind)

    def test_the_param_becomes_a_geometry_field(self):
        self.assertFalse(s.shadow_catcher_from_params({}))
        self.assertFalse(s.shadow_catcher_from_params(dict(shadow_catcher="off")))
        self.assertTrue(s.shadow_catcher_from_params(dict(shadow_catcher="on")))
        self.assertTrue(s.has_shadow_catchers(scene()))
        self.assertFalse(s.has_shadow_catchers(scene(catcher=False)))
        self.assertTrue(s.has_render_switches(s.Scene((floor(shadow_catcher=True),))))

    def test_an_instance_copy_takes_its_sources_switch(self):
        matrices = np.tile(np.eye(4), (2, 1, 1))
        matrices[:, 0, 3] = (-2, 2)
        copies = s.expand_instances(s.InstanceSet((floor(shadow_catcher=True),), matrices, np.zeros(2, np.int32)))
        self.assertTrue(all(copy.shadow_catcher for copy in copies))
        self.assertTrue(s.has_shadow_catchers(s.Scene(instances=(s.InstanceSet((floor(shadow_catcher=True),), matrices,
                                                                                np.zeros(2, np.int32)),))))

    def graph(self, **params):
        d = Dispatcher()
        for key, kind, p in (("ball", "Sphere3D", dict(sphere_radius=.8, ty=-.2)),
                             ("floor", "Card3D", dict(card_width=16, card_height=14, ty=-1, tz=-1, rx=-90, red=.8, green=.8,
                                                      blue=.8, **params)),
                             ("sun", "Light3D", dict(shadows="on", light_type="Directional", tx=-4, ty=5, tz=2)),
                             ("camera", "Camera3D", dict(ty=5, tz=8, target_y=-.5)),
                             ("scene", "Scene3D", {}), ("render", "Render3D", dict(width=96, height=60, samples=1))):
            d.execute(dict(op="create", id=key, type=kind, params=p))
        for slot, source in (("object0", "ball"), ("object1", "floor"), ("object2", "sun")):
            d.execute(dict(op="connect", id="scene", input=slot, source=source))
        for slot in ("scene", "camera"):
            d.execute(dict(op="connect", id="render", input=slot, source=slot))
        return d

    def test_the_node_carries_the_switch_into_the_render(self):
        d = self.graph(shadow_catcher="on")
        scene3 = Evaluator().evaluate_raster(d.document, "scene", typed=True)
        self.assertEqual([g.shadow_catcher for g in scene3.geometries], [False, True])
        caught = Evaluator().evaluate_raster(d.document, "render").pixels
        plain = Evaluator().evaluate_raster(self.graph().document, "render").pixels
        self.assertGreater(float(plain[..., :3].max()), .1)
        # the floor adds only alpha (its colour is black); the ball keeps its colour
        floor_pixels = (plain[..., 3] > .99) & ~ball_object(plain)
        self.assertTrue(np.all(caught[floor_pixels][..., :3] == 0))
        self.assertGreater(float(caught[floor_pixels][..., 3].max()), .5)

    def test_a_document_without_the_switch_renders_as_one_with_the_default(self):
        d = self.graph()
        new = Evaluator().evaluate_raster(d.document, "render").pixels
        d.document["nodes"]["floor"]["params"].pop("shadow_catcher")
        old = Evaluator().evaluate_raster(d.document, "render").pixels
        np.testing.assert_array_equal(old, new)
        self.assertGreater(float(new.max()), .1)

    def test_the_viewport_draws_a_catcher_as_the_floor_it_is(self):
        # the viewport reads `Geometry` meshes without the flag, so a catcher is an ordinary surface there
        import inspect
        from nodebased import viewport3d, viewportgpu
        for module in (viewport3d, viewportgpu):
            self.assertNotIn("shadow_catcher=True", inspect.getsource(module))


def ball_object(picture):
    """The pixels a lone ball covers (grown by a pixel), at `picture`'s size."""
    alone = s.render(s.Scene((ball(),)), CAMERA, picture.shape[1], picture.shape[0], mode="raster")[..., 3] > 0
    grown = alone.copy()
    for axis in (0, 1):
        for step in (-1, 1):
            grown |= np.roll(alone, step, axis)
    return grown


class CatcherTests:
    """The properties, for each renderer of `RENDERERS` (set by the subclasses below)."""
    RENDERERS = ()
    TOLERANCE = dict(mean=.003, edge=.45)

    def each(self):
        for renderer in self.RENDERERS:
            with self.subTest(renderer=renderer.name):
                yield renderer

    def check_against_reference(self, r, lights=(SUN,), ambient=0.0, mean=None, edge=None, extra=(), **more):
        got = r(s.Scene((ball(), floor(shadow_catcher=True), *extra), lights, **more), "rgba", ambient)
        alpha, on_floor = reference(r, lights, ambient, extra)
        mask = on_floor & ~ball_cover()
        difference = np.abs(got[..., 3] - alpha)[mask]
        self.assertGreater(float(alpha[mask].max()), .3, "the object does shadow the floor")
        self.assertLess(float(difference.mean()), mean if mean is not None else self.TOLERANCE["mean"], r.name)
        self.assertLess(float(difference.max()), edge if edge is not None else self.TOLERANCE["edge"], r.name)
        return got, alpha, mask

    def test_a_sphere_over_a_catcher_gives_the_reference_shadow_fraction_under_one_light(self):
        for r in self.each():
            got, alpha, mask = self.check_against_reference(r)
            shadow = inner(alpha > .99) & mask
            lit = inner(alpha < .001) & mask
            self.assertGreater(int(shadow.sum()), 30)
            self.assertGreater(int(lit.sum()), 1000)
            self.assertGreater(float(got[..., 3][shadow].min()), .985, "alpha 1 in the shadow")
            self.assertEqual(float(got[..., 3][lit].max()), 0.0, "alpha 0 in open light")

    def test_the_catcher_is_black_premultiplied_by_its_alpha(self):
        for r in self.each():
            got = r(scene())
            covered = (r(scene(), "depth")[..., 3] > 0) & ~ball_cover()
            self.assertGreater(int(covered.sum()), 1000)
            self.assertTrue(np.all(got[covered][..., :3] == 0), "no colour of its own")
            self.assertGreater(float(got[covered][..., 3].max()), .9)

    def test_the_object_is_unchanged_by_the_catcher(self):
        plain = scene(catcher=False)
        for r in self.each():
            with_floor = r(scene())
            without = r(s.Scene((ball(),), (SUN,)))
            core_pixels = inner(ball_cover() & (without[..., 3] > .999)) & inner(inner(ball_cover()))
            self.assertGreater(int(core_pixels.sum()), 50)
            # the ball is lit as it is with an ordinary floor under it (the floor's shadow-free bounce is not in the direct picture)
            np.testing.assert_allclose(with_floor[core_pixels], r(plain)[core_pixels], atol=2e-2 if "path" in r.name else 1e-5)

    def test_a_catcher_nothing_shadows_is_fully_transparent(self):
        alone = s.Scene((floor(shadow_catcher=True),), (SUN, LAMP, PANEL))
        for r in self.each():
            picture = r(alone)
            self.assertTrue(np.all(picture == 0), r.name)
            self.assertTrue(np.all(r(alone, ambient=.2) == 0))

    def test_ambient_light_dilutes_the_shadow_as_it_does_a_plate(self):
        for r in self.each():
            r_alpha = reference(r, ambient=.15)[0]
            got, _, mask = self.check_against_reference(r, ambient=.15, mean=.01)
            shadow = inner(r_alpha > .5) & mask
            self.assertGreater(int(shadow.sum()), 30)
            self.assertLess(float(got[..., 3][shadow].max()), .99, "an ambient fill leaves the shadow lighter than black")
            self.assertGreater(float(got[..., 3][shadow].mean()), .6)

    def test_a_point_light_throws_a_catcher_shadow(self):
        for r in self.each():
            self.check_against_reference(r, lights=(LAMP,))

    def test_a_soft_area_light_gives_a_soft_alpha_edge(self):
        for r in self.each():
            got, alpha, mask = self.check_against_reference(r, lights=(PANEL,), mean=.012, edge=.5)
            hard = r(scene())
            soft_pixels = int(((got[..., 3] > .08) & (got[..., 3] < .92) & mask).sum())
            hard_pixels = int(((hard[..., 3] > .08) & (hard[..., 3] < .92) & mask).sum())
            self.assertGreater(soft_pixels, 2 * hard_pixels + 20, f"penumbra {soft_pixels} against a hard edge's {hard_pixels}")
            # the alpha across the penumbra falls monotonically from the umbra to the open floor (rows through the ball's shadow)
            row = got[..., 3][np.nonzero(inner(alpha > .99) & mask)[0][0]]
            self.assertGreater(float(row.max()), .95)

    def test_a_catcher_that_takes_no_shadow_is_transparent(self):
        for r in self.each():
            picture = r(s.Scene((ball(), floor(shadow_catcher=True, receive_shadows=False)), (SUN,)))
            floor_only = (r(s.Scene((ball(), floor(shadow_catcher=True)), (SUN,)), "depth")[..., 3] > 0) & ~ball_cover()
            self.assertTrue(np.all(picture[floor_only] == 0))

    def test_an_object_that_casts_no_shadow_leaves_the_catcher_clear(self):
        for r in self.each():
            picture = r(s.Scene((ball(cast_shadows=False), floor(shadow_catcher=True)), (SUN,)))
            self.assertEqual(float(picture[..., 3][~ball_cover()].max()), 0.0)

    def test_the_depth_pass_marks_the_floor_and_the_id_pass_does_not_know_it(self):
        for r in self.each():
            depth = r(scene(), "depth")
            plain = r(scene(catcher=False), "depth")
            np.testing.assert_allclose(depth, plain, atol=1e-5)
            on_floor = (depth[..., 3] > 0) & ~ball_cover()
            self.assertGreater(int(on_floor.sum()), 1000)
            self.assertGreater(float(depth[on_floor][..., 0].min()), 1.0, "the ground has depth")
            ids = r(scene(), "object_id")
            floor_id = np.rint(ids[..., 0])
            self.assertEqual(int((floor_id == 2).sum()), 0, "the catcher (object 2) never appears in the id pass")
            self.assertTrue(np.all(ids[on_floor & ~ball_cover()] == 0))
            self.assertGreater(int((floor_id == 1).sum()), 50, "the ball does")
            for output in ("position", "normals", "uv"):
                np.testing.assert_allclose(r(scene(), output), r(scene(catcher=False), output), atol=1e-4)

    def test_a_translucent_object_leaves_the_catcher_shadow_behind_it(self):
        veil = s.Geometry(np.array([[-2, -.9, 1.5], [2, -.9, 1.5], [2, 1.2, 1.5], [-2, 1.2, 1.5]], "f4"),
                          np.array([[0, 1, 2], [0, 2, 3]], "i4"), (.2, .3, .9, .5), cast_shadows=False)
        pane = s.render(s.Scene((veil,)), CAMERA, *SIZE, mode="raster")[..., 3]
        behind = inner(inner(pane > .45)) & ~ball_cover()
        for r in self.each():
            held = r(s.Scene((ball(), floor(shadow_catcher=True), veil), (SUN,)))[..., 3]
            bare = r(s.Scene((ball(), floor(shadow_catcher=True)), (SUN,)))[..., 3]
            own = r(s.Scene((veil,), (SUN,)))[..., 3]                  # the pane's own alpha in this renderer
            shadow, open_floor = behind & (bare > .98), behind & (bare < .001)
            self.assertGreater(int(shadow.sum()), 20, "part of the ball's shadow lies behind the pane")
            self.assertGreater(int(open_floor.sum()), 100)
            # the pane's alpha over the catcher's: the catcher's shadow shows through the pane
            pane_open, pane_shadow = float(own[open_floor].mean()), float(own[shadow].mean())
            self.assertGreater(pane_open, .2)
            self.assertAlmostEqual(float(held[open_floor].mean()), pane_open, delta=.04)
            self.assertAlmostEqual(float(held[shadow].mean()), pane_shadow + (1 - pane_shadow), delta=.04)

    def test_merged_over_a_plate_the_shadow_darkens_it_and_nothing_else(self):
        plate = np.ones(SIZE[::-1] + (4,), np.float32)
        plate[..., :3] = (.7, .6, .4)
        for r in self.each():
            picture = np.asarray(r(scene()))
            merged = picture + plate * (1.0 - picture[..., 3:4])
            alpha, on_floor = reference(r)
            mask = on_floor & ~ball_cover()
            lit = mask & (alpha < .001)
            shadow = inner(mask & (alpha > .99))
            np.testing.assert_allclose(merged[lit], plate[lit], atol=1e-6)               # open floor: the plate untouched
            self.assertLess(float(merged[shadow][..., :3].max()), .02)                  # the shadow: black on the plate
            self.assertGreater(float(merged[shadow][..., 3].min()), .99)
            partial = mask & (alpha > .2) & (alpha < .8)
            if partial.any():                                                            # the plate is darkened by exactly `alpha`
                np.testing.assert_allclose(merged[partial][..., :3], plate[partial][..., :3] * (1 - picture[partial][..., 3:4]),
                                           atol=1e-6)


class CpuCatcherTests(CatcherTests, unittest.TestCase):
    RENDERERS = CPU


@unittest.skipUnless(gpu3d.available(), "no GPU adapter")
class GpuCatcherTests(CatcherTests, unittest.TestCase):
    RENDERERS = GPU


@unittest.skipUnless(gpu3d.available(), "no GPU adapter")
class GpuAgreesWithTheCpuTests(unittest.TestCase):
    """The GPU raster and ray-traced modes match the CPU's mode, the GPU path tracer the CPU's, with a catcher in the scene."""
    PATH = pt.PathSettings(samples=48, max_bounces=2, diffuse_bounces=2)

    def cases(self):
        return {"one light": scene(), "soft light": scene((PANEL,)), "point light": scene((LAMP,)),
                "catcher alone": s.Scene((floor(shadow_catcher=True),), (SUN,)),
                "holdout beside it": s.Scene((ball(), floor(shadow_catcher=True),
                                              s.Geometry(np.array([[1, -1, 2], [4, -1, 2], [4, 2, 2], [1, 2, 2]], "f4"),
                                                         np.array([[0, 1, 2], [0, 2, 3]], "i4"), (.2, .9, .2, 1), holdout=True)),
                                             (SUN,))}

    def test_raster_and_ray_traced(self):
        for label, case in self.cases().items():
            for mode in ("raster", "raytrace"):
                with self.subTest(case=label, mode=mode):
                    cpu, gpu = _cpu(mode)(case, "rgba", .05), _gpu(mode)(case, "rgba", .05)
                    wrong = (np.abs(gpu - cpu) > 5e-4 + 1e-3 * np.abs(cpu)).any(axis=-1)
                    self.assertLessEqual(int(wrong.sum()), 1 + cpu.shape[0] * cpu.shape[1] // 2000, int(wrong.sum()))
                    self.assertLess(float(np.abs(gpu - cpu).mean()), 2e-4 + 1e-3 * float(cpu.mean()))

    def test_the_path_tracer(self):
        for label, case in self.cases().items():
            with self.subTest(case=label):
                cpu = pt.render(case, CAMERA, *SIZE, BLACK, 0.0, "rgba", self.PATH, backend="cpu")
                gpu = pt.render(case, CAMERA, *SIZE, BLACK, 0.0, "rgba", self.PATH, backend="gpu")
                self.assertLess(float(np.abs(cpu[..., 3] - gpu[..., 3]).mean()), .004)
                self.assertLess(float(np.abs(cpu - gpu).mean()), .01 * float(cpu.mean()) + .004)

    def test_what_the_gpu_cannot_hold_with_a_catcher_it_refuses(self):
        smoky = s.Scene((floor(shadow_catcher=True),), (SUN,), volumes=(smoke_box(n=8),))
        with self.assertRaises(gpu3d.Unsupported):
            gpu3d.render(smoky, CAMERA, 32, 20, mode="raster")
        matrices = np.tile(np.eye(4), (2, 1, 1))
        matrices[:, 0, 3] = (-2, 2)
        instanced = s.Scene(instances=(s.InstanceSet((floor(shadow_catcher=True),), matrices, np.zeros(2, np.int32)),),
                            lights=(SUN,))
        picture = gpu3d.render(instanced, CAMERA, 32, 20, mode="raytrace")      # instances are expanded, then it renders
        self.assertEqual(picture.shape, (20, 32, 4))


class OtherCasterTests(unittest.TestCase):
    """Smoke, splats and meshes all shadow a catcher the way they shadow any mesh."""
    SMALL = (64, 40)
    CAM = s.Camera(s.Transform3D(position=s.Vec3(0, 3.0, 6.0)), s.Vec3(0, 0, 0), 40.0)
    TOP = s.Light("Directional", (1, 1, 1), 1.0, s.Vec3(0, 6, 0), s.Vec3(), shadows=True)

    def floor_alpha(self, extra, mode, lights=None, **more):
        catcher = floor(y=-1.0, shadow_catcher=True)
        scene_ = s.Scene((catcher,), lights or (self.TOP,), **{**more, **extra})
        if mode == "pathtrace":
            return pt.render(scene_, self.CAM, *self.SMALL, BLACK, 0.0, "rgba", pt.PathSettings(samples=12, max_bounces=1,
                                                                                               diffuse_bounces=1))[..., 3]
        return s.render(scene_, self.CAM, *self.SMALL, mode=mode, samples=1)[..., 3]

    def test_smoke_shadows_the_catcher(self):
        thick = smoke_box(n=12, density=40.0, size=1.2, matrix=np.eye(4, dtype=np.float32))
        for mode in ("raster", "raytrace", "pathtrace"):
            with self.subTest(mode=mode):
                clear = self.floor_alpha({}, mode)
                shaded = self.floor_alpha(dict(volumes=(thick,)), mode)
                self.assertEqual(float(clear.max()), 0.0)
                self.assertGreater(float(shaded.max()), .3, "the smoke darkens the floor under it")
                self.assertGreater(int((shaded > .05).sum()), 12)

    def test_a_splat_sheet_shadows_the_catcher(self):
        sheet = instance(plane_cloud(n=16, size=1.5, opacity=.99), relight=0.0)
        flat = replace(sheet, matrix=np.array([[1, 0, 0, 0], [0, 0, 1, 0.3], [0, -1, 0, 0], [0, 0, 0, 1]], np.float32))
        for mode in ("raster", "raytrace", "pathtrace"):
            with self.subTest(mode=mode):
                shaded = self.floor_alpha(dict(splats=(flat,)), mode)
                self.assertGreater(float(shaded.max()), .5, "the splat sheet shadows the floor")
                self.assertGreater(int((shaded > .1).sum()), 10)

    def test_an_excluded_light_leaves_no_shadow(self):
        # light linking: a ball excluded from the sun neither shadows the floor from it nor is shadowed by it
        unlit = (ball(light_link=("exclude", ("sun",))),)
        sun = replace(SUN, name="sun")
        for mode in ("raster", "raytrace", "pathtrace"):
            with self.subTest(mode=mode):
                scene_ = s.Scene((*unlit, floor(shadow_catcher=True)), (sun,))
                if mode == "pathtrace":
                    picture = pt.render(scene_, CAMERA, *SIZE, BLACK, 0.0, "rgba", DIRECT)
                else:
                    picture = s.render(scene_, CAMERA, *SIZE, mode=mode)
                self.assertEqual(float(picture[..., 3][~ball_cover()].max()), 0.0)

    def test_environment_light_in_the_path_tracer_gives_a_soft_contact_shadow(self):
        env = environment()
        scene_ = s.Scene((ball(), floor(shadow_catcher=True)), (), environments=(env,))
        got = pt.render(scene_, CAMERA, *SIZE, BLACK, 0.0, "rgba", pt.PathSettings(samples=64, max_bounces=1, diffuse_bounces=1))
        alpha = got[..., 3]
        under = (alpha > .02) & ~ball_cover()
        self.assertGreater(int(under.sum()), 15, "the sky is blocked under the ball")
        self.assertLess(float(alpha[~ball_cover()].max()), .99, "an HDRI shadow is soft, never black")
        self.assertGreater(float(alpha[under].max()), .15)
        # the raster modes do not shadow an environment: it only lights, so the floor stays clear
        self.assertEqual(float(s.render(scene_, CAMERA, *SIZE, mode="raster")[..., 3][~ball_cover()].max()), 0.0)


class CompositingTests(unittest.TestCase):
    """The catcher with a holdout, the returned depth, passes of a multichannel render, motion blur and instances."""

    def test_a_holdout_and_a_catcher_in_one_scene(self):
        post = s.Geometry(np.array([[.3, -1, 3], [.8, -1, 3], [.8, 2, 3], [.3, 2, 3]], "f4"),
                          np.array([[0, 1, 2], [0, 2, 3]], "i4"), (.2, .9, .2, 1), holdout=True, cast_shadows=False)
        both = s.Scene((ball(), floor(shadow_catcher=True), post), (SUN,))
        post_shape = replace(post, holdout=False)
        for mode in ("raster", "raytrace"):
            with self.subTest(mode=mode):
                picture = s.render(both, CAMERA, *SIZE, mode=mode)
                alone = s.render(s.Scene((post_shape,)), CAMERA, *SIZE, mode="raster")[..., 3] > .999
                cut = inner(alone)
                self.assertGreater(int(cut.sum()), 30)
                self.assertTrue(np.all(picture[cut] == 0), "the holdout cuts the catcher's shadow too")
                clear = s.render(s.Scene((ball(), floor(shadow_catcher=True)), (SUN,)), CAMERA, *SIZE, mode=mode)
                np.testing.assert_allclose(picture[~alone & ~np.roll(alone, 1, 1) & ~np.roll(alone, -1, 1)],
                                           clear[~alone & ~np.roll(alone, 1, 1) & ~np.roll(alone, -1, 1)], atol=1e-6)

    def test_return_depth_sees_the_floor(self):
        for mode in ("raster", "raytrace"):
            with self.subTest(mode=mode):
                image, depth = s.render(scene(), CAMERA, *SIZE, mode=mode, return_depth=True)
                plain_image, plain_depth = s.render(scene(catcher=False), CAMERA, *SIZE, mode=mode, return_depth=True)
                np.testing.assert_allclose(depth, plain_depth, atol=1e-5)
                self.assertTrue(np.isfinite(depth[(image[..., 3] > .5) & ~ball_cover()]).all())
        image, depth = s.render(scene(), CAMERA, *SIZE, mode="pathtrace", return_depth=True,
                                path=pt.PathSettings(samples=4, max_bounces=1, diffuse_bounces=1))
        self.assertTrue(np.isfinite(depth[(image[..., 3] > .5) & ~ball_cover()]).all())

    def test_the_background_colour_goes_on_under_the_shadow(self):
        background = (.4, .5, .6, 1.0)
        for mode in ("raster", "raytrace"):
            with self.subTest(mode=mode):
                with_bg = s.render(scene(), CAMERA, *SIZE, background, mode=mode)
                clear = s.render(scene(), CAMERA, *SIZE, mode=mode)
                np.testing.assert_allclose(with_bg, clear + np.array(background, np.float32) * np.float32(1) * (1 - clear[..., 3:4]),
                                           atol=1e-5)
                shadow = inner(clear[..., 3] > .99) & ~ball_cover()
                self.assertLess(float(with_bg[shadow][..., :3].max()), .01, "the shadow darkens the backdrop under it")

    def test_a_multichannel_render_has_the_floor_in_depth_and_not_in_the_object_id_pass(self):
        beauty, layers = s.render_multichannel(scene(), CAMERA, *SIZE, passes="beauty,depth,object_id", mode="raster")
        floor_pixels = (layers["depth"][..., 3] > 0) & ~ball_cover()
        self.assertGreater(int(floor_pixels.sum()), 1000)
        self.assertTrue(np.all(layers["object_id"][floor_pixels] == 0))
        self.assertGreater(float(beauty[..., 3][floor_pixels].max()), .9)
        self.assertTrue(np.all(beauty[floor_pixels][..., :3] == 0))

    def test_cryptomatte_never_names_a_catcher(self):
        named = replace(floor(shadow_catcher=True), name="Road")
        layers, metadata = cryptomatte3d.render_cryptomatte(s.Scene((replace(ball(), name="Ball"), named), (SUN,)), CAMERA, 48, 30)
        manifest = metadata["CryptoObject"]["manifest"]
        self.assertIn("Ball", manifest)
        self.assertNotIn("Road", manifest)
        ids = layers["CryptoObject00"][..., 0].view(np.uint32)
        self.assertEqual(int((ids == int(manifest["Ball"], 16)).any()), 1)

    def test_motion_blurred_path_tracing_keeps_the_catcher(self):
        moments = [(s.Scene((ball(x), floor(shadow_catcher=True)), (SUN,)), CAMERA) for x in (-.4, 0.0, .4)]
        picture = pt.render_motion(moments, 48, 30, BLACK, 0.0, "rgba", pt.PathSettings(samples=12, max_bounces=1, diffuse_bounces=1))
        self.assertEqual(picture.shape, (30, 48, 4))
        self.assertGreater(float(picture[..., 3].max()), .9)
        self.assertTrue(np.all(picture[..., :3][picture[..., 3] < .01] == 0))

    def test_the_denoised_path_tracer_keeps_the_catcher(self):
        picture = pt.render(scene(), CAMERA, 48, 30, BLACK, 0.0, "denoise", pt.PathSettings(samples=8, max_bounces=1, diffuse_bounces=1))
        self.assertGreater(float(picture[..., 3].max()), .9)

    def test_instances_of_a_catcher_render_on_the_cpu(self):
        matrices = np.tile(np.eye(4), (1, 1, 1))
        instanced = s.Scene((ball(),), (SUN,), instances=(s.InstanceSet((floor(shadow_catcher=True),), matrices,
                                                                        np.zeros(1, np.int32)),))
        picture = s.render(instanced, CAMERA, *SIZE, mode="raster")
        reference_picture = s.render(scene(), CAMERA, *SIZE, mode="raster")
        np.testing.assert_allclose(picture, reference_picture, atol=1e-6)

    def test_shadows_off_leaves_the_catcher_clear(self):
        picture = s.render(scene(), CAMERA, *SIZE, mode="raster", shadows=False)
        self.assertEqual(float(picture[..., 3][~ball_cover()].max()), 0.0)

    def test_the_inspection_headlight_draws_a_catcher_as_the_floor(self):
        shaded = s.render(scene(), CAMERA, *SIZE, shade=True)
        self.assertGreater(float(shaded[..., :3][~ball_cover()].max()), .1)


class LayerTests(unittest.TestCase):
    """`shadowcatch`'s own pieces: the two scene rewrites and the shadow layer."""

    def test_the_scene_rewrites_touch_only_the_catchers(self):
        held, plain = shadowcatch.hold_scene(scene()), shadowcatch.plain_scene(scene())
        self.assertEqual([(g.holdout, g.shadow_catcher) for g in held.geometries], [(False, False), (True, False)])
        self.assertEqual([(g.holdout, g.shadow_catcher) for g in plain.geometries], [(False, False), (False, False)])
        self.assertFalse(shadowcatch.active(held) or shadowcatch.active(plain))
        self.assertTrue(shadowcatch.active(scene()))

    def test_the_layer_is_the_floor_where_the_camera_sees_it_and_its_shadow(self):
        for engine in ("raster", "path"):
            with self.subTest(engine=engine):
                cover, shadow = shadowcatch.shadow_layer(scene(), CAMERA, *SIZE, engine=engine, samples=4)
                self.assertEqual(cover.shape, SIZE[::-1])
                self.assertTrue(((cover >= 0) & (cover <= 1)).all() and ((shadow >= 0) & (shadow <= 1)).all())
                grid = shadowcatch._grid(engine, 4)
                seen = s.render(s.Scene((floor(),)), CAMERA, *SIZE, mode="raster", samples=grid)[..., 3]
                np.testing.assert_allclose(cover, seen, atol=1e-6)        # the floor, antialiased on the layer's own grid
                self.assertGreater(float(shadow[seen > .99].max()), .99)
                self.assertEqual(float(shadow[seen == 0].max()), 0.0)

    def test_path_shadow_quality_budget_is_independent_of_picture_samples(self):
        measure = lambda _i, _g, points, _n: (np.ones(len(points)), np.ones(len(points)))
        with patch.object(shadowcatch, "_path_measure", return_value=measure) as build, \
             patch.object(shadowcatch, "_grid", wraps=shadowcatch._grid) as grid:
            shadowcatch.shadow_layer(scene(), CAMERA, *SIZE, engine="path", samples=128, shadow_samples=1)
        self.assertEqual(build.call_args.args[3], 128)
        self.assertEqual(build.call_args.kwargs["shadow_samples"], 1)
        self.assertEqual(grid.call_args.args, ("path", 1))

    def test_coarse_path_shadow_map_is_upsampled_to_the_picture_size(self):
        cover, shadow = shadowcatch.shadow_layer(scene(), CAMERA, *SIZE, engine="path", samples=4,
                                                  shadow_samples=1, shadow_resolution_scale=1.5)
        self.assertEqual(cover.shape, SIZE[::-1])
        self.assertEqual(shadow.shape, SIZE[::-1])
        self.assertTrue(np.isfinite(cover).all() and np.isfinite(shadow).all())

    def test_gpu_path_catcher_uses_its_own_quality_budget(self):
        with patch.object(shadowcatch, "render", return_value=np.zeros((*SIZE[::-1], 4), np.float32)) as wrapper:
            pt.render(scene(), CAMERA, 960, 540, BLACK, 0.0, "rgba", pt.PathSettings(samples=128), backend="gpu")
        self.assertEqual(wrapper.call_args.kwargs["shadow_samples"], 1)
        self.assertEqual(wrapper.call_args.kwargs["shadow_resolution_scale"], 1.5)

    def test_small_gpu_path_catcher_keeps_its_reference_budget(self):
        with patch.object(shadowcatch, "render", return_value=np.zeros((*SIZE[::-1], 4), np.float32)) as wrapper:
            pt.render(scene(), CAMERA, *SIZE, BLACK, 0.0, "rgba", pt.PathSettings(samples=24), backend="gpu")
        self.assertIsNone(wrapper.call_args.kwargs["shadow_samples"])
        self.assertEqual(wrapper.call_args.kwargs["shadow_resolution_scale"], 1.0)

    def test_a_mesh_that_is_no_catcher_is_left_alone_by_the_wrapper(self):
        plain = scene(catcher=False)
        np.testing.assert_array_equal(s.render(plain, CAMERA, *SIZE, mode="raster"),
                                      s.render(s.Scene(plain.geometries, plain.lights), CAMERA, *SIZE, mode="raster"))


class ExampleProjectTests(unittest.TestCase):
    FOLDER = ROOT / "examples" / "shadow_catcher"

    def test_the_example_loads_and_renders_cg_onto_the_plate(self):
        document = core.load_document(self.FOLDER / "shadow_catcher.nbcomp")
        kinds = {node["type"] for node in document["nodes"].values()}
        self.assertTrue({"Read", "Merge", "Render3D", "Sphere3D", "Card3D", "Light3D", "Camera3D"} <= kinds)
        catchers = [n for n in document["nodes"].values() if n["params"].get("shadow_catcher") == "on"]
        holdouts = [n for n in document["nodes"].values() if n["params"].get("holdout") == "on"]
        self.assertEqual((len(catchers), len(holdouts)), (1, 1))
        evaluator = Evaluator()
        merged = evaluator.evaluate_raster(document, "merge").pixels
        plate = evaluator.evaluate_raster(document, "plate").pixels
        cg = evaluator.evaluate_raster(document, "render").pixels
        self.assertEqual(merged.shape, plate.shape)
        shadow = (cg[..., 3] > .5) & (cg[..., :3].max(axis=-1) == 0)
        self.assertGreater(int(shadow.sum()), 100, "the catcher puts a shadow in the picture")
        self.assertTrue(np.all(merged[shadow][..., :3] <= plate[shadow][..., :3] + 1e-6))
        self.assertLess(float(merged[shadow][..., :3].mean()), float(plate[shadow][..., :3].mean()) * .6)
        clear = cg[..., 3] == 0
        np.testing.assert_allclose(merged[clear], plate[clear], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
