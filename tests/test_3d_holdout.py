"""Holdout objects (lane L4, plan "Rendering 7", step R2 of 3).

A mesh with **Holdout** on still hides whatever lies behind it from the camera, but is drawn as nothing: zero colour, zero
alpha, no depth, position, normal or id. It casts its shadow as its switches say, and every ray but the camera's passes
through it. The tests state that on the CPU raster and ray-traced modes, the CPU path tracer, the GPU raster and ray-traced
modes and the GPU path tracer, on whichever adapter the run forces (`run/force-adapter.py default|integrated|cpu`)."""
import unittest
from dataclasses import replace

import numpy as np

from nodebased import core, cryptomatte3d, gpu3d, gpuinstance, knobs, pathtrace as pt, scene3d as s
from nodebased.core import Dispatcher
from nodebased.imaging import Evaluator
from tests import gpu_precision
from tests.test_3d_light_linking import SUN, LAMP, ground
from tests.test_3d_pathtrace_splats import SPLAT_CAMERA, instance, plane_cloud
from tests.test_3d_render_switches import AMBIENT, BLACK, DIRECT, Renderer
from tests.test_volume_scene import box as smoke_box

HOLD_COLOR = (.2, .9, .2, 1.0)
EDGE_SAMPLES = 4
SIZE = (96, 60)
CAMERA = s.Camera(s.Transform3D(position=s.Vec3(0, 2.0, 6.0)), s.Vec3(0, 0, 0), 40.0)


def ball(x=0.0, **fields):
    """A ball 15 pixels in radius at the middle of the picture."""
    return replace(s._sphere(1.2, 32, (.75, .45, .3, 1), s.Transform3D(position=s.Vec3(x, .3, 0))), **fields)


def coverage(geometry, camera=CAMERA, size=SIZE):
    """Where the object itself is: the alpha of the object rendered alone (grown by a pixel so a silhouette edge belongs to it)."""
    alone = s.render(s.Scene((geometry,)), camera, *size, mode="raster")[..., 3] > 0
    grown = alone.copy()
    for axis in (0, 1):
        for step in (-1, 1):
            grown |= np.roll(alone, step, axis)
    return grown


def core_of(geometry, camera=CAMERA, size=SIZE):
    """The object's fully covered pixels whose four neighbours are covered too: pure object, no edge."""
    full = s.render(s.Scene((geometry,)), camera, *size, mode="raster")[..., 3] > .999
    return surrounded(full)


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


def quad(x0, x1, y0, y1, z, color=HOLD_COLOR, **fields):
    """A card facing the camera at depth `z` (the camera looks along -z from z = 6)."""
    return s.Geometry(np.array([[x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z]], "f4"),
                      np.array([[0, 1, 2], [0, 2, 3]], "i4"), color, **fields)


def holdout(**fields):
    """A card in front of the ball (the camera is at z = 6, the ball at z = 0) covering its right half and more; its left
    edge is slanted, so that the cut crosses the ball's pixels at every coverage."""
    fields.setdefault("holdout", True)
    fields.setdefault("name", "Holdout")
    return s.Geometry(np.array([[-.5, -2.5, 2.5], [3, -2.5, 2.5], [3, 2.5, 2.5], [.5, 2.5, 2.5]], "f4"),
                      np.array([[0, 1, 2], [0, 2, 3]], "i4"), fields.pop("color", HOLD_COLOR), **fields)


def scene(card=True, lights=(SUN, LAMP), floor=True, **card_fields):
    geometry = (ball(0),) + ((ground(),) if floor else ()) + ((holdout(**card_fields),) if card else ())
    return s.Scene(geometry, lights)


def surrounded(mask):
    """`mask` shrunk by a pixel: the pixels all four neighbours of which are in it too."""
    inner = mask.copy()
    for axis in (0, 1):
        for step in (-1, 1):
            inner &= np.roll(mask, step, axis)
    return inner


def card_area(**fields):
    """The pixels the holdout card covers, from the card drawn as an ordinary opaque card (alpha 1 inside)."""
    visible = replace(holdout(**fields), holdout=False)
    return s.render(s.Scene((visible,)), CAMERA, *SIZE, mode="raster")[..., 3] > .999


class Fields(unittest.TestCase):
    KINDS = ("Card3D", "Cube3D", "Sphere3D", "Cylinder3D", "ReadGeo3D", "ReadGLTF3D")

    def test_every_mesh_node_has_the_switch_off_by_default(self):
        for kind in self.KINDS:
            self.assertEqual(core.SPECS[kind]["params"]["holdout"], "off", kind)
        self.assertEqual(core.CHOICES["holdout"], ["off", "on"])

    def test_the_knob_is_laid_out_beside_the_other_switches(self):
        for kind in self.KINDS:
            labels = [g.label for g in knobs.KNOB_LAYOUT[kind]]
            self.assertIn("Holdout", labels, kind)
            self.assertEqual(labels.index("Holdout"), labels.index("Visible to camera") + 1, kind)

    def test_the_param_becomes_a_geometry_field(self):
        self.assertFalse(s.holdout_from_params({}))
        self.assertFalse(s.holdout_from_params(dict(holdout="off")))
        self.assertTrue(s.holdout_from_params(dict(holdout="on")))
        self.assertTrue(s.has_render_switches(s.Scene((ball(0, holdout=True),))))
        self.assertTrue(s.has_holdouts(s.Scene((ball(0, holdout=True),))))
        self.assertFalse(s.has_holdouts(s.Scene((ball(0),))))

    def test_an_instance_copy_takes_its_sources_holdout(self):
        matrices = np.tile(np.eye(4), (2, 1, 1))
        matrices[:, 0, 3] = (-2, 2)
        copies = s.expand_instances(s.InstanceSet((ball(0, holdout=True),), matrices, np.zeros(2, np.int32)))
        self.assertTrue(all(copy.holdout for copy in copies))

    def graph(self, **params):
        d = Dispatcher()
        for key, kind, p in (("ball", "Sphere3D", dict(sphere_radius=1.2, ty=.3)),
                             ("card", "Card3D", dict(card_width=6, card_height=5, tx=3, tz=2.5, **params)),
                             ("sun", "Light3D", dict(shadows="off", light_type="Directional", tx=-4, ty=5, tz=2)),
                             ("camera", "Camera3D", dict(ty=2, tz=6)),
                             ("scene", "Scene3D", {}), ("render", "Render3D", dict(width=96, height=60, samples=1))):
            d.execute(dict(op="create", id=key, type=kind, params=p))
        for slot, source in (("object0", "ball"), ("object1", "card"), ("object2", "sun")):
            d.execute(dict(op="connect", id="scene", input=slot, source=source))
        for slot in ("scene", "camera"):
            d.execute(dict(op="connect", id="render", input=slot, source=slot))
        return d

    def test_the_node_carries_the_switch_into_the_render(self):
        d = self.graph(holdout="on")
        scene3 = Evaluator().evaluate_raster(d.document, "scene", typed=True)
        self.assertEqual([g.holdout for g in scene3.geometries], [False, True])
        cut = Evaluator().evaluate_raster(d.document, "render").pixels
        drawn = Evaluator().evaluate_raster(self.graph().document, "render").pixels
        gone = Evaluator().evaluate_raster(self.graph(visible_to_camera="off").document, "render").pixels
        card = np.abs(drawn - gone).max(axis=-1) > 1e-6          # where the card is an ordinary object
        self.assertGreater(int(card.sum()), 200)
        self.assertTrue(np.all(cut[card] == 0), "the holdout card is nothing")
        ball_only = ~card & (gone[..., 3] > .999) & surrounded(~card)
        self.assertGreater(int(ball_only.sum()), 100)
        np.testing.assert_array_equal(cut[ball_only], gone[ball_only])      # the ball beside it is untouched

    def test_a_document_without_the_switch_renders_as_one_with_the_default(self):
        d = self.graph()
        new = Evaluator().evaluate_raster(d.document, "render").pixels
        for name in ("card",):
            d.document["nodes"][name]["params"].pop("holdout")
        old = Evaluator().evaluate_raster(d.document, "render").pixels
        np.testing.assert_array_equal(old, new)
        self.assertGreater(float(new.max()), .1)

    def test_the_viewport_draws_a_holdout_as_the_stand_in_it_is(self):
        # the viewport's CPU fallback replaces the flags of the geometries it draws (nodebased/viewport3d.py)
        import inspect
        from nodebased import viewport3d
        source = inspect.getsource(viewport3d)
        self.assertIn("holdout=False", source)



class HoldoutTests:
    """The properties, for each renderer of `RENDERERS` (set by the subclasses below)."""
    RENDERERS = ()

    def each(self):
        for renderer in self.RENDERERS:
            with self.subTest(renderer=renderer.name):
                yield renderer

    def test_a_ball_half_behind_the_card_is_cut_out_where_the_card_is(self):
        area = surrounded(card_area())
        inside_ball = core_of(ball(0))
        self.assertGreater(int((area & inside_ball).sum()), 30)                       # the card does cover part of the ball
        clear = inside_ball & ~coverage(replace(holdout(), holdout=False))            # and part of it is clear of the card
        self.assertGreater(int(clear.sum()), 30)
        for r in self.each():
            picture = r(scene(cast_shadows=False))
            self.assertTrue(np.all(picture[area] == 0), "colour and alpha are 0 over the whole card")
            self.assertTrue(np.all(picture[clear][..., 3] > .999), "and the ball is whole beside it")
            alone = r(scene(card=False))
            np.testing.assert_allclose(picture[clear], alone[clear], atol=r.atol)       # the ball is as if the card were absent

    def test_the_floor_behind_the_card_is_cut_too(self):
        for r in self.each():
            picture = r(scene(cast_shadows=False))
            floor_only = r(scene(card=False, cast_shadows=False))
            area = surrounded(card_area()) & ~coverage(ball(0)) & (floor_only[..., 3] > .99)     # floor behind the card
            self.assertGreater(int(area.sum()), 40)
            self.assertTrue(np.all(picture[area] == 0))                                  # is gone behind the card

    def test_where_alpha_is_zero_colour_is_zero(self):
        for r in self.each():
            picture = r(scene())
            empty = picture[..., 3] == 0
            self.assertGreater(int(empty.sum()), 100)
            self.assertTrue(np.all(picture[empty] == 0))

    def test_a_holdout_in_front_of_nothing_leaves_nothing(self):
        alone = s.Scene((holdout(),), (SUN, LAMP))
        for r in self.each():
            self.assertTrue(np.all(r(alone) == 0), "colour 0 and alpha 0 everywhere")
            for output in ("depth", "position", "normals", "object_id", "uv"):
                self.assertTrue(np.all(r(alone, output) == 0), output)

    def test_a_holdout_behind_the_ball_changes_nothing(self):
        # wholly behind the ball as seen from the camera; it casts no shadow, and the floor keeps the scene's extent (and so the
        # shadow bias) what it was
        hidden_behind = quad(-.3, .3, -.3, .3, -2.0, holdout=True, cast_shadows=False)
        without = s.Scene((ball(0), ground()), (SUN, LAMP))
        with_it = s.Scene((ball(0), ground(), hidden_behind), (SUN, LAMP))
        for r in self.each():
            np.testing.assert_allclose(r(with_it), r(without), atol=r.atol)
            for output in ("depth", "object_id"):
                np.testing.assert_allclose(r(with_it, output), r(without, output), atol=max(r.atol, 5e-4))

    def test_the_data_passes_carry_nothing_of_the_hidden_geometry(self):
        area = surrounded(card_area()) & surrounded(coverage(ball(0)))
        self.assertGreater(int(area.sum()), 20)
        clear = core_of(ball(0)) & ~coverage(replace(holdout(), holdout=False))
        for r in self.each():
            for output in ("depth", "position", "normals", "object_id", "uv"):
                picture = r(scene(floor=False), output)
                self.assertTrue(np.all(picture[area] == 0), output)                 # no value, and alpha 0: no coverage
                self.assertTrue(np.all(picture[clear][..., 3] == 1), output)       # the ball beside the card has its data
            ids = np.rint(r(scene(), "object_id")[..., 0])
            self.assertEqual(int((ids == 3).sum()), 0, "the holdout (object 3) never appears")

    def test_a_holdout_casts_the_shadow_its_switches_give_it(self):
        def floating(**fields):
            fields.setdefault("holdout", True)
            return quad(0.0, 1.8, .6, 1.6, 1.0, **fields)
        beside = ~coverage(replace(floating(), holdout=False))
        flat = ground()
        for r in self.each():
            cast = r(s.Scene((flat, floating()), (SUN,)))
            hidden = r(s.Scene((flat, floating(holdout=False, visible_to_camera=False)), (SUN,)))
            none = r(s.Scene((flat, floating(cast_shadows=False)), (SUN,)))
            clear_floor = r(s.Scene((flat,), (SUN,)))
            np.testing.assert_allclose(cast[beside], hidden[beside], atol=r.atol)       # the shadow is that of the same card, unseen
            self.assertGreater(float(np.abs(cast[beside] - clear_floor[beside]).max()), .05)   # and it is a real shadow
            np.testing.assert_allclose(none[beside], clear_floor[beside], atol=r.atol)  # cast_shadows off: none


class CpuHoldoutTests(HoldoutTests, unittest.TestCase):
    RENDERERS = CPU


@unittest.skipUnless(gpu3d.available(), "no GPU adapter")
class GpuHoldoutTests(HoldoutTests, unittest.TestCase):
    RENDERERS = GPU


class EdgeTests(unittest.TestCase):
    """The edge of a cut is antialiased: the ball's alpha beside the card edge is one minus the card's own coverage, and its
    colour is cut by the same fraction."""
    PATH = pt.PathSettings(samples=192, max_bounces=1, diffuse_bounces=1)

    def pictures(self, draw):
        both = draw(scene(floor=False, cast_shadows=False))
        card_alone = draw(s.Scene((replace(holdout(), holdout=False),), (SUN, LAMP)))
        ball_alone = draw(s.Scene((ball(0),), (SUN, LAMP)))
        return both, card_alone, ball_alone

    def check(self, both, card_alone, ball_alone, tolerance, mean_tolerance):
        edge = core_of(ball(0)) & (card_alone[..., 3] > .02) & (card_alone[..., 3] < .98)
        self.assertGreater(int(edge.sum()), 10, "the card's edge crosses the ball")
        expected = 1.0 - card_alone[..., 3]
        error = np.abs(both[..., 3] - expected)[edge]
        self.assertLessEqual(float(error.max()), tolerance)
        self.assertLessEqual(float(error.mean()), mean_tolerance)
        # the colour is cut with the same coverage (premultiplied: the ball's own colour times the alpha that is left)
        colour_error = np.abs(both[..., :3] - ball_alone[..., :3] * both[..., 3:4])[edge].max(axis=-1)
        self.assertLessEqual(float(colour_error.mean()), 0.02 + 3 * mean_tolerance)
        self.assertTrue(np.all(both[..., 3][~edge & (both[..., 3] == 0)] == 0))

    def test_cpu_raster_and_ray_traced(self):
        for mode in ("raster", "raytrace"):
            with self.subTest(mode=mode):
                draw = lambda sc: s.render(sc, CAMERA, *SIZE, ambient=AMBIENT, mode=mode, samples=EDGE_SAMPLES)
                self.check(*self.pictures(draw), tolerance=2e-6, mean_tolerance=2e-6)

    def test_cpu_path_tracer(self):
        draw = lambda sc: pt.render(sc, CAMERA, *SIZE, BLACK, 0.0, "rgba", self.PATH, backend="cpu")
        self.check(*self.pictures(draw), tolerance=.3, mean_tolerance=.06)

    @unittest.skipUnless(gpu3d.available(), "no GPU adapter")
    def test_gpu_modes(self):
        for mode in ("raster", "raytrace"):
            with self.subTest(mode=mode):
                draw = lambda sc: gpu3d.render(sc, CAMERA, *SIZE, ambient=AMBIENT, mode=mode, samples=EDGE_SAMPLES)
                self.check(*self.pictures(draw), tolerance=2e-3, mean_tolerance=1e-3)

    @unittest.skipUnless(gpu3d.available(), "no GPU adapter")
    def test_gpu_path_tracer(self):
        draw = lambda sc: pt.render(sc, CAMERA, *SIZE, BLACK, 0.0, "rgba", self.PATH, backend="gpu")
        self.check(*self.pictures(draw), tolerance=.3, mean_tolerance=.06)


class LensAndShutterTests(unittest.TestCase):
    """The cut edge blurs with the camera's lens and with the card's motion, like any other edge."""
    PATH = pt.PathSettings(samples=96, max_bounces=1, diffuse_bounces=1)
    SMALL = (64, 40)

    @staticmethod
    def soft_pixels(alpha):
        """How many pixels are neither empty nor whole: the length of the soft edges."""
        return int(((alpha > .1) & (alpha < .9)).sum())

    def test_depth_of_field_softens_the_edge_in_the_ray_traced_mode_and_the_path_tracer(self):
        sharp, soft = CAMERA, replace(CAMERA, fstop=0.05, focus_distance=6.0)    # the ball is in focus, the card in front of it is not
        sc = scene(floor=False, cast_shadows=False)
        for label, draw in (("ray traced", lambda camera: s.render(sc, camera, *self.SMALL, mode="raytrace", ambient=AMBIENT)[..., 3]),
                            ("path tracer", lambda camera: pt.render(sc, camera, *self.SMALL, BLACK, 0.0, "rgba", self.PATH)[..., 3])):
            with self.subTest(renderer=label):
                self.assertGreater(self.soft_pixels(draw(soft)), self.soft_pixels(draw(sharp)) + 30)

    @unittest.skipUnless(gpu3d.available(), "no GPU adapter")
    def test_depth_of_field_in_the_gpu_path_tracer(self):
        sharp, soft = CAMERA, replace(CAMERA, fstop=0.05, focus_distance=6.0)
        sc = scene(floor=False, cast_shadows=False)
        draw = lambda camera: pt.render(sc, camera, *self.SMALL, BLACK, 0.0, "rgba", self.PATH, backend="gpu")[..., 3]
        self.assertGreater(self.soft_pixels(draw(soft)), self.soft_pixels(draw(sharp)) + 30)

    def test_motion_blur_softens_the_edge_in_the_path_tracer(self):
        still = scene(floor=False, cast_shadows=False)
        moved = s.Scene((ball(0), replace(holdout(), transform=s.Transform3D(position=s.Vec3(-1.2, 0, 0)))), (SUN, LAMP))
        blurred = pt.render_motion([(still, CAMERA), (moved, CAMERA)], *self.SMALL, BLACK, 0.0, "rgba", self.PATH)
        sharp = pt.render(still, CAMERA, *self.SMALL, BLACK, 0.0, "rgba", self.PATH)
        self.assertGreater(self.soft_pixels(blurred[..., 3]), self.soft_pixels(sharp[..., 3]) + 10)


class DataPassTests(unittest.TestCase):
    """Cryptomatte coverage of the ball is what the card leaves of it; the holdout has none of its own."""

    @staticmethod
    def coverage_of(layers, metadata, name):
        wanted = int(metadata["CryptoObject"]["manifest"][name], 16)
        total = np.zeros(next(iter(layers.values())).shape[:2])
        for key, layer in layers.items():
            if key.startswith("CryptoObject"):
                for id_channel, coverage_channel in ((0, 1), (2, 3)):
                    total += np.where(layer[..., id_channel].view(np.uint32) == wanted, layer[..., coverage_channel], 0.0)
        return total

    def test_the_balls_coverage_is_reduced_by_the_holdouts(self):
        for mode in ("raster", "raytrace"):
            with self.subTest(mode=mode):
                def crypto(card):
                    geometry = (ball(0, name="Ball"),) + ((replace(holdout(holdout=card == "hold"), name="Card"),) if card else ())
                    return cryptomatte3d.render_cryptomatte(s.Scene(geometry, (SUN,)), CAMERA, *SIZE, samples=3, mode=mode)
                alone, ordinary, cut = crypto(None), crypto("ordinary"), crypto("hold")
                ball_alone, ball_ordinary, ball_cut = (self.coverage_of(*pair, "Ball") for pair in (alone, ordinary, cut))
                np.testing.assert_allclose(ball_cut, ball_ordinary, atol=1e-6)    # the holdout hides exactly what a card would
                self.assertGreater(float((ball_alone - ball_cut).max()), .99)       # and fully, where it covers the ball
                self.assertLess(float(ball_cut.sum()), float(ball_alone.sum()) - 10)
                self.assertTrue(np.all(ball_cut <= ball_alone + 1e-6))
                self.assertGreater(float(self.coverage_of(*ordinary, "Card").max()), .99)
                self.assertEqual(float(self.coverage_of(*cut, "Card").max()), 0.0, "the holdout has no coverage of its own")
