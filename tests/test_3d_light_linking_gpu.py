"""Light linking on the GPU everywhere (lane L4, plan "Rendering 8", step T1 of 3).

Rendering 6 linked meshes in the GPU raster and ray-traced modes and sent every scene with a linked splat set or instance
set to the CPU. These tests hold the GPU modes to the CPU reference for those objects, on whichever adapter the run forces
(`run/force-adapter.py default|integrated|cpu`): a linked splat set is lit only by its included light (within the splat
tolerance of 2e-3), an excluded light leaves an instance set unlit and shadowless while its neighbour keeps its light,
and the smoke, liquid and particle tests of the later parts sit beside them."""
import unittest
from dataclasses import replace

import numpy as np

from nodebased import gpu3d, scene3d as s
from tests import gpu_precision
from tests.test_3d_light_linking import AMBIENT, CAMERA, NOBODY, SIZE, SUN, ball, core_of, coverage, ground
from tests.test_3d_pathtrace import box
from tests.test_3d_pathtrace_splats import SPLAT_CAMERA, instance, plane_cloud

SPLAT_TOLERANCE = gpu_precision.tolerance(2e-3)   # x20 on an adapter that blends the rgba layers in half float
SPLAT_SIZE = (24, 24)
KEY = s.Light("Directional", (1, 1, 1), 1.0, s.Vec3(), s.Vec3(0.3, -0.2, -1.0), name="key", shadows=True)
LAMP = s.Light("Point", (.4, .5, 1), 3.0, s.Vec3(-1.2, 1.0, 2.5), shadows=True, name="lamp")


def draw(scene, mode, gpu, camera=SPLAT_CAMERA, size=SPLAT_SIZE, ambient=.1):
    return (gpu3d.render if gpu else s.render)(scene, camera, *size, ambient=ambient, mode=mode)


@unittest.skipUnless(gpu3d.available(), "no GPU adapter")
class SplatSetTests(unittest.TestCase):
    """A relit sheet, a mesh blocker in front of it and a wall behind a small splat caster, every link by every mode."""

    @staticmethod
    def blocker(link=s.LIGHT_LINK_ALL):
        return replace(box((.4, .4, .2), (.5, .5, .5, 1), (-.4, .3, 1.0)), light_link=link)

    def agree(self, scene, label):
        for mode in ("raster", "raytrace"):
            cpu, gpu = draw(scene, mode, False), draw(scene, mode, True)
            np.testing.assert_allclose(gpu[..., :3], cpu[..., :3], atol=SPLAT_TOLERANCE, rtol=0, err_msg=f"{label} {mode}")
            np.testing.assert_allclose(gpu[..., 3], cpu[..., 3], atol=SPLAT_TOLERANCE, rtol=0, err_msg=f"{label} {mode}")

    def test_a_linked_splat_set_is_lit_only_by_its_included_light(self):
        lights = (KEY, LAMP)
        everything = draw(s.Scene((), lights, splats=(instance(plane_cloud()),)), "raytrace", True)
        for link in (("include", ("lamp",)), ("exclude", ("key",)), ("include", ("key",)), NOBODY):
            with self.subTest(link=link):
                scene = s.Scene((), lights, splats=(instance(plane_cloud(), light_link=link),))
                self.agree(scene, str(link))
        only_lamp = draw(s.Scene((), lights, splats=(instance(plane_cloud(), light_link=("include", ("lamp",))),)),
                         "raytrace", True)
        lamp_alone = draw(s.Scene((), (LAMP,), splats=(instance(plane_cloud()),)), "raytrace", True)
        np.testing.assert_allclose(only_lamp, lamp_alone, atol=SPLAT_TOLERANCE, rtol=0)
        self.assertGreater(float(np.abs(everything - only_lamp)[..., :3].max()), .05)   # the link changes the picture

    def test_a_mesh_excluded_from_the_key_casts_no_shadow_on_a_relit_sheet(self):
        sheet = instance(plane_cloud())
        for link in (s.LIGHT_LINK_ALL, ("exclude", ("key",)), NOBODY):
            with self.subTest(link=link):
                self.agree(s.Scene((self.blocker(link),), (KEY,), splats=(sheet,)), str(link))
        outside = ~coverage(self.blocker(), SPLAT_CAMERA, SPLAT_SIZE)
        gone = draw(s.Scene((), (KEY,), splats=(sheet,)), "raytrace", True)
        linked = draw(s.Scene((self.blocker(("exclude", ("key",))),), (KEY,), splats=(sheet,)), "raytrace", True)
        plain = draw(s.Scene((self.blocker(),), (KEY,), splats=(sheet,)), "raytrace", True)
        np.testing.assert_allclose(linked[outside], gone[outside], atol=SPLAT_TOLERANCE, rtol=0)
        self.assertGreater(float(np.abs(plain[outside] - gone[outside]).max()), .1)

    def test_a_splat_set_excluded_from_the_key_casts_no_shadow_on_a_wall(self):
        wall = box((4, 4, .1), (.6, .6, .6, 1), (0, 0, -.8))
        matrix = np.array(((1, 0, 0, -.3), (0, 1, 0, .3), (0, 0, 1, .6), (0, 0, 0, 1)), np.float64)
        small = lambda link, **kw: instance(plane_cloud(n=8, size=.7), relight=0.0, light_link=link, cast_shadows=True,
                                            matrix=matrix, **kw)
        for link in (s.LIGHT_LINK_ALL, ("exclude", ("key",))):
            with self.subTest(link=link):
                self.agree(s.Scene((wall,), (KEY,), splats=(small(link),)), str(link))
        bare = draw(s.Scene((wall,), (KEY,), splats=(small(s.LIGHT_LINK_ALL, opacity_scale=0.0),)), "raytrace", True)
        plain = draw(s.Scene((wall,), (KEY,), splats=(small(s.LIGHT_LINK_ALL),)), "raytrace", True)
        linked = draw(s.Scene((wall,), (KEY,), splats=(small(("exclude", ("key",))),)), "raytrace", True)
        beside = draw(s.Scene((), (), splats=(small(s.LIGHT_LINK_ALL),)), "raytrace", False)[..., 3] <= 0
        np.testing.assert_allclose(linked[beside], bare[beside], atol=SPLAT_TOLERANCE, rtol=0)
        self.assertGreater(float(np.abs(plain[beside] - bare[beside]).max()), .02)

    def test_a_splat_set_excluded_from_the_key_casts_no_shadow_on_another_splat_set(self):
        sheet = instance(plane_cloud())
        floating = lambda link: instance(plane_cloud(n=8, size=.7), relight=0.0, light_link=link, cast_shadows=True,
                                         matrix=np.array(((1, 0, 0, -.3), (0, 1, 0, .3), (0, 0, 1, .6), (0, 0, 0, 1)),
                                                         np.float64))
        for link in (s.LIGHT_LINK_ALL, ("exclude", ("key",))):
            self.agree(s.Scene((), (KEY,), splats=(sheet, floating(link))), str(link))


def pair(link, left=True):
    """Two ball sets over a floor set: a `link` on the left one, the right one linked to every light."""
    def one(x, link):
        matrices = np.eye(4)[None].copy()
        matrices[0, 0, 3] = x
        return s.InstanceSet((ball(0),), matrices, np.zeros(1, np.int32), light_link=link)

    floor = s.InstanceSet((ground(),), np.eye(4)[None].copy(), np.zeros(1, np.int32))
    sets = (floor, one(-1.7, link), one(1.7, s.LIGHT_LINK_ALL))
    return s.Scene((), (SUN,), instances=sets if left else (floor, sets[2]))


@unittest.skipUnless(gpu3d.available(), "no GPU adapter")
class InstanceSetTests(unittest.TestCase):
    def render(self, scene, gpu):
        return (gpu3d.render if gpu else s.render)(scene, CAMERA, *SIZE, ambient=AMBIENT, mode="raytrace")

    def test_an_excluded_instance_set_is_unlit_and_shadowless_while_its_neighbour_keeps_the_light(self):
        left, right = ball(-1.7), ball(1.7)
        for link in (("exclude", ("sun",)), NOBODY, ("include", ("elsewhere",))):
            with self.subTest(link=link):
                cpu, gpu = self.render(pair(link), False), self.render(pair(link), True)
                np.testing.assert_allclose(gpu, cpu, atol=1e-4, rtol=0)
                bare = self.render(pair(link, left=False), True)
                plain = self.render(pair(s.LIGHT_LINK_ALL), True)
                mine = coverage(left)
                np.testing.assert_allclose(gpu[~mine], bare[~mine], atol=1e-5, rtol=0)    # no shadow on the floor
                self.assertGreater(float(np.abs(plain[~mine] - bare[~mine]).max()), .05)  # it shadowed before
                inside = core_of(left)
                expected = np.array((.75, .45, .3)) * AMBIENT                              # unlit: ambient on its colour
                np.testing.assert_allclose(gpu[inside][:, :3], np.broadcast_to(expected, (int(inside.sum()), 3)), atol=2e-3)
                neighbour = core_of(right)
                self.assertGreater(float(gpu[neighbour][:, :3].mean()), 3 * AMBIENT * .5)
                np.testing.assert_allclose(gpu[neighbour], bare[neighbour], atol=1e-5, rtol=0)

    def test_a_listed_light_still_reaches_an_included_instance_set(self):
        scene = pair(("include", ("sun",)))
        np.testing.assert_allclose(self.render(scene, True), self.render(pair(s.LIGHT_LINK_ALL), True), atol=1e-6)


if __name__ == "__main__":
    unittest.main()
