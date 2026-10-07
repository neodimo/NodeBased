"""Light linking (lane L4, plan "Rendering 6", step R2 of 3, part 2).

A mesh, splat set or instance set has `light_link` = (`all` | `include` | `exclude`, names): an excluded light neither
lights the object nor is shadowed by it. The property every renderer must keep is the same: **render the scene with an
object excluded from every light and you get the scene without that object everywhere outside the object itself, while the
object itself shows only what no light gives it** (ambient and emission). The tests state it that way, on the CPU raster
and ray-traced modes, both path tracers, the GPU raster and ray-traced modes, and the viewport, on whichever adapter the
test run forces (run/force-adapter.py)."""
import unittest
from dataclasses import replace

import numpy as np

from nodebased import core, envlight, gpu3d, gpupathtrace, pathtrace as pt, scene3d as s, viewportgpu
from nodebased.core import Dispatcher
from nodebased.imaging import Evaluator
from tests.test_3d_light_groups import environment
from tests.test_3d_pathtrace import box
from tests.test_3d_pathtrace_splats import SPLAT_CAMERA, instance, plane_cloud

SIZE = (64, 40)
CAMERA = s.Camera(s.Transform3D(position=s.Vec3(0, 5.0, 8.0)), s.Vec3(0, -0.5, 0), 40.0)
NOBODY = ("include", ())                       # lit by no light at all, so it casts no shadow either
AMBIENT = 0.1
BLACK = (0.0, 0.0, 0.0, 0.0)

SUN = s.Light("Directional", (1, .95, .9), 1.2, s.Vec3(-4, 5, 2), s.Vec3(), shadows=True, name="sun", light_group="key")
LAMP = s.Light("Point", (.5, .6, 1), 3.0, s.Vec3(3.5, 2.5, 3.0), shadows=True, name="lamp", light_group="fill")
PANEL = s.Light("Disc", (1, 1, 1), 2.5, s.Vec3(0, 3.5, 1), s.Vec3(0, 0, 0), area_radius=.5, light_samples=6, shadows=True,
                name="panel", light_group="fill")


def ground(y=-1.0):
    return s.Geometry(np.array([[-8, y, 6], [8, y, 6], [8, y, -8], [-8, y, -8]], "f4"),
                      np.array([[0, 1, 2], [0, 2, 3]], "i4"), (.6, .6, .6, 1))


def ball(x, link=s.LIGHT_LINK_ALL, **kw):
    return replace(s._sphere(.8, 24, (.75, .45, .3, 1), s.Transform3D(position=s.Vec3(x, -.2, 0))), light_link=link, **kw)


def two_balls(link, pbr=False, lights=(SUN, LAMP), environments=()):
    extra = dict(material="pbr", metallic=.2, pbr_roughness=.5) if pbr else dict(specular=.4)
    return s.Scene((ball(-1.7, link, **extra), ball(1.7, **extra), ground()), lights, environments=environments)


def without_the_left_ball(pbr=False, lights=(SUN, LAMP), environments=()):
    extra = dict(material="pbr", metallic=.2, pbr_roughness=.5) if pbr else dict(specular=.4)
    return s.Scene((ball(1.7, **extra), ground()), lights, environments=environments)


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
    inner = full.copy()
    for axis in (0, 1):
        for step in (-1, 1):
            inner &= np.roll(full, step, axis)
    return inner


class LinkSemanticsTests(unittest.TestCase):
    def test_all_include_and_exclude_by_node_name_or_group(self):
        key, fill = SUN, LAMP
        reaches = s.light_reaches
        self.assertTrue(reaches(s.LIGHT_LINK_ALL, key))
        self.assertTrue(reaches(("include", ("sun",)), key) and not reaches(("include", ("sun",)), fill))
        self.assertTrue(reaches(("include", ("fill",)), fill) and not reaches(("include", ("fill",)), key))   # by group
        self.assertFalse(reaches(("exclude", ("sun",)), key))
        self.assertTrue(reaches(("exclude", ("sun",)), fill))
        self.assertTrue(reaches(("exclude", ()), key))                       # exclude nothing is all
        self.assertFalse(reaches(NOBODY, key))                               # include nothing is none
        self.assertTrue(reaches(("include", ("default",)), s.Light()))      # an ungrouped light is in group "default"
        self.assertTrue(reaches(("exclude", ("sky",)), s.Light(name="other")))
        self.assertFalse(reaches(("exclude", ("sky",)), environment("sky")))   # environments link too

    def test_the_knobs_become_a_link(self):
        link = s.light_link_from_params
        self.assertEqual(link({}), s.LIGHT_LINK_ALL)
        self.assertEqual(link({"light_link": "all", "light_link_list": "sun"}), s.LIGHT_LINK_ALL)
        self.assertEqual(link({"light_link": "exclude", "light_link_list": " sun , fill;sun,"}), ("exclude", ("fill", "sun")))
        self.assertEqual(link({"light_link": "bogus", "light_link_list": "sun"}), s.LIGHT_LINK_ALL)
        self.assertFalse(s.has_light_links(s.Scene((ball(0),))))
        self.assertTrue(s.has_light_links(s.Scene((ball(0, NOBODY),))))

    def test_the_nodes_carry_the_link(self):
        d = Dispatcher()
        for key, kind, params in (("cube", "Cube3D", dict(light_link="exclude", light_link_list="sun")),
                                  ("light", "Light3D", dict(light_group="key")), ("scene", "Scene3D", {})):
            d.execute(dict(op="create", id=key, type=kind, params=params))
        for slot, source in (("object0", "cube"), ("object1", "light")):
            d.execute(dict(op="connect", id="scene", input=slot, source=source))
        scene = Evaluator().evaluate_raster(d.document, "scene", typed=True)
        self.assertEqual(scene.geometries[0].light_link, ("exclude", ("sun",)))
        for kind in ("Card3D", "Sphere3D", "Cylinder3D", "ReadGeo3D", "ReadSplat3D", "Instance3D"):
            self.assertEqual(core.SPECS[kind]["params"]["light_link"], "all", kind)
            self.assertEqual(core.SPECS[kind]["params"]["light_link_list"], "", kind)
        self.assertEqual(core.CHOICES["light_link"], ["all", "include", "exclude"])

    def test_an_instance_set_gives_its_link_to_every_copy(self):
        source = ball(0)
        matrices = np.tile(np.eye(4), (3, 1, 1))
        matrices[:, 0, 3] = (-2, 0, 2)
        link = ("exclude", ("sun",))
        copies = s.InstanceSet((source,), matrices, np.zeros(3, np.int32), light_link=link)
        self.assertEqual({g.light_link for g in s.expand_instances(copies)}, {link})
        self.assertEqual(s.expand_instances(replace(copies, light_link=s.LIGHT_LINK_ALL))[0].light_link, s.LIGHT_LINK_ALL)


class RasterAndRayTracedTests(unittest.TestCase):
    def check(self, mode, pbr=False, lights=(SUN, LAMP), environments=()):
        plain = s.render(two_balls(s.LIGHT_LINK_ALL, pbr, lights, environments), CAMERA, *SIZE, ambient=AMBIENT, mode=mode)
        linked = s.render(two_balls(NOBODY, pbr, lights, environments), CAMERA, *SIZE, ambient=AMBIENT, mode=mode)
        gone = s.render(without_the_left_ball(pbr, lights, environments), CAMERA, *SIZE, ambient=AMBIENT, mode=mode)
        mine = coverage(ball(-1.7))
        # no light, no shadow: everywhere but on the object the picture is the scene without it
        np.testing.assert_allclose(linked[~mine], gone[~mine], atol=1e-6)
        # the object keeps only what no light gives it: ambient on its own colour
        inside = core_of(ball(-1.7))
        self.assertGreater(int(inside.sum()), 20)
        expected = np.array((.75, .45, .3)) * AMBIENT * (1 if not pbr else 1 - .2)
        np.testing.assert_allclose(linked[inside][:, :3], np.broadcast_to(expected, (int(inside.sum()), 3)), atol=2e-3)
        # unlinked it is lit, and it did shadow the floor
        self.assertGreater(float(plain[inside][:, :3].mean()), 2 * float(linked[inside][:, :3].mean()))
        self.assertGreater(float(np.abs(plain[~mine] - gone[~mine]).max()), .05)
        # its neighbour is lit exactly as before
        neighbour = core_of(ball(1.7))
        self.assertGreater(float(linked[neighbour][:, :3].mean()), 3 * AMBIENT * .5)
        np.testing.assert_allclose(linked[neighbour], gone[neighbour], atol=1e-6)

    def test_raster_standard_material(self):
        self.check("raster")

    def test_raster_pbr_material(self):
        self.check("raster", pbr=True)

    def test_ray_traced_standard_material(self):
        self.check("raytrace")

    def test_ray_traced_pbr_material(self):
        self.check("raytrace", pbr=True)

    def test_area_light_and_environment_link_too(self):
        for mode in ("raster", "raytrace"):
            with self.subTest(mode=mode):
                self.check(mode, lights=(SUN, PANEL), environments=(environment(),))

    def test_one_light_excluded_leaves_the_others_on_the_object(self):
        for mode in ("raster", "raytrace"):
            plain = s.render(two_balls(s.LIGHT_LINK_ALL), CAMERA, *SIZE, ambient=AMBIENT, mode=mode)
            no_sun = s.render(two_balls(("exclude", ("sun",))), CAMERA, *SIZE, ambient=AMBIENT, mode=mode)
            no_key = s.render(two_balls(("exclude", ("key",))), CAMERA, *SIZE, ambient=AMBIENT, mode=mode)   # by group
            only_lamp = s.render(two_balls(("include", ("lamp",))), CAMERA, *SIZE, ambient=AMBIENT, mode=mode)
            mine = coverage(ball(-1.7))
            np.testing.assert_array_equal(no_sun, no_key)
            np.testing.assert_array_equal(no_sun, only_lamp)
            self.assertGreater(float(np.abs(plain[mine] - no_sun[mine]).max()), .05)
            lit_by_lamp = (no_sun[mine][:, :3].mean(), s.render(two_balls(NOBODY), CAMERA, *SIZE, ambient=AMBIENT,
                                                                 mode=mode)[mine][:, :3].mean())
            self.assertGreater(lit_by_lamp[0], lit_by_lamp[1])

    def test_the_relight_bundle_and_layers_follow_the_link(self):
        _, layers = s.render_multichannel(two_balls(("exclude", ("sun",))), CAMERA, *SIZE, BLACK,
                                          passes="beauty,lights", ambient=AMBIENT)
        inner = core_of(ball(-1.7))
        np.testing.assert_array_equal(layers["light.key"][inner][:, :3], 0)      # the sun reaches it nowhere
        self.assertGreater(float(layers["light.fill"][inner][:, :3].max()), 0)


class PathTracedTests(unittest.TestCase):
    """Direct light only (one bounce), so the picture outside the excluded object is exactly the scene without it."""
    PATH = pt.PathSettings(samples=32, max_bounces=1, diffuse_bounces=1)

    def render(self, scene, backend="cpu"):
        # no ambient: in the path tracer it is the sky a ray sees when it escapes, which an object blocks whether or not
        # a light is linked to it
        return pt.render(scene, CAMERA, *SIZE, BLACK, 0.0, "rgba", self.PATH, backend=backend)

    def check(self, backend, lights, environments=(), exact=True):
        linked = self.render(two_balls(NOBODY, True, lights, environments), backend)
        gone = self.render(without_the_left_ball(True, lights, environments), backend)
        plain = self.render(two_balls(s.LIGHT_LINK_ALL, True, lights, environments), backend)
        mine = coverage(ball(-1.7))
        if exact:
            np.testing.assert_allclose(linked[~mine], gone[~mine], atol=2e-5)
        else:
            # An area light or an environment that an object excludes is sampled by next-event estimation alone (see
            # pathtrace.PathScene.mis), a different, equally unbiased estimator from the mixed one the scene without the
            # object uses: the two agree in the mean, not pixel by pixel.
            a, b = linked[~mine][:, :3], gone[~mine][:, :3]
            self.assertLess(abs(float(a.mean()) - float(b.mean())), .05 * float(b.mean()))
            self.assertLess(float(np.abs(linked[~mine] - gone[~mine]).mean()), .35 * float(b.mean()) + .01)
        self.assertGreater(float(np.abs(plain[~mine] - gone[~mine]).max()), .05)       # unlinked, it shadowed the floor
        inner = core_of(ball(-1.7))
        self.assertEqual(float(linked[inner][:, :3].max()), 0.0)                       # nothing lights it
        self.assertGreater(float(plain[inner][:, :3].max()), .3)

    def test_cpu_point_and_directional_lights(self):
        self.check("cpu", (SUN, LAMP))

    def test_cpu_area_light_and_environment(self):
        self.check("cpu", (PANEL,), (environment(),), exact=False)

    @unittest.skipUnless(gpu3d.available(), "no GPU adapter")
    def test_gpu_point_and_directional_lights(self):
        self.check("gpu", (SUN, LAMP))

    @unittest.skipUnless(gpu3d.available(), "no GPU adapter")
    def test_gpu_area_light_and_environment(self):
        self.check("gpu", (PANEL,), (environment(),), exact=False)

    @unittest.skipUnless(gpu3d.available(), "no GPU adapter")
    def test_gpu_matches_the_cpu_reference_with_a_link(self):
        scene = two_balls(("exclude", ("sun",)), True, (SUN, LAMP, PANEL), (environment(),))
        cpu, gpu = self.render(scene), self.render(scene, "gpu")
        self.assertLess(float(np.abs(cpu - gpu).mean()), .01 * float(cpu.mean()) + .002)


@unittest.skipUnless(gpu3d.available(), "no GPU adapter")
class GpuRasterAndRayTracedTests(unittest.TestCase):
    def test_gpu_modes_keep_the_property_and_agree_with_the_cpu(self):
        for mode in ("raster", "raytrace"):
            for pbr in (False, True):
                with self.subTest(mode=mode, pbr=pbr):
                    lights, envs = (SUN, LAMP, PANEL), (environment(),)
                    for link in (NOBODY, ("exclude", ("sun", "sky")), ("include", ("panel",))):
                        scene = two_balls(link, pbr, lights, envs)
                        cpu = s.render(scene, CAMERA, *SIZE, ambient=AMBIENT, mode=mode)
                        gpu = gpu3d.render(scene, CAMERA, *SIZE, ambient=AMBIENT, mode=mode)
                        self.assertLess(float(np.abs(cpu - gpu)[..., :3].mean()), .004, (link, mode, pbr))
                    linked = gpu3d.render(two_balls(NOBODY, pbr, (SUN, LAMP)), CAMERA, *SIZE, ambient=AMBIENT, mode=mode)
                    gone = gpu3d.render(without_the_left_ball(pbr, (SUN, LAMP)), CAMERA, *SIZE, ambient=AMBIENT, mode=mode)
                    mine = coverage(ball(-1.7))
                    np.testing.assert_allclose(linked[~mine], gone[~mine], atol=2e-4)

    def test_a_linked_splat_or_instance_set_is_drawn_by_the_gpu_modes(self):
        """No refusal: the GPU modes take a scene with a linked splat set or instance set (tests.test_3d_light_linking_gpu
        holds their pictures to the CPU reference)."""
        scene = s.Scene((ground(),), (SUN,), splats=(instance(plane_cloud(), light_link=NOBODY),))
        for mode in ("raster", "raytrace"):
            self.assertEqual(gpu3d.render(scene, CAMERA, *SIZE, mode=mode).shape, (SIZE[1], SIZE[0], 4))
        copies = s.InstanceSet((ball(0),), np.eye(4)[None], np.zeros(1, np.int32), light_link=NOBODY)
        self.assertEqual(gpu3d.render(s.Scene((), (SUN,), instances=(copies,)), CAMERA, *SIZE, mode="raytrace").shape,
                         (SIZE[1], SIZE[0], 4))


@unittest.skipUnless(gpu3d.available(), "no GPU adapter")
class ViewportTests(unittest.TestCase):
    BACKGROUND = (0.02, 0.02, 0.03, 1.0)
    VIEW = s.Camera(s.Transform3D(s.Vec3(0, 4, 9)), s.Vec3(0, 0, 0), 45.0, 0.1, 100.0)

    def setUp(self):
        self.gpu = viewportgpu.renderer()
        self.assertIsNotNone(self.gpu, viewportgpu.failure())

    def draw(self, link, shadows=True):
        sun = s.Light("Directional", (1, 1, 1), 1.5, s.Vec3(4, 6, 2), s.Vec3(), shadows=shadows, name="sun")
        cube = replace(s._cube(1.5, (.8, .2, .2, 1), s.Transform3D(s.Vec3(0, .75, 0))), material="pbr", pbr_roughness=.5,
                       light_link=link)
        floor = replace(s._card(8, 8, (.8, .8, .8, 1), s.Transform3D(s.Vec3(0, -1, 0), s.Vec3(-90, 0, 0))),
                        material="pbr", pbr_roughness=.9)
        return self.gpu.render(s.Scene((floor, cube), (sun,)), self.VIEW, 160, 120, self.BACKGROUND,
                               headlight=False, ambient=.1).astype(int)

    def test_an_excluded_cube_is_dark_and_casts_no_shadow_in_the_viewport(self):
        plain = self.draw(s.LIGHT_LINK_ALL)
        linked = self.draw(("exclude", ("sun",)))
        unshadowed = self.draw(("exclude", ("sun",)), shadows=False)
        self.assertGreater(int(np.abs(plain - linked).max()), 60)                 # the cube is dark
        np.testing.assert_array_equal(linked, unshadowed)                         # and the sun's shadow map has no cube in it
        self.assertGreater(int(np.abs(plain - self.draw(s.LIGHT_LINK_ALL, shadows=False)).max()), 20)   # it did shadow before

    def test_an_instance_set_excluded_from_a_light_is_dark_and_casts_no_shadow_in_the_viewport(self):
        cube = replace(s._cube(1.5, (.8, .2, .2, 1), s.Transform3D()), material="pbr", pbr_roughness=.5)
        matrix = np.eye(4)
        matrix[1, 3] = .75
        floor = replace(s._card(8, 8, (.8, .8, .8, 1), s.Transform3D(s.Vec3(0, -1, 0), s.Vec3(-90, 0, 0))),
                        material="pbr", pbr_roughness=.9)

        def draw(link, shadows=True):
            sun = s.Light("Directional", (1, 1, 1), 1.5, s.Vec3(4, 6, 2), s.Vec3(), shadows=shadows, name="sun")
            copies = s.InstanceSet((cube,), matrix[None], np.zeros(1, np.int32), light_link=link)
            return self.gpu.render(s.Scene((floor,), (sun,), instances=(copies,)), self.VIEW, 160, 120, self.BACKGROUND,
                                   headlight=False, ambient=.1).astype(int)

        plain, linked = draw(s.LIGHT_LINK_ALL), draw(("exclude", ("sun",)))
        unshadowed = draw(("exclude", ("sun",)), shadows=False)
        self.assertGreater(int(np.abs(plain - linked).max()), 60)
        np.testing.assert_array_equal(linked, unshadowed)
        self.assertGreater(int(np.abs(plain - draw(s.LIGHT_LINK_ALL, shadows=False)).max()), 20)

    def test_a_splat_set_excluded_from_a_light_is_unlit_in_the_viewport(self):
        from tests.test_3d_viewport_shadows import _splat_floor
        sun = s.Light("Directional", (1, 1, 1), 1.5, s.Vec3(4, 6, 2), s.Vec3(), shadows=False, name="sun")
        view = lambda link: self.gpu.render(s.Scene((), (sun,), splats=(instance(_splat_floor(), light_link=link),)),
                                            self.VIEW, 160, 120, self.BACKGROUND, headlight=False, ambient=.1).astype(int)
        lit, dark = view(s.LIGHT_LINK_ALL), view(NOBODY)
        self.assertGreater(int(np.abs(lit - dark).max()), 40)
        self.assertLess(int(dark[60:100, 40:120, :3].mean()), int(lit[60:100, 40:120, :3].mean()))


class SplatTests(unittest.TestCase):
    """A relit sheet of splats under a slanted sun, with a mesh blocker in front of it that throws a visible shadow on it."""
    SUN = s.Light("Directional", (1, 1, 1), 1.0, s.Vec3(), s.Vec3(0.3, -0.2, -1.0), name="sun", shadows=True)
    PATH = pt.PathSettings(samples=12, max_bounces=1, diffuse_bounces=1)

    @staticmethod
    def blocker(link=s.LIGHT_LINK_ALL):
        return replace(box((.4, .4, .2), (.5, .5, .5, 1), (-.4, .3, 1.0)), light_link=link)

    def sheet(self, link=s.LIGHT_LINK_ALL):
        return instance(plane_cloud(), light_link=link)

    def mask(self, geometry):
        return coverage(geometry, SPLAT_CAMERA, (24, 24))

    def test_cpu_modes_a_mesh_excluded_from_the_sun_casts_no_shadow_on_a_relit_sheet(self):
        for mode in ("raster", "raytrace"):
            with self.subTest(mode=mode):
                draw = lambda scene: s.render(scene, SPLAT_CAMERA, 24, 24, ambient=.1, mode=mode)
                plain = draw(s.Scene((self.blocker(),), (self.SUN,), splats=(self.sheet(),)))
                linked = draw(s.Scene((self.blocker(("exclude", ("sun",))),), (self.SUN,), splats=(self.sheet(),)))
                gone = draw(s.Scene((), (self.SUN,), splats=(self.sheet(),)))
                outside = ~self.mask(self.blocker())
                np.testing.assert_allclose(linked[outside], gone[outside], atol=1e-5)
                self.assertGreater(float(np.abs(plain[outside] - gone[outside]).max()), .1)

    def test_cpu_modes_a_relit_sheet_excluded_from_the_sun_is_unlit(self):
        for mode in ("raster", "raytrace"):
            lit = s.render(s.Scene((), (self.SUN,), splats=(self.sheet(),)), SPLAT_CAMERA, 24, 24, ambient=.1, mode=mode)
            dark = s.render(s.Scene((), (self.SUN,), splats=(self.sheet(("exclude", ("sun",))),)), SPLAT_CAMERA, 24, 24,
                            ambient=.1, mode=mode)
            self.assertGreater(float(lit[12, 12, :3].max()), 5 * float(dark[12, 12, :3].max()))

    def check_path(self, backend):
        draw = lambda scene: pt.render(scene, SPLAT_CAMERA, 24, 24, BLACK, 0.0, "rgba", self.PATH, backend=backend)
        plain = draw(s.Scene((self.blocker(),), (self.SUN,), splats=(self.sheet(),)))
        linked = draw(s.Scene((self.blocker(NOBODY),), (self.SUN,), splats=(self.sheet(),)))
        gone = draw(s.Scene((), (self.SUN,), splats=(self.sheet(),)))
        outside = ~self.mask(self.blocker())
        np.testing.assert_allclose(linked[outside], gone[outside], atol=2e-5)
        self.assertGreater(float(np.abs(plain[outside] - gone[outside]).max()), .1)
        dark = draw(s.Scene((), (self.SUN,), splats=(self.sheet(NOBODY),)))
        lit = draw(s.Scene((), (self.SUN,), splats=(self.sheet(),)))
        self.assertGreater(float(lit[12, 12, :3].max()), 5 * float(dark[12, 12, :3].max()))

    def test_cpu_path_tracer(self):
        self.check_path("cpu")

    @unittest.skipUnless(gpu3d.available(), "no GPU adapter")
    def test_gpu_path_tracer(self):
        if not gpupathtrace.soft_supported(gpu3d._state()):
            self.skipTest("this adapter does not support GPU splat path tracing")
        self.check_path("gpu")

    def test_a_splat_set_excluded_from_the_sun_casts_no_shadow_on_a_mesh(self):
        wall = replace(box((4, 4, .1), (.6, .6, .6, 1), (0, 0, -.8)))
        small = lambda link: instance(plane_cloud(n=8, size=.7), relight=0.0, light_link=link, cast_shadows=True)
        # the slanted sun puts the caster's shadow beside it on the wall, where the camera sees it
        shifted = lambda link: replace(small(link), matrix=np.array(((1, 0, 0, -.3), (0, 1, 0, .3), (0, 0, 1, .6), (0, 0, 0, 1)), np.float64))
        for mode in ("raster", "raytrace"):
            with self.subTest(mode=mode):
                draw = lambda scene: s.render(scene, SPLAT_CAMERA, 24, 24, ambient=.1, mode=mode)
                plain = draw(s.Scene((wall,), (self.SUN,), splats=(shifted(s.LIGHT_LINK_ALL),)))
                linked = draw(s.Scene((wall,), (self.SUN,), splats=(shifted(("exclude", ("sun",))),)))
                bare = draw(s.Scene((wall,), (self.SUN,), splats=(replace(shifted(s.LIGHT_LINK_ALL), opacity_scale=0.0),)))
                splat_alone = draw(s.Scene((), (), splats=(shifted(s.LIGHT_LINK_ALL),)))[..., 3] > 0
                outside = ~splat_alone
                np.testing.assert_allclose(linked[outside], bare[outside], atol=1e-5)
                self.assertGreater(float(np.abs(plain[outside] - bare[outside]).max()), .02)


class InstanceTests(unittest.TestCase):
    def scene(self, link):
        matrices = np.tile(np.eye(4), (2, 1, 1))
        matrices[:, 0, 3] = (-1.7, 1.7)
        copies = s.InstanceSet((ball(0),), matrices, np.zeros(2, np.int32), light_link=link)
        return s.Scene((ground(),), (SUN,), instances=(copies,))

    def test_an_instance_set_excluded_from_the_sun_is_unlit_and_casts_no_shadow(self):
        for mode in ("raster", "raytrace"):
            plain = s.render(self.scene(s.LIGHT_LINK_ALL), CAMERA, *SIZE, ambient=AMBIENT, mode=mode)
            linked = s.render(self.scene(("exclude", ("sun",))), CAMERA, *SIZE, ambient=AMBIENT, mode=mode)
            bare = s.render(s.Scene((ground(),), (SUN,)), CAMERA, *SIZE, ambient=AMBIENT, mode=mode)
            mine = coverage(ball(-1.7)) | coverage(ball(1.7))
            np.testing.assert_allclose(linked[~mine], bare[~mine], atol=1e-6)
            self.assertGreater(float(np.abs(plain[~mine] - bare[~mine]).max()), .05)

    def test_the_path_tracer_keeps_instances_as_instances_and_honours_the_set_link(self):
        path = pt.PathSettings(samples=8, max_bounces=1, diffuse_bounces=1)
        draw = lambda scene: pt.render(scene, CAMERA, *SIZE, BLACK, 0.0, "rgba", path)
        linked = draw(self.scene(("exclude", ("sun",))))
        bare = draw(s.Scene((ground(),), (SUN,)))
        mine = coverage(ball(-1.7)) | coverage(ball(1.7))
        np.testing.assert_allclose(linked[~mine], bare[~mine], atol=2e-5)


class OldDocumentTests(unittest.TestCase):
    def test_a_document_without_the_new_knobs_renders_as_one_with_their_defaults(self):
        d = Dispatcher()
        for key, kind, params in (("ball", "Sphere3D", dict(sphere_radius=.8, spec_amount=.3)),
                                  ("light", "Light3D", dict(shadows="on")),
                                  ("camera", "Camera3D", {}), ("scene", "Scene3D", {}),
                                  ("render", "Render3D", dict(width=24, height=18, samples=1))):
            d.execute(dict(op="create", id=key, type=kind, params=params))
        for slot, source in (("object0", "ball"), ("object1", "light")):
            d.execute(dict(op="connect", id="scene", input=slot, source=source))
        for slot in ("scene", "camera"):
            d.execute(dict(op="connect", id="render", input=slot, source=slot))
        new = Evaluator().evaluate_raster(d.document, "render").pixels
        for key, names in (("ball", ("light_link", "light_link_list")), ("light", ("light_group",))):
            for name in names:
                d.document["nodes"][key]["params"].pop(name)
        old = Evaluator().evaluate_raster(d.document, "render").pixels
        np.testing.assert_array_equal(old, new)
        self.assertGreater(float(new.max()), .1)


if __name__ == "__main__":
    unittest.main()
