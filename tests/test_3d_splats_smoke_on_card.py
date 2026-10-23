"""Smoke together with splats is drawn on the render card (`gpu3d.render`, the splat layer between the mesh passes and the smoke march).

Before 10/23 a smoke volume in a scene with splats raised `Unsupported` in `gpu3d.render`, so Render3D's `auto` backend drew the whole
frame on the processor (about 20 seconds at 1080p). The CPU order is meshes, then the splat layer, then the smoke over both, cut at the
mesh depth (`scene3d.render` and `volumerender.composite_beauty`); the card does the same in one submission pair. Each test fails
without the change: the render raises Unsupported. Run on every adapter (`force-adapter.py integrated|cpu`)."""
import unittest
import unittest.mock

import numpy as np

import tests.isolation  # noqa: F401
from nodebased import gpu3d, gpusplat, scene3d as s
from tests import gpu_precision
from tests.test_3d_gpu_volumes import CAMERA, LAMP, SETTINGS, SUN, TOLERANCE as VOLUME_TOLERANCE, plume
from tests.test_3d_splats_on_card import card, disc, many, spread

W, H = 128, 96
TOLERANCE = 2e-2 if gpu_precision.half_float_target() else 4e-3      # the volume tests' own bound (TOLERANCE * 10)


def smoke_scene(*clouds, geometries=(), lights=(SUN,), relight=0.0, shadow_catch=0.0):
    return s.Scene(geometries=tuple(geometries), lights=tuple(lights), volumes=(plume(),),
                   splats=tuple(s.SplatInstance(c, relight=relight, shadow_catch=shadow_catch) for c in clouds))


def draw(scene, playing=True, **more):
    with gpu3d.playing(playing):
        return np.asarray(gpu3d.render(scene, CAMERA, W, H, volume=SETTINGS, ambient=.2, **more))


def reference(scene, **more):
    return np.asarray(s.render(scene, CAMERA, W, H, volume=SETTINGS, ambient=.2, **more))


@unittest.skipUnless(gpu3d.available(), 'no wgpu adapter')
class SmokeWithSplatsOnTheCard(unittest.TestCase):
    def cases(self):
        cloud = many(300, 3)
        return {'splats and smoke': smoke_scene(cloud),
                'splats, a card and smoke': smoke_scene(cloud, geometries=(card(-.5),)),
                'splats in front of a card and smoke': smoke_scene(cloud, geometries=(card(-1.5),)),
                'two clouds, a card and a lamp': smoke_scene(many(200, 9), disc((.2, .5, 1.0), (.9, .2, .1)),
                                                              geometries=(card(0.0),), lights=(SUN, LAMP))}

    def test_the_frame_is_drawn_on_the_card_with_no_readback_of_the_splat_layer(self):
        for name, scene in self.cases().items():
            with self.subTest(case=name), unittest.mock.patch.object(gpusplat, 'render_layer', side_effect=AssertionError(
                    'the splat layer was read back and composited on the processor')):
                frame = draw(scene)
            self.assertEqual(frame.shape, (H, W, 4))
            self.assertGreater(float(frame[..., 3].max()), .5)

    def test_the_card_frame_agrees_with_the_processor_frame(self):
        for name, scene in self.cases().items():
            with self.subTest(case=name):
                expected, actual = reference(scene), draw(scene)
                self.assertGreater(int((expected[..., 3] > .02).sum()), 400, 'the case must show smoke and splats')
                # the bounds of the smoke's own agreement tests (test_3d_gpu_volumes): worst pixel ten times the mean's bound
                self.assertLess(float(np.abs(actual - expected).max()), VOLUME_TOLERANCE * 10)
                self.assertLess(float(np.abs(actual - expected).mean()), VOLUME_TOLERANCE)

    def test_the_smoke_covers_the_splats_and_the_splats_show_through_it(self):
        scene = smoke_scene(many(300, 3))
        with_smoke, splats_only = draw(scene), draw(s.Scene(splats=scene.splats, lights=scene.lights))
        smoke_only = draw(s.Scene(volumes=scene.volumes, lights=scene.lights))
        self.assertGreater(float(np.abs(with_smoke - splats_only).max()), .05, 'the smoke changed nothing')
        self.assertGreater(float(np.abs(with_smoke - smoke_only).max()), .05, 'the splats changed nothing')

    def test_the_splats_lit_by_a_light_with_no_shadows_are_drawn_on_the_card(self):
        scene = smoke_scene(many(300, 3), relight=1.0)
        np.testing.assert_allclose(draw(scene), reference(scene), atol=TOLERANCE * 3, rtol=0)

    def test_smoke_shadows_on_lit_splats_stay_on_the_processor_outside_playback(self):
        sun = s.Light('Directional', position=s.Vec3(4, 5, 3), target=s.Vec3(), shadows=True)
        scene = smoke_scene(many(100, 3), relight=1.0, lights=(sun,))
        with self.assertRaises(gpu3d.Unsupported):
            draw(scene, playing=False)
        frame = draw(scene)          # a played frame lights its splats without shadows, with smoke or without
        self.assertGreater(float(frame[..., 3].max()), .5)

    def test_the_old_readback_route_leaves_smoke_with_splats_to_the_processor(self):
        with unittest.mock.patch.object(gpu3d, 'SPLATS_ON_CARD', False), self.assertRaises(gpu3d.Unsupported):
            draw(smoke_scene(many(100, 3)))

    def test_smoke_alone_and_splats_alone_are_unchanged(self):
        smoke = s.Scene(volumes=(plume(),), lights=(SUN,))
        np.testing.assert_allclose(draw(smoke), reference(smoke), atol=TOLERANCE, rtol=0)
        splats = s.Scene(splats=(s.SplatInstance(many(300, 3)),))
        np.testing.assert_allclose(draw(splats), reference(splats), atol=TOLERANCE, rtol=0)

    def test_a_frame_in_several_bands_agrees(self):
        # a smoke march too long for one submission goes out in row bands; each band gets its share of the splat layer
        scene = smoke_scene(many(300, 3), geometries=(card(-.5),))
        with unittest.mock.patch.object(gpu3d, 'GPU_FORCE_BANDS', 3), unittest.mock.patch.object(
                gpusplat, 'render_layer', side_effect=AssertionError('the splat layer was read back')):
            actual = draw(scene)
        np.testing.assert_allclose(actual, reference(scene), atol=TOLERANCE, rtol=0)

    def test_a_background_and_two_samples_agree(self):
        scene = smoke_scene(many(300, 3), geometries=(card(-.5),))
        expected = reference(scene, background=(.1, .2, .3, 1.0), samples=2)
        actual = draw(scene, background=(.1, .2, .3, 1.0), samples=2)
        self.assertLess(float(np.abs(actual - expected).max()), TOLERANCE * 3)


@unittest.skipUnless(gpu3d.available(), 'no wgpu adapter')
class SmokeWithMeshesIsResident(unittest.TestCase):
    """A scene with smoke and meshes took the per-frame path (the meshes packed and uploaded again every frame: 1.3 s for the mixed
    scene with a smoke puff at 1080p on the RTX 3080 Ti, against 22 ms without the smoke). The meshes now stay on the card
    and only the smoke's bind groups are made per frame."""

    def test_the_scene_is_resident_and_the_second_camera_reuses_it(self):
        scene = smoke_scene(geometries=(card(-.5),))
        scene = s.Scene(geometries=scene.geometries, lights=scene.lights, volumes=scene.volumes)
        self.assertTrue(gpu3d._resident_ok(scene, 'rgba', SETTINGS))
        state = gpu3d._state(None)
        draw(scene)
        builds, reuses = state.get('resident_builds', 0), state.get('resident_reuses', 0)
        with gpu3d.playing(True):
            gpu3d.render(scene, s.Camera(s.Transform3D(s.Vec3(0.5, 0.6, 3.2)), s.Vec3(0, 0.5, 0)), W, H, volume=SETTINGS, ambient=.2)
        self.assertEqual((state.get('resident_builds', 0), state.get('resident_reuses', 0)), (builds, reuses + 1))

    def test_meshes_with_smoke_and_a_shadowing_light_agree_with_the_processor(self):
        # the smoke shadows the card and the card shades the smoke: the mesh shader's volume variant on the resident path
        sun = s.Light('Directional', position=s.Vec3(4, 5, 3), target=s.Vec3(), shadows=True)
        floor = s._card(3, 3, (.7, .7, .6, 1), s.Transform3D(s.Vec3(0, 0, 0), s.Vec3(-90, 0, 0)), None)
        scene = s.Scene(geometries=(floor, card(-.8)), lights=(sun,), volumes=(plume(),))
        expected, actual = reference(scene), draw(scene)
        self.assertLess(float(np.abs(actual - expected).max()), TOLERANCE * 2)
        self.assertLess(float(np.abs(actual - expected).mean()), TOLERANCE / 10)

    def test_the_meshes_shadows_on_the_smoke_come_from_the_triangle_tree_and_agree_with_the_brute_force_walk(self):
        # a card between the smoke and the lamp, a translucent one beside it: the tree walk multiplies every card the ray crosses
        lamp = s.Light('Point', color=(1.0, .9, .8), intensity=4.0, position=s.Vec3(-1.5, 2.5, 1.0), target=s.Vec3(0, .5, 0),
                       falloff_type='Quadratic', shadows=True)
        blocker = s._card(1.0, 1.0, (.3, .3, .3, 1), s.Transform3D(s.Vec3(-.4, 1.0, .6), s.Vec3(0, 30, 0)), None)
        pane = s._card(1.0, 1.0, (.8, .9, 1, .5), s.Transform3D(s.Vec3(-.2, .8, .9), s.Vec3(0, 20, 0)), None)
        scene = s.Scene(geometries=(card(-1.0), blocker, pane), lights=(lamp,), volumes=(plume(),))
        brute = unittest.mock.patch.object(gpu3d, 'SHADOW_BVH_THRESHOLD', 10 ** 9)
        tree = unittest.mock.patch.object(gpu3d, 'SHADOW_BVH_THRESHOLD', 0)
        with brute:
            walked = draw(scene)
            self.assertEqual(gpu3d.last_shadow_path, 'brute')
        with tree:
            treed = draw(scene)
            self.assertEqual(gpu3d.last_shadow_path, 'bvh')
        expected = reference(scene)
        self.assertGreater(float(np.abs(walked - draw(s.Scene(geometries=(card(-1.0),), lights=(lamp,), volumes=(plume(),)))).max()), .02,
                           'the cards must shadow the smoke')
        np.testing.assert_allclose(treed, walked, atol=TOLERANCE / 4, rtol=0)
        self.assertLess(float(np.abs(treed - expected).max()), TOLERANCE * 2)

    def test_translucent_meshes_with_smoke_and_splats_are_drawn_on_the_card_in_one_band_or_several(self):
        # glass, splats around it and smoke over all of it: the splat layer goes around the glass band by band
        scene = smoke_scene(spread(70, 3), geometries=(card(0.0, alpha=.5),))
        expected = reference(scene)
        for bands in (1, 3):
            with self.subTest(bands=bands), unittest.mock.patch.object(gpu3d, 'GPU_FORCE_BANDS', bands), unittest.mock.patch.object(
                    gpusplat, 'render_layer', side_effect=AssertionError('the splat layer was read back')):
                actual = draw(scene)
            self.assertGreater(int((expected[..., 3] > .02).sum()), 400)
            np.testing.assert_allclose(actual, expected, atol=TOLERANCE * 3, rtol=0)

    def test_translucent_meshes_with_smoke_and_splats_outside_playback_stay_on_the_processor(self):
        with self.assertRaises(gpu3d.Unsupported):          # the ray tracer's layer merge cannot take smoke
            draw(smoke_scene(many(100, 3), geometries=(card(0.0, alpha=.5),)), playing=False)


if __name__ == '__main__':
    unittest.main()
