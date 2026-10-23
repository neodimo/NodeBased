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
from tests.test_3d_gpu_volumes import CAMERA, LAMP, SETTINGS, SUN, plume
from tests.test_3d_splats_on_card import card, disc, many

W, H = 128, 96
TOLERANCE = 2e-2 if gpu_precision.half_float_target() else 4e-3      # the volume tests' own bound (TOLERANCE * 10)


def smoke_scene(*clouds, geometries=(), lights=(SUN,), relight=0.0, shadow_catch=0.0):
    return s.Scene(geometries=tuple(geometries), lights=tuple(lights), volumes=(plume(),),
                   splats=tuple(s.SplatInstance(c, relight=relight, shadow_catch=shadow_catch) for c in clouds))


def draw(scene, **more):
    with gpu3d.playing(True):
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
                self.assertLess(float(np.abs(actual - expected).max()), TOLERANCE)

    def test_the_smoke_covers_the_splats_and_the_splats_show_through_it(self):
        scene = smoke_scene(many(300, 3))
        with_smoke, splats_only = draw(scene), draw(s.Scene(splats=scene.splats, lights=scene.lights))
        smoke_only = draw(s.Scene(volumes=scene.volumes, lights=scene.lights))
        self.assertGreater(float(np.abs(with_smoke - splats_only).max()), .05, 'the smoke changed nothing')
        self.assertGreater(float(np.abs(with_smoke - smoke_only).max()), .05, 'the splats changed nothing')

    def test_the_splats_lit_by_a_light_with_no_shadows_are_drawn_on_the_card(self):
        scene = smoke_scene(many(300, 3), relight=1.0)
        np.testing.assert_allclose(draw(scene), reference(scene), atol=TOLERANCE * 3, rtol=0)

    def test_smoke_shadows_on_lit_splats_stay_on_the_processor(self):
        sun = s.Light('Directional', position=s.Vec3(4, 5, 3), target=s.Vec3(), shadows=True)
        with self.assertRaises(gpu3d.Unsupported):
            draw(smoke_scene(many(100, 3), relight=1.0, lights=(sun,)))

    def test_the_old_readback_route_leaves_smoke_with_splats_to_the_processor(self):
        with unittest.mock.patch.object(gpu3d, 'SPLATS_ON_CARD', False), self.assertRaises(gpu3d.Unsupported):
            draw(smoke_scene(many(100, 3)))

    def test_smoke_alone_and_splats_alone_are_unchanged(self):
        smoke = s.Scene(volumes=(plume(),), lights=(SUN,))
        np.testing.assert_allclose(draw(smoke), reference(smoke), atol=TOLERANCE, rtol=0)
        splats = s.Scene(splats=(s.SplatInstance(many(300, 3)),))
        np.testing.assert_allclose(draw(splats), reference(splats), atol=TOLERANCE, rtol=0)

    def test_a_background_and_two_samples_agree(self):
        scene = smoke_scene(many(300, 3), geometries=(card(-.5),))
        expected = reference(scene, background=(.1, .2, .3, 1.0), samples=2)
        actual = draw(scene, background=(.1, .2, .3, 1.0), samples=2)
        self.assertLess(float(np.abs(actual - expected).max()), TOLERANCE * 3)


if __name__ == '__main__':
    unittest.main()
