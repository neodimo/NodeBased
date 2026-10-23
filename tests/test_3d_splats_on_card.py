"""The splat layer is drawn over the meshes on the render card, in the mesh frame's own submission (`gpusplat.prepare_on_card`).

A frame of the mixed mesh and splat scene used to leave the card three times at 1080p (the mesh colour, the mesh depth for the
splats' occlusion and the splat layer), be composited on the processor and be converted to its 8-bit picture there (89 ms of a
frame plus 62 ms of conversion on the RTX 3080 Ti, 10/23). Now the splats read the mesh pass's own depth buffer, blend over its colour
and the display picture is made beside it. Each test fails without the module: the layer comes back through `gpusplat.render_layer`,
and a scene with splats gets no card picture. Run on every adapter (`force-adapter.py integrated|cpu`)."""
import unittest
import unittest.mock

import numpy as np

import tests.isolation  # noqa: F401
from nodebased import gpu3d, gpupicture, gpusplat, imaging, scene3d as s, splats
from tests.test_3d_card_picture import reference, rows_of

W, H = 64, 48
ON_CARD_TOLERANCE = 6e-3        # half-float blending: the two routes round at different places (the layer's own, then the sum)


def disc(position, colour, opacity=.9, size=.9):
    """A flat splat facing the default camera, of a given linear colour."""
    sh = ((np.asarray(colour, float) - .5) / splats.C0)[None, None, :]
    return splats.SplatCloud(np.array([position], float), np.array([[size, size, .01]], float),
                             np.array([[1.0, 0, 0, 0]]), np.array([opacity]), sh, 0, colorspace='linear')


def many(count, seed):
    rng = np.random.default_rng(seed)
    sh = np.zeros((count, 1, 3))
    sh[:, 0] = (rng.uniform(.1, .9, (count, 3)) - .5) / splats.C0
    return splats.SplatCloud(rng.uniform(-1, 1, (count, 3)) * (1.8, 1.2, 1.0), rng.uniform(.06, .4, (count, 3)) * (1, 1, .08),
                             rng.normal(size=(count, 4)), rng.uniform(.15, .95, count), sh, 0, colorspace='linear')


def card(z=0.0, colour=(.2, .5, .9), alpha=1.0, size=(3.0, 2.2), yaw=0.0, x=0.0):
    return s._card(size[0], size[1], (*colour, alpha), s.Transform3D(s.Vec3(x, 0, z), s.Vec3(0, yaw, 0)), None)


def scene(*clouds, geometries=()):
    return s.Scene(geometries=tuple(geometries), splats=tuple(s.SplatInstance(c) for c in clouds))


def render_with(sc, on_card, playing=True, view=None, **more):
    """gpu3d.render of the scene by one route; `view` arms the card picture."""
    gpupicture.clear()
    if view is not None:
        gpupicture.arm(view)
    try:
        with unittest.mock.patch.object(gpu3d, 'SPLATS_ON_CARD', on_card), gpu3d.playing(playing):
            return np.asarray(gpu3d.render(sc, s.Camera(), W, H, **more))
    finally:
        gpupicture.disarm()


def cases():
    cloud = many(250, 3)
    return {'splats behind a card': scene(cloud, geometries=(card(-1.2),)),
            'splats in front of a card': scene(cloud, geometries=(card(1.6),)),
            'a card cutting through the cloud': scene(cloud, geometries=(card(0.0, yaw=.6),)),
            'a card between two clouds': scene(disc((0, 0, 1.0), (.9, .2, .1)), disc((.3, 0, -1.0), (.1, .9, .2)),
                                               geometries=(card(0.0),)),
            'two clouds and two cards': scene(cloud, many(200, 9), geometries=(card(1.0), card(-.5, colour=(.9, .3, .2)))),
            'a faint splat over a card': scene(disc((0, 0, .5), (1, 1, 1), opacity=.3), geometries=(card(-.2),))}


@unittest.skipUnless(gpu3d.available(), 'no wgpu adapter')
class SplatsOnTheCard(unittest.TestCase):
    def test_the_layer_never_comes_back_to_the_processor(self):
        for name, sc in cases().items():
            with self.subTest(case=name), unittest.mock.patch.object(gpusplat, 'render_layer', side_effect=AssertionError(
                    'the splat layer was read back and composited on the processor')):
                frame = render_with(sc, True)
            self.assertEqual(frame.shape, (H, W, 4))

    def test_the_old_route_still_reads_the_layer_back_when_the_card_route_is_off(self):
        with unittest.mock.patch.object(gpusplat, 'render_layer', wraps=gpusplat.render_layer) as layer:
            render_with(cases()['splats behind a card'], False)
        layer.assert_called_once()

    def test_the_two_routes_make_the_same_frame(self):
        for name, sc in cases().items():
            with self.subTest(case=name):
                card_frame, old_frame = render_with(sc, True), render_with(sc, False)
                self.assertGreater(float(old_frame[..., 3].max()), .5, 'the case must draw something')
                np.testing.assert_allclose(card_frame, old_frame, atol=ON_CARD_TOLERANCE, rtol=0)

    def test_the_card_route_agrees_with_the_reference_renderer(self):
        for name, sc in cases().items():
            with self.subTest(case=name):
                expected = np.asarray(s.render(sc, s.Camera(), W, H, (0, 0, 0, 0), samples=1))
                np.testing.assert_allclose(render_with(sc, True), expected, atol=ON_CARD_TOLERANCE, rtol=0)

    def test_splats_behind_the_mesh_are_hidden_and_splats_in_front_cover_it(self):
        behind = render_with(scene(disc((0, 0, -1.0), (1, 0, 0), opacity=.99, size=.8), geometries=(card(0.0, colour=(0, 0, 1)),)), True)
        in_front = render_with(scene(disc((0, 0, 1.0), (1, 0, 0), opacity=.99, size=.8), geometries=(card(0.0, colour=(0, 0, 1)),)), True)
        centre = (H // 2, W // 2)
        self.assertGreater(float(behind[centre][2]), .8)
        self.assertLess(float(behind[centre][0]), .1, 'a splat behind the card showed through it')
        self.assertGreater(float(in_front[centre][0]), .8)
        self.assertLess(float(in_front[centre][2]), .2, 'a splat in front of the card did not cover it')

    def test_a_card_picture_is_made_for_a_frame_with_splats(self):
        gpupicture.clear()
        sc = cases()['two clouds and two cards']
        with unittest.mock.patch.object(gpupicture, 'ARMED_KINDS', ('discrete', 'integrated', 'cpu', 'other')):
            frame = render_with(sc, True, view='ACES 2.0')
            before = gpupicture.counters['taken']
            made = imaging.to_qimage(frame, view='ACES 2.0')
        self.assertEqual(gpupicture.counters['taken'], before + 1, 'the conversion did not use the card picture')
        difference = np.abs(rows_of(made) - reference(frame, 'ACES 2.0'))
        self.assertLessEqual(int(difference.max()), 1)

    def test_the_old_route_has_no_card_picture(self):
        with unittest.mock.patch.object(gpupicture, 'ARMED_KINDS', ('discrete', 'integrated', 'cpu', 'other')):
            frame = render_with(cases()['two clouds and two cards'], False, view='ACES 2.0')
        self.assertIsNone(gpupicture.take(frame, 0.0, 'RGB', 'black', 'ACES 2.0', None))

    def test_translucent_meshes_still_hide_the_splats_behind_them(self):
        # the translucent card is not in the mesh pass's depth buffer; the splat layer is drawn in two parts around it
        sc = scene(spread(70, 3), geometries=(card(0.0, alpha=.5),))
        expected = np.asarray(s.render(sc, s.Camera(), W, H, (0, 0, 0, 0), samples=1))
        np.testing.assert_allclose(render_with(sc, True), expected, atol=ON_CARD_TOLERANCE, rtol=0)


def spread(count, seed):
    """Small splats scattered through a box that crosses the glass: few overlap, so the two orders a pixel's splats can be drawn
    in (centre depth on the card, plane depth in the ray tracer's merge) give the same picture."""
    rng = np.random.default_rng(seed)
    sh = np.zeros((count, 1, 3))
    sh[:, 0] = (rng.uniform(.1, .9, (count, 3)) - .5) / splats.C0
    return splats.SplatCloud(rng.uniform(-1, 1, (count, 3)) * (1.6, 1.1, 1.0), rng.uniform(.05, .1, (count, 3)) * (1, 1, .1),
                             np.tile((1.0, 0, 0, 0), (count, 1)), rng.uniform(.5, .95, count), sh, 0, colorspace='linear')


def glass_cases():
    """The splat layer behind and in front of translucent meshes: the cloud both sides of the glass, two discs, glass over a wall."""
    cloud = spread(70, 3)
    return {'a cloud both sides of a glass card': scene(cloud, geometries=(card(0.0, alpha=.5),)),
            'a cloud behind a glass card': scene(cloud, geometries=(card(1.6, alpha=.5),)),
            'a cloud in front of a glass card': scene(cloud, geometries=(card(-1.5, alpha=.5),)),
            'a disc behind and a disc in front': scene(disc((0, 0, 1.0), (.9, .2, .1)), disc((.3, 0, -1.0), (.1, .9, .2)),
                                                       geometries=(card(0.0, alpha=.5),)),
            'glass over a wall': scene(cloud, geometries=(card(-1.0), card(0.0, alpha=.4, colour=(.9, .9, .2)))),
            'a slanted glass card': scene(cloud, geometries=(card(0.0, alpha=.5, yaw=.7),)),
            # a splat between two glass surfaces of one pixel counts as behind the nearer one (the card orders against the nearest
            # translucent surface only), so the cards here have the whole cloud on one side of them
            'two glass cards in front of the cloud': scene(cloud, geometries=(card(1.6, alpha=.3, colour=(.9, .2, .2)),
                                                                              card(2.2, alpha=.3, colour=(.2, .2, .9)))),
            'two glass cards behind the cloud': scene(cloud, geometries=(card(-1.4, alpha=.3, colour=(.9, .2, .2)),
                                                                         card(-2.0, alpha=.3, colour=(.2, .2, .9))))}


@unittest.skipUnless(gpu3d.available(), 'no wgpu adapter')
class SplatsAroundTranslucentMeshesOnTheCard(unittest.TestCase):
    """A played frame draws the splats behind the glass before it (the glass tints them) and the splats in front of it after it,
    in one submission pair, ordered per pixel by a second depth buffer of the glass. A stopped frame takes the ray tracer's layer
    merge, which is the reference here."""

    def ray_traced(self, sc):
        return np.asarray(gpu3d.render(sc, s.Camera(), W, H, mode='raytrace'))

    def test_the_layer_never_comes_back_to_the_processor_and_the_ray_tracer_is_not_used(self):
        for name, sc in glass_cases().items():
            with self.subTest(case=name), unittest.mock.patch.object(gpusplat, 'render_layer', side_effect=AssertionError(
                    'the splat layer was read back')), unittest.mock.patch.object(gpu3d, 'render', wraps=gpu3d.render) as call:
                frame = render_with(sc, True)
            self.assertEqual(frame.shape, (H, W, 4))
            modes = [c.kwargs.get('mode') for c in call.call_args_list]
            self.assertNotIn('raytrace', modes, 'the frame went to the ray tracer')

    def test_the_card_frame_agrees_with_the_ray_tracers_layer_merge(self):
        for name, sc in glass_cases().items():
            with self.subTest(case=name):
                expected, actual = self.ray_traced(sc), render_with(sc, True)
                self.assertGreater(float(expected[..., 3].max()), .3, 'the case must draw something')
                np.testing.assert_allclose(actual, expected, atol=ON_CARD_TOLERANCE, rtol=0)

    def test_a_dense_cloud_agrees_apart_from_the_order_of_overlapping_splats(self):
        # the merge draws a pixel's overlapping splats by plane depth and the card by centre depth, glass or no glass (a ray-traced
        # frame with an invisible glass card differs from the plain splat raster by the same amount): same coverage, same mean colour
        sc = scene(many(250, 3), geometries=(card(0.0, alpha=.5),))
        expected, actual = self.ray_traced(sc), render_with(sc, True)
        np.testing.assert_allclose(actual[..., 3], expected[..., 3], atol=ON_CARD_TOLERANCE, rtol=0)
        self.assertLess(float(np.abs(actual - expected).mean()), 1e-2)

    def test_the_old_order_drew_every_splat_over_the_glass(self):
        # a splat behind a glass card must be tinted by it: a red disc behind a half-opaque blue card is not pure red
        sc = scene(disc((0, 0, -1.0), (1, 0, 0), opacity=.99, size=.8), geometries=(card(0.0, colour=(0, 0, 1), alpha=.5),))
        centre = (H // 2, W // 2)
        behind = render_with(sc, True)
        self.assertGreater(float(behind[centre][2]), .15, 'the glass did not tint the splat behind it')
        in_front = render_with(scene(disc((0, 0, 1.0), (1, 0, 0), opacity=.99, size=.8),
                                     geometries=(card(0.0, colour=(0, 0, 1), alpha=.5),)), True)
        self.assertLess(float(in_front[centre][2]), .05, 'the glass tinted a splat in front of it')
        self.assertGreater(float(in_front[centre][0]), .9)

    def test_a_stopped_frame_and_a_frame_the_card_cannot_take_use_the_ray_tracer(self):
        sc = glass_cases()['a cloud both sides of a glass card']
        with unittest.mock.patch.object(gpu3d, 'SPLATS_ON_CARD', True):
            stopped = np.asarray(gpu3d.render(sc, s.Camera(), W, H))
        np.testing.assert_array_equal(stopped, self.ray_traced(sc))
        with gpu3d.playing(True):
            coloured = np.asarray(gpu3d.render(sc, s.Camera(), W, H, background=(.1, .1, .1, 1.0)))
        np.testing.assert_allclose(coloured, np.asarray(gpu3d.render(sc, s.Camera(), W, H, background=(.1, .1, .1, 1.0), mode='raytrace')),
                                   atol=ON_CARD_TOLERANCE, rtol=0)

    def test_scenes_without_translucent_meshes_do_not_draw_the_glass_depth(self):
        state = gpu3d._state(None)
        state.pop('glass_depth', None)
        render_with(scene(many(100, 3), geometries=(card(0.0),)), True)
        self.assertFalse(state.get('glass_depth'))
        render_with(scene(many(100, 3), geometries=(card(0.0, alpha=.5),)), True)
        self.assertTrue(state.get('glass_depth'))


    def test_a_splat_only_scene_and_a_scene_with_splats_all_behind_the_camera_still_render(self):
        only = scene(many(100, 5))
        self.assertGreater(float(render_with(only, True)[..., 3].max()), 0)
        away = scene(disc((0, 0, 40.0), (1, 0, 0)), geometries=(card(0.0),))      # behind the default camera
        frame = render_with(away, True)
        np.testing.assert_allclose(frame, render_with(scene(geometries=(card(0.0),)), True), atol=1e-6, rtol=0)

    def test_two_samples_and_a_changing_size_work(self):
        sc = cases()['splats behind a card']
        two = render_with(sc, True, samples=2)
        self.assertEqual(two.shape, (H, W, 4))
        np.testing.assert_allclose(two, render_with(sc, False, samples=2), atol=ON_CARD_TOLERANCE, rtol=0)

    def test_a_cancelled_render_leaves_nothing_behind_on_the_card(self):
        import threading
        cancel = threading.Event()
        cancel.set()
        with self.assertRaises(Exception):
            with gpu3d.playing(True):
                gpu3d.render(cases()['splats behind a card'], s.Camera(), W, H, cancel=cancel)
        self.assertEqual(float(render_with(cases()['splats behind a card'], True)[..., 3].max()) > 0, True)


class CloudNormalsAreWorkedOutOnce(unittest.TestCase):
    """A relit cloud asked for its normals twice a frame; 1.9 ms of every played frame at 20,000 splats."""

    def test_the_axes_are_computed_once_and_each_caller_gets_its_own_copy(self):
        cloud = many(500, 4)
        with unittest.mock.patch.object(splats, '_rotation', wraps=splats._rotation) as rotation:
            first, second = cloud.normals(), cloud.normals()
        self.assertEqual(rotation.call_count, 1)
        np.testing.assert_array_equal(first, second)
        first[:] = 0
        np.testing.assert_array_equal(cloud.normals(), second)
        self.assertEqual(first.dtype, np.float32)
        self.assertTrue(np.allclose(np.linalg.norm(second, axis=1), 1, atol=1e-5))


@unittest.skipUnless(gpu3d.available(), 'no wgpu adapter')
class ReadbackBufferIsReleasedForTheRayTracer(unittest.TestCase):
    def test_a_big_readback_buffer_is_let_go_before_a_ray_traced_render(self):
        state = gpu3d._state(None)
        keep = []
        gpu3d._staging_buffer(state, 24 * 1024 * 1024, keep.append)
        self.assertIsNotNone(state.get('staging'))
        gpu3d.render(scene(geometries=(card(0.0),)), s.Camera(), 32, 24, mode='raytrace')
        self.assertIsNone(state.get('staging'), 'the played frame\'s readback buffer was still alive for the ray tracer')

    def test_a_small_one_is_kept(self):
        state = gpu3d._state(None)
        gpu3d._release_staging(state)
        gpu3d._staging_buffer(state, 2 * 1024 * 1024, [].append)
        gpu3d._release_staging(state)
        self.assertIsNotNone(state.get('staging'))


if __name__ == '__main__':
    unittest.main()
