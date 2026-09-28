"""Splats in the path tracer (lane L4 step R4 of 7): the CPU reference against the ray-traced relight, mirrors, glass,
shadows both ways and light thrown by a splat onto a mesh."""
import dataclasses
import unittest

import numpy as np

from nodebased import intrinsics as I, pathtrace as pt, ptsplats, scene3d as s, splats
from tests.test_3d_pathtrace import FRONT, box, card, center, sphere, trace, uniform_env


def plane_cloud(n=24, size=2.0, albedo=(0.6, 0.5, 0.4), roughness=0.6, delit=True, opacity=0.99, z=0.0,
                flip=False):
    """An n x n sheet of round splats facing +z (or -z), each with its own colour `albedo` (linear)."""
    xs = (np.arange(n) + 0.5) / n * size - size / 2
    gx, gy = np.meshgrid(xs, xs)
    count = n * n
    positions = np.stack((gx.ravel(), gy.ravel(), np.full(count, z)), 1)
    spacing = size / n
    quats = np.tile((0.0, 1.0, 0.0, 0.0) if flip else (1.0, 0.0, 0.0, 0.0), (count, 1))
    dc = np.zeros((count, 1, 3))
    dc[:] = (np.asarray(albedo, float) - 0.5) / splats.C0
    cloud = splats.SplatCloud(positions, np.tile((spacing * 0.75, spacing * 0.75, spacing * 0.02), (count, 1)), quats,
                              np.full(count, opacity), dc, 0, colorspace="linear")
    if delit:
        normals = np.tile((0, 0, -1.0 if flip else 1.0), (count, 1)).astype(np.float32)
        layer = I.Intrinsics(np.tile(albedo, (count, 1)), np.full(count, roughness), normals, np.ones(count),
                             np.ones(count), np.ones(count), np.zeros(3), np.zeros((0, 3)), np.zeros((0, 3)), 0)
        cloud = I.attach(cloud, layer)
    return cloud


SUN = s.Light("Directional", (1, 1, 1), 1.0, s.Vec3(0, 0, 0), s.Vec3(0.3, -0.2, -1.0))
SPLAT_CAMERA = s.Camera(s.Transform3D(position=s.Vec3(0, 0, 4)), s.Vec3(0, 0, 0), 30.0)


def instance(cloud, **fields):
    fields.setdefault("relight", 1.0)
    return s.SplatInstance(cloud, **fields)


def unpremultiplied(image, floor=0.5):
    covered = image[..., 3] > floor
    rgb = np.where(covered[..., None], image[..., :3] / np.maximum(image[..., 3:4], 1e-6), 0.0)
    return rgb, covered


class RelightParityTests(unittest.TestCase):
    """At one bounce the path tracer's relit splats are the ray-traced renderer's, up to sampling noise."""

    def compare(self, delit, tolerance):
        scene = s.Scene(lights=(SUN,), splats=(instance(plane_cloud(delit=delit)),))
        reference = s.render(scene, SPLAT_CAMERA, 24, 24, mode="raytrace", ambient=0.1)
        got = pt.render(scene, SPLAT_CAMERA, 24, 24, ambient=0.1, settings=pt.PathSettings(samples=96, max_bounces=1))
        ref_rgb, ref_cover = unpremultiplied(reference, 0.9)
        got_rgb, _ = unpremultiplied(got, 0.9)
        mask = ref_cover & (got[..., 3] > 0.9)
        self.assertGreater(int(mask.sum()), 100)
        # the sheet is uniformly lit, so the mean is the measurement and the pixel spread the noise
        self.assertLess(abs(float(got_rgb[mask].mean()) / float(ref_rgb[mask].mean()) - 1.0), tolerance)
        np.testing.assert_allclose(got_rgb[mask].mean(0) / ref_rgb[mask].mean(0), 1.0, atol=tolerance)
        return got, reference

    def test_captured_colour_splats_match_the_ray_traced_relight(self):
        self.compare(delit=False, tolerance=0.03)

    def test_de_lit_splats_match_the_ray_traced_relight(self):
        # the ray-traced relight weights the diffuse lobe by 1 - Fresnel(v.h); the path tracer by the split-sum albedo
        self.compare(delit=True, tolerance=0.08)

    def test_the_centre_pixel_agrees_closely(self):
        got, reference = self.compare(delit=False, tolerance=0.03)
        np.testing.assert_allclose(got[12, 12, :3], reference[12, 12, :3], rtol=0.03)

    def test_relight_zero_shows_the_capture_unlit(self):
        cloud = plane_cloud(albedo=(0.8, 0.2, 0.1), delit=False, n=32)
        scene = s.Scene(lights=(SUN,), splats=(instance(cloud, relight=0.0),))
        got = pt.render(scene, SPLAT_CAMERA, 16, 16, settings=pt.PathSettings(samples=64, max_bounces=1))
        rgb, covered = unpremultiplied(got, 0.9)
        np.testing.assert_allclose(rgb[covered].mean(0), (0.8, 0.2, 0.1), atol=0.02)

    def test_the_de_lit_layer_is_used_when_present(self):
        cloud = plane_cloud(albedo=(0.9, 0.9, 0.9), delit=True)
        base = plane_cloud(albedo=(0.9, 0.9, 0.9), delit=False)
        dark = I.attach(base, I.Intrinsics(np.full((len(base), 3), 0.2), np.full(len(base), 0.6),
                                           np.tile((0, 0, 1.0), (len(base), 1)).astype(np.float32), np.ones(len(base)),
                                           np.ones(len(base)), np.ones(len(base)), np.zeros(3), np.zeros((0, 3)),
                                           np.zeros((0, 3)), 0))
        render = lambda c, **kw: pt.render(s.Scene(lights=(SUN,), splats=(instance(c, **kw),)), SPLAT_CAMERA, 12, 12,
                                           settings=pt.PathSettings(samples=32, max_bounces=1))
        lit_dark, lit_captured = render(dark), render(dark, use_intrinsics=False)
        self.assertLess(float(center(lit_dark).mean()), 0.5 * float(center(lit_captured).mean()))
        self.assertIsNotNone(cloud.intrinsics)


class HitTests(unittest.TestCase):
    def test_coverage_is_the_splat_opacity(self):
        # one big splat of opacity 0.4 facing the camera: the alpha at its centre converges to 0.4
        cloud = plane_cloud(n=1, size=2.0, delit=False, opacity=0.4)
        cloud = dataclasses.replace(cloud, scales=np.full((1, 3), 3.0, np.float32))
        scene = s.Scene(splats=(instance(cloud, relight=0.0),))
        img = pt.render(scene, SPLAT_CAMERA, 8, 8, settings=pt.PathSettings(samples=800, max_bounces=1))
        self.assertAlmostEqual(float(img[3:5, 3:5, 3].mean()), 0.4 * np.exp(-0.5 * (0.0 / 3.0) ** 2), delta=0.04)

    def test_stacked_translucent_splats_composite_front_to_back(self):
        near = plane_cloud(n=1, size=2.0, albedo=(1, 0, 0), delit=False, opacity=0.5, z=1.0)
        far = plane_cloud(n=1, size=2.0, albedo=(0, 0, 1), delit=False, opacity=0.5, z=0.0)
        big = lambda c: dataclasses.replace(c, scales=np.full((1, 3), 4.0, np.float32))
        scene = s.Scene(splats=(instance(big(near), relight=0.0), instance(big(far), relight=0.0)))
        img = pt.render(scene, SPLAT_CAMERA, 8, 8, settings=pt.PathSettings(samples=1200, max_bounces=1))
        rgba = img[3:5, 3:5].reshape(-1, 4).mean(0)
        # red in front at 0.5, blue behind it at 0.5 * (1 - 0.5): premultiplied colours and total coverage
        np.testing.assert_allclose(rgba, (0.5, 0.0, 0.25, 0.75), atol=0.05)

    def test_the_data_passes_are_first_hit_and_deterministic(self):
        scene = s.Scene(splats=(instance(plane_cloud(delit=False), relight=0.0),))
        depth = pt.render(scene, SPLAT_CAMERA, 12, 12, output="depth")
        again = pt.render(scene, SPLAT_CAMERA, 12, 12, output="depth")
        np.testing.assert_array_equal(depth, again)
        covered = depth[..., 3] > 0
        self.assertGreater(int(covered.sum()), 60)
        np.testing.assert_allclose(depth[..., 0][covered], 4.0, atol=0.05)
        ids = pt.render(scene, SPLAT_CAMERA, 12, 12, output="object_id")
        self.assertTrue(np.all(ids[..., 0][covered] == 1))
        normals = pt.render(scene, SPLAT_CAMERA, 12, 12, output="normals")
        np.testing.assert_allclose(normals[..., :3][covered], np.tile((0, 0, 1), (int(covered.sum()), 1)), atol=0.02)   # the view leans in by 1 - confidence


class MirrorAndGlassTests(unittest.TestCase):
    def test_a_mirror_shows_a_splat_object(self):
        mirror = card(8, 8, (1, 1, 1, 1), (0, 0, 0), material="pbr", metallic=1.0, pbr_roughness=0.0)
        red = plane_cloud(n=16, size=4.0, albedo=(1.0, 0.1, 0.05), delit=False, z=6.0, flip=True)
        scene = s.Scene((mirror,), splats=(instance(red, relight=0.0),))
        camera = s.Camera(s.Transform3D(position=s.Vec3(0, 0, 4)), s.Vec3(0, 0, 0), 30.0)
        img = pt.render(scene, camera, 24, 24, settings=pt.PathSettings(samples=48, max_bounces=4))
        rgb, covered = unpremultiplied(img, 0.5)
        np.testing.assert_allclose(img[12, 12, :3], (1.0, 0.1, 0.05), atol=0.05)   # straight ahead, 10 away, in the glass
        self.assertGreater(float(img[12, 12, 0] - img[12, 12, 2]), 0.8)
        self.assertLess(float(img[1, 1, :3].max()), 0.02)                             # off the splat's image: the plain mirror, black

    def test_a_splat_is_seen_through_glass(self):
        slab = box((4, 4, 0.2), (1, 1, 1, 1), (0, 0, 0), material="liquid", ior=1.5, absorption_color=(1, 1, 1),
                   absorption_distance=0.2, reflection=1.0)
        behind = plane_cloud(n=24, size=6.0, albedo=(0.1, 0.9, 0.1), delit=False, z=-1.5)
        scene = s.Scene((slab,), splats=(instance(behind, relight=0.0),))
        img = pt.render(scene, FRONT, 12, 12, settings=pt.PathSettings(samples=96, max_bounces=8, diffuse_bounces=0))
        rgb, covered = unpremultiplied(img, 0.5)
        f = ((1.5 - 1) / (1.5 + 1)) ** 2
        # the clear slab passes (1 - F) / (1 + F) of the splat's green, the splat's own coverage aside
        self.assertAlmostEqual(float(rgb[5:7, 5:7, 1].mean()), 0.9 * (1 - f) / (1 + f), delta=0.06)
        self.assertLess(float(rgb[5:7, 5:7, 0].mean()), 0.15)


class ShadowAndBounceTests(unittest.TestCase):
    def _floor_scene(self, sheet=True, **fields):
        floor = card(8, 8, (0.8, 0.8, 0.8, 1), (0, -0.5, 0), (90, 0, 0))
        sheet_cloud = plane_cloud(n=12, size=1.5, albedo=(0.5, 0.5, 0.5), delit=False)
        # the sheet lies flat above the floor: rotate its splats onto the xz plane through the instance matrix
        matrix = np.eye(4)
        matrix[:3, :3] = ((1, 0, 0), (0, 0, -1), (0, 1, 0))
        matrix[1, 3] = 0.5
        # the sun leans 45 degrees, so the sheet's shadow lands 1 unit to its right, where the camera can see the floor
        sun = s.Light("Directional", (1, 1, 1), 1.0, s.Vec3(0, 0, 0), s.Vec3(1, -1, 0))
        splat_items = (instance(sheet_cloud, matrix=matrix, relight=1.0, **fields),) if sheet else ()
        return s.Scene((floor,), lights=(sun,), splats=splat_items)

    def _top(self):
        return s.Camera(s.Transform3D(position=s.Vec3(0.0, 5.0, 0.0001)), s.Vec3(0, -0.5, 0), 40.0)

    def _floor_ratio(self, **fields):
        camera, settings = self._top(), pt.PathSettings(samples=24, max_bounces=1)
        shaded = pt.render(self._floor_scene(**fields), camera, 24, 24, settings=settings, output="diffuse")
        open_floor = pt.render(self._floor_scene(sheet=False), camera, 24, 24, settings=settings, output="diffuse")
        depth = pt.render(self._floor_scene(), camera, 24, 24, output="depth")
        floor = (depth[..., 3] > 0) & (depth[..., 0] > 5.3)     # the sheet is 4.5 from the camera, the floor 5.5
        return (shaded[..., :3].sum(axis=2) / np.maximum(open_floor[..., :3].sum(axis=2), 1e-6))[floor]

    def test_a_splat_sheet_shadows_a_mesh(self):
        ratio = self._floor_ratio()
        self.assertGreater(int((ratio < 0.15).sum()), 8)            # the shadow the sheet throws
        self.assertGreater(int((ratio > 0.9).sum()), 100)           # and the lit floor around it

    def test_cast_shadows_off_lets_the_light_through(self):
        ratio = self._floor_ratio(cast_shadows=False)
        self.assertEqual(int((ratio < 0.5).sum()), 0)
        self.assertAlmostEqual(float(np.median(ratio)), 1.0, delta=0.03)

    def test_a_mesh_shadows_a_splat(self):
        blocker = card(1.0, 1.0, (0.3, 0.3, 0.3, 1), (0, 0, 1.0))
        sun = s.Light("Directional", (1, 1, 1), 1.0, s.Vec3(0, 0, 0), s.Vec3(0, 0, -1.0))
        lit = s.Scene(lights=(sun,), splats=(instance(plane_cloud(n=32, size=3.0, albedo=(0.8, 0.8, 0.8), delit=False)),))
        shadowed = s.Scene((blocker,), lights=(sun,), splats=lit.splats)
        settings = pt.PathSettings(samples=24, max_bounces=1)
        camera = s.Camera(s.Transform3D(position=s.Vec3(2.5, 0, 5)), s.Vec3(0, 0, 0), 30.0)
        with_blocker = pt.render(shadowed, camera, 20, 20, settings=settings, output="diffuse")
        without = pt.render(lit, camera, 20, 20, settings=settings, output="diffuse")
        ratio = with_blocker[..., :3].sum(axis=2) / np.maximum(without[..., :3].sum(axis=2), 1e-6)
        covered = without[..., 3] > 0.5
        self.assertLess(float(np.percentile(ratio[covered], 3)), 0.15)
        self.assertGreater(float(np.percentile(ratio[covered], 97)), 0.85)

    def test_a_glowing_splat_lights_a_mesh(self):
        # a captured red sheet at the right of a white wall, facing it (relight 0: it emits its own colour): the wall
        # turns red on that side and stays white-lit by nothing on the other
        red = plane_cloud(n=16, size=2.0, albedo=(1.0, 0.0, 0.0), delit=False)
        matrix = np.eye(4)
        matrix[:3, :3] = ((0, 0, -1), (0, 1, 0), (1, 0, 0))
        matrix[0, 3] = 1.2
        wall = card(2, 2, (0.8, 0.8, 0.8, 1), (0, 0, 0.0))
        scene = s.Scene((wall,), splats=(instance(red, matrix=matrix, relight=0.0),))
        camera = s.Camera(s.Transform3D(position=s.Vec3(0, 0, 4)), s.Vec3(0, 0, 0), 24.0)
        img = pt.render(scene, camera, 16, 16, settings=pt.PathSettings(samples=128, max_bounces=2),
                        output="diffuse")
        right, left = img[6:10, 12:14, :3].mean(axis=(0, 1)), img[6:10, 2:4, :3].mean(axis=(0, 1))
        self.assertGreater(float(right[0]), 0.05)
        self.assertLess(float(right[1]), 0.1 * float(right[0]))
        self.assertLess(float(right[2]), 0.1 * float(right[0]))
        self.assertGreater(float(right[0]), 2.0 * float(left[0]))


class InstanceTests(unittest.TestCase):
    def test_a_rotated_instance_with_view_dependent_colour_matches_the_ray_traced_relight_at_every_mix(self):
        base = plane_cloud(n=16, delit=False)
        dc = np.zeros((len(base), 4, 3))
        dc[:, 0] = base.sh[:, 0]
        dc[:, 1, 0] = 0.3                                          # a degree 1 lobe: the capture changes with the view
        cloud = splats.SplatCloud(base.positions, base.scales, base.rotations, base.opacity, dc, 1, colorspace="linear")
        turn = np.eye(4)
        turn[:3, :3] = ((0, 0, 1), (0, 1, 0), (-1, 0, 0))          # the sheet turned to face +x
        camera = s.Camera(s.Transform3D(position=s.Vec3(4, 0, 0)), s.Vec3(0, 0, 0), 30.0)
        for relight in (0.0, 0.5, 1.0):
            with self.subTest(relight=relight):
                scene = s.Scene(lights=(SUN,), splats=(instance(cloud, relight=relight, matrix=turn),))
                got = pt.render(scene, camera, 12, 12, ambient=0.1, settings=pt.PathSettings(samples=16, max_bounces=2))
                ref = s.render(scene, camera, 12, 12, mode="raytrace", ambient=0.1)
                np.testing.assert_allclose(got[6, 6, :3], ref[6, 6, :3], rtol=0.03, atol=0.005)

    def test_several_instances_and_one_too_faint_to_draw(self):
        faint = plane_cloud(n=4, delit=False, opacity=0.001)
        solid = plane_cloud(n=12, delit=True)
        both = s.Scene(lights=(SUN,), splats=(instance(faint), instance(solid)))
        only = s.Scene(lights=(SUN,), splats=(instance(solid),))
        settings = pt.PathSettings(samples=16, max_bounces=1)
        np.testing.assert_array_equal(pt.render(both, SPLAT_CAMERA, 8, 8, settings=settings),
                                      pt.render(only, SPLAT_CAMERA, 8, 8, settings=settings))
        nothing = pt.render(s.Scene(lights=(SUN,), splats=(instance(faint),)), SPLAT_CAMERA, 8, 8, settings=settings)
        self.assertEqual(float(nothing.max()), 0.0)

    def test_the_render_can_be_cancelled_inside_a_pass(self):
        import threading
        from nodebased.cancellation import Cancelled
        event = threading.Event()
        event.set()
        scene = s.Scene(lights=(SUN,), splats=(instance(plane_cloud(n=24)),))
        with self.assertRaises(Cancelled):
            pt.render(scene, SPLAT_CAMERA, 16, 16, settings=pt.PathSettings(samples=8), cancel=event)


class SplatRefusalTests(unittest.TestCase):
    def test_the_gpu_reports_that_it_does_not_draw_splats_yet(self):
        scene = s.Scene(lights=(SUN,), splats=(instance(plane_cloud(n=4)),))
        with self.assertRaisesRegex(ValueError, "does not draw splats"):
            pt.render(scene, SPLAT_CAMERA, 4, 4, backend="gpu")
        stats = {}
        img = pt.render(scene, SPLAT_CAMERA, 4, 4, backend="auto", stats=stats, settings=pt.PathSettings(samples=2))
        self.assertEqual(stats["backend"], "cpu")
        self.assertIn("splats", stats["fallback"])
        self.assertEqual(img.shape, (4, 4, 4))


if __name__ == "__main__":
    unittest.main()
