"""Smoke and fire in the path tracer (lane L4 step R4 of 7): delta tracking against the raymarch's closed forms,
a furnace, the phase function, shadows on meshes, fire that lights meshes and splats, smoke seen through glass."""
import math
import unittest
from dataclasses import replace

import numpy as np

from nodebased import pathtrace as pt, ptvolume, scene3d as s, volumerender as vr
from tests.test_3d_pathtrace import box as mesh_box, card, uniform_env
from tests.test_3d_pathtrace_splats import instance, plane_cloud
from tests.test_volume_fire_look import hot_cold
from tests.test_volume_scene import box

CAMERA = s.Camera()          # at (0, 0, 5) looking at the origin
SMOKE = vr.VolumeSettings(absorption=0.4, scattering=0.6, step_size=0.05, shadow_steps=16)


def trace(scene, size=12, samples=64, output="rgba", volume=SMOKE, camera=CAMERA, **settings):
    settings.setdefault("max_bounces", 8)
    settings.setdefault("diffuse_bounces", 8)
    return pt.render(scene, camera, size, size, (0, 0, 0, 0), settings.pop("ambient", 0.0), output,
                     pt.PathSettings(samples=samples, **settings), volume=volume)


def centre(img, half=1):
    h, w = img.shape[:2]
    return img[h // 2 - half:h // 2 + half, w // 2 - half:w // 2 + half].reshape(-1, img.shape[2]).mean(axis=0)


class TransmittanceTests(unittest.TestCase):
    def test_absorbing_slab_alpha_is_the_raymarchs_beer_lambert(self):
        settings = replace(SMOKE, absorption=1.0, scattering=0.0)
        scene = s.Scene(volumes=(box(32, 2.0),))
        got = centre(trace(scene, samples=256, max_bounces=2, volume=settings))
        ref = centre(s.render(scene, CAMERA, 12, 12, volume=settings))
        self.assertAlmostEqual(float(got[3]), float(ref[3]), delta=0.03)
        self.assertLess(float(got[:3].max()), 1e-9)               # nothing lights it or scatters: pure absorber

    def test_rays_that_miss_the_box_see_nothing_and_a_thin_box_is_faint(self):
        scene = s.Scene(volumes=(box(16, 0.2),))
        img = trace(scene, samples=64, volume=replace(SMOKE, absorption=1.0, scattering=0.0))
        self.assertEqual(float(img[0, 0, 3]), 0.0)
        self.assertAlmostEqual(float(centre(img)[3]), 1 - math.exp(-0.2), delta=0.05)

    def test_the_same_seed_gives_the_same_picture(self):
        scene = s.Scene(volumes=(box(16, 1.0),), environments=(uniform_env(),))
        a, b = trace(scene, size=6, samples=8), trace(scene, size=6, samples=8)
        np.testing.assert_array_equal(a, b)


class PlacementTests(unittest.TestCase):
    def test_a_scaled_volume_keeps_the_raymarchs_alpha(self):
        volume = replace(box(16, 2.0), matrix=np.diag((2.0, 2.0, 2.0, 1.0)))
        settings = replace(SMOKE, absorption=1.0, scattering=0.0)
        scene = s.Scene(volumes=(volume,))
        got = centre(trace(scene, samples=256, max_bounces=2, volume=settings))
        ref = centre(s.render(scene, CAMERA, 12, 12, volume=settings))
        self.assertAlmostEqual(float(got[3]), float(ref[3]), delta=0.03)

    def test_two_overlapping_volumes_take_out_light_together(self):
        one = box(16, 1.0)
        settings = replace(SMOKE, absorption=1.0, scattering=0.0)
        single = centre(trace(s.Scene(volumes=(box(16, 2.0),)), samples=512, max_bounces=2, volume=settings))
        double = centre(trace(s.Scene(volumes=(one, one)), samples=512, max_bounces=2, volume=settings))
        # two overlapping clouds of density 1 extinguish like one of density 2 (independent collision processes)
        self.assertAlmostEqual(float(double[3]), float(single[3]), delta=0.04)

    def test_the_render_can_be_cancelled(self):
        import threading
        from nodebased.cancellation import Cancelled
        event = threading.Event()
        event.set()
        with self.assertRaises(Cancelled):
            pt.render(s.Scene(volumes=(box(16, 2.0),)), CAMERA, 8, 8, settings=pt.PathSettings(samples=8), cancel=event,
                      volume=SMOKE)


class FurnaceTests(unittest.TestCase):
    def test_a_scattering_only_cloud_in_a_uniform_sky_shows_the_sky(self):
        scene = s.Scene(volumes=(box(16, 2.0),), environments=(uniform_env(),))
        img = trace(scene, samples=128, max_bounces=64, diffuse_bounces=64, volume=replace(SMOKE, absorption=0.0, scattering=1.0))
        got = centre(img)
        self.assertAlmostEqual(float(got[:3].mean() / got[3]), 1.0, delta=0.04)

    def test_a_partly_absorbing_cloud_is_darker_and_the_more_absorbing_the_darker(self):
        scene = s.Scene(volumes=(box(16, 2.0),), environments=(uniform_env(),))
        values = []
        for absorption in (0.0, 0.3, 1.0):
            img = trace(scene, samples=96, volume=replace(SMOKE, absorption=absorption, scattering=1.0))
            got = centre(img)
            values.append(float(got[:3].mean() / got[3]))
        self.assertGreater(values[0], values[1])
        self.assertGreater(values[1], values[2])
        self.assertLess(values[2], 0.6)

    def test_the_smoke_colour_tints_what_it_scatters(self):
        scene = s.Scene(volumes=(box(16, 2.0),), environments=(uniform_env(),))
        img = trace(scene, samples=96, volume=replace(SMOKE, absorption=0.0, scattering=1.0, color=(1.0, 0.5, 0.1)))
        got = centre(img)
        self.assertGreater(float(got[0]), 1.5 * float(got[1]))
        self.assertGreater(float(got[1]), 1.5 * float(got[2]))


class PhaseTests(unittest.TestCase):
    def test_the_phase_function_integrates_to_one_and_has_mean_cosine_g(self):
        cosine = np.linspace(-1, 1, 20001)
        for g in (-0.7, 0.0, 0.3, 0.8):
            density = ptvolume.phase(g, cosine) * 2 * math.pi
            self.assertAlmostEqual(float(np.trapezoid(density, cosine)), 1.0, delta=1e-3)
            self.assertAlmostEqual(float(np.trapezoid(density * cosine, cosine)), g, delta=2e-3)

    def test_sampling_follows_the_phase_function(self):
        rng = np.random.default_rng(3)
        d = np.tile((0.0, 0.0, -1.0), (40000, 1))
        for g in (-0.6, 0.0, 0.5, 0.9):
            new, pdf = ptvolume.sample_phase(g, d, rng.random(len(d)), rng.random(len(d)))
            np.testing.assert_allclose(np.linalg.norm(new, axis=1), 1.0, atol=1e-9)
            cosine = np.sum(new * d, axis=1)
            self.assertAlmostEqual(float(cosine.mean()), g, delta=0.02)
            np.testing.assert_allclose(pdf, ptvolume.phase(g, cosine), rtol=1e-9)

    def test_forward_scattering_brightens_smoke_lit_from_behind(self):
        # a sun behind the cloud shining toward the camera: forward scattering (positive g) shows it, back scattering hides it
        sun = s.Light("Directional", (1, 1, 1), 1.0, s.Vec3(0, 0, -5), s.Vec3(0, 0, 0))
        scene = s.Scene(volumes=(box(16, 0.6),), lights=(sun,))
        forward = centre(trace(scene, samples=64, max_bounces=1, volume=replace(SMOKE, anisotropy=0.8)))
        backward = centre(trace(scene, samples=64, max_bounces=1, volume=replace(SMOKE, anisotropy=-0.8)))
        self.assertGreater(float(forward[0]), 4 * float(backward[0]))


class DepthTests(unittest.TestCase):
    def test_the_depth_pass_sees_smoke_in_front_of_a_wall_and_the_wall_behind_thin_smoke(self):
        wall = card(6, 6, (1, 1, 1, 1), (0, 0, -2.0))
        cloud = s.Scene((wall,), volumes=(box(16, 2.0),))
        depth = pt.render(cloud, CAMERA, 12, 12, output="depth", volume=SMOKE)
        ref = s.render(cloud, CAMERA, 12, 12, output="depth", volume=SMOKE)
        np.testing.assert_allclose(depth[6, 6, 0], ref[6, 6, 0], atol=1e-4)
        self.assertLess(float(depth[6, 6, 0]), 5.5)                # the cloud (front at 4.5), not the wall (7)
        thin = pt.render(s.Scene((wall,), volumes=(box(16, 0.01),)), CAMERA, 12, 12, output="depth", volume=SMOKE)
        self.assertAlmostEqual(float(thin[6, 6, 0]), 7.0, delta=1e-4)


class ShadowTests(unittest.TestCase):
    def test_smoke_casts_the_raymarchs_shadow_on_a_mesh(self):
        floor = card(6, 6, (0.8, 0.8, 0.8, 1), (0, -1, 0), (90, 0, 0))
        sun = s.Light("Directional", (1, 1, 1), 1.0, s.Vec3(0, 0, 0), s.Vec3(1, -1, 0))
        scene = s.Scene((floor,), lights=(sun,), volumes=(box(16, 2.0),))
        top = s.Camera(s.Transform3D(position=s.Vec3(0, 6, 0.0001)), s.Vec3(0, -1, 0), 30.0)
        n = 48
        with_smoke = trace(scene, size=n, samples=4, output="diffuse", max_bounces=1, camera=top)
        open_floor = trace(s.Scene((floor,), lights=(sun,)), size=n, samples=4, output="diffuse", max_bounces=1, camera=top)
        position = pt.render(s.Scene((floor,)), top, n, n, output="position")
        points = position[..., :3].reshape(-1, 3).astype(np.float64)
        expected = vr.ShadowCasters(scene.volumes, SMOKE).transmittance(sun, points).reshape(n, n)
        # the pixels the camera sees the floor through the smoke itself are lit by the cloud's own scattering: skip them
        clear = (np.abs(position[..., 0]) > 0.8) | (np.abs(position[..., 2]) > 0.8)
        got = (with_smoke[..., 0] / np.maximum(open_floor[..., 0], 1e-9))
        # the picture antialiases and `expected` is read at pixel centres: leave out the pixels next to a steep edge
        padded = np.pad(expected, 1, mode="edge")
        around = np.stack([padded[1 + dy:n + 1 + dy, 1 + dx:n + 1 + dx] for dy in (-1, 0, 1) for dx in (-1, 0, 1)])
        flat = (around.max(axis=0) - around.min(axis=0)) < 0.06
        compared = clear & flat
        self.assertGreater(int(compared.sum()), 500)
        self.assertGreater(int((compared & (expected < 0.5)).sum()), 10)       # some of the compared floor is in the shadow
        np.testing.assert_allclose(got[compared], expected[compared], atol=0.03)
        self.assertLess(float(expected[clear].min()), 0.1)                      # the shadow is real: nearly dark in the middle

    def test_shadow_density_zero_lets_the_light_through(self):
        floor = card(6, 6, (0.8, 0.8, 0.8, 1), (0, -1, 0), (90, 0, 0))
        sun = s.Light("Directional", (1, 1, 1), 1.0, s.Vec3(0, 0, 0), s.Vec3(1, -1, 0))
        scene = s.Scene((floor,), lights=(sun,), volumes=(box(16, 2.0),))
        top = s.Camera(s.Transform3D(position=s.Vec3(0, 6, 0.0001)), s.Vec3(0, -1, 0), 30.0)
        clear = trace(scene, size=12, samples=4, output="diffuse", max_bounces=1, camera=top,
                      volume=replace(SMOKE, shadow_density=0.0))
        open_floor = trace(s.Scene((floor,), lights=(sun,)), size=12, samples=4, output="diffuse", max_bounces=1, camera=top)
        position = pt.render(s.Scene((floor,)), top, 12, 12, output="position")
        outside = (np.abs(position[..., 0]) > 0.8) | (np.abs(position[..., 2]) > 0.8)
        np.testing.assert_allclose(clear[..., 0][outside], open_floor[..., 0][outside], rtol=1e-6)

    def test_a_mesh_shadows_the_smoke(self):
        blocker = card(1.6, 1.6, (0.3, 0.3, 0.3, 1), (0, 1.0, 0), (90, 0, 0))
        sun = s.Light("Directional", (1, 1, 1), 1.0, s.Vec3(0, 3, 0), s.Vec3(0, 0, 0))
        smoke = box(16, 1.2)
        lit, dark = (trace(s.Scene(g, lights=(sun,), volumes=(smoke,)), samples=64, max_bounces=1, size=12,
                           camera=s.Camera(s.Transform3D(position=s.Vec3(0, 0, 5)), s.Vec3(0, 0, 0), 25.0))
                     for g in ((), (blocker,)))
        # the blocker is edge-on to the camera and above the cloud: the cloud below it is in its shadow
        self.assertLess(float(centre(dark)[0]), 0.25 * float(centre(lit)[0]))


class FireTests(unittest.TestCase):
    FIRE = vr.VolumeSettings(absorption=0.4, scattering=0.0, step_size=0.05, fire_intensity=1.0, shadow_steps=16)

    def test_fire_seen_directly_matches_the_raymarch(self):
        scene = s.Scene(volumes=(hot_cold(1.5, 0.0),))
        got = trace(scene, size=12, samples=384, max_bounces=1, volume=self.FIRE)
        ref = s.render(scene, CAMERA, 12, 12, volume=self.FIRE)
        hot = lambda a: a[3:5, 5:7].reshape(-1, 4).mean(axis=0)
        cold = lambda a: a[8:10, 5:7].reshape(-1, 4).mean(axis=0)
        np.testing.assert_allclose(hot(got)[:3], hot(ref)[:3], rtol=0.08)
        self.assertAlmostEqual(float(hot(got)[3]), float(hot(ref)[3]), delta=0.03)
        self.assertLess(float(cold(got)[:3].max()), 1e-9)          # the cold half holds smoke but does not glow
        self.assertGreater(float(hot(got)[0]), float(hot(got)[2]))  # a 1500 K flame is red

    def test_fire_lights_a_nearby_mesh_and_the_light_falls_with_distance(self):
        flame = hot_cold(1.5, 1.5, n=8, density=1.5)
        flame = replace(flame, matrix=np.array(((1, 0, 0, 0), (0, 1, 0, 1.6), (0, 0, 1, 0), (0, 0, 0, 1.0))))   # 1.1 to 2.1 up
        floor = card(8, 8, (0.8, 0.8, 0.8, 1), (0, 0, 0), (90, 0, 0))
        scene = s.Scene((floor,), volumes=(flame,))
        top = s.Camera(s.Transform3D(position=s.Vec3(0, 0.5, 6)), s.Vec3(0, 0, 0), 30.0)
        img = trace(scene, size=16, samples=192, output="diffuse", max_bounces=2, camera=top, volume=self.FIRE)
        near, far = img[10, 8, :3], img[13, 8, :3]                    # under the flame, and a way in front of it
        self.assertGreater(float(near[0]), 0.02)
        self.assertGreater(float(near[0]), 2 * float(near[2]))        # the warm colour of the fire, on the floor
        self.assertGreater(float(near[0]), 1.5 * float(far[0]))

    def test_fire_lights_a_splat_sheet(self):
        flame = hot_cold(1.5, 1.5, n=8, density=1.5)
        flame = replace(flame, matrix=np.array(((1, 0, 0, 0), (0, 1, 0, 1.6), (0, 0, 1, 0), (0, 0, 0, 1.0))))
        sheet = plane_cloud(n=24, size=4.0, albedo=(0.8, 0.8, 0.8), delit=False)
        matrix = np.eye(4)
        matrix[:3, :3] = ((1, 0, 0), (0, 0, -1), (0, 1, 0))
        scene = s.Scene(splats=(instance(sheet, matrix=matrix, relight=1.0),), volumes=(flame,))
        high = s.Camera(s.Transform3D(position=s.Vec3(0, 6, 6)), s.Vec3(0, 0, 0), 30.0)
        with_fire = trace(scene, size=20, samples=128, output="diffuse", max_bounces=2, camera=high, volume=self.FIRE)
        without = trace(replace(scene, volumes=()), size=20, samples=16, output="diffuse", max_bounces=2, camera=high,
                        volume=self.FIRE)
        self.assertEqual(float(without[..., :3].max()), 0.0)          # no lights: the sheet is dark without the fire
        where = pt.render(replace(scene, volumes=()), high, 20, 20, output="position")
        on_sheet = where[..., 3] > 0.5
        radius = np.hypot(where[..., 0], where[..., 2])
        near = with_fire[on_sheet & (radius < 0.8), :3].mean(axis=0)
        far = with_fire[on_sheet & (radius > 1.8), :3].mean(axis=0)
        self.assertGreater(float(near[0]), 0.02)
        self.assertGreater(float(near[0]), 2 * float(near[2]))        # the warm colour of the fire
        self.assertGreater(float(near[0]), 2 * float(far[0]))         # and it falls off with distance


class ThroughGlassTests(unittest.TestCase):
    def test_smoke_is_seen_through_a_glass_slab(self):
        slab = mesh_box((4, 4, 0.2), (1, 1, 1, 1), (0, 0, 2.0), material="liquid", ior=1.5, absorption_color=(1, 1, 1),
                        absorption_distance=0.2, reflection=1.0)
        fire = vr.VolumeSettings(absorption=1.0, scattering=0.0, step_size=0.05, fire_intensity=1.0)
        scene = s.Scene((slab,), volumes=(hot_cold(1.5, 1.5, n=8, density=2.0),))
        bare = s.Scene(volumes=scene.volumes)
        through = centre(trace(scene, size=12, samples=192, max_bounces=8, volume=fire))
        direct = centre(trace(bare, size=12, samples=192, max_bounces=8, volume=fire))
        f = ((1.5 - 1) / (1.5 + 1)) ** 2
        # a clear slab passes (1 - F) / (1 + F) of what is behind it, the smoke's own emission included
        self.assertAlmostEqual(float(through[0] / direct[0]), (1 - f) / (1 + f), delta=0.06)
        self.assertGreater(float(through[3]), 0.5)


class NodeTests(unittest.TestCase):
    """A Plume3D through Scene3D and Render3D in path tracer mode reads the volume knobs."""

    def graph(self, **render):
        from nodebased.core import Dispatcher
        d = Dispatcher()
        for key, (kind, params) in dict(p=("Plume3D", {"plume_resolution": 12}), s=("Scene3D", {}),
                                        c=("Camera3D", {"ty": .5, "tz": 3, "target_y": .5}),
                                        l=("Light3D", {"tx": -3.0, "ty": 2.0, "tz": 4.0}),
                                        r=("Render3D", {"width": 16, "height": 16, "render_mode": "pathtrace",
                                                        "pt_samples": 16, "max_bounces": 2, "volume_density_scale": 4.0,
                                                        **render})).items():
            d.execute({"op": "create", "id": key, "type": kind, "params": params})
        for target, slot, source in (("s", "object0", "p"), ("s", "object1", "l"), ("r", "scene", "s"), ("r", "camera", "c")):
            d.execute({"op": "connect", "id": target, "input": slot, "source": source})
        return d

    def image(self, d):
        from nodebased.imaging import Evaluator
        return Evaluator().evaluate(dict(d.document, view="r"), frame=1)

    def test_the_plume_renders_and_the_smoke_knobs_reach_the_tracer(self):
        d = self.graph()
        plain = self.image(d)
        self.assertEqual(plain.shape, (16, 16, 4))
        self.assertGreater(float(plain[..., 3].max()), 0.3)
        d.execute({"op": "set", "id": "r", "param": "volume_density_scale", "value": 1.0})
        thin = self.image(d)
        self.assertLess(float(thin[..., 3].sum()), float(plain[..., 3].sum()))
        d.execute({"op": "set", "id": "r", "param": "volume_density_scale", "value": 4.0})
        d.execute({"op": "set", "id": "r", "param": "volume_fire_intensity", "value": 2.0})
        d.execute({"op": "set", "id": "r", "param": "volume_temperature_scale", "value": 4000.0})   # a plume this hot glows
        fire = self.image(d)
        self.assertGreater(float(fire[..., 0].sum()), 3 * float(plain[..., 0].sum()))

    def test_volumes_off_leaves_an_empty_picture(self):
        d = self.graph(volumes="off")
        self.assertEqual(float(self.image(d)[..., 3].max()), 0.0)


class RefusalTests(unittest.TestCase):
    def test_without_a_usable_gpu_auto_uses_the_cpu_reference_and_gpu_says_why_not(self):
        from unittest import mock
        from nodebased import gpu3d
        scene = s.Scene(volumes=(box(8, 1.0),))
        with mock.patch.object(gpu3d, "available", return_value=False):
            with self.assertRaisesRegex(ValueError, "GPU Render3D unsupported"):
                pt.render(scene, CAMERA, 4, 4, backend="gpu")
            stats = {}
            img = pt.render(scene, CAMERA, 4, 4, backend="auto", stats=stats, settings=pt.PathSettings(samples=2))
        self.assertEqual(stats["backend"], "cpu")
        self.assertIn("fallback", stats)
        self.assertEqual(img.shape, (4, 4, 4))

    def test_particles_render_alongside_smoke(self):
        # R7 of 7 finish: the path tracer draws particles now (tests.test_3d_pathtrace.
        # ParticlesInThePathTracerTests); this only checks they do not upset a scene that also has smoke.
        particle = s.ParticleInstance(positions=np.array([[0, 0, 2.0]], np.float32), sizes=np.array([0.3], np.float32),
                                      colors=np.array([[1, 1, 1, 1]], np.float32), render_as="spheres")
        scene = s.Scene(volumes=(box(8, 1.0),), particles=(particle,))
        img = pt.render(scene, CAMERA, 8, 8, settings=pt.PathSettings(samples=4), volume=SMOKE)
        self.assertGreater(float(img[..., 3].max()), 0.0)


if __name__ == "__main__":
    unittest.main()
