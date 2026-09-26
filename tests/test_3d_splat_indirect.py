"""Indirect light and occlusion on relit splats (step D): the sampling, the traced terms, temporal stability,
the guided denoiser, the quality presets, the bundle passes and GPU parity (docs/SPLAT_RELIGHTING.md)."""
from dataclasses import replace
import unittest
from unittest import mock

import numpy as np

from nodebased import gpu3d, scene3d as s, splats, splatindirect as si, splatshade as sh, viewportgpu
from tests import gpu_precision


def _quat_z_to(n):
    n = np.asarray(n, float) / np.linalg.norm(n)
    axis = np.cross((0, 0, 1), n)
    norm = np.linalg.norm(axis)
    if norm < 1e-9:
        return np.array((1.0, 0, 0, 0)) if n[2] > 0 else np.array((0, 1.0, 0, 0))
    angle = np.arctan2(norm, n[2])
    axis = axis / norm
    return np.array((np.cos(angle / 2), *(np.sin(angle / 2) * axis)))


def plane(center, normal, half=1.0, spacing=0.1, colour=(0.8, 0.8, 0.8)):
    """A square sheet of thin discs facing `normal`, colour baked into the DC."""
    n = np.asarray(normal, float) / np.linalg.norm(normal)
    helper = np.array((0, 1.0, 0)) if abs(n[1]) < 0.9 else np.array((1.0, 0, 0))
    u = np.cross(helper, n)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    g = np.arange(-half + spacing / 2, half, spacing)
    a, b = np.meshgrid(g, g)
    points = np.asarray(center) + a.reshape(-1, 1) * u + b.reshape(-1, 1) * v
    count = len(points)
    scales = np.tile((spacing * 0.7, spacing * 0.7, spacing * 0.02), (count, 1))
    sh_dc = np.tile(((np.asarray(colour) - 0.5) / splats.C0)[None, None, :], (count, 1, 1))
    return splats.SplatCloud(points.astype(np.float32), scales.astype(np.float32),
                             np.tile(_quat_z_to(n), (count, 1)).astype(np.float32),
                             np.full(count, 0.99, np.float32), sh_dc.astype(np.float32), 0, colorspace='linear')


def merge(*clouds):
    return splats.SplatCloud(*(np.concatenate([getattr(c, name) for c in clouds])
                               for name in ('positions', 'scales', 'rotations', 'opacity', 'sh')),
                             0, colorspace='linear')


FLOOR = plane((0, 0, 0), (0, 1, 0))
WALL = plane((1.0, 1.0, 0), (-1, 0, 0), colour=(1.0, 1.0, 1.0))      # a bright wall at x = 1, facing the floor
CREASE = merge(FLOOR, plane((0, 1.0, -1.0), (0, 0, 1)))
CAMERA = s.Camera(s.Transform3D(s.Vec3(0, 3.2, 2.2)), s.Vec3(0, 0, -0.1), 45.0, 0.1, 100.0)
SUN_FROM_LEFT = s.Light('Directional', (1, 1, 1), 1.0, s.Vec3(-1, 0, 0), s.Vec3(0, 0, 0))   # grazes the floor


def instance(cloud, **kw):
    kw.setdefault('relight', 1.0)
    return s.SplatInstance(cloud, **kw)


def context(inst, lights=(), ambient=0.0):
    shadows = s._SplatShadows((inst,), tuple(lights), None, None, .001)
    return si.IndirectLight(shadows, ambient)


def facing_up(cloud):
    return np.tile((0.0, 1.0, 0.0), (len(cloud), 1))


class Sampling(unittest.TestCase):
    NORMALS = np.tile((0.0, 1.0, 0.0), (64, 1))

    def test_directions_are_unit_in_the_hemisphere_and_cosine_weighted(self):
        d = si.hemisphere_directions(np.arange(64), self.NORMALS, 32)
        np.testing.assert_allclose(np.linalg.norm(d, axis=2), 1.0, atol=1e-9)
        self.assertGreaterEqual(float(d[..., 1].min()), 0.0)
        self.assertAlmostEqual(float(d[..., 1].mean()), 2 / 3, delta=0.02)     # E[cos] of a cosine lobe

    def test_a_splat_gets_the_same_directions_on_every_call_and_frame(self):
        a = si.hemisphere_directions(np.arange(64), self.NORMALS, 8, seed=3)
        b = si.hemisphere_directions(np.arange(64), self.NORMALS, 8, seed=3, frame=None)
        c = si.hemisphere_directions(np.arange(10, 20), self.NORMALS[:10], 8, seed=3)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(a[10:20], c)            # anchored to the splat id, not to the batch
        self.assertGreater(float(np.abs(a - si.hemisphere_directions(np.arange(64), self.NORMALS, 8, seed=4)).max()), 0.1)
        moved = si.hemisphere_directions(np.arange(64), self.NORMALS, 8, seed=3, frame=5)
        self.assertGreater(float(np.abs(a - moved).max()), 0.1)      # only an explicit frame moves the pattern
        np.testing.assert_array_equal(moved, si.hemisphere_directions(np.arange(64), self.NORMALS, 8, seed=3, frame=5))


class Presets(unittest.TestCase):
    def test_presets_scale_the_sample_counts_and_never_turn_an_active_count_off(self):
        self.assertEqual([si.scaled_samples(16, q) for q in ('preview', 'medium', 'final')], [4, 16, 64])
        self.assertEqual([si.scaled_samples(1, q) for q in ('preview', 'medium', 'final')], [1, 1, 4])
        self.assertEqual([si.scaled_samples(0, q) for q in ('preview', 'medium', 'final')], [0, 0, 0])
        inst = instance(FLOOR, indirect_samples=8, reflection_samples=4, quality='final')
        self.assertEqual((si.effective_samples(inst, 'indirect_samples'), sh.reflection_sample_count(inst)), (32, 16))
        self.assertEqual(si.effective_samples(replace(inst, quality='preview'), 'indirect_samples'), 2)

    def test_the_preset_changes_the_number_of_rays_traced(self):
        counts = {}
        real = si.trace
        for quality in ('preview', 'medium', 'final'):
            inst = instance(FLOOR, indirect_samples=4, quality=quality)
            calls = []
            with mock.patch.object(si, 'trace', side_effect=lambda *a, **k: (calls.append(len(a[2])), real(*a, **k))[1]):
                context(inst)(inst, inst.cloud.positions.astype(float), facing_up(FLOOR))
            counts[quality] = sum(calls)
        n = len(FLOOR)
        self.assertEqual(counts, {'preview': n, 'medium': 4 * n, 'final': 16 * n})


class Occlusion(unittest.TestCase):
    def test_a_flat_plane_is_unoccluded_and_a_crease_darkens(self):
        inst = instance(FLOOR, indirect_samples=16, indirect_distance=0.8)
        ao, bounce = context(inst)(inst, inst.cloud.positions.astype(float), facing_up(FLOOR))
        np.testing.assert_allclose(ao, 1.0, atol=1e-9)
        self.assertEqual(float(np.abs(bounce).max()), 0.0)
        inst = instance(CREASE, indirect_samples=16, indirect_distance=0.8)
        positions = inst.cloud.positions.astype(float)
        normals = np.where((np.arange(len(positions)) < len(FLOOR))[:, None], (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        ao, _ = context(inst)(inst, positions, normals)
        floor_ao, z = ao[:len(FLOOR)], positions[:len(FLOOR), 2]
        self.assertLess(float(floor_ao[np.abs(z + 1) < 0.15].mean()), 0.85)      # in the corner
        self.assertGreater(float(floor_ao[np.abs(z + 1) > 0.6].mean()), 0.99)    # away from it
        far = instance(CREASE, indirect_samples=16, indirect_distance=0.05)
        ao_far, _ = context(far)(far, positions, normals)
        self.assertGreater(float(ao_far.min()), 0.99)                             # the rays stop short of the wall

    def test_occlusion_scales_ambient_but_not_the_direct_light(self):
        floor_only = s.Scene(splats=(instance(CREASE, indirect_samples=16, indirect_distance=0.8),))
        plain = s.Scene(splats=(instance(CREASE),))
        direct = s.Light('Directional', (1, 1, 1), 1.0, s.Vec3(0, 3, 2), s.Vec3(0, 0, 0))
        lit_traced = s.render(replace(floor_only, lights=(direct,)), CAMERA, 32, 24, ambient=0.0)
        lit_plain = s.render(replace(plain, lights=(direct,)), CAMERA, 32, 24, ambient=0.0)
        self.assertGreater(float(np.abs(lit_traced - lit_plain).max()), 1e-3)     # the bounce between the sheets
        dark_traced = s.render(floor_only, CAMERA, 32, 24, ambient=0.5)
        dark_plain = s.render(plain, CAMERA, 32, 24, ambient=0.5)
        self.assertLess(float(dark_traced[..., :3].sum()), float(dark_plain[..., :3].sum()))   # the crease is darker


class Bounce(unittest.TestCase):
    def scene(self, samples, **kw):
        return s.Scene(splats=(instance(merge(FLOOR, WALL), indirect_samples=samples, indirect_distance=3.0, **kw),),
                       lights=(SUN_FROM_LEFT,))

    def test_a_splat_facing_a_bright_wall_picks_up_its_light(self):
        # The sun grazes the floor (n.l = 0), so the floor is black unless the lit wall bounces onto it.
        without = s.render(self.scene(0), CAMERA, 40, 30, ambient=0.0)
        bounced = s.render(self.scene(16), CAMERA, 40, 30, ambient=0.0)
        self.assertGreater(float(bounced[..., :3].sum()), float(without[..., :3].sum()) + 1.0)
        inst = self.scene(16).splats[0]
        positions = inst.cloud.positions.astype(float)
        n = len(positions)
        up = np.where((np.arange(n) < len(FLOOR))[:, None], (0.0, 1.0, 0.0), (-1.0, 0.0, 0.0))
        shadows = s._SplatShadows((inst,), (SUN_FROM_LEFT,), None, None, .001)
        _, gathered = si.IndirectLight(shadows, 0.0)(inst, positions, up)
        near = gathered[:len(FLOOR)][positions[:len(FLOOR), 0] > 0.5].mean()
        far = gathered[:len(FLOOR)][positions[:len(FLOOR), 0] < -0.5].mean()
        self.assertGreater(float(near), 0.05)
        self.assertGreater(float(near), 2 * float(far))             # closer to the wall, more of the sky is wall

    def test_the_radiance_a_ray_picks_up_is_the_albedo_of_the_lit_wall(self):
        inst = self.scene(32).splats[0]
        shadows = s._SplatShadows((inst,), (SUN_FROM_LEFT,), None, None, .001)
        source = si.IndirectLight(shadows, 0.0)
        # The wall follows the floor in splat order; the unit sun lights it with n.l = 1 and its albedo is 1.
        wall_radiance = source._hit_radiance(np.arange(len(FLOOR), len(FLOOR) + len(WALL)),
                                             np.tile((1.0, 0, 0), (len(WALL), 1)))
        np.testing.assert_allclose(wall_radiance, 1.0, atol=1e-6)
        floor_radiance = source._hit_radiance(np.arange(len(FLOOR)), np.tile((0, -1.0, 0), (len(FLOOR), 1)))
        np.testing.assert_allclose(floor_radiance, 0.0, atol=1e-6)      # the grazing sun does not light the floor

    def test_the_bounce_stays_off_when_the_switch_is_off(self):
        a = s.render(self.scene(0), CAMERA, 24, 18, ambient=0.1)
        b = s.render(replace(self.scene(0), splats=(replace(self.scene(0).splats[0], indirect_distance=0.5),)),
                     CAMERA, 24, 18, ambient=0.1)
        np.testing.assert_array_equal(a, b)


class Stability(unittest.TestCase):
    def scene(self):
        return s.Scene(splats=(instance(merge(FLOOR, WALL), indirect_samples=4, indirect_distance=3.0),),
                       lights=(SUN_FROM_LEFT,))

    def test_two_frames_of_a_static_scene_are_identical(self):
        first = s.render(self.scene(), CAMERA, 32, 24, ambient=0.2)
        second = s.render(self.scene(), CAMERA, 32, 24, ambient=0.2)
        np.testing.assert_array_equal(first, second)

    def test_per_splat_terms_do_not_depend_on_the_camera(self):
        inst = self.scene().splats[0]
        cloud = inst.cloud.transformed(inst.matrix)
        shadows = s._SplatShadows((inst,), (SUN_FROM_LEFT,), None, None, .001)
        extras = mock.Mock(indirect=si.IndirectLight(shadows, 0.1))
        albedo = sh.splat_albedo(cloud)
        results = [sh.indirect_terms(inst, cloud, eye, extras, albedo, cloud.normals())
                   for eye in ((0, 3, 2), (0.05, 3, 2.05), (-0.1, 2.9, 2.1))]
        for other in results[1:]:
            np.testing.assert_array_equal(results[0][0], other[0])
            np.testing.assert_array_equal(results[0][1], other[1])

    def test_a_moving_camera_adds_no_noise_of_its_own(self):
        def render(scene, eye):
            camera = replace(CAMERA, transform=s.Transform3D(s.Vec3(*eye)))
            return s.render(scene, camera, 40, 30, ambient=0.2)[..., :3]
        eyes = [(0, 3.2, 2.2), (0.02, 3.2, 2.2), (0.04, 3.2, 2.2)]
        traced, plain = self.scene(), s.Scene(splats=(instance(merge(FLOOR, WALL)),), lights=(SUN_FROM_LEFT,))
        motion = [float(np.abs(render(traced, a) - render(traced, b)).mean()) for a, b in zip(eyes, eyes[1:])]
        baseline = [float(np.abs(render(plain, a) - render(plain, b)).mean()) for a, b in zip(eyes, eyes[1:])]
        for with_rays, without in zip(motion, baseline):
            self.assertLess(with_rays, 2.0 * without + 0.005)      # no frame-to-frame sparkle from the rays


class Denoise(unittest.TestCase):
    def test_zero_is_the_undenoised_render_and_the_direct_term_is_never_filtered(self):
        base = instance(merge(FLOOR, WALL), indirect_samples=4, indirect_distance=3.0)
        scene = lambda i: s.Scene(splats=(i,), lights=(SUN_FROM_LEFT,))
        raw = s.render(scene(base), CAMERA, 32, 24, ambient=0.1)
        zero = s.render(scene(replace(base, denoise=0.0)), CAMERA, 32, 24, ambient=0.1)
        np.testing.assert_array_equal(raw, zero)
        self.assertGreater(float(np.abs(raw - s.render(scene(replace(base, denoise=1.0)), CAMERA, 32, 24, ambient=0.1)).max()), 1e-5)
        # No indirect and no traced reflection: there is nothing for the denoiser to touch.
        direct = instance(merge(FLOOR, WALL))
        np.testing.assert_array_equal(s.render(scene(direct), CAMERA, 32, 24, ambient=0.1),
                                      s.render(scene(replace(direct, denoise=1.0)), CAMERA, 32, 24, ambient=0.1))

    def test_it_averages_noise_inside_a_surface_and_keeps_albedo_edges(self):
        rng = np.random.default_rng(1)
        cloud = plane((0, 0, 0), (0, 1, 0), half=1.0, spacing=0.1)
        positions = cloud.positions.astype(float)
        normals = facing_up(cloud)
        albedo = np.where((positions[:, 0] > 0)[:, None], 1.0, 0.1) * np.ones((len(positions), 3))
        signal = np.where((positions[:, 0] > 0)[:, None], 1.0, 0.0) * np.ones((len(positions), 3))
        noisy = signal + rng.normal(0, 0.2, signal.shape)
        clean = si.guided_denoise(noisy, positions, normals, albedo, 1.0)
        interior = np.abs(positions[:, 0]) > 0.4
        self.assertLess(float((clean - signal)[interior].std()), 0.5 * float((noisy - signal)[interior].std()))
        edge_left, edge_right = clean[(positions[:, 0] < 0) & (positions[:, 0] > -0.15)], clean[(positions[:, 0] > 0) & (positions[:, 0] < 0.15)]
        self.assertGreater(float(edge_right.mean() - edge_left.mean()), 0.7)     # the step survives
        self.assertIs(si.guided_denoise(noisy, positions, normals, albedo, 0.0), noisy)

    def test_a_partial_amount_blends(self):
        cloud = plane((0, 0, 0), (0, 1, 0), half=0.5, spacing=0.1)
        positions, normals = cloud.positions.astype(float), facing_up(cloud)
        noisy = np.random.default_rng(2).normal(0, 1, (len(positions), 3))
        half = si.guided_denoise(noisy, positions, normals, np.ones_like(noisy), 0.5)
        full = si.guided_denoise(noisy, positions, normals, np.ones_like(noisy), 1.0)
        np.testing.assert_allclose(half, 0.5 * noisy + 0.5 * full, atol=1e-12)


class Bundle(unittest.TestCase):
    def test_the_bundle_carries_indirect_and_the_traced_occlusion_and_sums_to_the_beauty(self):
        scene = s.Scene(splats=(instance(merge(FLOOR, WALL), indirect_samples=8, indirect_distance=3.0),),
                        lights=(SUN_FROM_LEFT,))
        beauty = s.render(scene, CAMERA, 32, 24, ambient=0.2)
        out, layers = s.render(scene, CAMERA, 32, 24, ambient=0.2, output='relight')
        np.testing.assert_allclose(out[..., :3], beauty[..., :3], atol=2e-4)
        self.assertIn('indirect', layers)
        self.assertGreater(float(layers['indirect'][..., :3].sum()), 0.5)
        plain = s.Scene(splats=(instance(merge(FLOOR, WALL)),), lights=(SUN_FROM_LEFT,))
        _, base = s.render(plain, CAMERA, 32, 24, ambient=0.2, output='relight')
        self.assertLess(float(np.abs(base['indirect'][..., :3]).max()), 1e-6)


class Budget(unittest.TestCase):
    def test_too_many_rays_raise_before_any_work(self):
        big = instance(FLOOR, indirect_samples=256)
        with mock.patch.object(s, 'SPLAT_SHADOW_BUDGET', 1000):
            with self.assertRaisesRegex(ValueError, 'indirect-light rays exceed'):
                s.render(s.Scene(splats=(big,)), CAMERA, 8, 8)

    def test_meshes_are_hit_by_the_rays_as_well(self):
        wall = s._card(2, 2, (1.0, 1.0, 1.0, 1.0), s.Transform3D(s.Vec3(1.0, 0.5, 0), s.Vec3(0, 90, 0)))
        scene = s.Scene((wall,), splats=(instance(FLOOR, indirect_samples=16, indirect_distance=3.0),),
                        lights=(SUN_FROM_LEFT,))
        plain = replace(scene, splats=(replace(scene.splats[0], indirect_samples=0),))
        lit, dark = s.render(scene, CAMERA, 32, 24, ambient=0.0), s.render(plain, CAMERA, 32, 24, ambient=0.0)
        self.assertGreater(float(lit[..., :3].sum()), float(dark[..., :3].sum()) + 0.3)


@unittest.skipUnless(gpu3d.available(), 'no wgpu adapter')
class GpuParity(unittest.TestCase):
    def test_gpu_matches_the_cpu_reference_with_indirect_light(self):
        scene = s.Scene(splats=(instance(merge(FLOOR, WALL), indirect_samples=8, indirect_distance=3.0, denoise=0.5),),
                        lights=(SUN_FROM_LEFT,))
        cpu = s.render(scene, CAMERA, 40, 30, ambient=0.1)
        gpu = gpu3d.render(scene, CAMERA, 40, 30, ambient=0.1)
        gpu_precision.note(self.id())
        self.assertLess(float(np.abs(cpu - gpu).max()), gpu_precision.tolerance(3e-3))

    def test_gpu_matches_the_cpu_reference_with_indirect_light_and_traced_shadows(self):
        sun = replace(SUN_FROM_LEFT, shadows=True)
        scene = s.Scene(splats=(instance(merge(FLOOR, WALL), indirect_samples=8, indirect_distance=3.0),), lights=(sun,))
        cpu = s.render(scene, CAMERA, 40, 30, ambient=0.1)
        gpu = gpu3d.render(scene, CAMERA, 40, 30, ambient=0.1)
        self.assertLess(float(np.abs(cpu - gpu).max()), gpu_precision.tolerance(3e-3))
        self.assertGreater(float(cpu[..., :3].sum()), 1.0)

    def test_gpu_refuses_indirect_light_with_meshes_so_callers_fall_back(self):
        wall = s._card(2, 2, (1.0, 1.0, 1.0, 1.0), s.Transform3D(s.Vec3(1.0, 0.5, 0), s.Vec3(0, 90, 0)))
        scene = s.Scene((wall,), splats=(instance(FLOOR, indirect_samples=4),), lights=(SUN_FROM_LEFT,))
        with self.assertRaises(gpu3d.Unsupported):
            gpu3d.render(scene, CAMERA, 16, 16)


@unittest.skipUnless(gpu3d.available(), 'no wgpu adapter')
class ViewportPreview(unittest.TestCase):
    """The editor viewport shows the relit result with the indirect terms traced at the preview preset."""
    SIZE = (96, 72)

    def setUp(self):
        self.gpu = viewportgpu.renderer()
        self.assertIsNotNone(self.gpu, viewportgpu.failure())

    def scene(self, samples=8, **kw):
        return s.Scene(splats=(instance(merge(FLOOR, WALL), indirect_samples=samples, indirect_distance=3.0, **kw),),
                       lights=(SUN_FROM_LEFT,))

    def frame(self, scene):
        return self.gpu.render(scene, CAMERA, *self.SIZE, (0, 0, 0, 1), headlight=False, ambient=0.0)

    def test_the_bounce_off_a_bright_wall_lights_the_floor_in_the_viewport(self):
        flat = self.frame(self.scene(samples=0)).astype(int)[..., :3]
        bounced = self.frame(self.scene()).astype(int)[..., :3]
        brighter = (bounced - flat).max(axis=2) > 40
        self.assertGreater(int(brighter.sum()), 200)                                  # the floor and the wall pick up the bounce
        self.assertEqual(int(((flat - bounced).max(axis=2) > 10).sum()), 0)           # and nothing gets darker

    def test_it_is_traced_once_at_the_preview_preset_and_reused(self):
        real = si.IndirectLight.__call__
        seen = []

        def spy(self, inst, positions, facing, albedo=None):
            seen.append(si.effective_samples(inst, 'indirect_samples'))
            return real(self, inst, positions, facing, albedo)
        scene = self.scene(samples=8, quality='final')       # the preset in the document would be 4x; the viewport uses preview
        with mock.patch.object(si.IndirectLight, '__call__', spy):
            self.frame(scene)
            self.frame(scene)
        self.assertEqual(seen, [2])
        with mock.patch.object(si.IndirectLight, '__call__', spy):
            self.frame(replace(scene, lights=(replace(SUN_FROM_LEFT, intensity=0.5),)))
        self.assertEqual(seen, [2, 2])                       # a changed light re-traces it

    def test_large_clouds_and_switched_off_instances_keep_the_plain_preview(self):
        plain = self.frame(self.scene(samples=0))
        with mock.patch.object(viewportgpu, 'INDIRECT_MAX_SPLATS', 10):
            np.testing.assert_array_equal(self.frame(self.scene()), plain)


class Defaults(unittest.TestCase):
    def test_old_documents_and_new_instances_keep_the_old_look(self):
        from nodebased import core
        params = core.NODE_SPECS["ReadSplat3D"]["params"] if hasattr(core, 'NODE_SPECS') else core.SPECS["ReadSplat3D"]["params"]
        self.assertEqual((params["splat_indirect_samples"], params["splat_denoise"], params["splat_quality"]),
                         (0, 0.0, 'medium'))
        base = instance(merge(FLOOR, WALL))
        self.assertEqual((base.indirect_samples, base.denoise, base.quality), (0, 0.0, 'medium'))
        image = s.render(s.Scene(splats=(base,), lights=(SUN_FROM_LEFT,)), CAMERA, 16, 12, ambient=0.1)
        again = s.render(s.Scene(splats=(replace(base, quality='final', indirect_distance=9.0),), lights=(SUN_FROM_LEFT,)),
                         CAMERA, 16, 12, ambient=0.1)
        np.testing.assert_array_equal(image, again)


if __name__ == '__main__':
    unittest.main()
