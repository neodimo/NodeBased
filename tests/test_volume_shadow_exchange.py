"""Lane L4 plan 3, step C part 2: meshes shadow volumes and volumes shadow meshes, splats and the catcher."""
import unittest
from dataclasses import replace

import numpy as np

from nodebased import gpu3d, scene3d as s, volumerender

W, H = 96, 72
CAMERA = s.Camera(s.Transform3D(s.Vec3(0.3, 1.6, 3.2)), s.Vec3(0, 0.3, 0))
SETTINGS = volumerender.VolumeSettings(step_size=0.04, density_scale=8.0, shadow_steps=12)
OVERHEAD = s.Light('Directional', position=s.Vec3(0, 6, 0.01), target=s.Vec3(), shadows=True)
SIDE = s.Light('Directional', position=s.Vec3(5, 1.5, 0.0), target=s.Vec3(0, .5, 0), shadows=True)


def floor():
    return s._card(4, 4, (.8, .8, .8, 1), s.Transform3D(s.Vec3(0, -0.05, 0), s.Vec3(-90, 0, 0)))


def wall(x=1.6, size=2.4, alpha=1.0):
    """A vertical card between the side light and the plume, seen edge-on by the camera."""
    return s._card(size, size, (.6, .3, .2, alpha), s.Transform3D(s.Vec3(x, .5, 0), s.Vec3(0, 90, 0)))


def render(scene, backend, **kwargs):
    kwargs.setdefault('ambient', 0.1)
    kwargs.setdefault('volume', SETTINGS)
    return (s.render if backend == 'cpu' else gpu3d.render)(scene, CAMERA, W, H, **kwargs)


class Exchange:
    backend = 'cpu'

    def test_a_dense_plume_darkens_the_floor_under_it(self):
        plume = s.analytic_plume(24, 0)
        scene = s.Scene(geometries=(floor(),), volumes=(plume,), lights=(OVERHEAD,))
        shadowed = render(scene, self.backend)
        clear = render(scene, self.backend, volume=replace(SETTINGS, shadow_density=0.0))
        bare = render(s.Scene(geometries=(floor(),), lights=(OVERHEAD,)), self.backend)
        smoke = render(s.Scene(volumes=(plume,)), self.backend)[..., 3]
        floor_only = (smoke < 0.002) & (bare[..., 3] > .99)          # floor pixels with no smoke in front
        darker = floor_only & (shadowed[..., 0] < clear[..., 0] - 0.08)
        self.assertGreater(int(darker.sum()), 20, 'the plume must shadow visible floor')
        self.assertTrue((shadowed[floor_only][:, :3] <= clear[floor_only][:, :3] + 2e-3).all(), 'smoke never brightens the floor')
        beside = floor_only & (shadowed[..., 0] >= clear[..., 0] - 1e-4)
        self.assertGreater(int(beside.sum()), 100)
        self.assertLess(float(shadowed[darker][:, 0].mean()), float(shadowed[beside][:, 0].mean()) - 0.08)
        # shadow_density scales the darkening
        heavier = render(scene, self.backend, volume=replace(SETTINGS, shadow_density=3.0))
        self.assertLess(float(heavier[darker][:, 0].mean()), float(shadowed[darker][:, 0].mean()))

    def test_a_card_between_the_light_and_the_plume_shadows_the_plume(self):
        plume = s.analytic_plume(24, 0)
        lit = s.Scene(volumes=(plume,), lights=(SIDE,))
        shaded = s.Scene(geometries=(wall(),), volumes=(plume,), lights=(SIDE,))
        open_ = render(lit, self.backend)
        blocked = render(shaded, self.backend)
        both = open_[..., 3] > 0.05
        self.assertGreater(int(both.sum()), 300)
        self.assertLess(float(blocked[both][:, :3].sum()), 0.35 * float(open_[both][:, :3].sum()),
                        'the card must take most of the light from the plume')
        # A card that does not block the light (behind the plume, on the far side from the light) leaves it alone.
        away = render(s.Scene(geometries=(wall(x=-1.6),), volumes=(plume,), lights=(SIDE,)), self.backend)
        np.testing.assert_allclose(away[both][:, :3], open_[both][:, :3], atol=2e-3)
        # A light with shadows off is not blocked.
        unshadowed = render(replace(shaded, lights=(replace(SIDE, shadows=False),)), self.backend)
        np.testing.assert_allclose(unshadowed[both][:, :3], open_[both][:, :3], atol=2e-3)


class CPUExchange(Exchange, unittest.TestCase):
    def test_the_plume_shadows_relit_splats_and_the_shadow_catcher(self):
        from tests.test_3d_splat_shadow_catch import field, sun, camera, W as SW, H as SH
        matrix = np.eye(4, dtype=np.float32)
        matrix[0, 3] = matrix[2, 3] = 0.8                     # between the light (from +x, +z) and the splats
        plume = replace(s.analytic_plume(24, 0), matrix=matrix)
        for kwargs in (dict(relight=1.0), dict(relight=0.0, shadow_catch=1.0)):
            scene = s.Scene((), (sun(),), (s.SplatInstance(field(), **kwargs),), volumes=(plume,))
            s.clear_splat_shadow_cache()
            shadowed = s.render(scene, camera(), SW, SH, output='splats', volume=SETTINGS, ambient=0.1)
            s.clear_splat_shadow_cache()
            clear = s.render(scene, camera(), SW, SH, output='splats',
                             volume=replace(SETTINGS, shadow_density=0.0), ambient=0.1)
            darker = shadowed[..., :3].sum(-1) < clear[..., :3].sum(-1) - 0.02
            self.assertGreater(int(darker.sum()), 40, kwargs)
            self.assertTrue((shadowed[..., :3] <= clear[..., :3] + 2e-3).all(), kwargs)
            self.assertGreater(int((shadowed[..., 3] > 0).sum()), 500)


@unittest.skipUnless(gpu3d.available(), 'no wgpu adapter')
class GPUExchange(Exchange, unittest.TestCase):
    backend = 'gpu'

    def test_the_gpu_matches_the_cpu_reference(self):
        plume = s.analytic_plume(24, 0)
        for scene in (s.Scene(geometries=(floor(),), volumes=(plume,), lights=(OVERHEAD,)),
                      s.Scene(geometries=(wall(),), volumes=(plume,), lights=(SIDE,)),
                      s.Scene(geometries=(floor(), wall()), volumes=(plume, replace(plume, matrix=np.array(
                          [[1, 0, 0, .9], [0, 1, 0, .2], [0, 0, 1, .3], [0, 0, 0, 1]], np.float32))),
                              lights=(OVERHEAD, SIDE))):
            cpu = render(scene, 'cpu')
            gpu = render(scene, 'gpu')
            # A card's diagonal edge rasterises differently on the two backends for a pixel or two (meshes alone too).
            self.assertLessEqual(int((np.abs(cpu - gpu).max(-1) > 0.02).sum()), 3)
            self.assertLess(float(np.abs(cpu - gpu).mean()), 1e-3)
        np.testing.assert_array_equal(render(scene, 'gpu'), render(scene, 'gpu'))

    def test_the_gpu_splat_shadow_provider_carries_the_smoke_like_the_cpu_one(self):
        from nodebased import gpurt_render
        from tests.test_3d_gpu_splat_shadows import field as splat_field, occluder, sun as splat_sun
        from tests.test_3d_splat_shadow_catch import mesh_parts
        matrix = np.eye(4, dtype=np.float32)
        matrix[:3, 3] = (1.0, 0.0, 1.0)
        plume = replace(s.analytic_plume(24, 0), matrix=matrix)
        casters = volumerender.ShadowCasters((plume,), SETTINGS)
        scene = s.Scene((occluder(),), (splat_sun(),), (s.SplatInstance(splat_field(), relight=1, shadow_catch=1),))
        s.clear_splat_shadow_cache()
        mesh, bvh, bias = mesh_parts(scene)
        reference = s._SplatShadows(scene.splats, scene.lights, mesh, bvh, bias)
        indices = np.arange(len(scene.splats[0].cloud))
        plain = {name: getattr(reference, name)(0, indices) for name in ('for_indices', 'catch_for_indices')}
        reference.volume_shadows = casters
        with gpurt_render.GpuSplatShadows(gpu3d._state(None), scene.splats, scene.lights, mesh, bvh, bias) as candidate:
            candidate.volume_shadows = casters
            for name in plain:
                expected = getattr(reference, name)(0, indices)
                self.assertLess(float(expected.sum()), float(plain[name].sum()) - 1.0, 'the smoke must darken splats')
                np.testing.assert_allclose(getattr(candidate, name)(0, indices), expected, atol=5e-3, rtol=0)

    def test_more_than_four_volumes_shadowing_meshes_fall_back_to_the_cpu(self):
        plume = s.analytic_plume(8, 0)
        scene = s.Scene(geometries=(floor(),), volumes=(plume,) * 5, lights=(OVERHEAD,))
        with self.assertRaises(gpu3d.Unsupported):
            render(scene, 'gpu')


if __name__ == '__main__':
    unittest.main()
