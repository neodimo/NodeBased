"""Burn-rate glow and fire-derived mesh lights on the CPU and GPU volume paths."""
from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np

from nodebased import gpu3d, gpupathtrace, pathtrace, scene3d, volumerender
from nodebased.core import Dispatcher, upgrade_document
from nodebased.imaging import Evaluator
from tests.test_volume_fire_look import BASE, hot_cold


def flame_scene():
    volume = hot_cold(0, 0, n=16, density=.25)
    flame = np.zeros(volume.shape, np.float32)
    flame[6:10, 6:10, 6:10] = 2
    return scene3d.Scene(volumes=(replace(volume, flame=flame, temperature=None),))


class FlameChannelTests(unittest.TestCase):
    def test_off_preserves_the_old_pixels_and_on_adds_only_energy(self):
        scene = flame_scene()
        without = replace(scene.volumes[0], flame=None)
        camera = scene3d.Camera()
        baseline = scene3d.render(replace(scene, volumes=(without,)), camera, 48, 48, volume=BASE)
        off = scene3d.render(scene, camera, 48, 48, volume=BASE)
        on = scene3d.render(scene, camera, 48, 48,
                            volume=replace(BASE, flame_intensity=2, fire_light=0))
        np.testing.assert_array_equal(off, baseline)
        np.testing.assert_array_equal(on[..., 3], off[..., 3])
        self.assertGreater(float((on - off)[..., :3].sum()), 0)

    @unittest.skipUnless(gpu3d.available(), 'no wgpu adapter')
    def test_gpu_raster_matches_cpu_and_the_off_path(self):
        scene = flame_scene()
        camera = scene3d.Camera()
        settings = replace(BASE, flame_intensity=2, fire_light=0)
        cpu = scene3d.render(scene, camera, 48, 48, volume=settings)
        gpu = gpu3d.render(scene, camera, 48, 48, volume=settings)
        np.testing.assert_allclose(gpu, cpu, atol=.005, rtol=.005)
        self_lit = replace(settings, fire_light=1)
        np.testing.assert_allclose(gpu3d.render(scene, camera, 48, 48, volume=self_lit),
                                   scene3d.render(scene, camera, 48, 48, volume=self_lit),
                                   atol=.005, rtol=.005)
        off = gpu3d.render(scene, camera, 48, 48, volume=BASE)
        no_field = gpu3d.render(replace(scene, volumes=(replace(scene.volumes[0], flame=None),)),
                                camera, 48, 48, volume=BASE)
        np.testing.assert_array_equal(off, no_field)
        bare = replace(scene.volumes[0], density=np.zeros_like(scene.volumes[0].density))
        bare_scene = replace(scene, volumes=(bare,))
        cpu_bare = scene3d.render(bare_scene, camera, 48, 48, volume=settings)
        gpu_bare = gpu3d.render(bare_scene, camera, 48, 48, volume=settings)
        self.assertGreater(float(cpu_bare[..., :3].sum()), 0)
        np.testing.assert_allclose(gpu_bare, cpu_bare, atol=.005, rtol=.005)

    def test_flame_only_pathtrace_emits_on_both_backends(self):
        scene = flame_scene()
        scene = replace(scene, volumes=(replace(scene.volumes[0],
                                               density=np.zeros_like(scene.volumes[0].density)),))
        camera = scene3d.Camera()
        settings = volumerender.VolumeSettings(flame_intensity=1, fire_light=0)
        path = pathtrace.PathSettings(samples=32, max_bounces=1)
        cpu = pathtrace.render(scene, camera, 24, 24, settings=path, backend='cpu', volume=settings)
        no_field = replace(scene, volumes=(replace(scene.volumes[0], flame=None),))
        np.testing.assert_array_equal(
            pathtrace.render(scene, camera, 24, 24, settings=path, backend='cpu',
                             volume=volumerender.VolumeSettings()),
            pathtrace.render(no_field, camera, 24, 24, settings=path, backend='cpu',
                             volume=volumerender.VolumeSettings()))
        if not gpu3d.available():
            return
        if not gpupathtrace.soft_supported(gpu3d._state()):
            with self.assertRaisesRegex(ValueError, 'GPU Render3D unsupported'):
                pathtrace.render(scene, camera, 24, 24, settings=path, backend='gpu', volume=settings)
            np.testing.assert_allclose(
                pathtrace.render(scene, camera, 24, 24, settings=path, backend='auto', volume=settings),
                cpu, atol=.005, rtol=.005)
            return
        gpu = pathtrace.render(scene, camera, 24, 24, settings=path, backend='gpu', volume=settings)
        np.testing.assert_allclose(gpu, cpu, atol=.005, rtol=.005)
        np.testing.assert_array_equal(
            pathtrace.render(scene, camera, 24, 24, settings=path, backend='gpu',
                             volume=volumerender.VolumeSettings()),
            pathtrace.render(no_field, camera, 24, 24, settings=path, backend='gpu',
                             volume=volumerender.VolumeSettings()))

    @unittest.skipUnless(gpu3d.available(), 'no wgpu adapter')
    def test_sparse_flame_field_emits_on_both_gpu_paths(self):
        dense = flame_scene().volumes[0]
        dense = replace(dense, density=np.zeros_like(dense.density))
        sparse = scene3d.Volume.from_sparse(dense.to_sparse(), voxel_size=dense.voxel_size,
                                            origin=dense.origin)
        scene = scene3d.Scene(volumes=(sparse,))
        camera = scene3d.Camera()
        settings = volumerender.VolumeSettings(flame_intensity=1, fire_light=0)
        cpu = scene3d.render(scene, camera, 24, 24, volume=settings)
        gpu = gpu3d.render(scene, camera, 24, 24, volume=settings)
        np.testing.assert_allclose(gpu, cpu, atol=.005, rtol=.005)
        path = pathtrace.PathSettings(samples=32, max_bounces=1)
        cpu = pathtrace.render(scene, camera, 24, 24, settings=path, backend='cpu', volume=settings)
        if not gpupathtrace.soft_supported(gpu3d._state()):
            with self.assertRaisesRegex(ValueError, 'GPU Render3D unsupported'):
                pathtrace.render(scene, camera, 24, 24, settings=path, backend='gpu', volume=settings)
            np.testing.assert_allclose(
                pathtrace.render(scene, camera, 24, 24, settings=path, backend='auto', volume=settings),
                cpu, atol=.005, rtol=.005)
            return
        gpu = pathtrace.render(scene, camera, 24, 24, settings=path, backend='gpu', volume=settings)
        np.testing.assert_allclose(gpu, cpu, atol=.005, rtol=.005)

    @unittest.skipUnless(gpu3d.available(), 'no wgpu adapter')
    def test_unsupported_adapter_keeps_cpu_flame_pathtrace_available(self):
        scene = flame_scene()
        camera = scene3d.Camera()
        settings = volumerender.VolumeSettings(flame_intensity=1, fire_light=0)
        path = pathtrace.PathSettings(samples=4, max_bounces=1)
        cpu = pathtrace.render(scene, camera, 16, 16, settings=path, backend='cpu', volume=settings)
        with patch.object(gpupathtrace, 'soft_supported', return_value=False):
            with self.assertRaisesRegex(ValueError, 'GPU Render3D unsupported'):
                pathtrace.render(scene, camera, 16, 16, settings=path, backend='gpu', volume=settings)
            auto = pathtrace.render(scene, camera, 16, 16, settings=path, backend='auto', volume=settings)
        np.testing.assert_array_equal(auto, cpu)


class MeshLightTests(unittest.TestCase):
    def setUp(self):
        volume = flame_scene().volumes[0]
        obj = scene3d._cube(.55, (.7, .7, .7, 1),
                            scene3d.Transform3D(position=scene3d.Vec3(1.25, 0, 0)))
        occluder = scene3d._cube(.4, (.3, .3, .3, 1),
                                 scene3d.Transform3D(position=scene3d.Vec3(.58, 0, -.2)))
        fill = scene3d.Light(kind='Directional', intensity=.1, position=scene3d.Vec3(0, 3, 4))
        self.scene = scene3d.Scene(geometries=(obj,), volumes=(volume,), lights=(fill,))
        self.blocked = replace(self.scene, geometries=(obj, occluder))
        self.camera = scene3d.Camera()
        self.settings = volumerender.VolumeSettings(flame_intensity=5, fire_light_meshes=100, fire_light=0)
        self.mask = scene3d.render(self.scene, self.camera, 96, 64, output='object_id')[..., 3] > .5

    def _check(self, renderer):
        on = renderer(self.scene, self.camera, 96, 64, volume=self.settings, ambient=.1)
        off = renderer(self.scene, self.camera, 96, 64,
                       volume=replace(self.settings, fire_light_meshes=0), ambient=.1)
        shadow = renderer(self.blocked, self.camera, 96, 64, volume=self.settings, ambient=.1)
        self.assertGreater(float((on - off)[self.mask, :3].mean()), .05)
        self.assertGreater(float((on - shadow)[self.mask, :3].mean()), .005)
        clear = replace(self.scene.volumes[0], density=np.zeros_like(self.scene.volumes[0].density))
        thick = replace(self.scene.volumes[0], density=np.full_like(self.scene.volumes[0].density, 2.0))
        clear_image = renderer(replace(self.scene, volumes=(clear,)), self.camera, 96, 64,
                               volume=self.settings, ambient=.1)
        thick_image = renderer(replace(self.scene, volumes=(thick,)), self.camera, 96, 64,
                               volume=self.settings, ambient=.1)
        self.assertGreater(float((clear_image - thick_image)[self.mask, :3].mean()), .01)
        np.testing.assert_array_equal(on, renderer(self.scene, self.camera, 96, 64,
                                                   volume=self.settings, ambient=.1))

    def test_cpu_mesh_light_and_occlusion(self):
        self._check(scene3d.render)

    @unittest.skipUnless(gpu3d.available(), 'no wgpu adapter')
    def test_gpu_mesh_light_and_occlusion(self):
        self._check(gpu3d.render)

    def test_cluster_is_bounded_deterministic_and_reused(self):
        first = volumerender.mesh_fire_lights(self.scene, self.settings)
        self.assertGreater(len(first), 0)
        self.assertLessEqual(len(first), 8)
        self.assertLessEqual(len(volumerender.mesh_fire_lights(
            replace(self.scene, volumes=self.scene.volumes * 3), self.settings)), 16)
        self.assertTrue(all(a is b for a, b in zip(first, volumerender.mesh_fire_lights(self.scene, self.settings))))
        changed = replace(self.scene.volumes[0], flame=self.scene.volumes[0].flame * 2)
        second = volumerender.mesh_fire_lights(replace(self.scene, volumes=(changed,)), self.settings)
        self.assertNotEqual([light.intensity for light in first], [light.intensity for light in second])

    def test_old_document_gets_safe_defaults(self):
        d = Dispatcher()
        d.execute({'op': 'create', 'id': 'r', 'type': 'Render3D'})
        for key in ('volume_flame_intensity', 'volume_flame_temperature', 'volume_fire_light_meshes'):
            d.document['nodes']['r']['params'].pop(key)
        params = upgrade_document(d.document)['nodes']['r']['params']
        self.assertEqual((params['volume_flame_intensity'], params['volume_flame_temperature'],
                          params['volume_fire_light_meshes']), (0.0, 1800.0, 0.0))

    def test_unedited_graph_keeps_its_cached_render_identity(self):
        d = Dispatcher()
        for key, kind, params in (('v', 'Plume3D', {}), ('s', 'Scene3D', {}),
                                  ('c', 'Camera3D', {}), ('r', 'Render3D', {'width': 16, 'height': 16})):
            d.execute({'op': 'create', 'id': key, 'type': kind, 'params': params})
        for target, slot, source in (('s', 'object0', 'v'), ('r', 'scene', 's'), ('r', 'camera', 'c')):
            d.execute({'op': 'connect', 'id': target, 'input': slot, 'source': source})
        evaluator = Evaluator()
        first = evaluator.evaluate_raster(d.document, 'r', frame=1)
        self.assertIs(evaluator.evaluate_raster(d.document, 'r', frame=1), first)


if __name__ == '__main__':
    unittest.main()
