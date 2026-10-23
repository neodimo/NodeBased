"""The path tracer's scene preparation is kept by content (lane 4 step B of 3, `nodebased/scenecache.py`): a camera move or a material
edit reuses the splat layer, the mesh trees and the packed scene; an edit to the geometry or the splats builds what changed; and the
picture is the one an uncached build gives."""
import dataclasses
import unittest
from dataclasses import fields, replace

import numpy as np

from nodebased import gpu3d, gpupathtrace, pathtrace as pt, ptsplats, scenecache, scene3d as s
from tests.test_3d_pathtrace import FRONT, box, gpu_ready, sphere
from tests.test_3d_pathtrace_splats import SPLAT_CAMERA, SUN, instance, plane_cloud

# the SplatInstance fields the layer does not read: kept apart so a new field has to be filed on one side or the other
IGNORED_SPLAT_FIELDS = {"shadow_catch", "specular", "reflection_samples", "indirect_samples", "indirect_distance", "denoise",
                        "quality", "name", "light_link"}


def moved(camera, x):
    return replace(camera, transform=replace(camera.transform, position=s.Vec3(x, camera.transform.position.y,
                                                                              camera.transform.position.z)))


def delta(before=None):
    now = scenecache.counters()
    return now if before is None else {k: now[k] - before[k] for k in now}


def layer_arrays(layer):
    return {f.name: getattr(layer, f.name) for f in fields(layer) if isinstance(getattr(layer, f.name), np.ndarray)}


def assert_layers_equal(test, a, b):
    test.assertEqual(layer_arrays(a).keys(), layer_arrays(b).keys())
    for name, value in layer_arrays(a).items():
        other = layer_arrays(b)[name]
        if value is None or other is None:
            test.assertIs(value, other)
            continue
        np.testing.assert_array_equal(value, other, err_msg=name)


def cloud_scene(**fields_):
    return s.Scene((box((1, 1, 1), (0.7, 0.3, 0.2, 1), (0, -1.2, 0)),), (SUN,),
                   splats=(instance(plane_cloud(n=10, delit=False), **fields_),))


class Isolated(unittest.TestCase):
    def setUp(self):
        scenecache.clear()
        self.addCleanup(scenecache.clear)


class FingerprintTests(Isolated):
    def test_equal_content_is_equal_and_an_edit_in_place_is_not(self):
        a = np.arange(12, dtype=np.float32).reshape(4, 3)
        b = a.copy()
        key = lambda x: scenecache.fingerprint(x, [])
        self.assertEqual(key(a), key(b))
        b[1, 1] += 1.0
        self.assertNotEqual(key(a), key(b))

    def test_a_large_frozen_array_is_identified_by_itself_and_a_changed_one_is_not_confused_with_it(self):
        big = np.zeros(scenecache.SMALL_ARRAY_BYTES, np.float32)
        big.flags.writeable = False
        same = scenecache.fingerprint(big, [])
        self.assertEqual(same, scenecache.fingerprint(big, []))
        other = np.zeros_like(big)
        other.flags.writeable = False
        self.assertNotEqual(same, scenecache.fingerprint(other, []))

    def test_a_view_of_a_writeable_array_is_hashed_by_content(self):
        base = np.zeros(scenecache.SMALL_ARRAY_BYTES, np.float32)
        view = base[:]
        view.flags.writeable = False
        before = scenecache.fingerprint(view, [])
        base[5] = 1.0
        self.assertNotEqual(before, scenecache.fingerprint(view, []))

    def test_dataclasses_compare_by_their_fields(self):
        box_a = box((1, 1, 1), (1, 0, 0, 1), (0, 0, 0))
        key = lambda g: scenecache.fingerprint(g, [])
        self.assertEqual(key(box_a), key(replace(box_a)))
        self.assertNotEqual(key(box_a), key(replace(box_a, pbr_roughness=0.9)))
        self.assertNotEqual(key(box_a), key(replace(box_a, transform=s.Transform3D(position=s.Vec3(1, 0, 0)))))


class SplatLayerTests(Isolated):
    def build(self, scene, camera=SPLAT_CAMERA):
        return pt.build_scene(scene, 0.1, eye=s._view_basis(camera)[0])

    def test_every_splat_instance_field_is_filed(self):
        names = {f.name for f in fields(s.SplatInstance)}
        self.assertEqual(names, set(scenecache.SPLAT_FIELDS) | IGNORED_SPLAT_FIELDS)

    def test_a_camera_move_builds_no_splat_layer(self):
        scene = cloud_scene()
        self.build(scene)
        before = delta()
        for x in (0.1, 0.2, 0.3):
            self.build(scene, moved(SPLAT_CAMERA, x))
        self.assertEqual(delta(before)["splat_builds"], 0)
        self.assertEqual(delta(before)["splat_hits"], 3)

    def test_a_material_or_light_edit_builds_no_splat_layer(self):
        scene = cloud_scene()
        self.build(scene)
        before = delta()
        rough = replace(scene, geometries=(replace(scene.geometries[0], pbr_roughness=0.9),))
        dim = replace(scene, lights=(replace(SUN, intensity=0.3),))
        self.build(rough)
        self.build(dim)
        self.assertEqual(delta(before)["splat_builds"], 0)

    def test_an_edit_to_a_field_the_layer_reads_builds_one_layer_equal_to_an_uncached_build(self):
        scene = cloud_scene()
        self.build(scene)
        variants = dict(opacity_scale=0.5, scale_scale=1.5, relight=0.4, cast_shadows=False, normal_smoothing=3, sh_degree=0,
                        use_intrinsics=False, metallic=0.5, roughness_scale=0.5, intrinsics_mix=0.2,
                        matrix=np.diag([1.0, 1.0, 1.0, 1.0]) + np.array([[0, 0, 0, 0.25], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]))
        self.assertEqual(set(variants) | {"cloud"}, set(scenecache.SPLAT_FIELDS))
        for name, value in variants.items():
            edited = replace(scene, splats=(replace(scene.splats[0], **{name: value}),))
            before = delta()
            cached = self.build(edited).splats
            self.assertEqual(delta(before)["splat_builds"], 1, name)
            scenecache.clear()
            fresh = self.build(edited).splats
            assert_layers_equal(self, cached, fresh)
            scenecache.clear()
            self.build(scene)

    def test_a_changed_cloud_builds_one_layer(self):
        scene = cloud_scene()
        self.build(scene)
        before = delta()
        other = replace(scene, splats=(instance(plane_cloud(n=10, delit=False, albedo=(0.1, 0.8, 0.3))),))
        self.build(other)
        self.assertEqual(delta(before)["splat_builds"], 1)

    def test_fields_the_layer_does_not_read_keep_it(self):
        scene = cloud_scene()
        self.build(scene)
        before = delta()
        edited = replace(scene, splats=(replace(scene.splats[0], reflection_samples=4, quality="final", name="other"),))
        self.build(edited)
        self.assertEqual(delta(before)["splat_builds"], 0)

    def test_the_eye_matters_only_with_normal_smoothing_on(self):
        flat = cloud_scene()
        self.build(flat)
        before = delta()
        self.build(flat, moved(SPLAT_CAMERA, 3.0))
        self.assertEqual(delta(before)["splat_builds"], 0)
        smoothed = replace(flat, splats=(replace(flat.splats[0], normal_smoothing=4),))
        behind = s.Camera(s.Transform3D(position=s.Vec3(0, 0, -4)), s.Vec3(0, 0, 0), 30.0)
        first = self.build(smoothed).splats
        before = delta()
        self.build(smoothed, moved(SPLAT_CAMERA, 0.0))
        self.assertEqual(delta(before)["splat_builds"], 0)             # the same eye: reused
        flipped = self.build(smoothed, behind).splats
        self.assertEqual(delta(before)["splat_builds"], 1)             # another eye: the normals are flipped toward it
        scenecache.clear()
        assert_layers_equal(self, flipped, self.build(smoothed, behind).splats)
        self.assertFalse(np.array_equal(first.normal, flipped.normal))

    def test_two_prepared_scenes_do_not_share_light_link_masks(self):
        scene = cloud_scene()
        a = self.build(scene)
        b = self.build(scene)
        self.assertIsNot(a.splats, b.splats)
        self.assertIs(a.splats.bvh, b.splats.bvh)           # the heavy parts are shared, the per-scene masks are not
        a.splats.excl = np.ones(len(a.splats.instance), np.int64)
        self.assertIsNone(b.splats.excl)
        self.assertIsNone(self.build(scene).splats.excl)


class MeshTreeTests(Isolated):
    def test_a_transform_or_material_edit_reuses_the_tree_and_a_vertex_edit_does_not(self):
        scene = s.Scene((sphere(1.0), box((1, 1, 1), (1, 0, 0, 1), (2, 0, 0))), (SUN,))
        pt.build_scene(scene, 0.1)
        before = delta()
        edited = replace(scene, geometries=(replace(scene.geometries[0], pbr_roughness=0.9,
                                                    transform=s.Transform3D(position=s.Vec3(0.5, 0, 0))),
                                            scene.geometries[1]))
        pt.build_scene(edited, 0.1)
        self.assertEqual(delta(before)["blas_builds"], 0)
        vertices = scene.geometries[0].vertices.copy()
        vertices[0] += 0.5
        before = delta()
        pt.build_scene(replace(scene, geometries=(replace(scene.geometries[0], vertices=vertices), scene.geometries[1])), 0.1)
        self.assertEqual(delta(before)["blas_builds"], 1)

    def test_a_mesh_edited_in_place_is_not_served_from_the_old_tree(self):
        geometry = sphere(1.0)
        vertices = np.array(geometry.vertices, np.float64)          # writeable: hashed by content
        geometry = replace(geometry, vertices=vertices)
        scene = s.Scene((geometry,), (SUN,))
        first = pt.build_scene(scene, 0.1).blases[0]
        vertices *= 2.0
        second = pt.build_scene(scene, 0.1).blases[0]
        self.assertIsNot(first, second)
        np.testing.assert_allclose(second.hi, 2 * first.hi, rtol=1e-6)


class LruTests(Isolated):
    def test_the_oldest_entries_leave_first(self):
        cache = scenecache._Lru(2, 1000)
        for key in "abc":
            cache.put(key, key, 10, [])
        self.assertIsNone(cache.get("a"))
        self.assertEqual(cache.get("b"), "b")
        cache.put("d", "d", 10, [])
        self.assertIsNone(cache.get("c"))              # b was used last, so c went
        cache.put("big", "big", 5000, [])
        self.assertIsNone(cache.get("big"))            # larger than the budget: not kept


def soft_ready():
    """The scenes here hold splats: the GPU path tracer takes them on cleared adapters only (`soft_supported`)."""
    return gpu_ready() and gpupathtrace.soft_supported(gpu3d._state())


@unittest.skipUnless(soft_ready(), "no wgpu adapter cleared to path trace splats and smoke")
class GpuSceneTests(Isolated):
    def render(self, scene, camera=SPLAT_CAMERA, samples=2):
        return pt.render(scene, camera, 24, 18, (0.02, 0.02, 0.03, 1.0), 0.1, "rgba",
                         pt.PathSettings(samples=samples, max_bounces=2), backend="gpu")

    def test_a_camera_move_after_the_first_build_performs_no_splat_or_scene_build(self):
        scene = cloud_scene()
        self.render(scene)
        before = delta()
        for x in (0.05, 0.1, 0.15):
            cached = self.render(scene, moved(SPLAT_CAMERA, x))
            self.assertTrue(np.isfinite(cached).all())
        change = delta(before)
        self.assertEqual((change["splat_builds"], change["scene_builds"], change["blas_builds"]), (0, 0, 0))
        self.assertEqual(change["scene_hits"], 3)

    def test_a_splat_edit_performs_one_splat_build(self):
        scene = cloud_scene()
        self.render(scene)
        before = delta()
        self.render(replace(scene, splats=(replace(scene.splats[0], opacity_scale=0.5),)))
        change = delta(before)
        self.assertEqual((change["splat_builds"], change["scene_builds"]), (1, 1))

    def test_a_material_edit_keeps_the_splat_layer_and_the_trees_but_repacks(self):
        scene = cloud_scene()
        self.render(scene)
        before = delta()
        self.render(replace(scene, geometries=(replace(scene.geometries[0], pbr_roughness=0.9),)))
        change = delta(before)
        self.assertEqual((change["splat_builds"], change["blas_builds"], change["scene_builds"]), (0, 0, 1))

    def test_the_cached_picture_is_the_uncached_picture(self):
        scene = cloud_scene()
        self.render(scene)
        for camera, edited in ((moved(SPLAT_CAMERA, 0.2), scene),
                               (SPLAT_CAMERA, replace(scene, splats=(replace(scene.splats[0], opacity_scale=0.5),))),
                               (SPLAT_CAMERA, replace(scene, geometries=(replace(scene.geometries[0], pbr_roughness=0.9),)))):
            cached = self.render(edited, camera)
            scenecache.clear()
            np.testing.assert_array_equal(cached, self.render(edited, camera))
            scenecache.clear()
            self.render(scene)

    def test_dropping_the_cache_builds_again(self):
        scene = cloud_scene()
        self.render(scene)
        gpupathtrace.release_frame_cache()
        before = delta()
        self.render(scene)
        self.assertEqual(delta(before)["scene_builds"], 1)


if __name__ == "__main__":
    unittest.main()
