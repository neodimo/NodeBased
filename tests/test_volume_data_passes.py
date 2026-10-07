"""The data passes see volumes (lane 4 Rendering, step T3 part 3): depth, normals, position, uv, object ids, Cryptomatte and
the motion pass read the smoke where its density first reaches the depth pass's threshold, in the path tracers and the
other renderers; and smoke honours `volume_multi_scatter` and `volume_fire_light` in the path tracers."""
import unittest
from dataclasses import replace

import numpy as np

from nodebased import cryptomatte3d, motionblur, pathtrace as pt, scene3d as s, volumerender
from tests.test_3d_pathtrace import sphere
from tests.test_3d_pathtrace_gpu_soft import READY

N = 40
VOXEL = 4.0 / N
FPS = 24.0
CAMERA = s.Camera(s.Transform3D(s.Vec3(0, 0, 9)), s.Vec3(0, 0, 0), 30)
SIZE = 48
SETTINGS = volumerender.VolumeSettings(absorption=.8, scattering=1.2, density_scale=3, fps=FPS, step_size=0.02)


def blob(voxels_per_frame=0.0, name="Plume"):
    """A soft sphere of density at the middle of a 4 unit box; its cells move `voxels_per_frame` voxels a frame along +x."""
    r = (np.arange(N) + .5 - N / 2) / (N * .3)
    density = np.clip(1 - (r[:, None, None] ** 2 + r[None, :, None] ** 2 + r[None, None, :] ** 2), 0, None).astype(np.float32)
    velocity = None
    if voxels_per_frame:
        velocity = np.zeros((N,) * 3 + (3,), np.float32)
        velocity[..., 0] = voxels_per_frame * VOXEL * FPS
    return s.Volume(density, VOXEL, (-2.0,) * 3, velocity=velocity, name=name)


def trace(scene, output, backend="cpu", camera=CAMERA):
    return pt.render(scene, camera, SIZE, SIZE, (0, 0, 0, 0), 0.0, output, pt.PathSettings(samples=4), backend=backend,
                     volume=SETTINGS)


def centre(image):
    return image[SIZE // 2, SIZE // 2]


class PathTracerPasses(unittest.TestCase):
    def test_the_depth_pass_is_the_first_sample_at_the_threshold_and_every_pass_covers_that_pixel(self):
        scene = s.Scene(volumes=(blob(),))
        depth = trace(scene, "depth")
        self.assertEqual(centre(depth)[3], 1.0)
        self.assertEqual(depth[0, 0, 3], 0.0)
        # the sphere's surface where the density reaches the threshold: density (1 - r^2) * 3 = 0.1
        radius = 0.3 * 4.0 * np.sqrt(1 - SETTINGS.depth_threshold / SETTINGS.density_scale)
        self.assertAlmostEqual(float(centre(depth)[0]), 9.0 - radius, delta=0.05)
        for output in ("normals", "position", "uv", "object_id"):
            image = trace(scene, output)
            np.testing.assert_array_equal(image[..., 3] > 0, depth[..., 3] > 0, err_msg=output)

    def test_the_normal_is_the_density_gradient_pointing_out_and_toward_the_eye(self):
        scene = s.Scene(volumes=(blob(),))
        normals, position = trace(scene, "normals"), trace(scene, "position")
        np.testing.assert_allclose(centre(normals)[:3], (0, 0, 1), atol=0.05)
        covered = np.argwhere(normals[..., 3] > 0)
        for y, x in covered[::25]:
            radial = position[y, x, :3] / np.linalg.norm(position[y, x, :3])      # the blob is a sphere around the origin
            self.assertGreater(float(normals[y, x, :3] @ radial), 0.95)
            self.assertAlmostEqual(float(np.linalg.norm(normals[y, x, :3])), 1.0, delta=1e-3)
            self.assertGreater(float(normals[y, x, 2]), 0.0)                       # facing the camera at (0, 0, 9)

    def test_position_is_where_the_depth_sample_sits(self):
        scene = s.Scene(volumes=(blob(),))
        position, depth = trace(scene, "position"), trace(scene, "depth")
        self.assertAlmostEqual(float(centre(position)[2]), 9.0 - float(centre(depth)[0]), delta=1e-3)
        self.assertAlmostEqual(float(centre(position)[0]), 0.0, delta=0.06)       # the pixel centre is half a pixel off the axis

    def test_the_volume_has_its_own_object_id_after_the_geometries_and_splats(self):
        mesh = sphere(radius=0.4, position=(-1.6, 0, 0))
        scene = s.Scene((mesh,), volumes=(blob(), blob(name="Second")))
        ids = trace(scene, "object_id")
        self.assertEqual(s.volume_id_base(scene), 2)
        self.assertEqual(float(centre(ids)[0]), 2.0)                       # the first volume
        self.assertIn(1.0, np.unique(ids[..., 0]))                          # the mesh
        self.assertEqual(s.particle_id_base(scene), 4)

    def test_a_surface_in_front_of_the_smoke_hides_it_in_every_pass(self):
        wall = sphere(radius=0.8, position=(0, 0, 3.0))
        scene = s.Scene((wall,), volumes=(blob(),))
        for output in ("depth", "normals", "position", "object_id"):
            behind = trace(s.Scene((wall,)), output)
            both = trace(scene, output)
            np.testing.assert_allclose(centre(both), centre(behind), atol=1e-4, err_msg=output)
        self.assertEqual(float(centre(trace(scene, "object_id"))[0]), 1.0)

    def test_smoke_in_front_of_a_surface_wins_where_its_threshold_is_met(self):
        wall = sphere(radius=0.8, position=(0, 0, -3.0))
        scene = s.Scene((wall,), volumes=(blob(),))
        self.assertEqual(float(centre(trace(scene, "object_id"))[0]), 2.0)
        self.assertLess(float(centre(trace(scene, "depth"))[0]), 9.0)


class OtherRenderers(unittest.TestCase):
    def test_the_raster_and_ray_traced_modes_see_the_smoke_too(self):
        scene = s.Scene(volumes=(blob(),))
        for mode in ("raster", "raytrace"):
            ids = s.render(scene, CAMERA, SIZE, SIZE, output="object_id", mode=mode, volume=SETTINGS)
            self.assertEqual(float(centre(ids)[0]), 1.0, mode)
            normals = s.render(scene, CAMERA, SIZE, SIZE, output="normals", mode=mode, volume=SETTINGS)
            np.testing.assert_allclose(centre(normals)[:3], (0, 0, 1), atol=0.05, err_msg=mode)
            reference = trace(scene, "normals")
            np.testing.assert_allclose(normals, reference, atol=1e-5, err_msg=mode)

    def test_cryptomatte_isolates_the_smoke_by_its_solver_name(self):
        mesh = sphere(radius=0.4, position=(-1.6, 0, 0))
        scene = s.Scene((mesh,), volumes=(blob(name="Kettle Steam"),))
        layers, metadata = cryptomatte3d.render_cryptomatte(scene, CAMERA, SIZE, SIZE, samples=1, mode="raytrace")
        manifest = metadata["CryptoObject"]["manifest"]
        self.assertIn("Kettle Steam", manifest)
        from nodebased import cryptomatte
        bits = np.float32(centre(layers["CryptoObject00"])[0]).view(np.uint32)
        self.assertEqual(int(bits), cryptomatte.name_to_bits("Kettle Steam"))
        self.assertAlmostEqual(float(centre(layers["CryptoObject00"])[1]), 1.0, delta=1e-6)
        self.assertIn("volume", [name for name in metadata["CryptoMaterial"]["manifest"]])

    def test_the_scene_state_manifest_lists_the_volume_with_its_cryptomatte_bits(self):
        from nodebased import cryptomatte, scene_state
        scene = s.Scene(volumes=(blob(name="Kettle Steam"), blob(name="")))
        rows = scene_state._object_manifest(scene)
        self.assertEqual([row["type"] for row in rows], ["volume", "volume"])
        self.assertEqual(rows[0]["id"], f"{cryptomatte.name_to_bits('Kettle Steam'):08x}")
        self.assertEqual(rows[1]["name"], "volume2")
        np.testing.assert_allclose(rows[0]["world_bounding_box"], [[-2, -2, -2], [2, 2, 2]], atol=1e-6)


class MotionPass(unittest.TestCase):
    def pixels_per_unit(self, depth):
        a, _ = s.project(CAMERA, SIZE, SIZE, np.array([[0, 0, 9 - depth], [1, 0, 9 - depth]], np.float32))
        return float(a[1][0] - a[0][0])

    def test_a_plume_moving_two_voxels_a_frame_reads_that_velocity(self):
        scene = s.Scene(volumes=(blob(2.0),))
        vectors = motionblur.motion_vectors(scene, CAMERA, scene, CAMERA, SIZE, SIZE, volume=SETTINGS)
        depth = float(centre(trace(scene, "depth"))[0])
        expected = 2 * VOXEL * self.pixels_per_unit(depth)
        self.assertEqual(vectors[SIZE // 2, SIZE // 2, 3], 1.0)
        self.assertAlmostEqual(float(vectors[SIZE // 2, SIZE // 2, 0]), expected, delta=expected * 0.05)
        self.assertAlmostEqual(float(vectors[SIZE // 2, SIZE // 2, 1]), 0.0, delta=0.02)
        covered = vectors[..., 3] > 0
        np.testing.assert_array_equal(covered, trace(scene, "depth")[..., 3] > 0)
        np.testing.assert_allclose(vectors[covered][:, 0], expected, rtol=0.1)

    def test_a_still_plume_has_zero_motion_under_a_still_camera_and_follows_the_camera_otherwise(self):
        scene = s.Scene(volumes=(blob(),))
        vectors = motionblur.motion_vectors(scene, CAMERA, scene, CAMERA, SIZE, SIZE, volume=SETTINGS)
        covered = vectors[..., 3] > 0
        self.assertGreater(int(covered.sum()), 50)
        np.testing.assert_allclose(vectors[covered][:, :2], 0.0, atol=1e-4)
        moved = replace(CAMERA, transform=s.Transform3D(s.Vec3(0.5, 0, 9)))
        panned = motionblur.motion_vectors(scene, CAMERA, scene, moved, SIZE, SIZE, volume=SETTINGS)
        point = trace(scene, "position")[SIZE // 2, SIZE // 2, :3][None]
        before, _ = s.project(CAMERA, SIZE, SIZE, point)
        after, _ = s.project(moved, SIZE, SIZE, point)
        expected = after[0] - before[0]
        self.assertLess(float(expected[0]), -0.5)
        np.testing.assert_allclose(panned[SIZE // 2, SIZE // 2, :2], expected, atol=0.02)

    def test_a_volume_that_moves_with_its_transform_shows_it(self):
        volume = blob()
        scene = s.Scene(volumes=(volume,))
        later = s.Scene(volumes=(replace(volume, matrix=np.array([[1, 0, 0, 0.5], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                                                              np.float32)),))
        vectors = motionblur.motion_vectors(scene, CAMERA, later, CAMERA, SIZE, SIZE, volume=SETTINGS)
        expected = 0.5 * self.pixels_per_unit(float(centre(trace(scene, "depth"))[0]))
        self.assertAlmostEqual(float(vectors[SIZE // 2, SIZE // 2, 0]), expected, delta=expected * 0.05)

    def test_a_mesh_in_front_keeps_its_own_vector(self):
        wall = sphere(radius=0.8, position=(0, 0, 3.0))
        scene = s.Scene((wall,), volumes=(blob(2.0),))
        vectors = motionblur.motion_vectors(scene, CAMERA, scene, CAMERA, SIZE, SIZE, volume=SETTINGS)
        self.assertEqual(vectors[SIZE // 2, SIZE // 2, 3], 1.0)
        self.assertAlmostEqual(float(vectors[SIZE // 2, SIZE // 2, 0]), 0.0, delta=1e-4)


@unittest.skipUnless(READY, "no wgpu adapter cleared for splats and smoke in the path tracer")
class CardAgrees(unittest.TestCase):
    def test_the_card_makes_the_same_data_passes_as_the_reference(self):
        wall = sphere(radius=0.6, position=(1.4, 0, 0))
        scene = s.Scene((wall,), volumes=(blob(),))
        for output in ("depth", "normals", "position", "uv", "object_id"):
            cpu, gpu = trace(scene, output), trace(scene, output, backend="gpu")
            np.testing.assert_array_equal(cpu[..., 3], gpu[..., 3], err_msg=output)
            np.testing.assert_allclose(cpu, gpu, atol=2e-3, err_msg=output)
        self.assertEqual(float(centre(trace(scene, "object_id", backend="gpu"))[0]), 2.0)


class Knobs(unittest.TestCase):
    def render(self, backend="cpu", **knobs):
        scene = s.Scene(volumes=(blob(),))
        settings = replace(SETTINGS, **knobs)
        return pt.render(scene, CAMERA, 24, 24, (0, 0, 0, 0), 0.8, "rgba", pt.PathSettings(samples=128, max_bounces=6, seed=2),
                         backend=backend, volume=settings)

    def test_multiple_scattering_amount_zero_is_the_physical_render_and_more_brightens_it(self):
        base = self.render()
        same = self.render(multi_scatter=0.0)
        np.testing.assert_array_equal(base, same)
        bright = self.render(multi_scatter=1.0)
        self.assertGreater(float(bright[..., :3].sum()), float(base[..., :3].sum()) * 1.05)
        self.assertLess(float(bright[..., :3].sum()), float(base[..., :3].sum()) * 2.0)      # only the later scatterings gain
        np.testing.assert_allclose(bright[..., 3], base[..., 3], atol=1e-6)                    # the smoke is as opaque

    def fire_scene(self):
        density = blob().density
        hot = np.zeros_like(density)
        hot[:, :, 20:30] = density[:, :, 20:30]            # a flame in the front half
        return s.Scene(volumes=(s.Volume(density, VOXEL, (-2.0,) * 3, temperature=hot),))

    def fire(self, light, backend="cpu"):
        settings = replace(SETTINGS, fire_intensity=2.0, temperature_scale=3000.0, fire_threshold=800.0, fire_light=light)
        return pt.render(self.fire_scene(), CAMERA, 24, 24, (0, 0, 0, 0), 0.0, "rgba",
                         pt.PathSettings(samples=192, max_bounces=6, seed=2), backend=backend, volume=settings)

    def test_fire_light_gains_the_fire_that_lights_the_smoke_and_leaves_the_fire_seen_directly_alone(self):
        off, normal, strong = self.fire(0.0), self.fire(1.0), self.fire(3.0)
        self.assertGreater(float(off[..., :3].sum()), 0.0)                  # the fire itself is still there
        self.assertLess(float(off[..., :3].sum()), float(normal[..., :3].sum()))
        gain_normal = float(normal[..., :3].sum() - off[..., :3].sum())
        gain_strong = float(strong[..., :3].sum() - off[..., :3].sum())
        self.assertGreater(gain_normal, 0.0)
        self.assertAlmostEqual(gain_strong / gain_normal, 3.0, delta=0.3)       # the added light is linear in the knob

    @unittest.skipUnless(READY, "no wgpu adapter cleared for splats and smoke in the path tracer")
    def test_the_card_honours_both_knobs(self):
        cpu = self.render(multi_scatter=1.0)
        gpu = self.render("gpu", multi_scatter=1.0)
        self.assertLess(abs(float(gpu[..., :3].sum() / cpu[..., :3].sum()) - 1), 0.05)
        off_cpu, off_gpu = self.fire(0.0), self.fire(0.0, "gpu")
        strong_cpu, strong_gpu = self.fire(3.0), self.fire(3.0, "gpu")
        self.assertLess(abs(float(off_gpu[..., :3].sum() / off_cpu[..., :3].sum()) - 1), 0.06)
        self.assertLess(abs(float(strong_gpu[..., :3].sum() / strong_cpu[..., :3].sum()) - 1), 0.06)
        self.assertGreater(float(strong_gpu[..., :3].sum()), float(off_gpu[..., :3].sum()) * 1.02)


if __name__ == "__main__":
    unittest.main()
