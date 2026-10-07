"""Volume motion blur in the path tracers (lane 4 Rendering, step T3 part 2): smoke carried along its velocity across
Render3D's shutter, or mixed between two cache frames where it has none, on the CPU reference and the GPU."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from nodebased import motionblur, pathtrace as pt, scene3d as s, vdbio, volumerender
from nodebased.core import Dispatcher
from nodebased.imaging import Evaluator
from tests.test_3d_pathtrace_gpu_soft import READY

N = 40
VOXEL = 4.0 / N
FPS = 24.0
CAMERA = s.Camera(s.Transform3D(s.Vec3(0, 0, 9)), s.Vec3(0, 0, 0), 30)
SETTINGS = volumerender.VolumeSettings(absorption=.8, scattering=1.2, density_scale=3, fps=FPS)


def blob(voxels_per_frame=2.0, axis=0):
    """A soft sphere in the middle of a 4 unit box, every cell moving `voxels_per_frame` voxels a frame along `axis`."""
    r = (np.arange(N) + .5 - N / 2) / (N * .16)
    density = np.clip(1 - (r[:, None, None] ** 2 + r[None, :, None] ** 2 + r[None, None, :] ** 2), 0, None).astype(np.float32)
    velocity = np.zeros((N,) * 3 + (3,), np.float32)
    velocity[..., axis] = voxels_per_frame * VOXEL * FPS
    return s.Volume(density, VOXEL, (-2.0,) * 3, velocity=velocity)


def centroid(volume):
    d = np.asarray(volume.density, np.float64)
    grid = np.indices(d.shape)
    return np.array([(g * d).sum() / d.sum() for g in grid])


def spread(image):
    """Standard deviation, in pixels, of the coverage along x and along y."""
    a = image[..., 3].astype(np.float64)
    ys, xs = np.indices(a.shape)
    out = []
    for axis in (xs, ys):
        mean = (axis * a).sum() / a.sum()
        out.append(float(np.sqrt((((axis - mean) ** 2) * a).sum() / a.sum())))
    return out


def moments(volume, times):
    scene = s.Scene(volumes=(volume,))
    return [(motionblur.advect_scene(scene, t, FPS), CAMERA) for t in times]


def render(shots, backend="cpu", samples=96):
    return pt.render_motion(shots, 48, 48, (0, 0, 0, 0), .8, "rgba",
                            pt.PathSettings(samples=samples, max_bounces=3, seed=3), backend=backend, volume=SETTINGS)


class Advection(unittest.TestCase):
    def test_a_cell_reads_the_density_the_flow_brings_to_it_and_the_mass_is_kept(self):
        volume = blob(2.0)
        moved = motionblur.advect_volume(volume, 0.5, FPS)       # half a frame at two voxels a frame: one voxel
        shift = centroid(moved) - centroid(volume)
        np.testing.assert_allclose(shift, (1.0, 0, 0), atol=0.02)
        self.assertAlmostEqual(float(moved.density.sum() / volume.density.sum()), 1.0, delta=0.01)
        back = motionblur.advect_volume(volume, -1.0, FPS)
        np.testing.assert_allclose(centroid(back) - centroid(volume), (-2.0, 0, 0), atol=0.02)

    def test_no_velocity_no_time_and_a_zero_field_leave_the_volume_alone(self):
        volume = blob(2.0)
        still = s.Volume(volume.density, VOXEL, volume.origin)
        self.assertIs(motionblur.advect_volume(still, 0.5, FPS), still)
        self.assertIs(motionblur.advect_volume(volume, 0.0, FPS), volume)
        zero = s.Volume(volume.density, VOXEL, volume.origin, velocity=np.zeros_like(volume.velocity))
        np.testing.assert_array_equal(motionblur.advect_volume(zero, 0.7, FPS).density, volume.density)

    def test_temperature_travels_with_the_density(self):
        volume = blob(2.0)
        hot = s.Volume(volume.density, VOXEL, volume.origin, velocity=volume.velocity, temperature=volume.density * 3)
        moved = motionblur.advect_volume(hot, 0.5, FPS)
        np.testing.assert_allclose(moved.temperature, moved.density * 3, atol=1e-5)

    def test_the_raymarch_keeps_its_own_shutter_and_the_scene_helper_leaves_volumes_alone_by_default(self):
        scene = s.Scene(volumes=(blob(2.0),))
        self.assertIs(motionblur.advect_scene(scene, 0.5).volumes[0], scene.volumes[0])
        self.assertIsNot(motionblur.advect_scene(scene, 0.5, FPS).volumes[0], scene.volumes[0])

    def test_two_cache_frames_stand_in_where_there_is_no_velocity(self):
        a = s.Volume(np.full((8, 8, 8), 1.0, np.float32), 0.5, (0, 0, 0))
        b = s.Volume(np.full((8, 8, 8), 3.0, np.float32), 0.5, (0, 0, 0))
        mixed = motionblur.blend_volumes(s.Scene(volumes=(a,)), s.Scene(volumes=(b,)), 0.25).volumes[0]
        np.testing.assert_allclose(mixed.density, 1.5)
        other = s.Volume(np.full((4, 4, 4), 3.0, np.float32), 0.5, (0, 0, 0))                    # another grid: left alone
        self.assertIs(motionblur.blend_volumes(s.Scene(volumes=(a,)), s.Scene(volumes=(other,)), 0.25).volumes[0], a)
        moving = blob(1.0)                                                                          # has a velocity: advected instead
        self.assertIs(motionblur.blend_volumes(s.Scene(volumes=(moving,)), s.Scene(volumes=(moving,)), 0.5).volumes[0], moving)


class Blur(unittest.TestCase):
    TIMES = (0.0, 0.25, 0.5, 0.75, 1.0)          # a one frame shutter at two voxels a frame

    def test_a_moving_plume_blurs_along_its_velocity_and_a_still_one_does_not(self):
        moving = blob(8.0, axis=0)
        still = s.Volume(moving.density, VOXEL, moving.origin, velocity=np.zeros_like(moving.velocity))
        sharp_x, sharp_y = spread(render(moments(moving, [0.0])))
        blur_x, blur_y = spread(render(moments(moving, self.TIMES)))
        self.assertGreater(blur_x, sharp_x * 1.15)                # smeared along x by 8 voxels (0.8 of the plume's width)
        self.assertAlmostEqual(blur_y, sharp_y, delta=sharp_y * 0.05)
        still_x, still_y = spread(render(moments(still, self.TIMES)))
        self.assertAlmostEqual(still_x, sharp_x, delta=sharp_x * 0.03)
        self.assertAlmostEqual(still_y, sharp_y, delta=sharp_y * 0.03)

    def test_the_blur_follows_the_direction_of_the_velocity(self):
        up = blob(8.0, axis=1)
        sharp_x, sharp_y = spread(render(moments(up, [0.0])))
        blur_x, blur_y = spread(render(moments(up, self.TIMES)))
        self.assertGreater(blur_y, sharp_y * 1.15)
        self.assertAlmostEqual(blur_x, sharp_x, delta=sharp_x * 0.05)

    def test_the_mean_picture_keeps_its_brightness(self):
        moving = blob(8.0)
        sharp = render(moments(moving, [0.0]))[..., :3].sum()
        blurred = render(moments(moving, self.TIMES))[..., :3].sum()
        self.assertAlmostEqual(float(blurred / sharp), 1.0, delta=0.06)

    @unittest.skipUnless(READY, "no wgpu adapter cleared for splats and smoke in the path tracer")
    def test_the_card_blurs_the_same_way(self):
        moving = blob(8.0)
        cpu_x, cpu_y = spread(render(moments(moving, self.TIMES)))
        gpu_x, gpu_y = spread(render(moments(moving, self.TIMES), backend="gpu"))
        self.assertAlmostEqual(gpu_x, cpu_x, delta=cpu_x * 0.04)
        self.assertAlmostEqual(gpu_y, cpu_y, delta=cpu_y * 0.04)
        sharp_x, _ = spread(render(moments(moving, [0.0]), backend="gpu"))
        self.assertGreater(gpu_x, sharp_x * 1.15)


class Render3DNode(unittest.TestCase):
    """Render3D's shutter reaches the volumes of a ReadVDB3D scene in path tracer mode (and only there)."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.path = Path(self.dir.name) / "plume.vdb"
        vdbio.write_scene(self.path, s.Scene(volumes=(blob(8.0),)))

    def document(self, mode, shutter):
        d = Dispatcher()
        for key, kind, params in (
                ("read", "ReadVDB3D", dict(vdb_path=str(self.path))), ("scene", "Scene3D", {}),
                ("camera", "Camera3D", dict(tz=9.0, focal=50.0)),
                ("render", "Render3D", dict(width=48, height=48, render_mode=mode, pt_samples=64, max_bounces=3, ambient=.8,
                                            motion_blur=1 if shutter else 0, shutter=shutter, motion_samples=5,
                                            volume_density_scale=3.0, volume_scattering=1.2, volume_absorption=.8))):
            d.execute(dict(op="create", id=key, type=kind, params=params))
        d.execute(dict(op="connect", id="scene", input="object0", source="read"))
        d.execute(dict(op="connect", id="render", input="scene", source="scene"))
        d.execute(dict(op="connect", id="render", input="camera", source="camera"))
        return d

    def picture(self, mode, shutter):
        return np.asarray(Evaluator().evaluate(self.document(mode, shutter).document, "render", frame=0))

    def test_the_shutter_smears_the_plume_in_path_tracer_mode(self):
        sharp_x, _ = spread(self.picture("pathtrace", 0.0))
        blur_x, _ = spread(self.picture("pathtrace", 1.0))
        self.assertGreater(blur_x, sharp_x * 1.05)

    def test_smoke_without_a_velocity_is_mixed_between_the_two_cache_frames(self):
        rng = np.random.default_rng(5)
        frames = [s.Volume((rng.random((12, 12, 12)) * 2).astype(np.float32), 0.25, (-1.5,) * 3) for _ in range(3)]
        for number, volume in enumerate(frames):
            vdbio.write_scene(Path(self.dir.name) / f"cache.{number:04d}.vdb", s.Scene(volumes=(volume,)))
        d = self.document("pathtrace", 1.0)
        d.document["nodes"]["read"]["params"]["vdb_path"] = str(Path(self.dir.name) / "cache.####.vdb")
        params = d.document["nodes"]["render"]["params"]
        params["motion_samples"] = 3
        evaluator = Evaluator()

        def moment_density(mode, index):
            shots, _, _ = evaluator._motion_inputs(d.document, d.document["nodes"]["render"], dict(params, render_mode=mode),
                                                   1, 1, None, (None, None))
            return np.asarray(shots[index][0].volumes[0].density)
        density = [np.asarray(v.density) for v in frames]
        # a one frame shutter centred on frame 1 opens at 0.5, closes at 1.5
        np.testing.assert_allclose(moment_density("pathtrace", 0), 0.5 * density[0] + 0.5 * density[1], atol=1e-5)
        np.testing.assert_allclose(moment_density("pathtrace", 1), density[1], atol=1e-5)          # on the frame: as cached
        np.testing.assert_allclose(moment_density("pathtrace", 2), 0.5 * density[1] + 0.5 * density[2], atol=1e-5)
        np.testing.assert_allclose(moment_density("raytrace", 0), density[0], atol=1e-5)           # the raymarch keeps the cache frame

if __name__ == "__main__":
    unittest.main()
