"""Lane 6 step N2: sparse volumes go to the GPU as a tile atlas and a tile index, never as a dense grid.

The viewport draw, the GPU raster Render3D and the raster mesh shadows read the atlas; each is compared with the dense
upload of the same grid. GPU cases skip without an adapter. Run the module on the other adapters with
`scratch/nb-lanes/run/force-adapter.py integrated|cpu tests.test_sparse_gpu_volumes`.
"""
import os
import unittest
from dataclasses import replace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QApplication

from nodebased import gpu3d, gpuvolume, scene3d as s, viewportgpu, volumerender
from nodebased.sparsevol import SparseField, SparseGrid
from tests import gpu_precision
from tools import benchmark_sparse_gpu as bench

APP = QApplication.instance() or QApplication([])
TOLERANCE = 5e-3 if gpu_precision.half_float_target() else 2e-4
CAMERA = bench.camera()
SETTINGS = volumerender.VolumeSettings(step_size=0.02, density_scale=8.0, shadow_steps=12)
SUN = s.Light('Directional', position=s.Vec3(4, 5, 3), target=s.Vec3())
OVERHEAD = s.Light('Directional', position=s.Vec3(0, 6, 0.01), target=s.Vec3(), shadows=True)
BACKGROUND = (0.025, 0.025, 0.03, 1.0)


def fields(n=40, shape='column', velocity=True, temperature=True):
    density = bench.SHAPES[shape](n)
    out = {'density': density}
    rng = np.random.default_rng(3)
    live = (density > 0)
    if temperature:
        out['temperature'] = np.where(live, 400.0 + 900.0 * density, 0.0).astype(np.float32)
    if velocity:
        v = np.zeros((n, n, n, 3), np.float32)
        v[..., 0], v[..., 1] = 1.2, 2.0
        v[..., 2] = rng.normal(0, 0.3, (n, n, n))
        out['velocity'] = v * live[..., None]
    return out


def twin(n=40, shape='column', **kwargs):
    """(dense Volume, sparse Volume) holding the same field values."""
    grid = SparseGrid.from_dense(fields(n, shape, **kwargs))
    sparse = s.Volume.from_sparse(grid, voxel_size=1.0 / n, origin=(-0.5, 0.0, -0.5))
    dense = s.Volume(**{name: array for name, array in grid.to_dense().items()}, voxel_size=1.0 / n, origin=(-0.5, 0.0, -0.5))
    return dense, sparse


def floor():
    return s._card(3, 3, (.8, .8, .8, 1), s.Transform3D(s.Vec3(0, -0.02, 0), s.Vec3(-90, 0, 0)))


def render(volume, *, settings=SETTINGS, lights=(SUN,), geometries=(), width=120, height=90, **kwargs):
    return gpu3d.render(s.Scene(geometries=geometries, volumes=(volume,), lights=lights), CAMERA, width, height,
                        volume=settings, ambient=kwargs.pop('ambient', 0.15), **kwargs)


class never_dense:
    """While active, expanding a sparse grid to a dense array fails the test."""

    def __enter__(self):
        error = AssertionError('a sparse volume was expanded to a dense grid')
        self.patches = [patch.object(SparseGrid, 'to_dense', side_effect=error),
                        patch.object(SparseField, '__array__', side_effect=error)]
        for p in self.patches:
            p.start()

    def __exit__(self, *exc):
        for p in self.patches:
            p.stop()


class GridPacking(unittest.TestCase):
    """The atlas and the index rebuild the dense grid exactly (no GPU needed)."""

    def test_index_and_atlas_rebuild_every_voxel_of_an_odd_sized_grid(self):
        rng = np.random.default_rng(1)
        density = np.zeros((37, 29, 21), np.float32)
        density[5:20, 3:12, 8:15] = rng.random((15, 9, 7))
        velocity = np.zeros(density.shape + (3,), np.float32)
        velocity[5:20, 3:12, 8:15] = rng.random((15, 9, 7, 3))
        grid = SparseGrid.from_dense({'density': density, 'velocity': velocity})
        index, atlas, vel = grid.gpu_index(), grid.gpu_atlas('density'), grid.gpu_atlas('velocity', 4)
        self.assertEqual(index.shape[:3], (3, 4, 5))
        self.assertEqual(vel.shape[3], 4)
        rebuilt = np.zeros_like(density)
        for x, y, z in np.ndindex(density.shape):
            info = index[z // 8, y // 8, x // 8]
            if info[3] > 0:
                ox, oy, oz = (int(v) for v in info[:3])
                rebuilt[x, y, z] = atlas[oz + z % 8, oy + y % 8, ox + x % 8]
        np.testing.assert_array_equal(rebuilt, density)
        self.assertEqual(int((index[..., 3] > 0).sum()), grid.tile_count)

    def test_the_atlas_holds_the_stored_tiles_and_a_layer_of_slack_at_most(self):
        for count in (1, 7, 8, 9, 64, 65, 513):
            coords = np.stack(np.unravel_index(np.arange(count), (9, 9, 9)), 1).astype(np.int32)
            grid = SparseGrid((72, 72, 72), coords, {'density': np.ones((count, 8, 8, 8), np.float32)})
            ax, ay, az = grid.atlas_layout()
            self.assertGreaterEqual(ax * ay * az, count)
            self.assertLess(ax * ay * (az - 1), count)

    def test_max_reads_the_stored_tiles_and_the_rest_value(self):
        grid = SparseGrid.from_dense({'density': bench.column_plume(32)})
        self.assertAlmostEqual(grid.max('density'), float(bench.column_plume(32).max()), places=6)
        self.assertEqual(grid.max('density'), SparseField(grid, 'density').max())
        empty = SparseGrid.from_dense({'density': np.zeros((16, 16, 16), np.float32)})
        self.assertEqual(empty.tile_count, 0)
        self.assertEqual(empty.max('density'), 0.0)


@unittest.skipUnless(gpu3d.available(), 'no wgpu adapter')
class SparseMatchesDense(unittest.TestCase):
    def agree(self, dense, sparse, **kwargs):
        expected = render(dense, **kwargs)
        with never_dense():
            actual = render(sparse, **kwargs)
        self.assertGreater(int((expected[..., 3] > 0.02).sum()), 150, 'the scene must show smoke')
        self.assertLess(float(np.abs(actual - expected).max()), TOLERANCE * 10)
        self.assertLess(float(np.abs(actual - expected).mean()), TOLERANCE)
        return expected, actual

    def test_a_lit_plume_renders_the_same_from_tiles_as_from_the_dense_grid(self):
        dense, sparse = twin()
        self.assertLess(sparse.sparse.tile_count, 0.4 * (40 // 8) ** 3, 'the plume must leave tiles empty')
        self.agree(dense, sparse)

    def test_three_lights_unlit_and_supersampled_renders_agree(self):
        dense, sparse = twin(32, 'block')
        lamp = s.Light('Point', color=(1.0, 0.6, 0.3), intensity=3.0, position=s.Vec3(-1.5, 1.2, 1.0), target=s.Vec3(0, .5, 0),
                       falloff_type='Quadratic')
        self.agree(dense, sparse, lights=(SUN, lamp, OVERHEAD))
        self.agree(dense, sparse, lights=())
        self.agree(dense, sparse, samples=2, background=(0.1, 0.2, 0.3, 1.0))

    def test_fire_emission_and_motion_blur_read_the_temperature_and_velocity_atlases(self):
        dense, sparse = twin()
        fire = replace(SETTINGS, fire_intensity=2.0, temperature_scale=1.0, fire_threshold=500.0, fire_light=0.0)
        self.agree(dense, sparse, settings=fire)
        blurred = replace(SETTINGS, motion_blur=1.5, motion_samples=4)
        self.agree(dense, sparse, settings=blurred)

    def test_a_volume_that_shadows_a_floor_agrees(self):
        dense, sparse = twin(32, 'column')
        expected, actual = self.agree(dense, sparse, lights=(OVERHEAD,), geometries=(floor(),))
        clear = render(dense, lights=(OVERHEAD,), geometries=(floor(),), settings=replace(SETTINGS, shadow_density=0.0))
        self.assertGreater(float(np.abs(expected - clear).max()), 0.05, 'the floor must carry the smoke shadow')

    def test_a_volume_with_a_nonzero_rest_value_and_an_odd_grid_agrees(self):
        density = np.full((37, 29, 21), 0.05, np.float32)
        density[5:20, 3:12, 8:15] = 0.9
        grid = SparseGrid.from_dense({'density': density}, rest={'density': 0.05})
        self.assertLess(grid.tile_count, 3 * 4 * 5)
        kwargs = dict(voxel_size=1.0 / 29, origin=(-0.6, 0.0, -0.4))
        sparse = s.Volume.from_sparse(grid, **kwargs)
        dense = s.Volume(grid.to_dense()['density'], **kwargs)
        self.agree(dense, sparse)

    def test_the_viewport_draw_agrees_and_never_expands_the_tiles(self):
        dense, sparse = twin()
        renderer = viewportgpu.renderer()
        expected = renderer.render(s.Scene(volumes=(dense,), lights=(SUN,)), CAMERA, 160, 120, BACKGROUND, ambient=0.15)
        with never_dense():
            actual = renderer.render(s.Scene(volumes=(sparse,), lights=(SUN,)), CAMERA, 160, 120, BACKGROUND, ambient=0.15)
        self.assertFalse(renderer.volume_note.startswith('volumes hidden'), renderer.volume_note)
        self.assertGreater(int((expected[..., 0] != expected[0, 0, 0]).sum()), 400)
        self.assertLessEqual(int(np.abs(expected.astype(int) - actual.astype(int)).max()), 1)

    def test_the_control_passes_agree(self):
        dense, sparse = twin()
        settings = replace(SETTINGS, motion_blur=1.0, motion_samples=3)
        for name in ('volume_density', 'volume_temperature', 'volume_motion', 'depth'):
            expected = gpu3d.render(s.Scene(volumes=(dense,)), CAMERA, 80, 60, output=name, volume=settings)
            with never_dense():
                actual = gpu3d.render(s.Scene(volumes=(sparse,)), CAMERA, 80, 60, output=name, volume=settings)
            self.assertGreater(float(np.abs(expected).max()), 0.05, name)
            self.assertLess(float(np.abs(actual - expected).max()), 2e-3 * max(1.0, float(np.abs(expected).max())), name)

    def test_the_vorticity_pass_and_mixed_storage_stay_on_the_cpu(self):
        dense, sparse = twin(24)
        with self.assertRaisesRegex(gpu3d.Unsupported, 'vorticity'):
            gpu3d.render(s.Scene(volumes=(sparse,)), CAMERA, 32, 24, output='volume_vorticity', volume=SETTINGS)
        mixed = replace(sparse, temperature=np.asarray(dense.temperature))
        with self.assertRaisesRegex(gpu3d.Unsupported, 'same tiles'):
            gpu3d.render(s.Scene(volumes=(mixed,)), CAMERA, 32, 24, volume=SETTINGS)


@unittest.skipUnless(gpu3d.available(), 'no wgpu adapter')
class SparseUploads(unittest.TestCase):
    @staticmethod
    def volume_bytes(state):
        """Bytes of the cached volume textures, without the one-texel stand-ins bound for unused fields."""
        return sum(entry[2] for key, entry in state.get('volume_textures', {}).items()
                   if not (isinstance(key, tuple) and key[0] == 'dummy'))

    def state(self):
        state = gpu3d._state()
        for entry in list(state.get('volume_textures', {}).values()):
            entry[0].destroy()
        state.get('volume_textures', {}).clear()
        return state

    def test_a_plume_filling_a_tenth_of_its_box_uses_under_a_quarter_of_the_dense_texture_memory(self):
        for size in (128, 256):
            for shape in bench.SHAPES:
                dense, sparse = bench.volumes(size, shape=shape)
                state = self.state()
                tracker = state['memory']
                before = tracker.current
                render(dense, width=48, height=36)
                dense_bytes = self.volume_bytes(state)
                dense_live = tracker.current - before
                state = self.state()
                before = tracker.current
                with never_dense():
                    render(sparse, width=48, height=36)
                sparse_bytes = self.volume_bytes(state)
                sparse_live = tracker.current - before
                self.assertEqual(dense_bytes, size ** 3 * 4)
                self.assertLess(sparse_bytes, 0.25 * dense_bytes, (size, shape))
                self.assertLess(sparse_live, 0.25 * dense_live, (size, shape))

    def test_an_all_empty_tile_set_draws_nothing_and_marches_nothing(self):
        grid = SparseGrid.from_dense({'density': np.zeros((32, 32, 32), np.float32)})
        self.assertEqual(grid.tile_count, 0)
        empty = s.Volume.from_sparse(grid, voxel_size=1 / 32, origin=(-0.5, 0.0, -0.5))
        state = self.state()
        scene = s.Scene(volumes=(empty,), lights=(SUN,))
        image = gpu3d.render(scene, CAMERA, 64, 48, volume=SETTINGS, background=(0.1, 0.2, 0.3, 1.0))
        np.testing.assert_allclose(image[..., :3], np.broadcast_to(np.float32([0.1, 0.2, 0.3]), image[..., :3].shape))
        self.assertEqual(float(gpuvolume.march_steps(state, scene, CAMERA, 64, 48, SETTINGS).sum()), 0.0)
        viewport = viewportgpu.renderer()
        frame = viewport.render(scene, CAMERA, 64, 48, BACKGROUND, ambient=0.15)
        self.assertEqual(int((frame[..., :3] != frame[0, 0, :3]).any(axis=2).sum()), 0)
        self.assertEqual(self.volume_bytes(state), 0, 'no atlas and no index is uploaded for an empty tile set')
        for name in ('volume_density', 'volume_motion'):
            self.assertEqual(float(np.abs(gpu3d.render(scene, CAMERA, 32, 24, output=name, volume=SETTINGS)).max()), 0.0)

    def test_empty_tiles_are_jumped_not_marched(self):
        dense, sparse = twin(64, 'column', velocity=False, temperature=False)
        state = self.state()
        fine = replace(SETTINGS, step_size=0.004)
        steps_dense = gpuvolume.march_steps(state, s.Scene(volumes=(dense,)), CAMERA, 64, 48, fine)
        steps_sparse = gpuvolume.march_steps(state, s.Scene(volumes=(sparse,)), CAMERA, 64, 48, fine)
        self.assertLess(float(steps_sparse.sum()), 0.6 * float(steps_dense.sum()))
        self.assertLessEqual(float(steps_sparse.max()), float(steps_dense.max()))
        rays_through_empty_space = (steps_dense > 20) & (steps_sparse < 0.5 * steps_dense)
        self.assertGreater(int(rays_through_empty_space.sum()), 100)
        same = np.abs(render(dense, settings=fine) - render(sparse, settings=fine)).max()
        self.assertLess(float(same), TOLERANCE * 10, 'jumping empty tiles must not change the picture')

    def test_a_frame_whose_tile_count_grows_uploads_its_own_atlas_without_a_stale_tile(self):
        n = 32
        order = [tuple(c) for c in np.argwhere(np.ones((4, 4, 4), bool))]
        centre = np.array((1.5, 1.5, 1.5))
        order.sort(key=lambda c: float(np.linalg.norm(np.array(c) - centre)))   # the plume grows outward from the middle
        axis = (np.arange(8) + 0.5) / 8.0 - 0.5
        blob = np.exp(-4.0 * (axis[:, None, None] ** 2 + axis[None, :, None] ** 2 + axis[None, None, :] ** 2)).astype(np.float32)

        def frame(count):
            density = np.zeros((n, n, n), np.float32)
            for index, (x, y, z) in enumerate(order[:count]):
                density[x * 8:x * 8 + 8, y * 8:y * 8 + 8, z * 8:z * 8 + 8] = blob * (0.5 + 0.5 * ((index % 3) + 1) / 3)
            kwargs = dict(voxel_size=1 / n, origin=(-0.5, 0.0, -0.5))
            grid = SparseGrid.from_dense({'density': density})
            return s.Volume.from_sparse(grid, **kwargs), s.Volume(density, **kwargs)

        state = self.state()
        layouts = []
        for count in (7, 8, 9, 20, 28, 9, 7):       # the adaptive domain grows, then a scrub goes back
            sparse, dense = frame(count)
            self.assertEqual(sparse.sparse.tile_count, count)
            before = gpuvolume.upload_count(state)
            with never_dense():
                actual = render(sparse, width=80, height=60)
            if count not in (9, 7) or layouts.count(sparse.sparse.atlas_layout()) == 0:
                self.assertGreater(gpuvolume.upload_count(state), before, 'a new tile set is a new upload')
            np.testing.assert_allclose(actual, render(dense, width=80, height=60), atol=TOLERANCE * 10, err_msg=str(count))
            layouts.append(sparse.sparse.atlas_layout())
        self.assertGreaterEqual(len(set(layouts)), 4, 'the atlas must have been laid out again as the tile count grew')

    def test_the_same_tiles_with_new_values_upload_new_values(self):
        density = fields(32, 'column', velocity=False, temperature=False)['density']
        first = s.Volume.from_sparse(SparseGrid.from_dense({'density': density}), voxel_size=1 / 32, origin=(-0.5, 0.0, -0.5))
        second = s.Volume.from_sparse(SparseGrid.from_dense({'density': density * 2.0}), voxel_size=1 / 32, origin=(-0.5, 0.0, -0.5))
        np.testing.assert_array_equal(first.sparse.coords, second.sparse.coords)
        self.state()
        a, b = render(first, width=64, height=48), render(second, width=64, height=48)
        self.assertGreater(float(np.abs(a - b).max()), 0.05)
        np.testing.assert_allclose(b, render(s.Volume(density * 2.0, voxel_size=1 / 32, origin=(-0.5, 0.0, -0.5)),
                                              width=64, height=48), atol=TOLERANCE * 10)

    def test_the_viewport_falls_back_to_the_cpu_reference_when_the_atlas_cannot_be_held(self):
        from nodebased.viewport3d import Viewport3D
        dense, sparse = twin(24)
        widget = Viewport3D()
        self.addCleanup(widget.close)
        widget.resize(40, 30)
        image = QImage(40, 30, QImage.Format.Format_RGBA8888)
        painter = QPainter(image)
        scene = s.Scene(volumes=(sparse,))
        kind = gpuvolume.adapter_kind(gpu3d._state())
        with patch.dict(gpuvolume.VOLUME_MEMORY_BUDGETS, {kind: 1000}):
            self.assertFalse(widget._paint_gpu(painter, viewportgpu.renderer(), scene, widget._camera(), None))
        self.assertTrue(viewportgpu.renderer().volume_note.startswith('volumes hidden'))
        self.assertTrue(widget._paint_gpu(painter, viewportgpu.renderer(), scene, widget._camera(), None))
        painter.end()


if __name__ == '__main__':
    unittest.main()
