"""Sparse fire caches build the same coarse fire-light grid as dense ones, from tiles alone, and light nearby meshes."""
from dataclasses import replace
import tracemalloc
import unittest
from unittest.mock import patch

import numpy as np

from nodebased import gpu3d, gpuvolume, scene3d, sparsevol, volumerender
from nodebased.sparsevol import SparseField, SparseGrid

SETTINGS = volumerender.VolumeSettings(fire_intensity=1.0, flame_intensity=2.0, fire_light=1.0, fire_light_meshes=50)


def seeded_fields(n, seed=3):
    """A seeded fire in an n-cubed box: a few hot, dense blobs and a flame core, empty elsewhere."""
    rng = np.random.default_rng(seed)
    density = np.zeros((n, n, n), np.float32)
    temperature = np.zeros((n, n, n), np.float32)
    flame = np.zeros((n, n, n), np.float32)
    for _ in range(4):
        x, y, z = rng.integers(2, n - 2, 3)
        r = int(rng.integers(2, 6))
        sl = (slice(max(x - r, 0), x + r), slice(max(y - r, 0), y + r), slice(max(z - r, 0), z + r))
        density[sl] = rng.uniform(.2, 1.5, density[sl].shape)
        temperature[sl] = rng.uniform(900, 2200, density[sl].shape)
        flame[sl] = rng.uniform(0, 1, density[sl].shape) * (rng.random(density[sl].shape) > .5)
    density[n - 3:, n - 3:, n - 3:] = 1.0            # the last, partial tile on every axis holds fire too
    temperature[n - 3:, n - 3:, n - 3:] = 1800.0
    flame[n - 3:, n - 3:, n - 3:] = .75
    return {'density': density, 'temperature': temperature, 'flame': flame}


def pair(n=37, rest=None, tile=8, fields=None, **volume_kwargs):
    """(dense Volume, sparse Volume) of the same fire; the dense one is built from the grid's own expansion."""
    fields = fields or seeded_fields(n)
    grid = SparseGrid.from_dense(fields, tile=tile, rest=rest)
    expanded = grid.to_dense()
    dense = scene3d.Volume(expanded['density'], temperature=expanded.get('temperature'), flame=expanded.get('flame'),
                           **volume_kwargs)
    return dense, scene3d.Volume.from_sparse(grid, **volume_kwargs)


class NoExpansion:
    """Raise if anything turns a sparse grid or field into a dense array."""
    def __enter__(self):
        def boom(*args, **kwargs):
            raise AssertionError('the sparse fire light expanded a sparse volume')
        self.patches = [patch.object(SparseGrid, 'to_dense', boom), patch.object(SparseField, '__array__', boom)]
        for p in self.patches:
            p.start()

    def __exit__(self, *exc):
        for p in self.patches:
            p.stop()


class SparseFireLightGridTests(unittest.TestCase):
    def check(self, dense, sparse, settings=SETTINGS):
        expected = volumerender.fire_light_grid(dense, settings)
        with NoExpansion():
            actual = volumerender.fire_light_grid(sparse, settings)
        if expected is None:
            self.assertIsNone(actual)
            return None
        self.assertIsNotNone(actual)
        self.assertEqual(actual[0].shape, expected[0].shape)
        self.assertEqual(actual[1], expected[1])
        np.testing.assert_allclose(actual[0], expected[0], rtol=2e-4, atol=1e-5 * float(expected[0].max()))
        return expected

    def test_matches_dense_with_a_partial_edge_tile_and_a_factor_of_two(self):
        dense, sparse = pair(37, voxel_size=.05)
        grid, cell = self.check(dense, sparse)
        self.assertEqual(grid.shape[:3], (19, 19, 19))
        self.assertAlmostEqual(cell, 2 * .05)
        self.assertGreater(float(grid.max()), 0)

    def test_matches_dense_when_the_coarse_cells_straddle_tiles(self):
        dense, sparse = pair(50, tile=8, voxel_size=.04)          # factor 2 over 8-voxel tiles, 50 = 6 * 8 + 2
        self.check(dense, sparse)
        dense, sparse = pair(100, tile=8, voxel_size=.02)         # factor 4; 100 = 12 * 8 + 4
        self.check(dense, sparse)

    def test_matches_dense_with_a_nonzero_rest_value(self):
        fields = seeded_fields(37)
        rest = {'density': .05, 'temperature': 2000.0, 'flame': .1}        # the empty space itself glows
        for name, value in rest.items():
            fields[name] = np.where(fields[name] == 0, np.float32(value), fields[name]).astype(np.float32)
        dense, sparse = pair(37, rest=rest, fields=fields, voxel_size=.05)
        self.assertGreater(sparse.sparse.tile_count, 0)
        self.check(dense, sparse)

    def test_matches_dense_with_a_cold_rest_value_and_missing_fields(self):
        fields = seeded_fields(37)
        rest = {'density': .2}
        fields['density'] = np.where(fields['density'] == 0, np.float32(.2), fields['density']).astype(np.float32)
        for drop in (('flame',), ('temperature',), ()):
            kept = {k: v for k, v in fields.items() if k not in drop}
            grid = SparseGrid.from_dense(kept, tile=8, rest=rest)
            expanded = grid.to_dense()
            dense = scene3d.Volume(expanded['density'], temperature=expanded.get('temperature'),
                                   flame=expanded.get('flame'), voxel_size=.05)
            self.check(dense, scene3d.Volume.from_sparse(grid, voxel_size=.05))

    def test_empty_and_dark_sparse_volumes_cast_no_light(self):
        empty = SparseGrid.from_dense({'density': np.zeros((40, 40, 40), np.float32),
                                       'temperature': np.zeros((40, 40, 40), np.float32)}, tile=8)
        self.assertEqual(empty.tile_count, 0)
        volume = scene3d.Volume.from_sparse(empty, voxel_size=.05)
        with NoExpansion():
            self.assertIsNone(volumerender.fire_light_grid(volume, SETTINGS))
            self.assertEqual(volumerender.mesh_fire_lights(scene3d.Scene(volumes=(volume,)), SETTINGS), ())
        dense, sparse = pair(37, voxel_size=.05)
        self.assertIsNone(volumerender.fire_light_grid(sparse, replace(SETTINGS, fire_light=0.0)))
        self.assertIsNone(volumerender.fire_light_grid(sparse, volumerender.VolumeSettings()))

    def test_one_stored_tile_in_a_large_empty_volume_builds_only_the_coarse_grid(self):
        n = 200
        data = {'density': np.full((1, 8, 8, 8), 1.0, np.float32),
                'temperature': np.full((1, 8, 8, 8), 1800.0, np.float32),
                'flame': np.zeros((1, 8, 8, 8), np.float32)}
        grid = SparseGrid((n, n, n), [[10, 12, 14]], data, tile=8, rest={'density': 0, 'temperature': 0, 'flame': 0})
        volume = scene3d.Volume.from_sparse(grid, voxel_size=.01)
        tracemalloc.start()
        try:
            with NoExpansion():
                result = volumerender.fire_light_grid(volume, SETTINGS)
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        self.assertEqual(result[0].shape[:3], (29, 29, 29))              # factor ceil(200 / 32) = 7
        self.assertLess(peak, 4_000_000)                 # one dense emission volume would be 192 MB
        weights = result[0].sum(-1)
        centroid = np.array([(weights.sum(axis=tuple(b for b in range(3) if b != a)) * np.arange(29)).sum()
                             for a in range(3)]) / weights.sum()
        tile_cells = np.array([(10 * 8 + 3.5) / 7, (12 * 8 + 3.5) / 7, (14 * 8 + 3.5) / 7]) - .5   # voxel centre to cell index
        np.testing.assert_allclose(centroid, tile_cells, atol=.6)                  # the light sits at the stored tile

    def test_mesh_light_positions_and_intensities_match_dense(self):
        dense, sparse = pair(37, voxel_size=.05, origin=(-.9, -.9, -.9))
        with NoExpansion():
            lights = volumerender.mesh_fire_lights(scene3d.Scene(volumes=(sparse,)), SETTINGS)
        expected = volumerender.mesh_fire_lights(scene3d.Scene(volumes=(dense,)), SETTINGS)
        self.assertGreater(len(expected), 1)
        self.assertEqual(len(lights), len(expected))
        for a, b in zip(lights, expected):
            np.testing.assert_allclose((a.position.x, a.position.y, a.position.z),
                                       (b.position.x, b.position.y, b.position.z), rtol=1e-4, atol=1e-6)
            self.assertAlmostEqual(a.intensity, b.intensity, delta=1e-3 * b.intensity)
            np.testing.assert_allclose(a.color, b.color, rtol=1e-4, atol=1e-6)

    def test_changing_the_flame_changes_the_cached_lights(self):
        _, sparse = pair(37, voxel_size=.05)
        scene = scene3d.Scene(volumes=(sparse,))
        first = volumerender.mesh_fire_lights(scene, SETTINGS)
        self.assertTrue(all(a is b for a, b in zip(first, volumerender.mesh_fire_lights(scene, SETTINGS))))
        grid = sparse.sparse
        louder = SparseGrid(grid.shape, grid.coords, {**grid.data, 'flame': grid.data['flame'] * 3}, grid.tile, grid.rest)
        second = volumerender.mesh_fire_lights(
            scene3d.Scene(volumes=(scene3d.Volume.from_sparse(louder, voxel_size=.05),)), SETTINGS)
        self.assertNotEqual([light.intensity for light in first], [light.intensity for light in second])


def flame_pair():
    """The 16-cubed flame plume of the dense mesh-light tests as a dense and an equivalent sparse volume."""
    from tests.test_volume_flame_mesh import flame_scene
    dense = flame_scene().volumes[0]
    sparse = scene3d.Volume.from_sparse(dense.to_sparse(), voxel_size=dense.voxel_size, origin=dense.origin)
    return dense, sparse


class SparseMeshLightTests(unittest.TestCase):
    def setUp(self):
        self.dense, self.sparse = flame_pair()
        obj = scene3d._cube(.55, (.7, .7, .7, 1), scene3d.Transform3D(position=scene3d.Vec3(1.25, 0, 0)))
        blocker = scene3d._cube(.4, (.3, .3, .3, 1), scene3d.Transform3D(position=scene3d.Vec3(.58, 0, -.2)))
        fill = scene3d.Light(kind='Directional', intensity=.1, position=scene3d.Vec3(0, 3, 4))
        self.scene = scene3d.Scene(geometries=(obj,), volumes=(self.sparse,), lights=(fill,))
        self.blocked = replace(self.scene, geometries=(obj, blocker))
        self.camera = scene3d.Camera()
        self.settings = volumerender.VolumeSettings(flame_intensity=5, fire_light_meshes=100, fire_light=0)
        self.mask = scene3d.render(self.scene, self.camera, 96, 64, output='object_id')[..., 3] > .5

    def check(self, renderer):
        on = renderer(self.scene, self.camera, 96, 64, volume=self.settings, ambient=.1)
        off = renderer(self.scene, self.camera, 96, 64, volume=replace(self.settings, fire_light_meshes=0), ambient=.1)
        shadow = renderer(self.blocked, self.camera, 96, 64, volume=self.settings, ambient=.1)
        self.assertGreater(float((on - off)[self.mask, :3].mean()), .05)          # the sparse fire lights the mesh
        self.assertGreater(float((on - shadow)[self.mask, :3].mean()), .005)      # a blocker darkens it
        dense_on = renderer(replace(self.scene, volumes=(self.dense,)), self.camera, 96, 64,
                            volume=self.settings, ambient=.1)
        np.testing.assert_allclose(on, dense_on, atol=5e-3, rtol=5e-3)           # same picture as the dense cache

    def test_cpu(self):
        with NoExpansion():
            lights = volumerender.mesh_fire_lights(self.scene, self.settings)
        self.assertGreater(len(lights), 0)
        self.check(scene3d.render)

    @unittest.skipUnless(gpu3d.available(), 'no wgpu adapter')
    def test_gpu(self):
        self.check(gpu3d.render)

    def test_both_gains_zero_renders_as_before(self):
        settings = volumerender.VolumeSettings(flame_intensity=5, fire_light=0, fire_light_meshes=0)
        image = scene3d.render(self.scene, self.camera, 48, 32, volume=settings)
        real = volumerender.fire_light_grid
        built = []
        with patch.object(volumerender, 'fire_light_grid', lambda *a: built.append(real(*a)) or built[-1]):
            again = scene3d.render(self.scene, self.camera, 48, 32, volume=settings)
        self.assertEqual(built, [None])                   # nothing was built for the sparse plume
        np.testing.assert_array_equal(image, again)
        self.assertEqual(volumerender.mesh_fire_lights(self.scene, settings), ())


@unittest.skipUnless(gpu3d.available(), 'no wgpu adapter')
class SparseSmokeLightTests(unittest.TestCase):
    """The coarse grid also lights the sparse smoke itself (fire_light), on the CPU march and on the GPU."""
    def setUp(self):
        n = 16
        density = np.full((n, n, n), .6, np.float32)
        temperature = np.zeros((n, n, n), np.float32)
        temperature[:, :4, :] = 2200
        self.fields = {'density': density, 'temperature': temperature}
        self.settings = volumerender.VolumeSettings(step_size=.05, fire_intensity=1.0, fire_light=2.0,
                                                    absorption=.4, shadow_steps=8)
        self.camera = scene3d.Camera()

    def volumes(self, fields):
        grid = SparseGrid.from_dense(fields, tile=8, rest={'density': .6})
        expanded = grid.to_dense()
        dense = scene3d.Volume(expanded['density'], temperature=expanded['temperature'], voxel_size=.125)
        return dense, scene3d.Volume.from_sparse(grid, voxel_size=.125)

    def test_gpu_sparse_smoke_light_matches_dense_and_cpu(self):
        dense, sparse = self.volumes(self.fields)
        results = {}
        for name, volume in (('dense', dense), ('sparse', sparse)):
            scene = scene3d.Scene(volumes=(volume,))
            results[name] = (scene3d.render(scene, self.camera, 40, 40, volume=self.settings),
                             gpu3d.render(scene, self.camera, 40, 40, volume=self.settings))
        dark = scene3d.render(scene3d.Scene(volumes=(sparse,)), self.camera, 40, 40,
                              volume=replace(self.settings, fire_light=0.0))
        self.assertGreater(float(np.abs(results['sparse'][0] - dark).max()), .01)    # the halo is really there
        np.testing.assert_allclose(results['sparse'][0], results['dense'][0], atol=2e-3, rtol=2e-3)
        np.testing.assert_allclose(results['sparse'][1], results['dense'][1], atol=5e-3, rtol=5e-3)
        np.testing.assert_allclose(results['sparse'][1], results['sparse'][0], atol=5e-3, rtol=5e-3)

    def test_the_gpu_light_texture_follows_the_flame_and_stays_coarse(self):
        n = 16
        fields = {**self.fields, 'flame': np.zeros((n, n, n), np.float32)}
        fields['flame'][:, :3, :] = 1.0
        settings = replace(self.settings, flame_intensity=1.0)
        grid = SparseGrid.from_dense(fields, tile=8, rest={'density': .6})
        weak = scene3d.Volume.from_sparse(grid, voxel_size=.125)
        grid2 = SparseGrid(grid.shape, grid.coords, {**grid.data, 'flame': grid.data['flame'] * 4}, grid.tile, grid.rest)
        strong = scene3d.Volume.from_sparse(grid2, voxel_size=.125)
        state = gpu3d._state()
        before = len(state.get('fire_light_grids', {}))
        with NoExpansion():
            a = gpuvolume.fire_light_view(state, weak, settings, set())
            b = gpuvolume.fire_light_view(state, strong, settings, set())
            again = gpuvolume.fire_light_view(state, weak, settings, set())
        self.assertIsNotNone(a[1])
        self.assertEqual(len(state['fire_light_grids']) - before, 2)       # a different flame is a different key
        self.assertIs(a[0], again[0])                                       # the same flame reuses its texture
        self.assertIsNot(a[0], b[0])
        self.assertLessEqual(max(a[1][:3]), volumerender.FIRE_LIGHT_CELLS)
        origin_moved = replace(weak, origin=(1.0, 0.0, 0.0))
        self.assertIsNot(gpuvolume.fire_light_view(state, origin_moved, settings, set())[0], a[0])


if __name__ == '__main__':
    unittest.main()
