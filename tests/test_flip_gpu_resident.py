"""Lane 6 Fluids 7 step N1: the GPU-resident FLIP liquid solver (nodebased/flip_gpu_resident.py) against the CPU
reference (flip3d.Liquid3D), plus the GPU level set (nodebased/flip_gpu_levelset.py).

Agreement with the reference on a dam break, a pouring source, a collider, an open face and a wind force; bit-identical
checkpoint and restart; a cancel inside a substep that leaves the committed state and the cache untouched; the scan,
sort and top-up kernels on their own; the surface field against `liquid_surface.level_set`.
Run on all three local adapters: the default (RTX 3080 Ti), then integrated (AMD) and cpu (llvmpipe) through the lane's
force-adapter.py runner.
"""
import unittest

import numpy as np

from nodebased import fluid_gpu_solver as fgs
from nodebased import flip3d, liquid_surface, simcache
from nodebased.cancellation import Cancelled
from nodebased.simcache import State
from tests.test_flip3d import BoxCollider, BoxSource

DAM = {"nx": 16, "ny": 16, "nz": 8, "gravity": 0.08, "substeps": 1, "flip_ratio": 0.8, "max_iterations": 300,
       "tolerance": 1.0e-5}


def sorted_arrays(state):
    a = state.arrays
    order = np.argsort(a["id"], kind="stable")
    return {k: v[order] for k, v in a.items()}


class CancelAfter:
    """A cancel flag that reads as set from its `after`-th look onward."""

    def __init__(self, after):
        self.after, self.looks = after, 0

    def is_set(self):
        self.looks += 1
        return self.looks > self.after


def resident(params, **kwargs):
    from nodebased.flip_gpu_resident import GpuLiquid3D
    return GpuLiquid3D(params, **kwargs)


def assert_states_equal(case, a, b):
    sa, sb = sorted_arrays(a), sorted_arrays(b)
    case.assertEqual(sa.keys(), sb.keys())
    for name in sa:
        np.testing.assert_array_equal(sa[name], sb[name], err_msg=name)
    case.assertEqual(a.meta, b.meta)


@unittest.skipUnless(fgs.available(), "no wgpu compute adapter")
class KernelTests(unittest.TestCase):
    def test_scan_is_an_exclusive_prefix_sum_at_every_size(self):
        from nodebased.flip_gpu_resident import _scan
        ctx = fgs._ctx()
        rng = np.random.default_rng(5)
        for n in (1, 7, 1024, 1025, 5000, 70000, 300001):
            data = rng.integers(0, 40, n).astype(np.uint32)
            src = ctx.buffer(4 * n)
            dst = ctx.buffer(4 * (n + 1))
            bsum = ctx.buffer(4 * (-(-(n + 1) // 1024) + 2))
            ctx.write(src, data)
            _scan(ctx, src, dst, bsum, n)
            got = ctx.read(dst, 4 * (n + 1)).view(np.uint32)
            expected = np.concatenate(([0], np.cumsum(data, dtype=np.uint64)))
            np.testing.assert_array_equal(got, expected.astype(np.uint32), err_msg=str(n))

    def test_bins_sort_particles_by_cell_then_id(self):
        from nodebased.flip_gpu_resident import Bins, PART
        ctx = fgs._ctx()
        rng = np.random.default_rng(11)
        shape = (9, 7, 5)
        n = 4000
        pos = rng.uniform(-0.5, 10.0, (n, 3)).astype(np.float32)          # some outside the grid: clamped to the edge
        rec = np.zeros(n, PART)
        rec["px"], rec["py"], rec["pz"] = pos[:, 0], pos[:, 1], pos[:, 2]
        rec["id"] = rng.permutation(n).astype(np.uint32)
        rec["id"][::97] = 0xFFFFFFFF                                      # dead particles are not binned
        part = ctx.buffer(36 * n)
        ctx.write(part, rec)
        bins = Bins(ctx, shape)
        bins.ensure(n)
        bins.begin()
        common = ((0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0, 1.0))
        bins.count(part, 0, n, common)
        bins.sort(part, n, common)
        cells = np.clip(np.floor(pos).astype(int), 0, np.array(shape) - 1)
        flat = (cells[:, 0] * shape[1] + cells[:, 1]) * shape[2] + cells[:, 2]
        alive = np.flatnonzero(rec["id"] != 0xFFFFFFFF)
        expected = alive[np.lexsort((rec["id"][alive], flat[alive]))]
        offsets = ctx.read(bins.offsets, 4 * (bins.cells + 1)).view(np.uint32)
        order = ctx.read(bins.order, 4 * n).view(np.uint32)[:len(alive)]
        np.testing.assert_array_equal(order, expected)
        np.testing.assert_array_equal(np.diff(offsets), np.bincount(flat[alive], minlength=bins.cells))

    def test_a_dispatch_past_one_row_of_workgroups_still_covers_every_particle(self):
        from nodebased.flip_gpu_resident import Bins, PART
        ctx = fgs._ctx()
        n = 4_300_000                                         # more threads than one 65,535-workgroup row holds
        if 36 * n > min(ctx.max_binding, ctx.max_buffer):
            self.skipTest("the adapter's largest buffer is too small for this many particles")
        rng = np.random.default_rng(8)
        shape = (16, 16, 16)
        pos = rng.uniform(0.0, 16.0, (n, 3)).astype(np.float32)
        rec = np.zeros(n, PART)
        rec["px"], rec["py"], rec["pz"] = pos[:, 0], pos[:, 1], pos[:, 2]
        rec["id"] = np.arange(n, dtype=np.uint32)
        part = ctx.buffer(36 * n)
        ctx.write(part, rec)
        bins = Bins(ctx, shape)
        bins.ensure(n)
        bins.begin()
        common = ((0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0, 1.0))
        bins.count(part, 0, n, common)
        bins.sort(part, n, common)
        cells = np.clip(np.floor(pos).astype(int), 0, 15)
        flat = (cells[:, 0] * 16 + cells[:, 1]) * 16 + cells[:, 2]
        offsets = ctx.read(bins.offsets, 4 * (bins.cells + 1)).view(np.uint32)
        np.testing.assert_array_equal(np.diff(offsets), np.bincount(flat, minlength=bins.cells))
        order = ctx.read(bins.order, 4 * n).view(np.uint32)
        np.testing.assert_array_equal(order, np.lexsort((np.arange(n), flat)).astype(np.uint32))

    def test_top_up_and_thinning_follow_the_reference_rule(self):
        # a 6x6x6 block of cells holding 8 particles each; one interior cell left with 1 particle (a gap, topped up to 3)
        # and one corner cell crowded with 20 (thinned to the 12 lowest ids). No gravity, no motion.
        params = {"nx": 8, "ny": 8, "nz": 8, "gravity": 0.0, "substeps": 1}
        rng = np.random.default_rng(2)
        cells = np.array([(i, j, k) for i in range(1, 7) for j in range(1, 7) for k in range(1, 7)])
        pos = np.repeat(cells, 8, axis=0) + rng.random((len(cells) * 8, 3))
        ids = np.arange(len(pos))
        gap_cell = (3, 3, 3)
        in_gap = np.all(np.floor(pos).astype(int) == gap_cell, axis=1)
        keep = ~in_gap
        keep[np.flatnonzero(in_gap)[0]] = True
        pos, ids = pos[keep], ids[keep]
        crowd = (1, 1, 1) + rng.random((12, 3))
        pos = np.concatenate([pos, crowd])
        ids = np.concatenate([ids, np.arange(len(ids), len(ids) + 12)])
        arrays = flip3d.empty_arrays()
        arrays.update(position=pos.astype(np.float32), velocity=np.zeros((len(pos), 3), np.float32),
                      id=ids.astype(np.int64), age=np.zeros(len(pos), np.int32), temperature=np.ones(len(pos), np.float32))
        state = State(arrays, {"next_id": int(ids.max()) + 1, "substep_count": 0})
        reference = flip3d.Liquid3D(params).step(state, 1, 0, 0)
        got = resident(params).step(state, 1, 0, 0)

        def per_cell(s):
            c = np.floor(s.arrays["position"]).astype(int)
            return np.bincount((c[:, 0] * 8 + c[:, 1]) * 8 + c[:, 2], minlength=512)

        np.testing.assert_array_equal(per_cell(got), per_cell(reference))
        np.testing.assert_array_equal(np.sort(got.arrays["id"]), np.sort(reference.arrays["id"]))
        self.assertEqual(got.meta["next_id"], reference.meta["next_id"])
        # the topped-up particles took the neighbours' mean velocity (zero here) and sit inside the gap cell
        fresh = got.arrays["id"] >= int(ids.max()) + 1
        self.assertGreater(int(fresh.sum()), 0)
        inside = np.floor(got.arrays["position"][fresh]).astype(int)
        self.assertTrue(np.all(inside == gap_cell))


@unittest.skipUnless(fgs.available(), "no wgpu compute adapter")
class ReferenceAgreementTests(unittest.TestCase):
    def advance(self, solver, state, frames, substeps=1):
        for frame in range(1, frames + 1):
            for sub in range(substeps):
                state = solver.step(state, frame, sub, 0)
        return state

    def compare(self, cpu_state, gpu_state, atol=4e-3, rtol=2e-3, vatol=None):
        a, b = sorted_arrays(cpu_state), sorted_arrays(gpu_state)
        self.assertEqual(len(a["id"]), len(b["id"]))
        np.testing.assert_array_equal(a["id"], b["id"])
        np.testing.assert_allclose(b["position"], a["position"], atol=atol, rtol=rtol)
        np.testing.assert_allclose(b["velocity"], a["velocity"], atol=vatol or atol, rtol=rtol)

    def test_dam_break_tracks_the_cpu_reference(self):
        source = BoxSource((0, 0, 0), (5, 8, 8))
        cpu = flip3d.Liquid3D(DAM, sources=[source])
        gpu = resident(DAM, sources=[source])
        a, b = cpu.initial_state(0), gpu.initial_state(0)
        for frame in range(1, 4):
            a, b = cpu.step(a, frame, 0, 0), gpu.step(b, frame, 0, 0)
        self.compare(a, b)
        self.assertEqual(b.meta["next_id"], a.meta["next_id"])
        self.assertEqual(b.meta["substep_count"], a.meta["substep_count"])

    def test_volume_is_conserved_and_the_per_cell_bounds_hold(self):
        params = {**DAM, "gravity": 0.3, "substeps": 2, "tolerance": 1.0e-3}
        gpu = resident(params, sources=[BoxSource((0, 0, 0), (5, 10, 8))])
        state = gpu.initial_state(0)
        for frame in range(1, 7):
            for sub in range(2):
                state = gpu.step(state, frame, sub, 0)
        pos = state.arrays["position"]
        self.assertAlmostEqual(len(pos) / 8.0, 5 * 10 * 8, delta=0.03 * 5 * 10 * 8)
        cell = np.floor(pos).astype(int)
        counts = np.bincount((cell[:, 0] * 16 + cell[:, 1]) * 8 + cell[:, 2])
        self.assertLessEqual(int(counts.max()), gpu.max_per_cell)
        self.assertTrue(np.all(pos >= 0.0) and np.all(pos[:, 0] <= 16) and np.all(pos[:, 1] <= 16))

    def test_pouring_source_matches_the_reference(self):
        params = {**DAM, "nx": 12, "ny": 12, "nz": 8, "gravity": 0.05, "flip_ratio": 0.9}
        source = BoxSource((4, 9, 3), (7, 11, 5), velocity=(0.0, -0.4, 0.0))
        cpu = flip3d.Liquid3D(params, sources=[source])
        gpu = resident(params, sources=[source])
        a, b = cpu.initial_state(0), gpu.initial_state(0)
        for frame in range(1, 7):
            a, b = cpu.step(a, frame, 0, 0), gpu.step(b, frame, 0, 0)
        self.assertGreater(len(b.arrays["id"]), 8 * 3 * 2 * 2)               # it kept pouring
        # the pour tops up from device cell counts; the count of particles agrees with the reference
        self.assertAlmostEqual(len(b.arrays["id"]), len(a.arrays["id"]), delta=0.02 * len(a.arrays["id"]))
        self.assertAlmostEqual(float(b.arrays["position"][:, 1].mean()), float(a.arrays["position"][:, 1].mean()), delta=0.05)

    def test_a_collider_blocks_particles_like_the_reference(self):
        block = ((6, 0, 0), (10, 5, 8))
        cpu = flip3d.Liquid3D(DAM, sources=[BoxSource((0, 0, 0), (5, 10, 8))], colliders=[BoxCollider(*block)])
        gpu = resident(DAM, sources=[BoxSource((0, 0, 0), (5, 10, 8))], colliders=[BoxCollider(*block)])
        a, b = cpu.initial_state(0), gpu.initial_state(0)
        for frame in range(1, 7):
            a, b = cpu.step(a, frame, 0, 0), gpu.step(b, frame, 0, 0)
        cells = np.floor(b.arrays["position"]).astype(int)
        inside = np.all((cells >= block[0]) & (cells < block[1]), axis=1)
        self.assertEqual(int(inside.sum()), 0)
        self.assertAlmostEqual(float(b.arrays["position"][:, 0].mean()), float(a.arrays["position"][:, 0].mean()), delta=0.05)

    def test_open_bottom_records_drained_liquid_mass(self):
        params = {"nx": 8, "ny": 8, "nz": 8, "gravity": 0.0, "boundary_y_min": "open"}
        arrays = flip3d.empty_arrays()
        arrays.update(position=np.tile((3.5, 0.2, 3.5), (8, 1)).astype(np.float32),
                      velocity=np.tile((0.0, -20.0, 0.0), (8, 1)).astype(np.float32),
                      id=np.arange(8, dtype=np.int64), age=np.zeros(8, np.int32), temperature=np.ones(8, np.float32))
        result = resident(params).step(State(arrays, {"next_id": 8, "substep_count": 0}), 1, 0, 0)
        self.assertEqual(len(result.arrays["position"]), 0)
        self.assertEqual(result.meta["escaped_mass"], 1.0)

    def test_wind_and_drag_forces_match_the_reference(self):
        from nodebased.fluid3d import Force
        wind = Force("wind", {"dir_x": 1.0, "dir_y": 0.0, "dir_z": 0.0, "strength": 1.0})
        drag = Force("drag", {"drag": 0.2})
        params = {**DAM, "gravity": 0.02, "boundary_x_max": "open"}
        cpu = flip3d.Liquid3D(params, sources=[BoxSource((0, 0, 0), (5, 8, 8))], forces=[wind, drag])
        gpu = resident(params, sources=[BoxSource((0, 0, 0), (5, 8, 8))], forces=[wind, drag])
        a, b = cpu.initial_state(0), gpu.initial_state(0)
        for frame in range(1, 4):
            a, b = cpu.step(a, frame, 0, 0), gpu.step(b, frame, 0, 0)
        self.compare(a, b, atol=8e-3, rtol=3e-3, vatol=3e-2)

    def test_unsupported_features_are_named(self):
        from nodebased.flip_gpu_resident import create_solver, unsupported_reason
        self.assertIsNone(unsupported_reason({}))
        for params, word in (({"viscosity": 0.1}, "viscosity"), ({"narrow_band": 2.0}, "narrow band")):
            self.assertIn(word, unsupported_reason(params))
            fallback = create_solver({"nx": 8, "ny": 8, "nz": 8, **params})
            self.assertIn(word, fallback.fallback_reason)
            self.assertIsInstance(fallback, flip3d.Liquid3D)


def drop_state(cells, per_cell=8, seed=0):
    rng = np.random.default_rng(seed)
    points = np.concatenate([np.asarray(cell)[None, :] + rng.random((per_cell, 3)) for cell in cells]).astype(np.float32)
    arrays = flip3d.empty_arrays()
    arrays.update(position=points, velocity=np.zeros_like(points), id=np.arange(len(points), dtype=np.int64),
                  age=np.zeros(len(points), np.int32), temperature=np.ones(len(points), np.float32))
    return State(arrays, {"next_id": len(points), "substep_count": 0})


def count_pieces(positions):
    occupied = set(map(tuple, np.floor(positions).astype(np.int64)))
    pieces = 0
    while occupied:
        pieces += 1
        stack = [occupied.pop()]
        while stack:
            x, y, z = stack.pop()
            for cell in ((x + 1, y, z), (x - 1, y, z), (x, y + 1, z), (x, y - 1, z), (x, y, z + 1), (x, y, z - 1)):
                if cell in occupied:
                    occupied.remove(cell)
                    stack.append(cell)
    return pieces


@unittest.skipUnless(fgs.available(), "no wgpu compute adapter")
class SurfaceTensionTests(unittest.TestCase):
    """Lane 6 step N3: surface tension on the card. The Hot pour liquid needs it and the resident solver used to
    refuse it, so a scene with surface tension fell back to the slow transfer path."""

    def test_the_resident_solver_accepts_surface_tension(self):
        solver = resident({"nx": 8, "ny": 8, "nz": 8, "surface_tension": 0.3})
        self.assertEqual(solver.backend, "resident")

    def test_zero_tension_is_bit_identical_to_a_run_without_the_knob(self):
        cells = [(x, y, z) for x in range(7, 17) for y in range(9, 14) for z in range(3, 5)]
        base = {"nx": 24, "ny": 24, "nz": 8, "gravity": 0.0, "max_iterations": 60}
        a, b = resident(base), resident({**base, "surface_tension": 0.0})
        sa, sb = drop_state(cells), drop_state(cells)
        for frame in range(1, 4):
            sa, sb = a.step(sa, frame, 0, 0), b.step(sb, frame, 0, 0)
        assert_states_equal(self, sa, sb)

    def test_a_2d_drop_rounds_up_and_matches_the_reference(self):
        cells = [(x, y, z) for x in range(7, 17) for y in range(9, 14) for z in range(3, 5)
                 if ((x - 12) / 5) ** 2 + ((y - 11) / 2.5) ** 2 <= 1.0]
        params = {"nx": 24, "ny": 24, "nz": 8, "gravity": 0.0, "surface_tension": 0.2, "max_iterations": 150}
        cpu, gpu = flip3d.Liquid3D(params), resident(params)
        a, b = drop_state(cells), drop_state(cells)
        start = float(np.sqrt(np.var(a.arrays["position"][:, 0]) / np.var(a.arrays["position"][:, 1])))
        for frame in range(1, 7):
            a, b = cpu.step(a, frame, 0, 0), gpu.step(b, frame, 0, 0)

        def aspect(state):
            pos = state.arrays["position"]
            return float(np.sqrt(np.var(pos[:, 0]) / max(np.var(pos[:, 1]), 1e-12)))
        self.assertLess(abs(aspect(b) - 1.0), abs(start - 1.0))
        self.assertAlmostEqual(aspect(b), aspect(a), delta=0.03)
        np.testing.assert_allclose(sorted_arrays(b)["position"].mean(axis=0), sorted_arrays(a)["position"].mean(axis=0),
                                   atol=0.03)

    def test_a_thin_stream_breaks_into_drops_on_the_card(self):
        cells = [(x, 7, z) for x in range(3, 17) for z in (3, 4)]
        pieces = {}
        for tension in (0.0, 20.0):
            solver = resident({"nx": 20, "ny": 16, "nz": 8, "gravity": 0.0, "surface_tension": tension,
                               "max_iterations": 100})
            state = drop_state(cells, per_cell=4, seed=3)
            for frame in range(1, 9):
                state = solver.step(state, frame, 0, 0)
            pieces[tension] = count_pieces(state.arrays["position"])
        self.assertEqual(pieces[0.0], 1)
        self.assertGreater(pieces[20.0], 1)

    def test_the_force_field_matches_the_reference_cell_for_cell(self):
        cells = [(x, y, z) for x in range(5, 12) for y in range(4, 10) for z in range(3, 8) if (x + y + z) % 5]
        params = {"nx": 16, "ny": 16, "nz": 12, "gravity": 0.0}
        cpu, gpu = flip3d.Liquid3D(params), resident(params)
        state = drop_state(cells, per_cell=2, seed=9)
        gpu.step(gpu.initial_state(0), 1, 0, 0)                        # allocate the grid
        g = gpu._grid
        liquid = cpu._classify(state.arrays["position"].astype(np.float64), None)
        fields = {k: np.zeros(s) for k, s in zip("uvw", ((17, 16, 12), (16, 17, 12), (16, 16, 13)))}
        cpu._apply_surface_tension(fields, liquid, 0.7)
        gpu.ctx.write(g.dens, liquid.reshape(-1).astype(np.float32))
        for buf in g.f:
            gpu.ctx.clear(buf)
        gpu._surface_tension(g, 0.7)
        for axis, name in enumerate("uvw"):
            got = gpu.ctx.read(g.f[axis], 4 * g.face_n[axis]).view(np.float32).reshape(g.face_shapes[axis])
            np.testing.assert_allclose(got, fields[name], atol=2e-5, err_msg=name)


@unittest.skipUnless(fgs.available(), "no wgpu compute adapter")
class AdaptiveDomainTests(unittest.TestCase):
    """Lane 6 step N3: an adaptive domain on the card (the resident solver refused `auto_resize`)."""

    PARAMS = {"nx": 16, "ny": 16, "nz": 16, "origin_x": -1.0, "origin_y": 0.0, "origin_z": -1.0, "voxel_size": 0.125,
              "auto_resize": 1, "padding": 8, "max_size": 64, "particles_per_cell": 2, "substeps": 2, "gravity": 0.02,
              "start_frame": 1, "max_iterations": 100}

    def source(self):
        return fluid3d_source()

    def run_frames(self, solver, frames, state=None, first=1):
        state = state or solver.initial_state()
        for frame in range(first, first + frames):
            state = solver.step(state, frame=frame)
        return state

    def test_the_resident_solver_accepts_auto_resize(self):
        self.assertEqual(resident(self.PARAMS).backend, "resident")

    def test_the_box_follows_the_liquid_in_whole_tiles_and_every_particle_stays_inside(self):
        solver = resident(self.PARAMS, sources=[fluid3d_source()])
        state = self.run_frames(solver, 24)
        shape, origin = state.meta["domain_shape"], np.asarray(state.meta["domain_origin"])
        self.assertTrue(all(n % 8 == 0 for n in shape), shape)
        pos = state.arrays["position"]
        top = origin + np.asarray(shape) * 0.125
        self.assertTrue(np.all(pos >= origin - 1e-6) and np.all(pos <= top + 1e-6))
        self.assertGreater(float(pos[:, 1].min()), -1.0e-4)
        self.assertGreaterEqual(origin[1], solver.floor_y - 1e-8)             # the floor stays anchored

    def test_the_adaptive_box_matches_the_reference_box_and_the_particles_agree(self):
        cpu = flip3d.Liquid3D(self.PARAMS, sources=[fluid3d_source()])
        gpu = resident(self.PARAMS, sources=[fluid3d_source()])
        a, b = cpu.initial_state(), gpu.initial_state()
        for frame in range(1, 10):
            for sub in range(2):
                a, b = cpu.step(a, frame, sub, 0), gpu.step(b, frame, sub, 0)
        self.assertAlmostEqual(len(b.arrays["id"]), len(a.arrays["id"]), delta=0.03 * len(a.arrays["id"]))
        self.assertAlmostEqual(float(b.arrays["position"][:, 1].mean()), float(a.arrays["position"][:, 1].mean()), delta=0.03)
        self.assertLessEqual(abs(int(np.prod(b.meta["domain_shape"])) - int(np.prod(a.meta["domain_shape"]))),
                             int(np.prod(a.meta["domain_shape"])) // 2)

    def test_the_box_shrinks_back_around_a_small_blob(self):
        solver = resident({**self.PARAMS, "nx": 48, "ny": 48, "nz": 48, "max_size": 96}, sources=[fluid3d_source()])
        state = self.run_frames(solver, 4)
        self.assertLess(int(np.prod(state.meta["domain_shape"])), 48 ** 3)

    def test_a_checkpoint_restart_across_resizes_matches_the_uninterrupted_run(self):
        whole = resident(self.PARAMS, sources=[fluid3d_source()])
        full = self.run_frames(whole, 12)
        first = resident(self.PARAMS, sources=[fluid3d_source()])
        mid = self.run_frames(first, 6)
        checkpoint = State({k: v.copy() for k, v in mid.arrays.items()}, dict(mid.meta))
        second = resident(self.PARAMS, sources=[fluid3d_source()])
        end = self.run_frames(second, 6, state=checkpoint, first=7)
        assert_states_equal(self, full, end)


@unittest.skipUnless(fgs.available(), "no wgpu compute adapter")
class SolidMaskUploadTests(unittest.TestCase):
    """The collider mask was uploaded only when `id(mask)` changed. A new mask for a moved box could take the id of the
    freed old one, so the card kept the old box's solids: bakes of an adaptive scene with a collider differed run to run."""

    def test_a_new_mask_that_takes_the_freed_masks_id_is_still_uploaded(self):
        # Python reuses the id of a freed array for the next one; to make that certain the modules' `id` says every object
        # is the same object, which is what an id collision looks like to the upload check
        from nodebased import flip_gpu_resident as resident_module
        for module, solver in ((resident_module, resident({"nx": 8, "ny": 8, "nz": 8})),
                               (fgs, fgs.GpuSmoke3D({"nx": 8, "ny": 8, "nz": 8, "default_source": 0}))):
            module.id = lambda obj: 1
            try:
                g = solver._alloc()
                first = np.zeros((8, 8, 8), bool)
                first[:2] = True
                solver._put_solid(g, first, None)
                second = np.zeros((8, 8, 8), bool)
                second[4:] = True
                solver._put_solid(g, second, None)
                on_card = solver.ctx.read(g.solid, 4 * g.n).view(np.uint32).reshape(8, 8, 8)
                np.testing.assert_array_equal(on_card.astype(bool), second, err_msg=type(solver).__name__)
            finally:
                del module.id


@unittest.skipUnless(fgs.available(), "no wgpu compute adapter")
class CardMemoryTests(unittest.TestCase):
    """Lane 6 step N3: the card's memory stays bounded over a long bake (24 frames of a 128-cell liquid took 4.5 GB: every
    resize left the old grid on the card, and every frame's solver was kept alive by its cached state)."""

    def tracker(self):
        from nodebased import gpu3d
        return gpu3d._state()["memory"]

    def test_a_resize_gives_the_old_grid_back(self):
        solver = resident({"nx": 16, "ny": 16, "nz": 16, "voxel_size": 0.125, "auto_resize": 1})
        solver.step(solver.initial_state(), 1, 0, 0)
        tracker = self.tracker()
        solver._alloc()
        after_first = tracker.current
        for shape in ((24, 24, 24), (16, 16, 16), (24, 24, 24), (16, 16, 16), (32, 24, 24)) * 3:
            solver._adopt(shape, (0.0, 0.0, 0.0))
            solver._alloc()
        solver.ctx.read(solver._grid.dummy, 4)                  # the next readback frees what was dropped
        self.assertLess(tracker.current, after_first + 6 * 2 ** 20)

    def test_a_read_state_does_not_keep_the_solver_alive(self):
        solver = resident({"nx": 8, "ny": 8, "nz": 8})
        state = solver.step(solver.initial_state(), 1, 0, 0)
        self.assertIsNotNone(state._solver)
        state.arrays
        self.assertIsNone(state._solver)

    def test_successive_frames_of_one_run_share_a_solver_and_the_card_stays_bounded(self):
        import gc
        from nodebased import fluid3d
        from nodebased.core import Dispatcher
        from nodebased.imaging import Evaluator
        d = Dispatcher()
        d.execute({"op": "batch", "commands": [
            {"op": "create", "id": "src", "type": "FluidSource3D", "params": {
                "fluid_type": "liquid", "src_center_y": 0.7, "src_radius": 0.3, "src_vel_y": -0.5}},
            {"op": "create", "id": "liq", "type": "FluidLiquidSolver3D", "params": {
                "division_size": 0.1, "bounds_min_x": -0.5, "bounds_max_x": 0.5, "bounds_min_z": -0.5, "bounds_max_z": 0.5,
                "bounds_max_y": 1.2, "auto_resize": 1, "pressure": "resident", "particles_per_cell": 2, "substeps": 1}},
            {"op": "connect", "id": "liq", "input": "fluid", "source": "src"},
            {"op": "create", "id": "cache", "type": "ParticleCache3D", "params": {}},
            {"op": "connect", "id": "cache", "input": "particles", "source": "liq"}]})
        evaluator = Evaluator()
        gc.collect()
        before = sum(1 for o in gc.get_objects() if type(o).__name__ == "GpuLiquid3D")
        solvers = set()
        for frame in range(1, 9):
            value = evaluator.evaluate_raster(d.document, "cache", frame=frame, typed=True)
            solvers.add(id(value.stream.solver()))
        self.assertEqual(len(solvers), 1, "every frame built its own solver, and its own grid on the card")
        gc.collect()
        live = sum(1 for o in gc.get_objects() if type(o).__name__ == "GpuLiquid3D")
        self.assertLessEqual(live - before, 1, "the frames' cached states kept their solvers alive")


def fluid3d_source():
    from nodebased import fluid3d
    return fluid3d.Source(center=(0.0, 1.0, 0.0), radius=0.45, fluid_type="liquid", end_frame=1)


@unittest.skipUnless(fgs.available(), "no wgpu compute adapter")
class ResumeAndCancelTests(unittest.TestCase):
    PARAMS = {**DAM, "gravity": 0.06, "tolerance": 1.0e-3}

    def make(self, cancel=None):
        return resident(self.PARAMS, sources=[BoxSource((0, 0, 0), (5, 8, 8)),
                                              BoxSource((9, 9, 2), (12, 11, 5), velocity=(0.0, -0.3, 0.0))],
                        cancel=cancel)

    def test_a_resident_run_checkpoints_and_restarts_bit_identically(self):
        straight = self.make()
        state = straight.initial_state(0)
        checkpoints = {}
        for frame in range(1, 9):
            state = straight.step(state, frame, 0, 0)
            if frame in (3, 5):
                checkpoints[frame] = straight.checkpoint(state)
        for frame in (3, 5):
            restarted = self.make()                                   # a fresh solver: nothing carried but the state
            resumed = restarted.restore(checkpoints[frame])
            for later in range(frame + 1, 9):
                resumed = restarted.step(resumed, later, 0, 0)
            assert_states_equal(self, state, resumed)

    def test_the_run_is_deterministic_across_solvers(self):
        a, b = self.make(), self.make()
        sa, sb = a.initial_state(0), b.initial_state(0)
        for frame in range(1, 6):
            sa, sb = a.step(sa, frame, 0, 0), b.step(sb, frame, 0, 0)
        assert_states_equal(self, sa, sb)

    def test_cancel_inside_a_substep_leaves_the_committed_state_untouched(self):
        for warm in (0, 3):
            solver = self.make()
            state = solver.initial_state(0)
            for frame in range(1, warm + 1):
                state = solver.step(state, frame, 0, 0)
            before = {k: v.copy() for k, v in state.arrays.items()} if warm else None
            token = solver._token
            reference = self.make()
            ref_state = reference.initial_state(0)
            for frame in range(1, warm + 2):
                ref_state = reference.step(ref_state, frame, 0, 0)
            # looks: 1 at the top of the step, 2 after binning, then inside the pressure loop on the first solve
            for after in (1, 2, 3):
                solver.cancel = CancelAfter(after)
                with self.assertRaises(Cancelled):
                    solver.step(state, warm + 1, 0, 0)
                self.assertEqual(solver._token, token, "a cancelled substep must not commit")
            solver.cancel = None
            if warm:
                for name, value in before.items():
                    np.testing.assert_array_equal(state.arrays[name], value, err_msg=name)
            done = solver.step(state, warm + 1, 0, 0)
            assert_states_equal(self, done, ref_state)

    def test_a_cancelled_frame_is_not_cached(self):
        solver = self.make()
        solver.substeps = 3
        solver.dt = 1.0 / 3
        cache = simcache.SimCache(enabled=False)
        solver.cancel = CancelAfter(40)
        with self.assertRaises(Cancelled):
            simcache.solve_to_frame(cache, "run", 4, 1, 3, 0, solver.initial_state, solver.step, solver.cancel)
        frames = [f for f in range(0, 6) if cache.get("run", f) is not None]
        self.assertLess(len(frames), 4)
        self.assertNotIn(4, frames)
        solver.cancel = None
        again = simcache.solve_to_frame(cache, "run", 4, 1, 3, 0, solver.initial_state, solver.step, None)
        fresh = self.make()
        fresh.substeps = 3
        fresh.dt = 1.0 / 3
        clean = simcache.solve_to_frame(simcache.SimCache(enabled=False), "run2", 4, 1, 3, 0, fresh.initial_state,
                                        fresh.step, None)
        assert_states_equal(self, again, clean)


@unittest.skipUnless(fgs.available(), "no wgpu compute adapter")
class GraphTests(unittest.TestCase):
    def test_a_liquid_node_on_the_resident_backend_matches_the_reference_and_uses_the_gpu_surface(self):
        from nodebased.imaging import Evaluator
        from tests.test_liquid_nodes import at, liquid
        got = {}
        for backend in ("cpu", "resident"):
            instance = at(Evaluator(), liquid(pressure=backend), "sol", 4)
            got[backend] = instance
        cpu, gpu = got["cpu"], got["resident"]
        self.assertEqual(type(gpu.stream._solver).__name__, "GpuLiquid3D")
        self.assertEqual(gpu.stream.backend, "resident")
        self.assertEqual(len(cpu.positions), len(gpu.positions))
        np.testing.assert_allclose(gpu.positions.mean(axis=0), cpu.positions.mean(axis=0), atol=1e-3)
        np.testing.assert_allclose(gpu.surface.density, cpu.surface.density, atol=5e-2)

    def test_the_run_identity_names_the_backend_so_cpu_and_resident_caches_stay_apart(self):
        from tests.test_liquid_nodes import liquid
        runs = set()
        for backend in ("cpu", "resident"):
            d = liquid(pressure=backend)
            runs.add(flip3d.build_stream(d.document, "sol", d.document["nodes"]["sol"], None).run)
        self.assertEqual(len(runs), 2)


@unittest.skipUnless(fgs.available(), "no wgpu compute adapter")
class LevelSetTests(unittest.TestCase):
    def test_the_gpu_field_matches_the_cpu_field(self):
        from nodebased import flip_gpu_levelset
        rng = np.random.default_rng(3)
        shape, origin, voxel = (24, 20, 16), (0.5, -1.0, 2.0), 0.25
        pos = rng.uniform(origin, np.array(origin) + np.array(shape) * voxel, (3000, 3)).astype(np.float32)
        spacing = voxel / 2
        for smoothing in (0, 2):
            cpu = liquid_surface.level_set(pos, origin, voxel, shape, spacing, 3 * spacing, smoothing)
            gpu = flip_gpu_levelset.level_set(pos, origin, voxel, shape, spacing, 3 * spacing, smoothing)
            np.testing.assert_allclose(gpu, cpu, atol=2e-5)
            np.testing.assert_array_equal(gpu < 0, cpu < 0)

    def test_the_resident_state_makes_the_field_without_leaving_the_card(self):
        from nodebased import flip_gpu_levelset
        solver = resident({**DAM, "gravity": 0.05}, sources=[BoxSource((0, 0, 0), (5, 8, 8))])
        state = solver.initial_state(0)
        for frame in range(1, 4):
            state = solver.step(state, frame, 0, 0)
        spacing = 0.5
        phi_buffer, shape = solver.level_set_device(state, spacing, 3 * spacing)
        gpu = flip_gpu_levelset.read_field(phi_buffer, shape)
        cpu = liquid_surface.level_set(state.arrays["position"], (0.0, 0.0, 0.0), 1.0, (16, 16, 8), spacing, 3 * spacing)
        np.testing.assert_allclose(gpu, cpu, atol=5e-5)

    def test_an_empty_state_gives_a_far_air_field(self):
        from nodebased import flip_gpu_levelset
        phi = flip_gpu_levelset.level_set(np.zeros((0, 3)), (0, 0, 0), 1.0, (4, 4, 4), 0.5, 1.5)
        self.assertTrue(np.all(phi == 3.0))


if __name__ == "__main__":
    unittest.main()
