"""Memory samples emitted by the production Hot pour refinement benchmark."""
import unittest
import tempfile
import io
from contextlib import redirect_stdout
from unittest import mock
import numpy as np

from tools import benchmark_fluid_edit_stages as benchmark


class FluidEditMemoryTests(unittest.TestCase):
    def test_stage_sampler_reports_process_and_card_peaks(self):
        with mock.patch.object(benchmark, "process_rss_mb", return_value=123.0), \
             mock.patch.object(benchmark, "process_swap_mb", return_value=2.0), \
             mock.patch.object(benchmark, "card_memory_mb", return_value=456.0):
            sample = benchmark.StageMemoryPeak("default")
            sample.done = mock.Mock()
            sample.done.is_set.side_effect = [False, True]
            sample.done.wait.return_value = None
            sample._sample()
        row = sample.result()
        self.assertEqual(row["peak_process_rss_mb"], 123.0)
        self.assertEqual(row["peak_process_swap_mb"], 2.0)
        self.assertEqual(row["peak_card_memory_mb"], 456.0)
        self.assertEqual(row["card_memory_scope"], "process allocation")

    def test_cpu_adapter_marks_card_memory_unavailable(self):
        with mock.patch.object(benchmark, "process_rss_mb", return_value=91.0), \
             mock.patch.object(benchmark, "process_swap_mb", return_value=0.0), \
             mock.patch.object(benchmark, "card_memory_mb", return_value=None):
            sample = benchmark.StageMemoryPeak("cpu")
            sample.done = mock.Mock()
            sample.done.is_set.side_effect = [False, True]
            sample.done.wait.return_value = None
            sample._sample()
        self.assertEqual(sample.result()["peak_card_memory_mb"], None)
        self.assertEqual(sample.result()["card_memory_scope"], "unavailable")

    def test_new_probe_clears_a_prior_failure_marker(self):
        previous = {"memory_probe_failure": {"frame": 90}}
        benchmark.begin_memory_probe(previous, 16384)
        self.assertNotIn("memory_probe_failure", previous)
        self.assertEqual(previous["max_process_rss_mb"], 16384)

    def test_whitewater_over_budget_selects_bounded_deterministic_sample_and_status(self):
        from nodebased.fluid_gpu_whitewater import (_budgeted_sample_indices, neighbor_lists,
                                                    NeighborListBudgetExceeded)

        sample, message = _budgeted_sample_indices(1000, 128 * 1024)
        self.assertEqual(len(sample), 128)
        self.assertEqual(len(set(sample.tolist())), 128)
        self.assertIn("reducing neighbor evaluation", message)
        self.assertIsNone(_budgeted_sample_indices(10, 128 * 1024)[0])
        points = np.zeros((8, 3), dtype=np.float64)
        with self.assertRaises(NeighborListBudgetExceeded):
            neighbor_lists(points, 1.0, max_member_bytes=32)

    def test_gpu_allocation_failure_retries_at_half_budget_with_status(self):
        from nodebased import fluid_gpu_whitewater as gpu

        class Buffer:
            def __init__(self, size):
                self.size = size
            def destroy(self):
                pass

        class Pass:
            def set_pipeline(self, _value): pass
            def set_bind_group(self, *_args): pass
            def dispatch_workgroups(self, _value): pass
            def end(self): pass

        class Encoder:
            def begin_compute_pass(self): return Pass()
            def finish(self): return object()

        class Queue:
            def submit(self, _commands): pass
            def read_buffer(self, buffer): return bytes(buffer.size)

        class Device:
            limits = {"max-storage-buffer-binding-size": 1024}
            def __init__(self):
                self.failed_once = False
                self.queue = Queue()
            def create_buffer_with_data(self, data, usage):
                if not self.failed_once:
                    self.failed_once = True
                    raise RuntimeError("Not enough memory left.")
                return Buffer(len(data))
            def create_buffer(self, size, usage): return Buffer(size)
            def create_bind_group(self, **_kwargs): return object()
            def create_command_encoder(self): return Encoder()

        class Pipeline:
            def get_bind_group_layout(self, _index): return object()

        class Usage:
            STORAGE, UNIFORM, COPY_SRC = 1, 2, 4

        solver = object.__new__(gpu.GpuWhitewater3D)
        solver.adapter_name = "budget-test-adapter"
        solver.device = Device()
        solver.pipeline = Pipeline()
        solver.wgpu = mock.Mock(BufferUsage=Usage)
        gpu._budget_notices.discard(solver.adapter_name)
        positions = np.column_stack((np.arange(100, dtype=float) * 2, np.zeros((100, 2))))
        output = io.StringIO()
        with mock.patch.object(gpu, "_free_card_memory_bytes", return_value=8 << 30), \
             redirect_stdout(output):
            result = solver.potentials(positions, np.zeros_like(positions), np.ones_like(positions), 0.1)
        self.assertEqual(tuple(len(row) for row in result), (100, 100, 100))
        self.assertIn("reducing neighbor evaluation from 100 to 50", output.getvalue())

    def test_cache_write_spills_frame_and_releases_memory(self):
        from nodebased.simcache import SimCache, State

        with tempfile.TemporaryDirectory() as folder:
            cache = SimCache(root=folder, memory_budget=1, disk_budget=1024 * 1024)
            state = State({"position": np.arange(12, dtype="f4").reshape(4, 3)}, {})
            cache.put("stream-test", 7, state)
            self.assertEqual(cache._memory, {})
            self.assertTrue(cache._path(("stream-test", 7)).exists())
            np.testing.assert_array_equal(cache.get("stream-test", 7).arrays["position"], state.arrays["position"])


if __name__ == "__main__":
    unittest.main()
