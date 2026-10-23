"""Memory samples emitted by the production Hot pour refinement benchmark."""
import unittest
from unittest import mock

from tools import benchmark_fluid_edit_stages as benchmark


class FluidEditMemoryTests(unittest.TestCase):
    def test_stage_sampler_reports_process_and_card_peaks(self):
        with mock.patch.object(benchmark, "process_rss_mb", return_value=123.0), \
             mock.patch.object(benchmark, "card_memory_mb", return_value=456.0):
            sample = benchmark.StageMemoryPeak("default")
            sample.done = mock.Mock()
            sample.done.is_set.side_effect = [False, True]
            sample.done.wait.return_value = None
            sample._sample()
        row = sample.result()
        self.assertEqual(row["peak_process_rss_mb"], 123.0)
        self.assertEqual(row["peak_card_memory_mb"], 456.0)
        self.assertEqual(row["card_memory_scope"], "process allocation")

    def test_cpu_adapter_marks_card_memory_unavailable(self):
        with mock.patch.object(benchmark, "process_rss_mb", return_value=91.0), \
             mock.patch.object(benchmark, "card_memory_mb", return_value=None):
            sample = benchmark.StageMemoryPeak("cpu")
            sample.done = mock.Mock()
            sample.done.is_set.side_effect = [False, True]
            sample.done.wait.return_value = None
            sample._sample()
        self.assertEqual(sample.result()["peak_card_memory_mb"], None)
        self.assertEqual(sample.result()["card_memory_scope"], "unavailable")


if __name__ == "__main__":
    unittest.main()
