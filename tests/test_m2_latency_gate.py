"""M2 gate (docs/VISION.md, docs/BENCHMARKS-v0.33-m2.md): time to first pixel and edit p95 at
1080p, gated at the measurement plus 30 percent headroom -- see the doc's "Gates" section for
the exact numbers and why only these two of the doc's measurements are gated.

Slow (a real 1080p `Read` -> `Viewer` evaluation, and ten full 1080p `Grade` edits, each a
cache-miss recompute) and skipped on a software/no-GPU adapter: these are wall-clock budgets
calibrated on the workstation named in docs/BENCHMARKS-v0.33-m2.md, and a software-rendered or
GPU-less runner (the same class of machine the project's other slow/GPU tests already skip on)
is not that machine -- running this gate there would be measuring the runner, not the code.
"""
import unittest

from tools.benchmark_4k_viewport import measure_edit_kind, measure_time_to_first_pixel

TTFP_1080P_BUDGET_MS = 43.8   # measured 33.7 ms, docs/BENCHMARKS-v0.33-m2.md's "Gates" table
GRADE_EDIT_P95_1080P_BUDGET_MS = 54.2   # measured 41.7 ms, same table


def _on_software_or_missing_adapter():
    """True on a CPU/software wgpu adapter, or when no adapter can be probed at all (the common
    CI case -- no GPU present is the same "not this workstation" signal a software adapter is)."""
    try:
        from nodebased import gpu3d
        info = gpu3d._state()["info"]
        return str(info.get("adapter_type", "")).lower() == "cpu"
    except Exception:
        return True


@unittest.skipIf(_on_software_or_missing_adapter(),
                 "software/no-GPU adapter: not the workstation these wall-clock budgets are calibrated on")
class M2LatencyGateTests(unittest.TestCase):
    def test_read_to_viewer_time_to_first_pixel_1080p_under_budget(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            ms = measure_time_to_first_pixel(1920, 1080, tmpdir)
        self.assertLess(ms, TTFP_1080P_BUDGET_MS,
                        f"Read -> Viewer time to first pixel at 1080p took {ms:.1f} ms, over the "
                        f"{TTFP_1080P_BUDGET_MS} ms budget (docs/BENCHMARKS-v0.33-m2.md)")

    def test_grade_edit_p95_1080p_under_budget(self):
        result = measure_edit_kind("Grade", 1920, 1080, 10)
        p95 = result["edit_p95_ms"]
        self.assertLess(p95, GRADE_EDIT_P95_1080P_BUDGET_MS,
                        f"Grade edit p95 at 1080p took {p95:.1f} ms, over the "
                        f"{GRADE_EDIT_P95_1080P_BUDGET_MS} ms budget (docs/BENCHMARKS-v0.33-m2.md)")


if __name__ == "__main__":
    unittest.main()
