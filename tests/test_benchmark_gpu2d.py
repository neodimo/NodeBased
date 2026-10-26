"""tools/benchmark_gpu2d.py (R6-1 part 4): the frame spec, the statistics, the island-to-frame join, the gate table, and a
tiny end-to-end `kernels` and `graph` run (96 by 54 pixels, whatever adapter is the default; skipped when there is none)."""
import argparse
import contextlib
import io
import json
import os
import pathlib
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from nodebased import gpu2d
from tools import benchmark_2d_playback as bench
from tools import benchmark_gpu2d as tool


def namespace(**kw):
    return argparse.Namespace(**kw)


class PureFunctionTests(unittest.TestCase):
    def test_frame_specs(self):
        self.assertEqual(tool.parse_frames("1-4"), [1, 2, 3, 4])
        self.assertEqual(tool.parse_frames("49-240:60"), [49, 109, 169, 229])
        self.assertEqual(tool.parse_frames("1,5,9"), [1, 5, 9])
        self.assertEqual(tool.parse_size("480x270"), (480, 270))
        self.assertEqual(tool.parse_size("1080p"), (1920, 1080))

    def test_statistics(self):
        row = tool.stats(list(range(1, 101)))
        self.assertEqual((row["n"], row["min"], row["max"], row["p50"]), (100, 1.0, 100.0, 50.5))
        self.assertAlmostEqual(row["p95"], 95.05, places=2)
        self.assertEqual(tool.stats([]), {"n": 0})

    def test_islands_join_the_frame_whose_evaluation_contains_them(self):
        frame = lambda b, e: {"points_ms": {"eval_begin": b, "eval_end": e, "shown": e + 10}, "stages_ms": {"node evaluation": e - b}, "interval_ms": 50.0}
        island = lambda t, wall: {"t_ms": t, "wall_ms": wall, "upload_ms": 1.0, "encode_ms": 2.0, "gpu_ms": 3.0, "readback_ms": 4.0}
        timeline = {"trace_offset_ms": 1000.0,
                    "fresh_frame_timeline": {"frames": [frame(0, 100), frame(100, 200), frame(300, 400)]},
                    "gpu2d": {"islands": [island(1050, 30.0), island(1150, 40.0)]}}
        rows = tool.join_islands(timeline)                    # the third frame has no island and is left out
        self.assertEqual([(r["node_evaluation_ms"], r["island_wall_ms"], r["outside_island_ms"]) for r in rows], [(100, 30.0, 70.0), (100, 40.0, 60.0)])
        self.assertEqual(tool.join_islands({"fresh_frame_timeline": {"frames": []}, "gpu2d": None}), [])

    def test_the_gate_table_marks_each_gate_from_the_numbers(self):
        def kernel(exec_p95, total_p95, readbacks=0, violations=0, within_budget=True):
            mode = lambda p95: {"total_ms": {"p95": p95}, "exec_ms": {"p95": p95}}
            fresh = {"total_ms": {"p95": total_p95}, "exec_ms": {"p95": exec_p95 + 1.0}, "intermediate_readbacks": readbacks, "cpu_boundaries": []}
            pixels = {"violations": violations, "windows_match": True, "nonfinite_match": True, "max_gate_ratio": 0.01}
            return {"budget_bytes": 1 << 30, "memory": {"peak_bytes": 1 << 28, "within_budget": within_budget},
                    "pixels": {"resident": pixels, "roundtrip": pixels},
                    "summary": {"resident_warm": mode(exec_p95), "resident_fresh": fresh, "cpu": mode(100.0)}}
        with tempfile.TemporaryDirectory() as directory:
            good, bad = pathlib.Path(directory, "good.json"), pathlib.Path(directory, "bad.json")
            good.write_text(json.dumps(kernel(12.0, 28.0)))
            bad.write_text(json.dumps(kernel(25.0, 140.0, readbacks=1, violations=2, within_budget=False)))
            ok = tool.cmd_gates(namespace(kernels=[str(good)]))
            no = tool.cmd_gates(namespace(kernels=[str(bad)]))
        self.assertEqual([row["result"] for row in ok], ["pass"] * 5)
        self.assertEqual([row["result"] for row in no], ["fail"] * 5)
        self.assertIn("| Gate | Limit | Measured | Result |", tool.markdown(ok))


class EndToEndTests(unittest.TestCase):
    """A real (tiny) run on the default adapter: the output file holds every mode, the pixel check and the counters."""

    @classmethod
    def setUpClass(cls):
        backend = gpu2d.GPU2D.open("default", 256 << 20)
        if backend is None:
            raise unittest.SkipTest(f"no GPU 2D adapter: {gpu2d.GPU2D.open_error}")
        cls.directory = tempfile.TemporaryDirectory()
        bench.make_inputs(pathlib.Path(cls.directory.name), 96, 54, 4)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def args(self, name, **extra):
        base = dict(inputs=pathlib.Path(self.directory.name), size="96x54", adapter="default", budget_mb=256, label="unit",
                    output=pathlib.Path(self.directory.name, name), warmup=1, frames="1-4", last=None, reps=2, truth_frames=2, backend="gpu2d", source_workers=0)
        base.update(extra)
        return namespace(**base)

    def test_kernels_report_every_mode_and_agree_with_the_cpu(self):
        args = self.args("k.json")
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(tool.cmd_kernels(args), 0)
        report = json.loads(args.output.read_text())
        self.assertEqual(set(report["summary"]), {"cpu", "roundtrip", "resident_fresh", "resident_warm"})
        self.assertTrue(report["cpu_reference_is_evaluator"]["bit_identical"])
        for mode in ("resident", "roundtrip"):
            self.assertEqual(report["pixels"][mode]["violations"], 0)
            self.assertTrue(report["pixels"][mode]["windows_match"])
        fresh, warm, trip = (report["summary"][m] for m in ("resident_fresh", "resident_warm", "roundtrip"))
        self.assertEqual((fresh["downloads"], fresh["intermediate_readbacks"], fresh["cpu_boundaries"], fresh["upload_cache_hits"]), (1, 0, [], 0))
        self.assertEqual(trip["downloads"], 4)                            # a round trip reads every node back
        hits = warm["upload_cache_hits"]
        self.assertGreater(hits if isinstance(hits, int) else hits["max"], 0)      # the warm mode really reuses resident sources
        self.assertEqual(len(report["samples"]["resident_warm"]), 4 * 2)  # frames times reps
        self.assertTrue(report["memory"]["within_budget"])
        self.assertIn("bounded completion", report["environment"]["timing_method"])

    def test_graph_reports_the_node_times_for_both_backends(self):
        for backend in ("cpu", "gpu2d"):
            args = self.args(f"g-{backend}.json", backend=backend)
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(tool.cmd_graph(args), 0)
            report = json.loads(args.output.read_text())
            self.assertEqual(report["wall_ms"]["n"], 4)
            self.assertTrue(all(r["finite"] for r in report["samples"]))
            kinds = set(report["kernel_ms_by_node_kind"])
            self.assertTrue({"Read", "Roto", "ColorCorrect"} <= kinds)
            self.assertEqual({"Grade", "Transform", "Blur", "Merge"} <= kinds, backend == "cpu")
            if backend == "gpu2d":
                self.assertTrue(all(r["island"]["downloads"] == 1 and r["island"]["intermediate_readbacks"] == 0 for r in report["samples"]))

    def test_graph_with_source_workers_gives_the_serial_pixels_and_records_the_count(self):
        reports = {}
        for workers in (0, 2):
            args = self.args(f"g-workers-{workers}.json", backend="cpu", source_workers=workers)
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(tool.cmd_graph(args), 0)
            reports[workers] = json.loads(args.output.read_text())
            self.assertEqual(reports[workers]["source_workers"], workers)
            self.assertTrue(all(r["finite"] for r in reports[workers]["samples"]))
        self.assertEqual(reports[0]["frames"], reports[2]["frames"])
        self.assertEqual({"Read", "Roto"} <= set(reports[2]["kernel_ms_by_node_kind"]), True)


if __name__ == "__main__":
    unittest.main()
