"""The end-to-end demo shot (examples/demo_shot): plate, solved camera, graph and the scripted hand run.

Everything runs on the CPU at small sizes; the full-size numbers are in the example's README."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

import tests.isolation  # noqa: F401  (Qt settings of the test run stay out of the user's real ones)

DEMO = Path(__file__).resolve().parent.parent / "examples" / "demo_shot"
sys.path.insert(0, str(DEMO))
import compare_runs  # noqa: E402
import make_plate  # noqa: E402
import run_by_agent  # noqa: E402
import run_by_hand  # noqa: E402
import shot  # noqa: E402
import solve_camera  # noqa: E402

SMALL, FRAMES = (160, 90), 4


def truth_keys(frames):
    return {f: {**shot.camera_at(f), "roll": 0.0, "focal": shot.FOCAL} for f in range(1, frames + 1)}


class DemoShotFiles(unittest.TestCase):
    def test_committed_pieces_are_there_and_hold_no_frames_or_home_paths(self):
        names = {path.name for path in DEMO.iterdir() if path.is_file()}
        for name in ("README.md", "BRIEF.md", "demo_shot.nbcomp", "make_plate.py", "run_by_hand.py",
                     "solve_camera.py", "run_by_agent.py", "compare_runs.py", "plate.sha256.json", "solve.json"):
            self.assertIn(name, names)
        self.assertFalse([n for n in names if n.endswith(".exr")])
        for name in ("demo_shot.nbcomp", "solve.json", "README.md", "BRIEF.md"):
            self.assertNotIn("/home/", (DEMO / name).read_text(encoding="utf-8"))

    def test_hash_file_covers_every_frame(self):
        recorded = json.loads((DEMO / "plate.sha256.json").read_text())
        self.assertEqual(sorted(map(int, recorded["frames"])), list(range(1, shot.FRAMES + 1)))
        self.assertEqual(recorded["size"], list(shot.PLATE_SIZE))

    def test_graph_has_every_stage_and_the_brief_in_the_text_control(self):
        document = json.loads((DEMO / "demo_shot.nbcomp").read_text())
        kinds = {key: node["type"] for key, node in document["nodes"].items()}
        for kind in ("Read", "Camera3D", "Scene3D", "WriteGeo3D", "Render3D", "Generate", "Grade", "Merge", "Write"):
            self.assertIn(kind, kinds.values())
        generate = document["nodes"]["generate"]["params"]
        self.assertEqual((generate["provider"], generate["seed"], generate["text"]),
                         ("reproject", 7, shot.BRIEF))
        self.assertEqual(len(document["animation"]["curves"]["camera"]["tx"]["keys"]), shot.FRAMES)


class DemoShotRun(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        cls.root = Path(cls.folder.name)
        cls.plate = cls.root / "plate"
        cls.hashes = make_plate.render_plate(cls.plate, SMALL, FRAMES)
        cls.graph = cls.root / "demo_shot.nbcomp"
        shot.save_graph(cls.graph, *SMALL, truth_keys(FRAMES), FRAMES, cls.plate)

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def test_plate_is_repeatable_and_carries_every_pass(self):
        again = make_plate.render_plate(self.root / "again", SMALL, FRAMES)
        self.assertEqual(again, self.hashes)
        from tools.mixed_scene_gate import exr_channels
        names, fmt, size = exr_channels(self.plate / "plate.0001.exr")
        self.assertEqual(size, SMALL)
        for channel in ("R", "A", "depth.Z", "normals.X", "motion.X", "object_id.R"):
            self.assertIn(channel, names)
        self.assertEqual(make_plate.check(self.hashes, {str(k): v for k, v in self.hashes.items()}), [])
        self.assertEqual(make_plate.check(self.hashes, {}), list(self.hashes))

    def test_camera_solve_recovers_the_rendering_camera(self):
        frames = solve_camera.read_plate(self.plate, FRAMES)
        # the solve needs the larger plate's pixels; this checks the machinery and a loose bound
        report, solved = solve_camera.solve(frames, *SMALL)
        summary = solve_camera.summarise(report)
        self.assertGreaterEqual(summary["landmarks"], 20)
        self.assertLess(summary["max_position_error"], 0.25)
        self.assertLess(summary["max_rotation_error_degrees"], 1.0)
        keys = solve_camera.camera_keys(solved)
        self.assertEqual(sorted(keys), list(range(1, FRAMES + 1)))

    def test_hand_run_twice_gives_the_same_ids_and_pixels(self):
        runs = []
        for name in ("one", "two"):
            logged = []
            summary = run_by_hand.run(self.plate, self.root / f"out-{name}", self.root / f"store-{name}",
                                      self.graph, frames=FRAMES, log=logged.append)
            summary["output_pixel_digest"] = run_by_hand.output_digest(self.root / f"out-{name}", FRAMES)
            runs.append(summary)
        self.assertEqual(runs[0], runs[1])
        first = runs[0]
        for key in ("scene_state", "control_bundle", "generated_sequence", "report"):
            self.assertRegex(first[key], r"^[0-9a-f]{64}$")
        self.assertIn(first["verdict"], ("PASS", "FAIL"))
        self.assertEqual(first["controls"]["text"], "ignored")
        self.assertEqual(first["ignored"], ["text"])
        self.assertEqual(first["controls"]["seed"], "used")
        from nodebased.media import read_media_raster
        pixels = np.asarray(read_media_raster(str(self.root / "out-one" / "shot.0001.exr")).to_display())
        self.assertEqual(pixels.shape[:2], (SMALL[1], SMALL[0]))
        self.assertTrue(np.isfinite(pixels).all())

    def test_agent_run_through_mcp_matches_the_hand_run(self):
        logged = []
        hand, agent, rows = compare_runs.run(self.plate, self.root / "compare", self.graph, frames=FRAMES,
                                              log=logged.append)
        by_label = {label: (a, b, same) for label, _, a, b, same in rows}
        self.assertEqual(list(by_label), ["SceneState", "ControlBundle", "Generated sequence",
                                          "Verification report", "Verdict", "Determinism", "Written output"])
        for label, (a, b, same) in by_label.items():
            self.assertTrue(same, f"{label}: hand {a} agent {b}")
        # the contract's two ids, exactly
        self.assertEqual(agent["generated_sequence"], hand["generated_sequence"])
        self.assertEqual(agent["report"], hand["report"])
        self.assertRegex(agent["generated_sequence"], r"^[0-9a-f]{64}$")
        self.assertEqual((agent["controls"]["text"], agent["ignored"]), ("ignored", ["text"]))
        self.assertEqual(agent["provenance_links"], hand["provenance_links"])
        # the agent used tools only, in the shot-workflow order
        order = [name for name in dict.fromkeys(agent["tool_sequence"])]
        self.assertEqual(order, ["knowledge", "describe", "inspect", "edit", "export_conditioning", "generate",
                                 "verify", "artifacts"])
        self.assertEqual(agent["tool_sequence"].count("edit"), 2 + FRAMES)
        from nodebased.media import read_media_raster
        pixels = np.asarray(read_media_raster(str(self.root / "compare" / "out-agent" / "shot.0001.exr")).to_display())
        self.assertEqual(pixels.shape[:2], (SMALL[1], SMALL[0]))

    def test_agent_script_reaches_the_shot_through_tools_alone(self):
        import ast
        tree = ast.parse((DEMO / "run_by_agent.py").read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                imported.add(node.module)
            elif isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
        # the plan sees NodeBased only through the MCP client; no document, evaluator or pipeline import
        self.assertEqual({m for m in imported if m.startswith("nodebased")}, {"nodebased.mcpclient"})


if __name__ == "__main__":
    unittest.main()
