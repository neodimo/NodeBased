"""The seed contract (stage 2 of the end-to-end sprint) and the loop verifier's pass handling."""
import json
import os
import tempfile
import threading
import unittest
import zipfile
import io
from pathlib import Path
from unittest import mock

import numpy as np

from nodebased import scene3d
from nodebased.artifacts import ArtifactStore
from nodebased.control_bundle import write_control_bundle
from nodebased.conditioned_read import _read_rgba
from nodebased.generative import (GENERATION_MANIFEST, ProviderDescription, _check_frame_readable,
                                  _write_frame, generate, normalize_sampler_settings, normalize_seed)
from nodebased.jobs import Queue
from nodebased.loops import LoopRun, verify_task
from nodebased.scene_state import write_scene_state


def make_shot(root, width=32, height=24, frames=(1,)):
    camera = scene3d.Camera(transform=scene3d.Transform3D(position=scene3d.Vec3(0, 0, 5)))
    state, scene_id = write_scene_state(
        root / "shot.scene.json",
        {f: {"scene": scene3d.Scene(), "camera": camera, "resolution": (width, height)} for f in frames},
        resolution=(width, height), return_artifact_id=True)
    plate = np.zeros((height, width, 4), np.float32)
    plate[..., 0] = np.linspace(0.1, 0.9, width, dtype=np.float32)[None, :]
    plate[..., 3] = 1
    samples = {f: {"beauty": plate.copy(), "depth": np.linspace(1, 3, width * height, dtype=np.float32).reshape(height, width),
                   "normals": np.tile(np.array([0, 0, 1], np.float32), (height, width, 1)),
                   "motion_forward": np.zeros((height, width, 2), np.float32),
                   "motion_backward": np.zeros((height, width, 2), np.float32),
                   "object_ids": np.zeros((height, width), np.float32)} for f in frames}
    bundle, bundle_id = write_control_bundle(state, root / "controls", samples, return_artifact_id=True)
    return state, scene_id, bundle, bundle_id


def unpack(store, artifact_id):
    with zipfile.ZipFile(io.BytesIO(store.get(artifact_id))) as archive:
        return {info.filename: archive.read(info) for info in archive.infolist()}


class SeedContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        patcher = mock.patch.dict(os.environ, {"NODEBASED_CACHE": str(self.root / "cache")})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.store = ArtifactStore()
        self.state, self.scene_id, self.bundle, self.bundle_id = make_shot(self.root)

    def run_generate(self, name, provider="reproject", **kwargs):
        # The frame file names are part of the archive, so identical briefs use identical names
        # in different folders.
        return generate(self.bundle, self.state, self.root / name / "out.####.exr", provider, **kwargs)

    def test_normalisation_rejects_bad_seed_and_settings(self):
        self.assertEqual(normalize_seed(None), 0)
        self.assertEqual(normalize_sampler_settings(None), {})
        self.assertEqual(normalize_sampler_settings('{"steps": 20}'), {"steps": 20})
        for bad in (-1, 1.5, "7", True):
            with self.assertRaises(ValueError):
                normalize_seed(bad)
        for bad in ([1], "[1]", "{not json", {"x": float("nan")}, {"blob": "x" * 5000}):
            with self.assertRaises(ValueError):
                normalize_sampler_settings(bad)

    def test_descriptions_declare_seed_honoured_and_settings_ignored(self):
        for name in ("null", "reproject"):
            description = ProviderDescription.load(name)
            report = description.control_report()
            self.assertEqual(report["seed"], "used")
            self.assertEqual(report["sampler_settings"], "ignored")
            self.assertIn("Seed: used", description.panel_lines())
            self.assertIn("Sampler Settings: ignored", description.panel_lines())

    def test_an_ignoring_provider_refuses_settings_it_cannot_honour(self):
        with self.assertRaisesRegex(ValueError, "sampler_settings"):
            self.run_generate("refused", "null", sampler_settings={"steps": 4})

    def test_old_descriptions_without_the_new_controls_read_as_ignored(self):
        data = json.loads((Path(__file__).parents[1] / "nodebased/providers/null.json").read_text())
        del data["accepts"]["seed"], data["accepts"]["sampler_settings"]
        data["returns"]["honoured"] = []
        path = self.root / "old.json"
        path.write_text(json.dumps(data))
        old = ProviderDescription.load(path)
        self.assertEqual(old.control_report()["seed"], "ignored")
        with self.assertRaisesRegex(ValueError, "seed"):
            generate(self.bundle, self.state, self.root / "old.####.exr", "null", description=path, seed=3)

    def test_manifest_inside_the_artifact_covers_every_input(self):
        result = self.run_generate("covered", seed=7, text=None)
        files = unpack(self.store, result["artifact_id"])
        manifest = json.loads(files[GENERATION_MANIFEST])
        self.assertEqual((manifest["provider"], manifest["version"], manifest["seed"]), ("reproject", "1.1", 7))
        self.assertEqual(manifest["sampler_settings"], {})
        self.assertEqual(manifest["scene_state"], self.scene_id)
        self.assertEqual(manifest["control_bundle"], self.bundle_id)
        self.assertTrue(manifest["seed_honoured"])
        self.assertFalse(manifest["sampler_settings_honoured"])
        self.assertIn("text", manifest["options"])
        self.assertEqual(result["seed"], 7)

    def test_provenance_chain_shows_seed_and_provider(self):
        result = self.run_generate("prov", seed=11)
        row = self.store.provenance(result["artifact_id"])[-1]["provenance"]
        self.assertEqual((row["seed"], row["provider"], row["provider_version"]), (11, "reproject", "1.1"))
        self.assertEqual(row["sampler_settings"], {})
        import subprocess, sys
        shown = subprocess.check_output([sys.executable, "-m", "nodebased.artifacts", "provenance",
                                         result["artifact_id"]], text=True,
                                        env={**os.environ, "PYTHONPATH": str(Path(__file__).parents[1])})
        self.assertIn("seed: 11 (honoured)", shown)

    def test_same_seed_is_byte_identical_and_a_new_seed_changes_only_the_id(self):
        """The stand-ins ignore the seed's value, so changing it leaves the frames byte-identical;
        the artifact id still differs because the hashed generation manifest records the seed."""
        for provider in ("null", "reproject"):
            first = self.run_generate(f"{provider}-a", provider, seed=5)
            second = self.run_generate(f"{provider}-b", provider, seed=5)
            other = self.run_generate(f"{provider}-c", provider, seed=6)
            self.assertEqual(first["artifact_id"], second["artifact_id"])
            self.assertNotEqual(first["artifact_id"], other["artifact_id"])
            frames = lambda r: {k: v for k, v in unpack(self.store, r["artifact_id"]).items() if k.endswith(".exr")}
            self.assertEqual(sorted(frames(first).values()), sorted(frames(second).values()))
            self.assertEqual(sorted(frames(first).values()), sorted(frames(other).values()))

    def test_report_has_a_determinism_section_and_the_verdict_ignores_it(self):
        reports = []
        for seed in (0, 9):
            generated = self.run_generate(f"verify{seed}", "null", seed=seed)["artifact_id"]
            result = verify_task({"scene_state_id": self.scene_id, "control_bundle_id": self.bundle_id,
                                  "_artifact_root": str(self.store.root)}, [generated],
                                 lambda *_: None, threading.Event())
            reports.append(json.loads(self.store.get(result["artifact_id"])))
        for seed, report in zip((0, 9), reports):
            self.assertEqual(report["determinism"], {"provider": "null", "version": "1.0", "seed": seed,
                                                     "sampler_settings": {}, "seed_honoured": True})
        self.assertEqual(reports[0]["score_card"], reports[1]["score_card"])


class LoopSeedTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.store = ArtifactStore(root / "artifacts")
        self.queue = Queue(root / "jobs.sqlite", self.store, max_workers=1)
        self.addCleanup(self.queue.close)
        self.loop = LoopRun(root / "loops.sqlite", self.queue, generate_operation="tests.loop_fixtures:generate",
                            verify_operation="tests.loop_fixtures:verify")
        self.scene = self.store.put(b"scene", "scene_state", {"producer": "fixture", "inputs": []})
        self.bundle = self.store.put(b"bundle", "control_bundle", {"producer": "fixture", "inputs": []})

    def run_loop(self, **kwargs):
        ident = self.loop.create(scene_state_id=self.scene, control_bundle_id=self.bundle, provider_id="null",
                                 provider_options={"fail_verifications": 2}, max_attempts=3,
                                 max_estimated_spend=3, estimated_spend_per_attempt=1, spend_unit="credits (estimate)",
                                 **kwargs)
        return self.loop.run(ident)

    def test_attempts_reuse_the_loop_seed_by_default(self):
        snap = self.run_loop(seed=42)
        self.assertEqual([a["seed"] for a in snap["attempts"]], [42, 42, 42])
        self.assertEqual(snap["attempts"][0]["outputs"]["generated_sequence"] is None, False)

    def test_vary_seed_increments_per_attempt(self):
        snap = self.run_loop(seed=42, vary_seed=True)
        self.assertEqual([a["seed"] for a in snap["attempts"]], [42, 43, 44])
        self.assertEqual(snap["config"]["vary_seed"], True)

    def test_bad_seed_options_are_rejected(self):
        with self.assertRaises(ValueError):
            self.run_loop(seed=-3)
        with self.assertRaises(ValueError):
            self.run_loop(vary_seed="yes")


class VerifierPassTests(unittest.TestCase):
    """verify_task judges what the sequence carries and labels the rest "not conditioned"."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        patcher = mock.patch.dict(os.environ, {"NODEBASED_CACHE": str(self.root / "cache")})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.store = ArtifactStore()
        self.state, self.scene_id, self.bundle, self.bundle_id = make_shot(self.root)

    def verify(self, generated):
        result = verify_task({"scene_state_id": self.scene_id, "control_bundle_id": self.bundle_id,
                              "_artifact_root": str(self.store.root)}, [generated],
                             lambda *_: None, threading.Event())
        return json.loads(self.store.get(result["artifact_id"]))["score_card"]

    def sequence_from_bundle(self, mutate=None):
        """A generated sequence carrying every pass: the bundle's own frame, optionally altered."""
        import OpenImageIO as oiio
        frame = next((self.bundle.parent).glob("*.exr"))
        image = oiio.ImageInput.open(str(frame))
        spec = image.spec()
        names = list(spec.channelnames)
        pixels = np.asarray(image.read_image(oiio.FLOAT), np.float32).reshape(spec.height, spec.width, spec.nchannels)
        image.close()
        if mutate:
            mutate(pixels, names)
        target = self.root / "full.0001.exr"
        out_spec = oiio.ImageSpec(spec.width, spec.height, spec.nchannels, oiio.FLOAT)
        out_spec.channelnames = names
        out_spec.attribute("oiio:ColorSpace", spec.get_string_attribute("oiio:ColorSpace") or "ACEScg")
        output = oiio.ImageOutput.create(str(target))
        self.assertTrue(output.open(str(target), out_spec))
        self.assertTrue(output.write_image(np.ascontiguousarray(pixels)))
        output.close()
        return self.store.put_files({"full.0001.exr": target}, "generated_sequence", {"producer": "test"})

    def test_beauty_only_sequence_marks_missing_passes_and_gets_a_verdict(self):
        generated = generate(self.bundle, self.state, self.root / "null.####.exr", "null",
                             artifact_store=self.store)["artifact_id"]
        card = self.verify(generated)
        bindings = card["bindings"]
        for name in ("depth", "motion", "normals"):
            self.assertEqual(bindings[name].get("status"), "not conditioned", name)
            self.assertIsNone(bindings[name]["pass"], name)
        self.assertIn(card["verdict"], ("PASS", "FAIL"))
        applicable = [b["pass"] for b in bindings.values() if b["pass"] is not None]
        self.assertEqual(card["verdict"], "PASS" if all(applicable) else "FAIL")

    def test_passes_in_the_sequence_are_judged_against_the_export(self):
        card = self.verify(self.sequence_from_bundle())
        for name in ("depth", "motion", "normals"):
            self.assertIs(card["bindings"][name]["pass"], True, name)
            self.assertNotIn("status", card["bindings"][name], name)

    def test_a_wrong_depth_or_normals_pass_fails_that_check(self):
        def wrong(pixels, names):
            pixels[..., names.index("depth.Z")] = pixels[..., names.index("depth.Z")][::-1, ::-1]
            pixels[..., names.index("normals.Z")] *= -1
        card = self.verify(self.sequence_from_bundle(wrong))
        self.assertIs(card["bindings"]["depth"]["pass"], False)
        self.assertIs(card["bindings"]["normals"]["pass"], False)
        self.assertEqual(card["verdict"], "FAIL")


class UnreadableFrameTests(unittest.TestCase):
    """10/10: at 160x120 the isolated provider worker (2 GiB address-space cap) sometimes wrote a
    header-only EXR and reported success; verify then died with "cannot reshape array of size 1"."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        patcher = mock.patch.dict(os.environ, {"NODEBASED_CACHE": str(self.root / "cache")})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_small_case_through_the_isolated_worker_is_always_readable(self):
        from nodebased.workers import Job, Worker
        state, scene_id, bundle, bundle_id = make_shot(self.root, 160, 120)
        for run in range(8):
            worker = Worker(Job("null", bundle_id, {"manifest_path": str(bundle), "scene_state_path": str(state),
                                                    "output_pattern": str(self.root / f"run{run}.####.exr")}))
            result = worker.run()
            self.assertEqual(result["type"], "result", result)
            self.assertEqual(_read_rgba(self.root / f"run{run}.0001.exr")[0].shape, (120, 160, 4))

    def test_truncated_frames_fail_loudly(self):
        frame = self.root / "frame.exr"
        _write_frame(frame, np.ones((120, 160, 4), np.float32), "ACEScg")
        data = frame.read_bytes()
        frame.write_bytes(data[:543])
        with self.assertRaisesRegex(ValueError, "unreadable"):
            _read_rgba(frame)
        with self.assertRaises(ValueError):
            _check_frame_readable(frame, 160, 120)
