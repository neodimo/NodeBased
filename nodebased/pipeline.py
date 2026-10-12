"""The generative pipeline as agent bridge operations.

Both the desktop window (`Window.agent_command`) and the headless dispatcher (`nodebased.agent`)
route these ops here, so an attached agent can go from a scene to a verified, generated sequence
without a human touching a panel. Everything stays inside the artifact store except an explicit
`output_pattern` on `generate`, which the MCP server gates behind the disk-ops setting.
"""
from __future__ import annotations

import copy
import io
import re
import threading
import uuid
import zipfile
from pathlib import Path

from .artifacts import ArtifactStore
from .generative import PROVIDER_DIR, ProviderDescription, normalize_sampler_settings, normalize_seed

OPS = frozenset({"export_conditioning", "generate", "verify", "loop", "artifacts"})
_ID = re.compile(r"[0-9a-f]{64}")
_TERMINAL = ("pass", "provider_failure", "attempts_exhausted", "budget_exhausted", "cancelled",
             "feedback_unsupported")


def _artifact_id(value, name):
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ValueError(f"{name} must be a 64-character artifact id")
    return value


def _provider(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", value) \
            or not (PROVIDER_DIR / f"{value}.json").is_file():
        known = sorted(path.stem for path in PROVIDER_DIR.glob("*.json"))
        raise ValueError(f"provider must be one of {known}")
    return value


def _require_local(provider):
    if ProviderDescription.load(provider).locality != "local":
        raise ValueError(f"provider {provider!r} is queued; start it with the loop tool and poll loop status")


class Pipeline:
    """`queue` and `loops` are created on first use; `runner` (a thread pool) makes loops async."""

    def __init__(self, store=None, queue=None, loops=None, runner=None, evaluator_factory=None):
        self._store = store
        self._queue = queue
        self._loops = loops
        self.runner = runner
        self._evaluator_factory = evaluator_factory

    @property
    def store(self):
        if self._store is None:
            self._store = self._queue.store if self._queue is not None else ArtifactStore()
        return self._store

    @property
    def queue(self):
        if self._queue is None:
            from .queuepanel import create_default_queue
            self._queue = create_default_queue()
        return self._queue

    @property
    def loops(self):
        if self._loops is None:
            from .loops import LoopRun
            self._loops = LoopRun(self.queue.path.parent / "conditioning-loops.sqlite", self.queue)
        return self._loops

    def execute(self, cmd, document):
        op = cmd.get("op")
        if op == "export_conditioning":
            return self.export_conditioning(cmd, document)
        if op == "generate":
            return self.generate(cmd)
        if op == "verify":
            return self.verify(cmd)
        if op == "loop":
            return self.loop(cmd)
        if op == "artifacts":
            return self.artifacts(cmd)
        raise ValueError(f"unknown pipeline op {op!r}")

    # -- export -------------------------------------------------------------------------------
    def export_conditioning(self, cmd, document):
        from .geoexport import export_obj
        key = cmd.get("id")
        nodes = document["nodes"]
        if key not in nodes:
            raise ValueError(f"export_conditioning: unknown node {key!r}")
        time = document["time"]
        first = int(cmd.get("first", time["current"]))
        last = int(cmd.get("last", first))
        if last < first:
            raise ValueError("export_conditioning: last must not be before first")
        if last - first > 999:
            raise ValueError("export_conditioning: at most 1000 frames per export")
        scratch = copy.deepcopy(document)
        node = scratch["nodes"][key]
        target = self.store.root / "agent-exports" / uuid.uuid4().hex / "shot.scene.json"
        if node["type"] == "WriteGeo3D":
            writer = key
            node["params"]["geo_write_path"] = str(target)
        else:
            if node["type"] == "Render3D":
                scene, camera = node["inputs"].get("scene"), node["inputs"].get("camera")
            else:
                scene, camera = key, cmd.get("camera")
            if not scene or not camera or camera not in nodes:
                raise ValueError("export_conditioning needs a scene and a camera: pass a WriteGeo3D or "
                                 "Render3D node, or a scene node with a camera node id in `camera`")
            writer = "_nb_export"
            scratch["nodes"][writer] = {**copy.deepcopy(node), "type": "WriteGeo3D",
                                        "params": {"geo_write_path": str(target)},
                                        "inputs": {"scene": scene, "camera": camera}}
            if node["type"] == "Render3D":
                # export_obj takes the render size from a Render3D fed by the writer.
                scratch["nodes"]["_nb_export_render"] = {
                    **copy.deepcopy(node), "inputs": {"scene": writer, "camera": camera}}
        evaluator = self._evaluator_factory() if self._evaluator_factory else None
        written = export_obj(scratch, writer, range(first, last + 1), str(target), evaluator)
        scene_id = _sidecar_id(target.with_suffix(target.suffix + ".artifact.json"))
        bundle_id = _sidecar_id(Path(written[2]).with_suffix(".artifact.json"))
        return {"scene_state": scene_id, "control_bundle": bundle_id, "frames": [first, last]}

    # -- generate / verify --------------------------------------------------------------------
    def generate(self, cmd):
        from .loops import generate_task
        scene = _artifact_id(cmd.get("scene_state"), "scene_state")
        bundle = _artifact_id(cmd.get("bundle"), "bundle")
        provider = _provider(cmd.get("provider"))
        _require_local(provider)
        options = cmd.get("options") or {}
        if not isinstance(options, dict):
            raise ValueError("generate options must be an object")
        options = {k: v for k, v in options.items() if k in ("text",)}
        options["seed"] = normalize_seed(cmd.get("seed", 0))
        options["sampler_settings"] = normalize_sampler_settings(cmd.get("sampler_settings"))
        if isinstance(cmd.get("output_pattern"), str) and cmd["output_pattern"]:
            options["output_pattern"] = cmd["output_pattern"]
        options.update({"provider_id": provider, "_artifact_root": str(self.store.root)})
        result = generate_task(options, [scene, bundle], lambda *_: None, threading.Event())
        description = ProviderDescription.load(provider)
        reply = {"generated_sequence": result["artifact_id"], "provider": provider,
                 "version": description.version, "seed": options["seed"],
                 "sampler_settings": options["sampler_settings"], "controls": description.control_report(),
                 "ignored": result.get("ignored", [])}
        # The same unpacked ConditionedRead parameters a finished loop attempt hands out (`use_result`).
        return {**reply, **_conditioned_read(self.store, scene, bundle, result["artifact_id"])}

    def verify(self, cmd):
        import json
        from .loops import verify_task
        scene = _artifact_id(cmd.get("scene_state"), "scene_state")
        bundle = _artifact_id(cmd.get("bundle"), "bundle")
        generated = _artifact_id(cmd.get("generated_sequence"), "generated_sequence")
        options = {"scene_state_id": scene, "control_bundle_id": bundle,
                   "_artifact_root": str(self.store.root)}
        survey = cmd.get("survey")
        if survey is not None:
            if not isinstance(survey, dict) or not isinstance(survey.get("points"), list):
                raise ValueError("verify: survey must be {\"points\": [{\"name\", \"world\", \"track\"}, ...]}")
            options["survey"] = survey
        result = verify_task(options, [generated], lambda *_: None, threading.Event())
        report = json.loads(self.store.get(result["artifact_id"]))
        card = report.get("score_card", report)
        return {"report": result["artifact_id"], "verdict": card.get("verdict"), "score_card": card,
                "determinism": report.get("determinism"),
                **({"survey": report["survey"]} if "survey" in report else {})}

    # -- loops --------------------------------------------------------------------------------
    def loop(self, cmd):
        action = cmd.get("action")
        if action == "start":
            return self._loop_start(cmd)
        if action == "status":
            return self._loop_status(cmd.get("loop"))
        if action == "cancel":
            loop_id = self._loop_id(cmd)
            self.loops.cancel(loop_id)
            return self._loop_status(loop_id)
        if action == "use_result":
            return self._loop_use_result(cmd)
        raise ValueError("loop action must be start, status, cancel or use_result")

    @staticmethod
    def _loop_id(cmd):
        loop_id = cmd.get("loop")
        if not isinstance(loop_id, str) or not loop_id:
            raise ValueError("loop requires a loop id")
        return loop_id

    def _loop_start(self, cmd):
        provider = _provider(cmd.get("provider"))
        options = {"estimated_spend": cmd.get("estimate"), "spend_unit": cmd.get("unit")}
        if cmd.get("text"):
            options["text"] = cmd["text"]
        controls = list(cmd.get("feedback") or [])
        if options.get("text") and "text" not in controls:
            controls.append("text")
        loop_id = self.loops.create(
            scene_state_id=_artifact_id(cmd.get("scene_state"), "scene_state"),
            control_bundle_id=_artifact_id(cmd.get("bundle"), "bundle"),
            provider_id=provider, provider_options=options,
            max_attempts=cmd.get("max_attempts", 3), max_estimated_spend=cmd.get("spend_cap"),
            estimated_spend_per_attempt=cmd.get("estimate"), spend_unit=cmd.get("unit"),
            feedback_controls=controls, seed=cmd.get("seed", 0),
            sampler_settings=cmd.get("sampler_settings"), vary_seed=cmd.get("vary_seed", False))
        if self.runner is not None:
            self.runner.submit(self.loops.run, loop_id)
        else:
            self.loops.run(loop_id)
        return self._loop_status(loop_id)

    def _loop_status(self, loop_id):
        if not loop_id:
            return {"loops": [{"loop": row["id"], "state": row["state"]}
                              for row in self.loops.list_snapshots()[:20]]}
        snap = self.loops.snapshot(loop_id)
        attempts = [{"number": a["number"], "state": a["state"], "verdict": a["verdict"], "seed": a["seed"],
                     "generated_sequence": a["outputs"].get("generated_sequence"),
                     "verification_report": a["outputs"].get("verification_report"),
                     "spend": a["spend"], "spend_unit": a["spend_unit"],
                     "queue_chain": a["queue_chain"], "error": a["error"]}
                    for a in snap["attempts"]]
        done = snap["state"] in _TERMINAL
        return {"loop": loop_id, "state": snap["state"], "finished": done,
                "stop_reason": snap["state"] if done else None, "error": snap["error"],
                "cumulative_estimate": snap["cumulative"], "attempts": attempts,
                "seed": snap["config"].get("seed", 0),
                "sampler_settings": snap["config"].get("sampler_settings", {}),
                "vary_seed": snap["config"].get("vary_seed", False),
                "controls": ProviderDescription.load(snap["config"]["provider_id"]).control_report(),
                "scene_state": snap["config"]["scene_state_id"],
                "bundle": snap["config"]["control_bundle_id"]}

    def _loop_use_result(self, cmd):
        """Unpack a finished attempt's inputs so a ConditionedRead node can be created with `edit`."""
        status = self._loop_status(self._loop_id(cmd))
        wanted = cmd.get("attempt")
        done = [a for a in status["attempts"] if a["generated_sequence"]
                and (wanted is None or a["number"] == wanted)]
        if wanted is None:
            done = [a for a in done if a["state"] == "pass"] or done
        if not done:
            raise ValueError("loop has no completed attempt to use")
        chosen = done[-1]
        return {"attempt": chosen["number"],
                **_conditioned_read(self.store, status["scene_state"], status["bundle"],
                                    chosen["generated_sequence"])}

    # -- artifacts ----------------------------------------------------------------------------
    def artifacts(self, cmd):
        action = cmd.get("action")
        if action == "list":
            kind = cmd.get("kind")
            rows = sorted(self.store.find(kind), key=lambda r: r["created"], reverse=True)
            limit = max(1, min(int(cmd.get("limit", 20)), 200))
            return {"artifacts": [{"id": r["id"], "kind": r["kind"], "size": r["size"],
                                   "created": r["created"],
                                   "producer": r["provenance"].get("producer")}
                                  for r in rows[:limit]]}
        if action == "provenance":
            return {"chain": self.store.provenance(_artifact_id(cmd.get("id"), "id"))}
        raise ValueError("artifacts action must be list or provenance")


def _sidecar_id(path):
    import json
    return json.loads(Path(path).read_text(encoding="utf-8"))["artifact_id"]


def _conditioned_read(store, scene_id, bundle_id, sequence_id):
    """The ConditionedRead node that reads a generated sequence against its inputs, unpacked.

    `generate` and a loop's `use_result` both answer with this, so either path creates the same node.
    """
    paths = unpack_inputs(store, scene_id, bundle_id)
    return {"path": sequence_id, "manifest": paths["bundle"], "scene_state": paths["scene"],
            "node": {"type": "ConditionedRead", "params": {
                "path": sequence_id, "manifest": paths["bundle"], "scene_state": paths["scene"]}}}


def unpack_inputs(store, scene_id, bundle_id):
    """Extract a SceneState and ControlBundle artifact to the store's loop-inputs folder."""
    root = store.root / "loop-inputs"
    paths = {}
    for label, aid in (("scene", scene_id), ("bundle", bundle_id)):
        folder = root / aid
        folder.mkdir(parents=True, exist_ok=True)
        try:
            with zipfile.ZipFile(io.BytesIO(store.get(aid))) as archive:
                for info in archive.infolist():
                    name = Path(info.filename)
                    if name.is_absolute() or ".." in name.parts or len(name.parts) != 1:
                        raise ValueError("conditioning artifact contains an unsafe path")
                    if not info.is_dir():
                        (folder / name.name).write_bytes(archive.read(info))
        except zipfile.BadZipFile as error:
            raise ValueError(f"Cannot open {label} artifact {aid}") from error
        wanted = [p for p in folder.glob("*.json")
                  if (p.name == "manifest.json") == (label == "bundle")]
        if not wanted:
            raise ValueError(f"{label} artifact has no JSON document")
        paths[label] = str(wanted[0])
    return paths
