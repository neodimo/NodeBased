"""Durable, bounded Generate -> VerifyConditioning loops over the job queue."""
from __future__ import annotations

import json
import io
import re
import sqlite3
import tempfile
import threading
import time
import uuid
from pathlib import Path
import zipfile
from contextlib import contextmanager

from .generative import ProviderDescription, normalize_sampler_settings, normalize_seed


class LoopError(ValueError):
    pass


class LoopRun:
    """A resumable loop. Operation callables follow the queue's module:function contract."""
    def __init__(self, path, queue, *, generate_operation="nodebased.loops:generate_task",
                 verify_operation="nodebased.loops:verify_task"):
        self.path = Path(path); self.path.parent.mkdir(parents=True, exist_ok=True)
        self.queue = queue
        self.generate_operation = generate_operation
        self.verify_operation = verify_operation
        with self._db() as db:
            db.executescript("""CREATE TABLE IF NOT EXISTS loops(
                id TEXT PRIMARY KEY, config TEXT NOT NULL, state TEXT NOT NULL,
                next_attempt INTEGER NOT NULL DEFAULT 1, cumulative REAL NOT NULL DEFAULT 0,
                last_output TEXT NOT NULL DEFAULT '', error TEXT NOT NULL DEFAULT '');
                CREATE TABLE IF NOT EXISTS attempts(
                loop_id TEXT NOT NULL, number INTEGER NOT NULL, state TEXT NOT NULL,
                inputs TEXT NOT NULL, outputs TEXT NOT NULL DEFAULT '{}', provider TEXT NOT NULL,
                version TEXT NOT NULL, options TEXT NOT NULL, verdict TEXT NOT NULL DEFAULT '',
                spend REAL NOT NULL, spend_unit TEXT NOT NULL, queue_chain TEXT NOT NULL,
                error TEXT NOT NULL DEFAULT '', PRIMARY KEY(loop_id,number));""")

    @contextmanager
    def _db(self):
        db=sqlite3.connect(self.path, timeout=30); db.row_factory=sqlite3.Row
        try:
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def create(self, *, scene_state_id, control_bundle_id, provider_id, provider_options,
               max_attempts, max_estimated_spend, estimated_spend_per_attempt=None,
               spend_unit=None, feedback_controls=(), loop_id=None, seed=0, sampler_settings=None,
               vary_seed=False):
        if not isinstance(max_attempts, int) or isinstance(max_attempts, bool) or max_attempts <= 0:
            raise LoopError("max_attempts must be a positive finite integer")
        if max_estimated_spend is None or float(max_estimated_spend) <= 0:
            raise LoopError("max_estimated_spend must be positive and finite")
        if not (float(max_estimated_spend) < float("inf")):
            raise LoopError("max_estimated_spend must be finite")
        provider=ProviderDescription.load(provider_id)
        options=dict(provider_options or {})
        estimate = estimated_spend_per_attempt
        if estimate is None: estimate=options.get("estimated_spend")
        unit=spend_unit or options.get("spend_unit")
        if estimate is None or not unit: raise LoopError("a declared per-attempt spend estimate and unit are required")
        estimate=float(estimate)
        if estimate < 0 or not estimate < float("inf"): raise LoopError("per-attempt estimate must be finite and non-negative")
        controls=list(feedback_controls)
        if any(c not in ("reference_frames", "text") for c in controls):
            raise LoopError("unsupported feedback control name")
        if "text" in controls and not options.get("text"):
            raise LoopError("text feedback was requested but provider_options has no text value")
        seed=normalize_seed(seed); sampler_settings=normalize_sampler_settings(sampler_settings)
        if not isinstance(vary_seed, bool): raise LoopError("vary_seed must be true or false")
        ident=loop_id or uuid.uuid4().hex
        config={"scene_state_id":scene_state_id,"control_bundle_id":control_bundle_id,
                "provider_id":provider_id,"provider_name":provider.name,"provider_version":provider.version,
                "provider_options":options,"max_attempts":max_attempts,
                "max_estimated_spend":float(max_estimated_spend),"estimate":estimate,
                "spend_unit":unit,"feedback_controls":controls,
                "seed":seed,"sampler_settings":sampler_settings,"vary_seed":vary_seed}
        with self._db() as db:
            db.execute("INSERT OR IGNORE INTO loops(id,config,state) VALUES(?,?,'queued')",
                       (ident,json.dumps(config,sort_keys=True)))
            row=db.execute("SELECT config FROM loops WHERE id=?",(ident,)).fetchone()
            if row is None or json.loads(row[0]) != config: raise LoopError("loop id already exists with different inputs")
        return ident

    def cancel(self, loop_id):
        with self._db() as db:
            row=db.execute("SELECT state FROM loops WHERE id=?",(loop_id,)).fetchone()
            if row is None: raise LoopError("unknown loop")
            db.execute("UPDATE loops SET state='cancelling' WHERE id=?",(loop_id,))
            attempt=db.execute("SELECT queue_chain FROM attempts WHERE loop_id=? AND state='running' ORDER BY number DESC LIMIT 1",(loop_id,)).fetchone()
        if attempt: self.queue.cancel(attempt[0])
        else:
            with self._db() as db: db.execute("UPDATE loops SET state='cancelled' WHERE id=?",(loop_id,))

    def run(self, loop_id):
        while True:
            with self._db() as db:
                row=db.execute("SELECT * FROM loops WHERE id=?",(loop_id,)).fetchone()
                if row is None: raise LoopError("unknown loop")
                config=json.loads(row["config"]); state=row["state"]
                if state in ("pass","provider_failure","attempts_exhausted","budget_exhausted","cancelled","feedback_unsupported"):
                    return self.snapshot(loop_id)
                if state == "cancelling":
                    db.execute("UPDATE loops SET state='cancelled' WHERE id=?",(loop_id,)); db.commit(); return self.snapshot(loop_id)
                n=int(row["next_attempt"]); cumulative=float(row["cumulative"]); previous=row["last_output"]
                if n>config["max_attempts"]:
                    db.execute("UPDATE loops SET state='attempts_exhausted' WHERE id=?",(loop_id,)); db.commit(); return self.snapshot(loop_id)
                if cumulative+config["estimate"] > config["max_estimated_spend"]+1e-12:
                    db.execute("UPDATE loops SET state='budget_exhausted' WHERE id=?",(loop_id,)); db.commit(); return self.snapshot(loop_id)
                provider=ProviderDescription.load(config["provider_id"])
                requested=config["feedback_controls"]
                if n>1 and requested and (not previous or any(not provider.accepted(c) for c in requested)):
                    db.execute("UPDATE loops SET state='feedback_unsupported',error=? WHERE id=?",
                               ("provider does not declare support for requested feedback: "+", ".join(requested),loop_id))
                    db.commit(); return self.snapshot(loop_id)
                inputs=[config["scene_state_id"],config["control_bundle_id"]]
                if n>1 and requested: inputs.append(previous)
                chain=f"loop-{loop_id}-{n}"; gen_id=f"{chain}-generate"; ver_id=f"{chain}-verify"
                # Every attempt reuses the loop's seed (reproducible) unless vary_seed asks for seed+n-1.
                attempt_seed=config.get("seed",0)+(n-1 if config.get("vary_seed") else 0)
                args={**config["provider_options"],"seed":attempt_seed,
                      "sampler_settings":config.get("sampler_settings",{}),"provider_id":config["provider_id"],"provider_version":config["provider_version"],
                      "feedback_controls":requested,"attempt":n,"estimated_spend":config["estimate"],"spend_unit":config["spend_unit"]}
                links=[{"id":gen_id,"name":"Generate","operation":self.generate_operation,
                        "options":args,"inputs":inputs,"provider":config["provider_id"]},
                       {"id":ver_id,"name":"VerifyConditioning","operation":self.verify_operation,
                        "options":{**config["provider_options"],"attempt":n,
                                   "scene_state_id":config["scene_state_id"],
                                   "control_bundle_id":config["control_bundle_id"]},
                        "inputs":[config["scene_state_id"],config["control_bundle_id"]],
                        "dependencies":[gen_id]}]
                db.execute("INSERT OR IGNORE INTO attempts(loop_id,number,state,inputs,provider,version,options,spend,spend_unit,queue_chain) VALUES(?,?,'running',?,?,?,?,?,?,?)",
                    (loop_id,n,json.dumps(inputs),config["provider_name"],config["provider_version"],json.dumps(args,sort_keys=True),config["estimate"],config["spend_unit"],chain))
                # Existing matching completed queue work is reused by Queue.add_chain on reopen.
                db.execute("UPDATE loops SET state='running' WHERE id=?",(loop_id,))
            self.queue.add_chain(f"Generate VerifyConditioning {loop_id} attempt {n}",links,chain_id=chain)
            result=self.queue.run_until_idle()
            qlinks={x["id"]:x for x in result["links"] if x["chain_id"]==chain}
            gen=qlinks.get(gen_id); ver=qlinks.get(ver_id)
            with self._db() as db:
                if gen is None or gen["state"]!="done":
                    terminal="cancelled" if gen and gen["state"]=="cancelled" else "provider_failure"
                    db.execute("UPDATE attempts SET state=?,error=? WHERE loop_id=? AND number=?",(terminal,(gen or {}).get("error","missing generate result"),loop_id,n))
                    db.execute("UPDATE loops SET state=?,error=?,next_attempt=?,cumulative=cumulative+? WHERE id=?",
                               (terminal,(gen or {}).get("error","missing generate result"),n+1,config["estimate"],loop_id))
                    continue
                out=gen["artifact_id"]
                if ver is None or ver["state"]!="done":
                    terminal="cancelled" if ver and ver["state"]=="cancelled" else "verification_failed"
                    db.execute("UPDATE attempts SET state=?,outputs=?,error=? WHERE loop_id=? AND number=?",(terminal,json.dumps({"generated_sequence":out}),(ver or {}).get("error","missing verification result"),loop_id,n))
                    db.execute("UPDATE loops SET state=?,last_output=?,next_attempt=?,cumulative=cumulative+? WHERE id=?",(terminal,out,n+1,config["estimate"],loop_id))
                    continue
                report=json.loads(self.queue.store.get(ver["artifact_id"]))
                verdict=report.get("score_card",{}).get("verdict",report.get("verdict",""))
                state="pass" if verdict=="PASS" else "failed_verification"
                db.execute("UPDATE attempts SET state=?,outputs=?,verdict=? WHERE loop_id=? AND number=?",
                    (state,json.dumps({"generated_sequence":out,"verification_report":ver["artifact_id"]}),verdict,loop_id,n))
                next_loop_state = "pass" if verdict=="PASS" else "queued"
                db.execute("UPDATE loops SET state=?,last_output=?,next_attempt=?,cumulative=cumulative+? WHERE id=?",
                    (next_loop_state,out,n+1,config["estimate"],loop_id))
                # Session reference pins every successful sequence through subsequent failures.
                ids=[x["outputs"] for x in db.execute("SELECT outputs FROM attempts WHERE loop_id=?",(loop_id,))]
                pinned=[]
                for item in ids:
                    try: pinned.append(json.loads(item).get("generated_sequence"))
                    except ValueError: pass
                self.queue.store.reference("session:conditioning-loop:"+loop_id,[x for x in pinned if x])
            if verdict=="PASS": return self.snapshot(loop_id)

    def snapshot(self, loop_id):
        with self._db() as db:
            loop=db.execute("SELECT * FROM loops WHERE id=?",(loop_id,)).fetchone()
            if loop is None: raise LoopError("unknown loop")
            attempts=[dict(x) for x in db.execute("SELECT * FROM attempts WHERE loop_id=? ORDER BY number",(loop_id,))]
        for row in attempts:
            row["inputs"]=json.loads(row["inputs"]); row["outputs"]=json.loads(row["outputs"]); row["options"]=json.loads(row["options"])
            row["seed"]=row["options"].get("seed",0)
        return {**dict(loop),"config":json.loads(loop["config"]),"attempts":attempts}

    def list_snapshots(self):
        """Return durable loop history newest-first for queue-panel restoration."""
        with self._db() as db:
            ids = [row[0] for row in db.execute("SELECT id FROM loops ORDER BY rowid DESC")]
        return [self.snapshot(loop_id) for loop_id in ids]


def generate_task(options, input_ids, progress, cancelled):
    """Run the installed local deterministic provider on the supplied exported artifacts."""
    from .artifacts import ArtifactStore
    from .generative import generate
    store=ArtifactStore(options.get("_artifact_root"))
    if cancelled.is_set(): raise RuntimeError("cancelled")
    scene_id, bundle_id = input_ids[:2]
    with tempfile.TemporaryDirectory(prefix="nodebased-loop-generate-") as temporary:
        root=Path(temporary); scene_dir=root/"scene"; bundle_dir=root/"bundle"; previous_dir=root/"previous"
        scene_dir.mkdir(); bundle_dir.mkdir(); previous_dir.mkdir()
        scene_path=_unpack_artifact(store.get(scene_id),scene_dir)
        manifest_path=_unpack_artifact(store.get(bundle_id),bundle_dir)
        scene_path.with_suffix(scene_path.suffix+".artifact.json").write_text(json.dumps({"artifact_id":scene_id}))
        manifest_path.with_suffix(".artifact.json").write_text(json.dumps({"artifact_id":bundle_id}))
        reference_frames=options.get("reference_frames") or []
        if reference_frames and "reference_frames" in options.get("feedback_controls", ()):
            raise ValueError("choose either explicit reference frames or loop feedback")
        if "reference_frames" in options.get("feedback_controls", ()):
            if len(input_ids)<3: raise ValueError("reference-frame feedback artifact is missing")
            _unpack_artifact(store.get(input_ids[2]),previous_dir)
            reference_frames=sorted(str(path) for path in previous_dir.glob("*.exr"))
            if not reference_frames: raise ValueError("reference-frame feedback artifact has no EXR frames")
        pattern=options.get("output_pattern") or str(root/"generated.####.exr")
        from .workers import Job, Worker
        worker_options = {"manifest_path": str(manifest_path), "scene_state_path": str(scene_path),
            "output_pattern": pattern, "text": options.get("text"),
            "reference_frames": reference_frames or None,
            "seed": options.get("seed", 0), "sampler_settings": options.get("sampler_settings")}
        worker = Worker(Job(options["provider_id"], bundle_id, worker_options))
        watcher_done = threading.Event()
        def watch_cancel():
            while not watcher_done.wait(.05):
                if cancelled.is_set(): worker.cancel(); return
        watcher = threading.Thread(target=watch_cancel, daemon=True)
        watcher.start()
        try:
            result = worker.run(lambda message: progress(message.get("fraction", 0), message.get("text", "Working"))
                                if message.get("type") == "progress" else None)
        finally:
            watcher_done.set(); watcher.join(timeout=.2)
        if result.get("type") != "result":
            detail = result.get("error", "Provider worker failed")
            if result.get("worker_log_id"):
                detail += f"; worker log artifact {result['worker_log_id']}"
            if result.get("log_tail"):
                detail += f"; log: {result['log_tail']}"
            raise RuntimeError(detail)
        result = result.get("result", result)
        if cancelled.is_set(): raise RuntimeError("cancelled")
        return {"artifact_id":result["artifact_id"], "ignored":list(result.get("ignored", []))}


def verify_task(options, input_ids, progress, cancelled):
    """Verify generated frames against their exact SceneState with the CPU score-card verifier.

    The verifier is given beauty and every pass the sequence carries (depth, motion, IDs, normals)
    together with the exported reference for it. A check whose pass is absent is labelled
    "not conditioned" and left out of the verdict.
    """
    from .artifacts import ArtifactStore
    from .conditioning_verify import verify_conditioning
    from .conditioned_read import _read_rgba, read_generated_passes
    from .generative import GENERATION_MANIFEST
    from .scene_state import read_scene_state
    store=ArtifactStore(options.get("_artifact_root"))
    scene_id=options["scene_state_id"]; bundle_id=options["control_bundle_id"]
    generated_id=input_ids[-1]
    if cancelled.is_set(): raise RuntimeError("cancelled")
    with tempfile.TemporaryDirectory(prefix="nodebased-loop-verify-") as temporary:
        root=Path(temporary); scene_dir=root/"scene"; sequence_dir=root/"sequence"; bundle_dir=root/"bundle"
        scene_dir.mkdir(); sequence_dir.mkdir(); bundle_dir.mkdir()
        scene_path=_unpack_artifact(store.get(scene_id),scene_dir)
        _unpack_artifact(store.get(generated_id),sequence_dir)
        scene_path.with_suffix(scene_path.suffix+".artifact.json").write_text(json.dumps({"artifact_id":scene_id}))
        state=read_scene_state(scene_path)
        frames=[int(row["frame"]) for row in state["frames"]]
        files=sorted(sequence_dir.glob("*.exr"))
        if len(files)!=len(frames): raise ValueError("generated sequence frame count does not match SceneState")
        observations={}; carried=set()
        from .control_bundle import read_control_bundle
        manifest=_unpack_artifact(store.get(bundle_id),bundle_dir)
        manifest.with_suffix(".artifact.json").write_text(json.dumps({"artifact_id":bundle_id}))
        references={c.frame:c for c in read_control_bundle(manifest,scene_path)}
        for frame,path in zip(frames,files):
            observations[frame]={"beauty":_read_rgba(path)[0]}
            passes=read_generated_passes(path)
            if passes:
                _attach_passes(observations[frame],passes,references[frame],
                               [row for row in state["frames"] if int(row["frame"])==frame][0])
            carried.update(passes)
        missing=sorted(("objects" if name=="ids" else name) for name in {"depth","motion","normals","ids"}-carried)
        # A provider that does not declare camera pose honoured is not held to the camera tolerance.
        honoured=_manifest_honoured(sequence_dir/GENERATION_MANIFEST)
        if honoured is not None and "camera_pose" not in honoured: missing.append("camera")
        for row in observations.values(): row["not_conditioned"]=missing
        determinism=_determinism(sequence_dir/GENERATION_MANIFEST)
        temporal=temporal_measure(frames,{frame:row["beauty"] for frame,row in observations.items()},references)
        report_path=root/"verification.json"
        report=verify_conditioning(scene_path,observations,report_path,
            artifact_ids=[bundle_id,generated_id],artifact_store=store,determinism=determinism,temporal=temporal,
            survey=options.get("survey"))
        if cancelled.is_set(): raise RuntimeError("cancelled")
        return {"artifact_id":report["artifact_id"]}


def temporal_measure(frames, generated, references):
    """The flicker measure of a generated sequence (`nodebased.temporal.flicker`) against its ControlBundle.

    Each generated frame N-1 is warped into frame N with N's exported backward motion, occlusions and changed
    objects are masked with the exported depth and IDs, and the colour difference is taken in display sRGB.
    """
    import numpy as np
    from . import color, temporal
    to_srgb=color.processor(color.WORKING,color.INPUT_SPACES["sRGB"])
    def to_display(rgb):
        rgba=np.ones((*np.shape(rgb)[:-1],4),np.float32); rgba[...,:3]=rgb
        flat=np.ascontiguousarray(rgba.reshape(-1,4)); to_srgb.applyRGBA(flat)
        return np.clip(flat[:,:3].reshape(rgba[...,:3].shape),0.0,1.0)*255.0
    controls={c.frame:{"motion_backward":c.motion_backward.values,"depth":c.depth.values,"ids":c.object_ids.values}
              for c in references.values()}
    present=[frame for frame in frames if frame in controls and frame in generated]
    return temporal.flicker(present,generated,controls,to_display=to_display)


def _manifest_honoured(manifest_path):
    """The controls the provider declared honoured, from the sequence's manifest (None when it has none)."""
    if not manifest_path.is_file(): return None
    honoured=json.loads(manifest_path.read_text(encoding="utf-8")).get("honoured")
    return None if honoured is None else list(honoured)


def _determinism(manifest_path):
    """The determinism section of a verification report, read from the sequence's manifest."""
    from .generative import ProviderDescription
    if not manifest_path.is_file():
        return {"provider": None, "version": None, "seed": None, "sampler_settings": None,
                "seed_honoured": None, "note": "sequence has no generation manifest"}
    data=json.loads(manifest_path.read_text(encoding="utf-8"))
    return {"provider": data.get("provider"), "version": data.get("version"), "seed": data.get("seed"),
            "sampler_settings": data.get("sampler_settings"), "seed_honoured": bool(data.get("seed_honoured"))}


def _attach_passes(observation, passes, reference, frame_row):
    """Turn generated passes and the exported reference into verifier observations."""
    import numpy as np
    if "depth" in passes:
        generated=passes["depth"][...,0]; exported=np.asarray(reference.depth.values)[...,0]
        valid=np.argwhere(np.isfinite(exported)&(exported>0)&np.isfinite(generated))
        step=max(1,len(valid)//64)
        observation["depth_landmarks"]={f"{y},{x}":{"pixel":(int(x),int(y)),"expected":float(exported[y,x]),
                                                    "observed":float(generated[y,x])} for y,x in valid[::step]}
    if "motion" in passes:
        observation["motion_observed"]=passes["motion"]
        observation["motion_reference"]=np.asarray(reference.motion_forward.values)[...,:2]
    if "normals" in passes:
        observation["normals"]=passes["normals"]
        observation["normals_reference"]=np.asarray(reference.normals.values)[...,:3]
    if "ids" in passes:
        bits=np.ascontiguousarray(passes["ids"][...,0]).view(np.uint32)
        observation["object_ids"]={row["id"]:bits==int(row["id"],16) for row in frame_row["objects"]}


def _unpack_artifact(data, destination):
    """Extract a content package after rejecting paths outside its temporary directory."""
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names=[]
        for info in archive.infolist():
            path=Path(info.filename)
            if path.is_absolute() or ".." in path.parts or len(path.parts)!=1:
                raise ValueError("conditioning artifact contains an unsafe path")
            target=destination/path.name
            if not info.is_dir(): target.write_bytes(archive.read(info)); names.append(target)
    json_files=[path for path in names if path.suffix==".json" and path.name!="manifest.json"]
    if json_files: return json_files[0]
    manifest=destination/"manifest.json"
    if manifest.is_file(): return manifest
    sequences=[p for p in names if p.suffix==".exr"]
    if sequences: return sequences[0]
    raise ValueError("conditioning artifact has no recognized files")
