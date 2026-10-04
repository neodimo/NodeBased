import json
from pathlib import Path
import tempfile
import threading
import time
import unittest

from nodebased.artifacts import ArtifactStore
from nodebased.jobs import Queue
from nodebased.loops import LoopError, LoopRun


class LoopTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.root=Path(self.temp.name)
        self.store=ArtifactStore(self.root/"artifacts")
        self.queue=Queue(self.root/"jobs.sqlite",self.store,max_workers=1)
        self.loop=LoopRun(self.root/"loops.sqlite",self.queue,
            generate_operation="tests.loop_fixtures:generate",verify_operation="tests.loop_fixtures:verify")
        self.scene=self.store.put(b"scene","scene_state",{"producer":"fixture","inputs":[]})
        self.bundle=self.store.put(b"bundle","control_bundle",{"producer":"fixture","inputs":[]})
        self.provider=self.root/"provider.json"
        self.write_provider(True)

    def tearDown(self): self.queue.close(); self.temp.cleanup()

    def write_provider(self, references):
        controls=("depth","normals","motion","ids","camera_pose","text","reference_frames")
        accepts={name:{"accepted": name=="reference_frames" and references} for name in controls}
        data={"schema":1,"name":"fixture","version":"1.2","locality":"local","accepts":accepts,
            "limits":{"max_resolution":[64,64],"max_frames":2},"color_spaces":["ACEScg"],
            "returns":{"honoured":[name for name in controls if accepts[name]["accepted"]]}}
        self.provider.write_text(json.dumps(data))

    def create(self, **kw):
        args=dict(scene_state_id=self.scene,control_bundle_id=self.bundle,provider_id=str(self.provider),
            provider_options={"fail_verifications":1},max_attempts=3,max_estimated_spend=3,
            estimated_spend_per_attempt=.5,spend_unit="credits",feedback_controls=("reference_frames",))
        args.update(kw); return self.loop.create(**args)

    def test_pass_stops_after_one_and_persists_provenance_and_estimate(self):
        ident=self.create(provider_options={"fail_verifications":0})
        snap=self.loop.run(ident)
        self.assertEqual(snap["state"],"pass"); self.assertEqual(len(snap["attempts"]),1)
        attempt=snap["attempts"][0]
        self.assertEqual(attempt["inputs"],[self.scene,self.bundle]); self.assertEqual(attempt["verdict"],"PASS")
        self.assertEqual(attempt["spend_unit"],"credits"); self.assertEqual(snap["cumulative"],.5)
        output=attempt["outputs"]["generated_sequence"]
        self.assertIn(self.scene,[r["id"] for r in self.store.provenance(output)])
        self.assertIn(self.bundle,[r["id"] for r in self.store.provenance(output)])

    def test_failed_score_schedules_second_only_with_supported_reference_feedback(self):
        ident=self.create(); snap=self.loop.run(ident)
        self.assertEqual(snap["state"],"pass"); self.assertEqual([a["state"] for a in snap["attempts"]],
            ["failed_verification","pass"])
        self.assertEqual(snap["attempts"][1]["inputs"], [self.scene,self.bundle,snap["attempts"][0]["outputs"]["generated_sequence"]])
        self.write_provider(False)
        other=self.create(loop_id="unsupported",max_attempts=3)
        blocked=self.loop.run(other)
        self.assertEqual(blocked["state"],"feedback_unsupported"); self.assertEqual(len(blocked["attempts"]),1)

    def test_attempt_and_budget_caps_prevent_excess_dispatch(self):
        ident=self.create(max_attempts=1); snap=self.loop.run(ident)
        self.assertEqual(snap["state"],"attempts_exhausted"); self.assertEqual(len(snap["attempts"]),1)
        other=self.create(loop_id="budget",max_attempts=5,max_estimated_spend=.5)
        snap=self.loop.run(other)
        self.assertEqual(snap["state"],"budget_exhausted"); self.assertEqual(len(snap["attempts"]),1)
        with self.assertRaises(LoopError): self.create(loop_id="bad",max_attempts=0)
        with self.assertRaises(LoopError): self.create(loop_id="missing",estimated_spend_per_attempt=None)

    def test_reopen_keeps_full_history_and_failing_second_keeps_first_output(self):
        ident=self.create(provider_options={"fail_verifications":1,"fail_attempt":2},max_attempts=4)
        snap=self.loop.run(ident)
        self.assertEqual(snap["state"],"provider_failure"); self.assertEqual(len(snap["attempts"]),2)
        first=snap["attempts"][0]["outputs"]["generated_sequence"]
        self.assertEqual(self.store.meta(first)["kind"],"generated_sequence")
        reopened=LoopRun(self.root/"loops.sqlite",self.queue,
            generate_operation="tests.loop_fixtures:generate",verify_operation="tests.loop_fixtures:verify")
        again=reopened.snapshot(ident)
        self.assertEqual(len(again["attempts"]),2); self.assertEqual(again["attempts"][0]["outputs"]["generated_sequence"],first)

    def test_reopen_reuses_completed_queue_attempt_with_same_artifact_ids(self):
        ident=self.create(provider_options={"fail_verifications":0})
        run_queue=self.queue.run_until_idle
        def interrupted():
            run_queue()
            raise RuntimeError("simulated process stop after queue completion")
        self.queue.run_until_idle=interrupted
        with self.assertRaisesRegex(RuntimeError,"simulated process stop"):
            self.loop.run(ident)
        before=self.loop.snapshot(ident)["attempts"][0]
        inputs=list(before["inputs"])
        output=next(row for row in self.queue.snapshot()["links"] if row["name"]=="Generate")["artifact_id"]
        self.queue.close()
        self.queue=Queue(self.root/"jobs.sqlite",self.store,max_workers=1)
        reopened=LoopRun(self.root/"loops.sqlite",self.queue,
            generate_operation="tests.loop_fixtures:generate",verify_operation="tests.loop_fixtures:verify")
        done=reopened.run(ident)
        self.assertEqual(done["state"],"pass"); self.assertEqual(len(done["attempts"]),1)
        self.assertEqual(done["attempts"][0]["inputs"],inputs)
        self.assertEqual(done["attempts"][0]["outputs"]["generated_sequence"],output)
        generated=next(row for row in self.queue.snapshot()["links"] if row["name"]=="Generate")
        self.assertEqual(generated["attempts"],1)

    def test_cancel_running_attempt_prevents_later_attempts(self):
        ident=self.create(provider_options={"sleep":1,"fail_verifications":0})
        worker=threading.Thread(target=lambda:self.loop.run(ident)); worker.start()
        deadline=time.monotonic()+3
        while time.monotonic()<deadline:
            snap=self.loop.snapshot(ident)
            if snap["attempts"]: break
            time.sleep(.01)
        self.loop.cancel(ident); worker.join(3)
        self.assertFalse(worker.is_alive()); self.assertEqual(self.loop.snapshot(ident)["state"],"cancelled")
        self.assertEqual(len(self.loop.snapshot(ident)["attempts"]),1)


if __name__=="__main__": unittest.main()
