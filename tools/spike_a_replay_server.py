"""Replays recorded step previews with the recorded timing, speaking `sdxl_controlnet.py serve`'s line protocol (no card needed).

    python tools/spike_a_replay_server.py RECORDED_PREVIEW_DIR RECORDED_EVIDENCE_DIR

Used by `spike_a_interactive.py --replay` to measure the viewer's own responsiveness (event-loop gaps, paint cost) on a
machine where the card is busy: the pictures are the ones a real run made, the schedule is the real run's medians (first
preview 3.99 s after the edit, 0.31 s per step, final at 7.95 s). Supersession works like the real server's.
"""
import glob
import json
import os
import shutil
import sys
import threading
import time

import numpy as np

recorded, evidence = sys.argv[1], sys.argv[2]
SCHEDULE = [3.99, 4.30, 4.61, 4.92]
FINAL = 7.95
state = {"pending": None, "quit": False}
lock, wake = threading.Lock(), threading.Event()


def reader():
    for line in sys.stdin:
        request = json.loads(line)
        with lock:
            if request["op"] == "quit":
                state["quit"] = True
            else:
                if state["pending"] is not None:
                    print("coalesced " + json.dumps({"id": state["pending"]["id"], "revision": state["pending"]["revision"]}), flush=True)
                state["pending"] = request
        wake.set()
    state["quit"] = True
    wake.set()


threading.Thread(target=reader, daemon=True).start()
print("ready " + json.dumps({"replay": True}), flush=True)
while True:
    wake.wait()
    with lock:
        request, state["pending"] = state["pending"], None
        if request is None:
            wake.clear()
            if state["quit"]:
                break
            continue
    job = json.load(open(request["job"]))
    tags = {"id": request["id"], "revision": request["revision"]}
    began = time.monotonic()
    print("started " + json.dumps({**tags, "mono": began, "queued_seconds": 0.0}), flush=True)
    cancelled = False
    for row in job["frames"]:
        frame = row["frame"]
        for step in range(1, 5):
            while time.monotonic() - began < SCHEDULE[step - 1]:
                time.sleep(0.002)
                with lock:
                    if state["quit"] or (state["pending"] is not None and state["pending"]["revision"] > request["revision"]):
                        cancelled = True
                if cancelled:
                    break
            if cancelled:
                break
            sources = sorted(glob.glob(os.path.join(recorded, f"r*_f{frame:04d}_text2img_{step}of4.npy")))
            array = np.load(sources[-1])
            path = os.path.join(job["preview_dir"], f"r{request['revision']:04d}_f{frame:04d}_text2img_{step}of4.npy")
            np.save(path, array)
            print("preview " + json.dumps({**tags, "frame": frame, "stage": "text2img", "step": step, "of": 4, "path": path,
                  "decode_seconds": 0.17, "done": time.monotonic()}), flush=True)
        if cancelled:
            break
        while time.monotonic() - began < FINAL:
            time.sleep(0.002)
        shutil.copyfile(os.path.join(evidence, f"frame{frame:02d}_delivered.npy"), row["result"])
        print("final " + json.dumps({**tags, "frame": frame, "path": row["result"], "done": time.monotonic()}), flush=True)
    print(("cancelled " if cancelled else "done ") + json.dumps({**tags, "mono": time.monotonic()}), flush=True)
