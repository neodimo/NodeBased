"""Frame rate at 1920 x 1080 on each graphics adapter (docs/PERFORMANCE.md, "Adapter choice").

    python tools/benchmark_adapter_choice.py [--adapters "RTX,Radeon,llvmpipe"] [--seconds 6] [--json out.json]
    python tools/benchmark_adapter_choice.py --case empty3d|cards3d|checker --seconds 6      (one case, in-process)

Each adapter and case is its own subprocess with NODEBASED_ADAPTER set to part of the adapter's name, under a turn of
/tmp/nb-gpu.lock. Cases:

* empty3d: an empty 3D scene rendered by the GPU renderer (what a 3D viewer pays before any geometry).
* cards3d: 3000 lit cards rasterized at 4 samples: a scene where the card matters.
* checker: a Checker -> Viewer graph at 1920 x 1080 played in the real window (drawn frames per second).
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def case_empty3d(seconds):
    from nodebased import gpu3d, scene3d
    scene, camera = scene3d.Scene(()), scene3d.Camera()
    gpu3d.render(scene, camera, 1920, 1080)
    times = []
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        start = time.perf_counter()
        gpu3d.render(scene, camera, 1920, 1080)
        times.append(time.perf_counter() - start)
    return dict(frames=len(times), fps=round(len(times) / sum(times), 2), median_ms=round(statistics.median(times) * 1000, 1))


def case_cards3d(seconds):
    """3000 lit cards at 4 samples: a scene where the card matters."""
    from nodebased import gpu3d, scene3d
    cards = tuple(scene3d._card(0.5, 0.5, (.2, .6, .9, 1), scene3d.Transform3D(scene3d.Vec3((i % 60) * .2 - 6, (i // 60) * .2 - 5, -(i % 7) * .3)))
                  for i in range(3000))
    scene, camera = scene3d.Scene(cards, (scene3d.Light(),)), scene3d.Camera()
    kwargs = dict(ambient=.2, samples=4)
    gpu3d.render(scene, camera, 1920, 1080, **kwargs)
    times = []
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        start = time.perf_counter()
        gpu3d.render(scene, camera, 1920, 1080, **kwargs)
        times.append(time.perf_counter() - start)
    return dict(frames=len(times), fps=round(len(times) / sum(times), 2), median_ms=round(statistics.median(times) * 1000, 1))


def case_checker(seconds):
    from PySide6.QtCore import QEventLoop, QTimer, Qt
    from PySide6.QtWidgets import QApplication
    from nodebased.app import Window
    app = QApplication.instance() or QApplication([])
    window = Window()
    window.resize(1600, 1000)
    window.show()
    window.agent_command({"op": "batch", "commands": [
        {"op": "create", "type": "Checker", "id": "qa_checker", "params": {"width": 1920, "height": 1080}},
        {"op": "view", "id": "qa_checker"},
        {"op": "set_key", "id": "qa_checker", "param": "size", "frame": 1, "value": 16},
        {"op": "set_key", "id": "qa_checker", "param": "size", "frame": 100, "value": 240}]})
    window.set_time(first=1, last=100, current=1, fps=24.0)
    drawn = []
    timer_ticks = []
    window.signals.finished.connect(lambda payload, frame, image, status, region=None:
                                    drawn.append(time.perf_counter()) if image is not None and frame is not None else None)
    end = time.perf_counter() + 30
    while not drawn and time.perf_counter() < end:
        app.processEvents()
        time.sleep(0.01)
    drawn.clear()
    original_tick = window.playback_tick
    window.playback_timer.timeout.disconnect(window.playback_tick)
    window.playback_timer.timeout.connect(lambda: (timer_ticks.append(time.perf_counter()), original_tick()))
    window.toggle_playback(True)
    loop = QEventLoop()
    sample = {}

    def finish_sample():
        sample.setdefault("start", time.perf_counter())
        sample["end"] = time.perf_counter()
        loop.quit()

    def begin_sample():
        if drawn:
            # The steady-state window is anchored at the first live picture, which is itself a
            # displayed frame at t=0. Starting the clock just after its signal drops that frame.
            sample["start"] = drawn[-1]
            sample["cutoff"] = sample["start"] + seconds
            sample_timer = QTimer(window)
            sample_timer.setSingleShot(True)
            sample_timer.setTimerType(Qt.TimerType.PreciseTimer)
            sample_timer.timeout.connect(finish_sample)
            sample["timer"] = sample_timer
            sample_timer.start(int(seconds * 1000))
        else:
            QTimer.singleShot(5, begin_sample)

    # Let the first live picture arrive before opening the measurement window. This excludes
    # playback-start latency and any in-flight frame that completes after the sample timer fires.
    QTimer.singleShot(0, begin_sample)
    QTimer.singleShot(int((seconds + 30) * 1000), finish_sample)
    loop.exec()
    elapsed = float(seconds)
    measured = [stamp for stamp in drawn if sample.get("start", float("inf")) <= stamp
                < sample.get("cutoff", 0.0)]
    measured_ticks = [stamp for stamp in timer_ticks if sample.get("start", float("inf")) <= stamp
                      < sample.get("cutoff", 0.0)]
    display_intervals = [(b - a) * 1000.0 for a, b in zip(measured, measured[1:])]
    timer_intervals = [(b - a) * 1000.0 for a, b in zip(measured_ticks, measured_ticks[1:])]
    window.toggle_playback(False)
    window.confirm_discard = lambda *a, **k: True
    window.close()
    fps = len(measured) / elapsed if measured and elapsed > 0 else 0.0
    return dict(frames=len(measured), fps=round(fps, 2), elapsed_s=round(elapsed, 3),
                display_cadence_fps=round((len(measured) - 1) /
                                          (measured[-1] - measured[0]), 2) if len(measured) > 1 else 0.0,
                display_interval_ms_p50=round(statistics.median(display_intervals), 3) if display_intervals else None,
                timer_ticks=len(measured_ticks), timer_interval_ms_p50=round(statistics.median(timer_intervals), 3)
                if timer_intervals else None,
                median_ms=None)


def run_case(case, adapter, seconds):
    from nodebased import gpu3d
    result = {"empty3d": case_empty3d, "cards3d": case_cards3d, "checker": case_checker}[case](seconds)
    result.update(case=case, adapter=gpu3d.chosen_adapter_name(), requested=adapter)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapters", default="RTX,Radeon,llvmpipe")
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--case")
    parser.add_argument("--json")
    args = parser.parse_args()
    if args.case:
        result = run_case(args.case, os.environ.get("NODEBASED_ADAPTER", ""), args.seconds)
        print("@@JSON@@" + json.dumps(result))
        if args.json:
            Path(args.json).write_text(json.dumps(result, indent=2) + "\n")
        return
    rows = []
    for adapter in args.adapters.split(","):
        for case in ("empty3d", "cards3d", "checker"):
            env = {**os.environ, "NODEBASED_ADAPTER": adapter, "PYTHONPATH": str(ROOT)}
            env.setdefault("QT_QPA_PLATFORM", "offscreen")
            command = ["flock", "-w", "1800", "/tmp/nb-gpu.lock", sys.executable, __file__, "--case", case,
                       "--seconds", str(args.seconds)]
            done = subprocess.run(command, env=env, capture_output=True, text=True, timeout=900)
            line = next((l for l in done.stdout.splitlines() if l.startswith("@@JSON@@")), None)
            row = json.loads(line[8:]) if line else dict(case=case, requested=adapter, error=(done.stderr or done.stdout)[-300:])
            print(json.dumps(row), flush=True)
            rows.append(row)
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
