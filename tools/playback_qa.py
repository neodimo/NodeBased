"""Native-display playback QA: real Window on a real display, 4K EXR sequence.

Usage: QT_QPA_PLATFORM=xcb DISPLAY=:0 .venv/bin/python tools/playback_qa.py [VIEW] [SECONDS] [--plate PATTERN]
Prints one JSON line with drawn-frame counts, render times and the display backend seen in status.

Promoted from scratch/v016-qa/playback_qa.py (gitignored) for v0.17 playback-perf QA. Behaviour is
unchanged: local QEventLoop (app.quit() would hit the confirm_discard modal because the harness
adds unsaved nodes), collision-free node ids, and the optional NODEBASED_QA_SOURCE for comparing
another checkout (e.g. the v0.16.0 baseline) against this repo's own 4K plate.
"""
import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path

from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtWidgets import QApplication

ROOT = Path(__file__).resolve().parents[1]
# NODEBASED_QA_SOURCE runs the same harness against another checkout (e.g. the v0.16.0 baseline),
# while the 4K plate always comes from this repository root unless --plate overrides it.
SOURCE = Path(__import__("os").environ.get("NODEBASED_QA_SOURCE", str(ROOT)))
sys.path.insert(0, str(SOURCE))

from nodebased.app import Window  # noqa: E402

try:
    from nodebased import gpudisplay  # noqa: E402
except ImportError:  # pre-0.16 checkouts have no GPU display module
    gpudisplay = None

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("view", nargs="?", default="ACES 2.0")
parser.add_argument("seconds", nargs="?", type=float, default=8.0)
parser.add_argument("--plate", default=str(ROOT / "noise_test_4k.####.exr"),
                    help="Sequence pattern to play (default: this repo's 4K noise test sequence)")
parser.add_argument("--full", action="store_true",
                    help="Uncheck 'Proxy while playing' so playback stays at the Full/tier-1 "
                         "resolution the artist selected, instead of the standard auto-proxy.")
args = parser.parse_args()

app = QApplication.instance() or QApplication([])
window = Window()
window.resize(1600, 1000)
window.show()

window.agent_command({"op": "batch", "commands": [
    {"op": "create", "type": "Read", "id": "qa_plate_4k", "params": {"path": args.plate}},
    {"op": "view", "id": "qa_plate_4k"},
]})
window.set_time(first=1, last=100, current=1, fps=24.0)
window.display_view.setCurrentText(args.view)
if args.full:
    window.playback_proxy.setChecked(False)

drawn = []
statuses = []


def on_finished(payload, frame, image, status, render_region=None):
    request, _cancel = payload
    statuses.append(status)
    if image is not None and frame is not None:
        drawn.append((time.perf_counter(), request.frame, status))


window.signals.finished.connect(on_finished)

# Let the first frame render before the transport starts, as an artist would.
first_ready = time.perf_counter()
deadline = first_ready + 30
while not drawn and time.perf_counter() < deadline:
    app.processEvents()
    time.sleep(0.01)
first_frame_s = time.perf_counter() - first_ready
drawn.clear()

start = time.perf_counter()
window.toggle_playback(True)
# A local loop: app.quit() would close the window mid-run, and closeEvent's unsaved-changes
# dialog (the harness just added nodes) would then block forever with nobody to click it.
loop = QEventLoop()
QTimer.singleShot(int(args.seconds * 1000), loop.quit)
loop.exec()
window.toggle_playback(False)
elapsed = time.perf_counter() - start

render_ms = []
hits = 0
for _t, _frame, status in drawn:
    if "display cache hit" in status:
        hits += 1
        continue
    match = re.search(r"·\s+(\d+) ms", status)
    if match:
        render_ms.append(int(match.group(1)))
frames = [f for _t, f, _s in drawn]
backend = next((s.split("display ")[-1] for s in reversed(statuses) if "  ·  display " in s), None)
proxy_used = any("proxy 1/" in s for s in statuses)
gaps = [b[0] - a[0] for a, b in zip(drawn, drawn[1:])]
result = {
    "view": args.view,
    "plate": args.plate,
    "seconds": round(elapsed, 2),
    "first_frame_s": round(first_frame_s, 2),
    "drawn": len(drawn),
    "distinct_frames": len(set(frames)),
    "drawn_fps": round(len(drawn) / elapsed, 2) if elapsed else None,
    "in_order": all(b >= a for a, b in zip(frames, frames[1:])),
    "longest_gap_s": round(max(gaps), 3) if gaps else None,
    "render_ms_median_uncached": statistics.median(render_ms) if render_ms else None,
    "display_cache_hits": hits,
    "backend_status": backend,
    "playback_proxy_used": proxy_used,
    "gpudisplay_status": gpudisplay.status() if gpudisplay else None,
    "errors": [s for s in statuses if s and "ms" not in s][:5],
}
print(json.dumps(result), flush=True)
window.confirm_discard = lambda *args, **kwargs: True
window.close()
