"""Manual QA: does the viewer actually play an EXR sequence, and how fast?

Why this exists
---------------
`tests/test_desktop.py::SlowPlaybackTests` proves the *transport rule* (a tick must
not cancel the render in flight) by injecting `time.sleep` into `Evaluator.evaluate`.
That is not the tile path the viewer really uses, and trusting it produced a wrong
public claim once already — "the 1600px case displays 0 frames" was an artifact of the
injected sleep, not of real rendering. See docs/PLAYBACK.md, "Corrected measurement."

This harness measures the real thing:

* real linear float32 EXR sequences, generated here, not PNG stand-ins
* a real `Window` on a real X server (run it under `xvfb-run`), not `QT_QPA_PLATFORM=offscreen`
* frame identity decoded from the **displayed pixels**, so the transport cannot be
  credited for a frame it never actually drew

Each generated plate encodes ``phase = (frame - 1) / FRAMES`` in its blue channel as
``blue_premultiplied = (0.02 + phase * 0.5) * alpha``; unpremultiplying any pixel with
meaningful alpha recovers the source frame number. `qa_decoder_check.py` verifies that
decoder against known scrub positions — run it first, or the numbers here mean nothing.

Usage
-----
    xvfb-run -a -s "-screen 0 1920x1080x24" uv run python tests/manual/qa_exr_playback.py /tmp/exrqa
    xvfb-run -a -s "-screen 0 1920x1080x24" uv run python tests/manual/qa_exr_playback.py /tmp/exrqa --uhd

To measure an older build, point PYTHONPATH at a worktree of that tag and run this same
file, so both sides of the comparison use identical measurement code:

    git worktree add /tmp/v091 v0.9.1
    xvfb-run -a env PYTHONPATH=/tmp/v091 .venv/bin/python tests/manual/qa_exr_playback.py /tmp/exrqa --uhd
"""
import sys, time, uuid, pathlib
import numpy as np
from PySide6.QtWidgets import QApplication

APP = QApplication.instance() or QApplication([])
APP.setStyle('Fusion')

from nodebased.app import Window
from nodebased.media import write_exr

FRAMES = 12
SIZES = {'small': (512, 512), 'big': (1600, 1600), 'uhd': (3840, 2160)}


def generate(root, name):
    """Write a linear float32 EXR sequence with premultiplied alpha and real HDR values."""
    width, height = SIZES[name]
    folder = pathlib.Path(root) / name
    folder.mkdir(parents=True, exist_ok=True)
    if len(list(folder.glob('*.exr'))) >= FRAMES:
        return folder
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    for frame in range(1, FRAMES + 1):
        phase = (frame - 1) / FRAMES
        rgba = np.zeros((height, width, 4), np.float32)
        bar = ((xx / width + phase) % 1.0 < 0.12).astype(np.float32)
        rgba[..., 0] = 0.05 + bar * 8.0 + phase * 0.3      # above 1.0 on purpose: real linear HDR
        rgba[..., 1] = 0.18 * (yy / max(height - 1, 1)) + phase * 0.4
        rgba[..., 2] = 0.02 + phase * 0.5                  # the frame tag the decoder reads
        alpha = np.clip(0.25 + 0.75 * (xx / max(width - 1, 1)), 0, 1).astype(np.float32)
        rgba[..., 3] = alpha
        rgba[..., :3] *= alpha[..., None]                  # premultiplied, matching EXR convention
        write_exr(folder / f'plate.{frame:04d}.exr', rgba)
    return folder


def decode_frame(pixels):
    """Recover the source frame number from displayed premultiplied float32 RGBA."""
    if pixels is None:
        return None
    alpha = pixels[..., 3]
    usable = alpha > 0.5
    if not usable.any():
        return None
    blue = pixels[..., 2][usable] / alpha[usable]
    phase = (float(np.median(blue)) - 0.02) / 0.5
    return int(round(phase * FRAMES + 1))


def run(pattern, label, seconds=6.0, blur=False, fps=24.0):
    window = Window(agent_name='qa-' + uuid.uuid4().hex)
    window.show()
    deadline = time.monotonic() + 20
    while window.frame is None and time.monotonic() < deadline:
        APP.processEvents()

    document = window.dispatcher
    document.execute({'op': 'create', 'id': 'read', 'type': 'Read',
                      'params': {'path': str(pattern), 'missing': 'error', 'frame_offset': 0}})
    target = 'read'
    if blur:
        document.execute({'op': 'create', 'id': 'blur', 'type': 'Blur', 'params': {'radius': 24.0}})
        document.execute({'op': 'connect', 'id': 'blur', 'input': 'image', 'source': 'read'})
        target = 'blur'
    document.execute({'op': 'view', 'id': target})
    window.set_time(first=1, last=FRAMES, current=1, fps=fps)
    window.request_preview()

    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        APP.processEvents()
        if window.frame is not None and decode_frame(window.frame) is not None:
            break

    shown, timeline, stamps = [], [], []
    last_generation = window.frame_generation
    window.toggle_playback(True)
    start = time.monotonic()
    while time.monotonic() < start + seconds:
        APP.processEvents()
        timeline.append(window.dispatcher.document['time']['current'])
        if window.frame_generation != last_generation:
            last_generation = window.frame_generation
            decoded = decode_frame(window.frame)
            if decoded is not None:
                shown.append(decoded)
                stamps.append(time.monotonic() - start)
    window.toggle_playback(False)
    APP.processEvents()
    window.saved_document = window.dispatcher.document
    window.close()
    APP.processEvents()

    # Counting distinct frames hides the reported failure, which is a viewer that stops
    # updating while the timeline runs on. The load-bearing number is the longest interval
    # with no update at all.
    gaps = [b - a for a, b in zip(stamps, stamps[1:])] or [seconds]
    tail = seconds - stamps[-1] if stamps else seconds
    worst = max(max(gaps), tail)
    distinct = sorted(set(shown))
    print(f'--- {label} ---')
    print(f'  timeline advanced          : {min(timeline)} -> {max(timeline)} '
          f'({len(set(timeline))} distinct positions)')
    print(f'  frames DISPLAYED (decoded) : {shown}')
    print(f'  display times (s)          : {[round(s, 2) for s in stamps]}')
    print(f'  distinct displayed         : {len(distinct)} of {FRAMES}  {distinct}')
    print(f'  sustained display rate     : {len(shown) / seconds:.1f} fps over {seconds:.0f}s')
    print(f'  longest gap with NO update : {worst:.2f}s')
    stalled = worst > max(1.5, seconds * 0.3)
    print(f'  VERDICT                    : {"STALLED" if stalled else "playing"}')
    print()
    return not stalled


if __name__ == '__main__':
    root = sys.argv[1]
    if '--uhd' in sys.argv:
        generate(root, 'uhd')
        run(f'{root}/uhd/plate.####.exr', '3840x2160 EXR + Blur (heavy)', seconds=12.0, blur=True)
    else:
        generate(root, 'small')
        generate(root, 'big')
        fast = run(f'{root}/small/plate.####.exr', '512px EXR, no blur (fast frames)')
        slow = run(f'{root}/big/plate.####.exr', '1600px EXR + Blur (slow frames)', blur=True)
        sys.exit(0 if (fast and slow) else 1)
