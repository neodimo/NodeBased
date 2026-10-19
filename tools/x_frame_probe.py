"""Reads the frame code that `viewer_direct_screen.py --playback` draws, straight from an X window, as fast as it can.

Runs under a Python that has python-xlib (the system one here; the project venv does not carry it):

    python3 tools/x_frame_probe.py WINDOW_ID SECONDS CELL_PX BITS

Prints one JSON line per change of the code: {"t": time.time(), "code": n} (or "code": null when the strip
does not read as a code). The strip is BITS cells of CELL_PX x CELL_PX along the window's top-left edge;
a cell is a 1 when its centre pixel is bright.
"""
import json
import sys
import time

from Xlib import X, display

window_id, seconds, cell, bits = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
d = display.Display()
window = d.create_resource_object("window", window_id)
width, y = cell * bits, cell // 2
last = object()
end = time.time() + seconds
samples = 0
while time.time() < end:
    raw = window.get_image(0, y, width, 1, X.ZPixmap, 0xFFFFFFFF).data
    samples += 1
    values = []
    for k in range(bits):
        x = k * cell + cell // 2
        b, g, r = raw[4 * x], raw[4 * x + 1], raw[4 * x + 2]
        level = (r + g + b) / 3
        values.append(1 if level > 180 else 0 if level < 75 else None)
    code = None if None in values else sum(v << k for k, v in enumerate(values))
    if code != last:
        print(json.dumps({"t": time.time(), "code": code}), flush=True)
        last = code
print(json.dumps({"samples": samples, "seconds": seconds}), flush=True)
