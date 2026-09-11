"""Verify the frame decoder that `qa_exr_playback.py` builds all of its verdicts on.

If the decoder is wrong, every playback number is worthless. Scrub the viewer to known
frames and confirm the decoded value matches what was requested. Run this before
trusting a playback measurement.

    xvfb-run -a -s "-screen 0 1920x1080x24" uv run python tests/manual/qa_decoder_check.py /tmp/exrqa
"""
import sys, time, uuid, pathlib
from PySide6.QtWidgets import QApplication

APP = QApplication.instance() or QApplication([])
APP.setStyle('Fusion')

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from nodebased.app import Window
from qa_exr_playback import decode_frame, generate, FRAMES

root = sys.argv[1]
generate(root, 'small')

window = Window(agent_name='qa-' + uuid.uuid4().hex)
window.show()
deadline = time.monotonic() + 20
while window.frame is None and time.monotonic() < deadline:
    APP.processEvents()

window.dispatcher.execute({'op': 'create', 'id': 'read', 'type': 'Read',
                           'params': {'path': f'{root}/small/plate.####.exr',
                                      'missing': 'error', 'frame_offset': 0}})
window.dispatcher.execute({'op': 'view', 'id': 'read'})

failures = 0
for frame in (1, 3, 7, 11, 12):
    window.set_time(first=1, last=FRAMES, current=frame)
    window.request_preview()
    decoded, deadline = None, time.monotonic() + 30
    while time.monotonic() < deadline:
        APP.processEvents()
        if window.frame is not None and window.frame_generation == window.generation:
            decoded = decode_frame(window.frame)
            if decoded is not None:
                break
    failures += decoded != frame
    print(f'  scrubbed to {frame:2d} -> decoder read {decoded}  '
          f'[{"ok" if decoded == frame else "MISMATCH"}]')

window.saved_document = window.dispatcher.document
window.close()
APP.processEvents()
print('decoder check:', 'PASS' if not failures else f'FAIL ({failures} mismatches)')
sys.exit(1 if failures else 0)
