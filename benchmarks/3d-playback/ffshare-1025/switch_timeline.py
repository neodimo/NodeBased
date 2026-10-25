"""The timeline of the frame where play returns to full size after the half-size start frames.

usage: switch_timeline.py events.json [--all]

Reads the raw playback trace (`tools/benchmark_3d_playback.py --events`), splits it into worker frames at `worker_start`, finds the
size each frame rendered at (stamps of the worker thread only) (the `size WxH` stamp) and prints, for the last half-size frame and the first full-size frame after it,
the time between consecutive stamps, so the stage that holds the repack shows by its name. Stamps under 0.5 ms are left out
unless --all is given.
"""
import json
import sys


def frames(events):
    out, cur = [], None
    for t, thread, label in events:
        if label == 'worker_start':
            cur = {'t': t, 'thread': thread, 'marks': [], 'size': None}
            out.append(cur)
        elif cur is not None and thread == cur['thread']:      # the worker's own stamps; the window thread's are another story
            cur['marks'].append((t, label))
            if label.startswith('size '):
                cur['size'] = label[5:]
    return out


def switch_frames(events):
    fs = [f for f in frames(events) if f['size']]
    for a, b in zip(fs, fs[1:]):
        wa, wb = [int(v) for v in a['size'].split('x')], [int(v) for v in b['size'].split('x')]
        if wb[0] > wa[0]:
            return a, b
    return None, None


def describe(frame):
    prev, rows = frame['t'], []
    for t, label in frame['marks']:
        rows.append((round((t - prev) * 1000, 1), label))
        prev = t
    last = frame['marks'][-1][0] if frame['marks'] else frame['t']
    return round((last - frame['t']) * 1000, 1), rows


def main(argv):
    events = json.load(open(argv[1]))
    before, switch = switch_frames(events)
    if switch is None:
        print('no switch frame')
        return 1
    for name, frame in (('last small frame', before), ('switch frame', switch)):
        total, rows = describe(frame)
        print(f"{name} ({frame['size']}) {total} ms")
        print('  ' + ', '.join(f'{label} {ms}' for ms, label in rows if ms >= 0.5 or '--all' in argv))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
