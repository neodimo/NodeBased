"""Summarise a gate directory: <root>/<adapter label>/<scene>-<run>.json from tools/benchmark_3d_playback.py (play pressed 4 s
after the first picture, Proxy while playing on, 1080p, first 3 s).

usage: summarize_gate.py <root> [out.json]

Prints, for every adapter label and scene, the longest interval of each run in ms (runs in order), the runs over 54 ms and where
(ms after the play) the intervals over 54 ms fell, and writes the same as JSON.
"""
import json
import sys
from pathlib import Path

GATE_MS = 54.0


def run_row(path):
    data = json.loads(path.read_text())
    first = data.get('first_seconds') or {}
    cold = first.get('cold', first)
    shown = cold.get('shown_at_ms') or []
    intervals = cold.get('intervals_ms') or []
    late = [{'after_play_ms': shown[i + 1] if i + 1 < len(shown) else None, 'ms': v} for i, v in enumerate(intervals) if v > GATE_MS]
    return {'run': int(path.stem.rsplit('-', 1)[1]), 'max_ms': cold.get('interval_max_ms'), 'over_54': late,
            'shown': cold.get('shown'), 'fps': cold.get('fps')}


def main(argv):
    root = Path(argv[1])
    table = {}
    for label_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for path in sorted(label_dir.glob('*-[0-9].json')):
            if path.name.endswith('.events.json'):
                continue
            scene = path.stem.rsplit('-', 1)[0]
            table.setdefault(label_dir.name, {}).setdefault(scene, []).append(run_row(path))
    for label, scenes in table.items():
        for scene, rows in scenes.items():
            rows.sort(key=lambda r: r['run'])
            maxima = [r['max_ms'] for r in rows]
            over = [r['run'] for r in rows if r['max_ms'] is not None and r['max_ms'] > GATE_MS]
            print(f"{label:8s} {scene:11s} {len(rows)} runs, longest interval {min(maxima):.1f} to {max(maxima):.1f} ms, "
                  f"{len(over)} over {GATE_MS:.0f}: {maxima}")
            for r in rows:
                for late in r['over_54']:
                    print(f"           run {r['run']}: {late['ms']} ms ending {late['after_play_ms']} ms after the play")
    if len(argv) > 2:
        Path(argv[2]).write_text(json.dumps(table, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
