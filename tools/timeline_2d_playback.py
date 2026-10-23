"""Reduce a tools/benchmark_2d_playback.py result to its fresh-frame timeline and check the accounting.

    python tools/timeline_2d_playback.py raw.json --output benchmarks/2d-playback/name.timeline.json

The stage durations come from the playback trace marks the app records for each request (nodebased/playback_trace.py);
the frame interval they must add up to comes from the benchmark's own handoff stamps for the first pass through the
48 fresh frames, taken in a different place (the Viewer's finished signal), so the two do not share a clock reading.
"""
import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from tools.benchmark_2d_playback import first_unique_pass  # noqa: E402

KEEP = ('case', 'pipeline', 'direct_picture', 'adapter', 'force_adapter', 'resolution', 'cache_mode', 'duration_s',
        'achieved_fps', 'first_pass_fps', 'frame_interval_ms', 'handoff_ms', 'frames_dropped_counted',
        'conversion_stage_drops', 'display_conversions_8bit', 'fresh_frame_timeline')


def reduce(raw):
    shown = [(event['t_ms'], event['frame']) for event in raw['timeline']
             if event['stage'] == 'ui_handoff' and event.get('presented')]
    cold = first_unique_pass(shown)
    measured = (cold[-1][0] - cold[0][0]) / (len(cold) - 1) if len(cold) > 1 else None
    out = {key: raw[key] for key in KEEP if key in raw}
    summary = out['fresh_frame_timeline'].get('summary', {})
    out['fresh_frame_timeline'] = dict(out['fresh_frame_timeline'])
    out['fresh_frame_timeline'].pop('measured_mean_interval_ms', None)
    accounted = summary.get('accounted_interval_ms')
    out['accounting'] = {
        'fresh_frames_in_first_pass': len(cold),
        'accounted_interval_ms': accounted,
        'measured_first_pass_interval_ms': round(measured, 3) if measured else None,
        'difference_percent': round(100.0 * (accounted - measured) / measured, 2) if accounted and measured else None,
        'rule': 'stage durations + transport wait - overlap with the previous picture, against the first-pass interval',
    }
    return out


def table_row(out):
    summary = out['fresh_frame_timeline']['summary']
    return (out['case'], summary['stages_ms'], summary['overlap_ms'], summary['accounted_interval_ms'],
            out['accounting']['measured_first_pass_interval_ms'], out['accounting']['difference_percent'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('raw', type=pathlib.Path)
    parser.add_argument('--output', type=pathlib.Path, required=True)
    args = parser.parse_args()
    out = reduce(json.loads(args.raw.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(out['accounting']))


if __name__ == '__main__':
    main()
