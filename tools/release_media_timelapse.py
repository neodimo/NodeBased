"""Shorten a recorded release clip by speeding up the render waits and nothing else.

Run: python tools/release_media_timelapse.py OUT_DIR NAME [SPEED]
Reads the step log release_media_clips.py wrote to manifest-clips.json. Every wait longer than
WAIT seconds plays at SPEED with a caption burned in saying so; the holds, and the last LEAD
seconds before each new picture, stay in real time. Writes NAME-timelapse.mp4 beside the original
and records the cut list in the manifest.
"""
import json
import subprocess
import sys
from pathlib import Path

WAIT, LEAD = 5.0, .5


def segments(entry):
    """(start, end, fast) spans covering the whole clip, from the recorder-clock step log."""
    spans = []
    for step in entry["steps"]:
        start, shown = step["cmd_at_s"], step["picture_at_s"]
        if shown - start > WAIT:
            spans += [(start, shown - LEAD, True), (shown - LEAD, shown + step["hold_s"], False)]
        else:
            spans.append((start, shown + step["hold_s"], False))
    spans[-1] = (spans[-1][0], entry["seconds"], spans[-1][2])
    merged = [spans[0]]
    for start, end, fast in spans[1:]:                # adjoining real-time spans become one
        if fast == merged[-1][2] == False:
            merged[-1] = (merged[-1][0], end, False)
        else:
            merged.append((start, end, fast))
    return merged


def main(out, name, speed=40):
    path = out / "manifest-clips.json"
    manifest = json.loads(path.read_text())
    entry = manifest["files"][f"{name}.mp4"]
    spans = segments(entry)
    waits = sorted(s["until_picture_ms"] / 1000 for s in entry["steps"] if s["until_picture_ms"] > WAIT * 1000)
    slow = waits[len(waits) // 2]                     # the median wait: one disturbed step does not set it
    what = "each step re-renders for" if len(waits) > 1 else "this render takes"
    caption = f"{speed}x  -  {what} about {slow:.0f} s on the CPU"
    font = subprocess.check_output(["fc-match", "-f", "%{file}", "sans:bold"], text=True)
    parts = []
    for i, (start, end, fast) in enumerate(spans):
        chain = f"[0:v]trim={start:.3f}:{end:.3f},setpts=PTS-STARTPTS"
        if fast:
            chain += (f",select='not(mod(n\\,{speed}))',setpts=N/{entry['fps']}/TB,"
                      f"drawtext=fontfile={font}:text='{caption}':fontsize=34:fontcolor=white:"
                      f"box=1:boxcolor=black@0.6:boxborderw=14:x=(w-text_w)/2:y=h-150")
        parts.append(chain + f"[s{i}]")
    graph = ";".join(parts) + ";" + "".join(f"[s{i}]" for i in range(len(spans))) + f"concat=n={len(spans)}[v]"
    target = out / f"{name}-timelapse.mp4"
    subprocess.check_call(["ffmpeg", "-loglevel", "error", "-y", "-i", str(out / f"{name}.mp4"),
                           "-filter_complex", graph, "-map", "[v]", "-r", str(entry["fps"]),
                           "-c:v", "libx264", "-preset", "slow", "-crf", "21", "-pix_fmt", "yuv420p",
                           "-movflags", "+faststart", str(target)])
    manifest["files"][target.name] = dict(
        edit_of=f"{name}.mp4", speed=speed, caption=caption, credit=entry.get("credit"),
        spans=[dict(start=round(a, 2), end=round(b, 2), speed=speed if fast else 1) for a, b, fast in spans])
    path.write_text(json.dumps(manifest, indent=2))
    print(target, sum((b - a) / (speed if fast else 1) for a, b, fast in spans))


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 40)
