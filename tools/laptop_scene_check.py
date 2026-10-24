"""3D scene check for a laptop or a workstation: one command, a short result file to paste back (Lane 4, 10/23).

Runs the four test scenes (cube, furnished room, mixed, smoke-splats-glass) at 1080p in the real window, each in a process of
its own: the time to the first picture, the first 3 s of playback pressed at that picture, the same again after the card has
rested 4 s (the idle clock of a discrete card), and the rate over the length of the run. When the machine lists a discrete and an
integrated card, both are measured: the card the application chooses itself, then the other one.

    python tools/laptop_scene_check.py                       (about 6 minutes; writes scene-check/result.txt and result.json)
    python tools/laptop_scene_check.py --quick               (cube and furnished room only, 3 s runs)
    python tools/laptop_scene_check.py --self-check          (a few seconds on the software renderer; measures nothing about a GPU)
    python tools/laptop_scene_check.py --read scene-check    (print the summary of an earlier run, on any machine)

Paste the contents of `scene-check/result.txt` back. Nothing is claimed for a machine until its own result file exists: this is
not a measurement of the Windows laptop or the A6000 machine until it has been run there. Run it from the repository or the
install folder with the field kit's Python, with the laptop on mains power and the window left alone while it runs.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCENES = ("cube", "furnished", "mixed", "mixedglass")
LABELS = {"cube": "cube", "furnished": "furnished room", "mixed": "mixed (room + splats)", "mixedglass": "smoke, splats, glass"}
GAP_GATE_MS = 54.0          # the longest interval between pictures a 24 a second scene may show (41.7 ms budget)
RATE_GATE_FPS = 23.5        # pictures a second between the first and last picture of the first 3 s
IDLE_REST_S = 4.0           # the card rests this long before play on the "after a rest" runs


def run_scene(scene, adapter, out_dir, args, rest=0.0):
    """One run of tools/benchmark_3d_playback.py in a process of its own; returns its JSON, or a dict with an `error`."""
    tag = "".join(c if c.isalnum() else "-" for c in adapter.split()[0].lower())
    target = Path(out_dir) / f"{scene}-{tag}-{'rest' if rest else 'warm'}.json"
    cmd = [sys.executable, str(ROOT / "tools" / "benchmark_3d_playback.py"), "--scene", scene, "--size", args.size,
           "--seconds", str(args.seconds), "--first-seconds", "3", "--replay", "--play-at", "first", "--play-delay", str(rest),
           "--adapter", adapter, "--settle-wait", "5", "--proxy", args.proxy, "--out", str(target)]
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    if args.self_check:
        env.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        subprocess.run(cmd, check=False, env=env, timeout=args.timeout, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return json.loads(target.read_text())
    except subprocess.TimeoutExpired:
        return {"scene": scene, "error": f"no result in {args.timeout:.0f} s"}
    except (OSError, ValueError):
        return {"scene": scene, "error": "the run produced no result"}


def row_of(result):
    """The numbers worth pasting back from one run's JSON."""
    if "error" in result:
        return {"scene": result.get("scene", "?"), "error": result["error"]}
    cold = (result.get("first_seconds") or {}).get("cold") or {}
    warm = (result.get("first_seconds") or {}).get("warm") or {}
    picture = cold.get("first_picture") or {}
    tiers = result.get("tier_pictures") or {}
    shown = sum(tiers.values())
    reduced = round(100.0 * sum(n for tier, n in tiers.items() if str(tier) != "1") / shown) if shown else None
    return {
        "reduced_percent": reduced, "scene": result.get("scene"), "adapter": result.get("adapter"), "proxy": result.get("proxy"),
        "first_picture_s": picture.get("open_to_first_picture_s"),
        "between_fps": cold.get("fps_between_pictures"), "worst_ms": cold.get("interval_max_ms"),
        "worst_warm_ms": warm.get("interval_max_ms"), "skipped": cold.get("frames_skipped"),
        "fps_whole_run": result.get("fps"), "interval_p95_ms": (result.get("interval_ms") or {}).get("p95"),
        "tier_pictures": result.get("tier_pictures"),
    }


def verdict(warm, rested):
    """OK when the first 3 s hold the rate with no interval over the gate, warm and after a rest."""
    if "error" in warm or "error" in rested:
        return "no result"
    problems = []
    if (warm.get("between_fps") or 0) < RATE_GATE_FPS:
        problems.append(f"{warm.get('between_fps')} pictures a second")
    for name, row in (("warm", warm), ("after a rest", rested)):
        worst = row.get("worst_ms")
        if worst is None or worst > GAP_GATE_MS:
            problems.append(f"{worst} ms gap {name}")
    return "OK" if not problems else "; ".join(problems)


def machine_info():
    info = {"platform": platform.platform(), "python": platform.python_version(), "when": time.strftime("%Y-%m-%d %I:%M %p")}
    try:
        from nodebased import gpu3d
        adapters = gpu3d.enumerate_adapters()
        info["adapters"] = adapters
        choice = gpu3d.adapter_choice(adapters=adapters)
        info["chosen"] = choice
    except Exception as exc:          # a machine without wgpu still gets a file saying so
        info["adapters"], info["chosen"] = [], {"name": "", "type": "", "reason": f"the adapter probe failed: {exc}"}
    return info


def adapters_to_run(info):
    """The app's own choice first; the other kind (integrated when it chose a discrete card) second."""
    chosen = info["chosen"].get("name") or "default"
    runs = [chosen]
    kinds = {a["name"]: a["type"] for a in info.get("adapters", [])}
    other = next((n for n, k in kinds.items() if k in ("integrated", "discrete") and k != info["chosen"].get("type")), None)
    if other and other not in runs:
        runs.append(other)
    return runs


def summary_text(report):
    info = report["machine"]
    lines = [f"NodeBased 3D scene check, {info['when']}", f"{info['platform']}"]
    for entry in info.get("adapters", []):
        lines.append(f"  card: {entry['name']} ({entry['type']}, {entry['backend']})")
    chosen = info["chosen"]
    lines.append(f"The app chooses {chosen.get('name') or 'nothing'}: {chosen.get('reason')}")
    for run in report["runs"]:
        lines += ["", f"On {run['adapter']} ({run['size']}, pictures a second between the first and last of the first 3 s; "
                      f"longest gap, warm and after a {IDLE_REST_S:g} s rest; gate {GAP_GATE_MS:g} ms):"]
        for scene in run["scenes"]:
            warm, rested = scene["warm"], scene["rested"]
            if "error" in warm:
                lines.append(f"  {LABELS.get(scene['scene'], scene['scene'])}: {warm['error']}")
                continue
            lines.append(f"  {LABELS.get(scene['scene'], scene['scene'])}: first picture {warm.get('first_picture_s')} s, "
                         f"{warm.get('between_fps')} fps, gap {warm.get('worst_ms')} ms, after a rest {rested.get('worst_ms', '-')} ms, "
                         f"whole run {warm.get('fps_whole_run')} fps"
                         + (f", {warm['reduced_percent']} percent of the pictures at a lower resolution" if warm.get("reduced_percent") else "")
                         + f": {scene['verdict']}")
    return "\n".join(lines) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="scene-check", help="folder for result.txt, result.json and the per-run JSON")
    p.add_argument("--scenes", default=",".join(SCENES))
    p.add_argument("--size", default="1920x1080")
    p.add_argument("--seconds", type=float, default=6.0)
    p.add_argument("--proxy", default="on", choices=("on", "off"),
                   help="on (default): as the application ships, with Proxy while playing; off: the raw cost at the size set")
    p.add_argument("--timeout", type=float, default=900.0, help="seconds one run may take")
    p.add_argument("--quick", action="store_true", help="cube and furnished room only, 3 s runs")
    p.add_argument("--only-chosen", action="store_true", help="measure only the card the application chooses")
    p.add_argument("--self-check", action="store_true", help="tiny runs on the software renderer: proves the script runs")
    p.add_argument("--read", default="", help="print the summary of an earlier run's folder")
    args = p.parse_args(argv)
    if args.read:
        print((Path(args.read) / "result.txt").read_text(encoding="utf-8"))
        return 0
    scenes = [s for s in args.scenes.split(",") if s]
    if args.quick:
        scenes, args.seconds = ["cube", "furnished"], 3.0
    if args.self_check:
        scenes, args.seconds, args.size, args.timeout = ["cube"], 2.0, "320x180", 180.0
    for scene in scenes:
        if scene not in SCENES:
            raise SystemExit(f"unknown scene {scene!r}; choose from {', '.join(SCENES)}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT))
    info = machine_info()
    adapters = ["llvmpipe"] if args.self_check else adapters_to_run(info)
    if args.only_chosen and not args.self_check:
        adapters = adapters[:1]
    report = {"machine": info, "args": {"size": args.size, "seconds": args.seconds, "proxy": args.proxy}, "runs": []}
    for adapter in adapters:
        run = {"adapter": adapter, "size": args.size, "scenes": []}
        for scene in scenes:
            print(f"{adapter}: {scene} ...", flush=True)
            warm = row_of(run_scene(scene, adapter, out, args))
            rested = row_of(run_scene(scene, adapter, out, args, rest=IDLE_REST_S))
            run["scenes"].append({"scene": scene, "warm": warm, "rested": rested, "verdict": verdict(warm, rested)})
        report["runs"].append(run)
    (out / "result.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    text = summary_text(report)
    (out / "result.txt").write_text(text, encoding="utf-8")
    print("\n" + text)
    print(f"Paste back {out / 'result.txt'}")
    failed = [s for r in report["runs"] for s in r["scenes"] if "error" in s["warm"]]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
