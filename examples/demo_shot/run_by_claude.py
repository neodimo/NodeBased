"""The demo shot by a real Claude Code session attached over MCP (stage 4b, run once, by hand).

Starts an offscreen NodeBased window on demo_shot.nbcomp with the agent bridge open, registers the
nodebased MCP server for one `claude -p` session the way the Agent panel does (--mcp-config), gives
it the brief and the folders, and records the tool sequence it chose from the session's stream-json.
Needs the `claude` CLI on PATH with a login; it reaches the network, so it is not part of any test.

    python examples/demo_shot/run_by_claude.py [--plate DIR] [--out DIR] [--cache DIR] [--log FILE]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
import shot  # noqa: E402
from nodebased.mcpclient import HeadlessSession  # noqa: E402

PROMPT = """{brief}

You are attached to a running NodeBased session over MCP (server "nodebased") with the demo shot open. \
The plate frames are in {plate} (plate.%04d.exr, frames 1 to {frames}); the Read node's path does not point \
there yet. Write the finished shot as EXR frames shot.%04d.exr in {out}. Call the knowledge tool first. \
Use only the nodebased tools. When done, state the artifact ids you produced."""


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--plate", default=str(shot.plate_dir()))
    parser.add_argument("--out", default=str(shot.default_cache() / "out-claude"))
    parser.add_argument("--cache", default=str(shot.default_cache() / "store-claude"))
    parser.add_argument("--log", default=str(shot.default_cache() / "claude-stream.jsonl"))
    parser.add_argument("--frames", type=int, default=shot.FRAMES)
    args = parser.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    session = HeadlessSession(HERE / "demo_shot.nbcomp", args.cache, attach_client=False)
    try:
        config = {"mcpServers": {"nodebased": {
            "command": sys.executable, "args": ["-m", "nodebased.mcpserver", "--endpoint", session.endpoint,
                                                 "--agent-name", "claude-code"],
            "env": {key: session.env[key] for key in ("HOME", "XDG_CONFIG_HOME", "NODEBASED_CACHE",
                                                      "QT_QPA_PLATFORM", "PYTHONPATH")}}}}
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump(config, handle)
        command = ["claude", "-p", PROMPT.format(brief=shot.BRIEF, plate=args.plate, frames=args.frames, out=args.out),
                   "--mcp-config", handle.name, "--strict-mcp-config", "--allowedTools", "mcp__nodebased",
                   "--output-format", "stream-json", "--verbose", "--permission-mode", "dontAsk"]
        with open(args.log, "w", encoding="utf-8") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, cwd=tempfile.gettempdir(),
                           env={k: v for k, v in os.environ.items() if k != "CLAUDECODE"}, timeout=3600)
    finally:
        session.close()
    print(f"stream log: {args.log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
