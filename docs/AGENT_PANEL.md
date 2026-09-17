# Agent panel

The Agent panel (Agent menu → Show Agent panel, docked at the bottom) is where an artist runs
their own coding agent CLI alongside the graph, wired to talk to this NodeBased session over MCP.
NodeBased never sees or handles an Anthropic/OpenAI credential: the CLI is the artist's own
install, under their own subscription/login, and only the CLI's process reaches the network.

## What it does

Opening the dock lazily opens the same `LocalBridge` local endpoint that `--agent <name>` opens at
startup (skipped if the GUI was already started with `--agent`). "Start Claude Code" and "Start
Codex" each:

1. Check the CLI is installed (`shutil.which`); if not, the panel says so and does nothing further.
2. Build a launch command that registers `nodebased-mcp` (see `docs/AGENT_PROTOCOL.md`) as an MCP
   server for that one session only — Claude Code via `--mcp-config <temp JSON>`, Codex via
   repeated `-c mcp_servers.nodebased...` overrides — pointed at this GUI's LocalBridge endpoint.
3. Run it in a real PTY embedded in the dock (`nodebased.agentpanel.PtyTerminal`, rendered through
   `pyte`), so the artist sees and can type into the actual CLI session.
4. Send an initial note asking the agent to call the `knowledge` tool with topic `overview` before
   doing anything else, so it orients itself before touching the graph.

Starting a second CLI replaces the running terminal (only one agent session at a time). The
activity log beneath the terminal lists every bridge request the attached agent makes (tool name,
arguments, and whether it succeeded), independent of whether it went through raw JSON-lines,
`agentloop`, or the MCP server — `Window.agent_command` is the single point every LocalBridge
client passes through, so anything reaching the graph shows up there.

## Settings

Two checkboxes live in the panel itself (stored in the same per-machine `QSettings` store as the
interface theme, never in the `.nbcomp`):

- **Allow save / load / render from the agent** (`agent/allow_disk_ops`, off by default). With it
  off, the MCP `edit` tool refuses any `save`/`load`/`render` command outright.
- **Allow the agent to file GitHub issues** (`agent/file_issue_enabled`, on by default, shared with
  `nodebased.issues`). With it off, the MCP `file_issue` tool is refused before any `gh`/network
  call is attempted.

## Platform notes

The PTY terminal uses the stdlib `pty`/`os` on Linux and macOS. On Windows it needs the optional
`pywinpty` package; without it the panel shows a clear message instead of attempting to spawn
anything, and the rest of the app is unaffected.
