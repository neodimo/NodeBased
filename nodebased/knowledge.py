"""Human-readable knowledge surfaces for NodeBased and its agent protocol."""

from __future__ import annotations

import json
from pathlib import Path
import re

import nodebased
from .core import CHOICES, LIMITS, SPECS


TOPICS = (
    "overview",
    "nodes",
    "protocol",
    "colour",
    "playback",
    "animation",
    "roto_tracking",
    "time_model",
    "three_d",
    "simulation",
    "version",
    "limits",
    "live_state",
)

_DOC_NAMES = {
    "overview": ("VISION.md", "ARCHITECTURE.md"),
    "protocol": ("AGENT_PROTOCOL.md",),
    "colour": ("COLOR_MANAGEMENT.md",),
    "playback": ("PLAYBACK.md",),
    "animation": ("ANIMATION.md",),
    "roto_tracking": ("ROTO_TRACKING.md",),
    "time_model": ("TIME_MODEL.md",),
    "three_d": ("3D_FOUNDATION.md", "3D_ROADMAP.md"),
    "simulation": ("SIMULATION.md",),
}


def _read_doc(name):
    """Read a bundled document, with the repository checkout as a dev fallback."""
    here = Path(__file__).resolve().parent
    candidates = [here / "data" / "docs" / name]
    candidates.extend(parent / "docs" / name for parent in here.parents)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8")
    raise FileNotFoundError(
        f"Bundled document {name!r} was not found at nodebased/data/docs or in a parent docs directory"
    )


def _json_value(value):
    return json.dumps(value, sort_keys=True)


def _node_reference():
    sections = ["# Node reference"]
    for node_type, spec in SPECS.items():
        kind = spec.get("category", spec.get("kind"))
        title = f"## {node_type}"
        if kind is not None:
            title += f" ({kind})"
        sections.append(title)
        params = spec.get("params", {})
        if not params:
            sections.append("No parameters.")
            continue
        for name, default in params.items():
            detail = f"- `{name}`: default `{_json_value(default)}`"
            if name in LIMITS:
                detail += f"; range `{_json_value(LIMITS[name])}`"
            if name in CHOICES:
                detail += f"; choices `{_json_value(CHOICES[name])}`"
            sections.append(detail)
    return "\n".join(sections)


def _limits():
    lines = ["# Limits and choices"]
    for name, value in LIMITS.items():
        lines.append(f"- `{name}`: range `{_json_value(value)}`")
    for name, value in CHOICES.items():
        lines.append(f"- `{name}`: choices `{_json_value(value)}`")
    return "\n".join(lines)


def _release_section(version, release_notes):
    headings = list(re.finditer(r"(?m)^# NodeBased\s+([^\n]+)", release_notes))
    selected = None
    for match in headings:
        if version in match.group(1):
            selected = match
            break
    note = ""
    if selected is None:
        selected = headings[0] if headings else None
        note = "\n> Note: no exact release-notes match was found; this is the topmost section.\n"
    if selected is None:
        section = release_notes
    else:
        end = next((match.start() for match in headings if match.start() > selected.start()), len(release_notes))
        section = release_notes[selected.start():end].rstrip()
    return note + section


def _known_limits(release_notes):
    lines = release_notes.splitlines()
    sections = []
    current = []
    for line in lines:
        if line.startswith("#") and current:
            sections.append(current)
            current = []
        current.append(line)
    if current:
        sections.append(current)
    bullets = []
    for section in sections:
        if any("limitation" in line.lower() or "known limit" in line.lower() for line in section[:5]):
            bullets.extend(line for line in section if re.match(r"^\s*[-*+]\s+", line))
    if not bullets:
        return "No known-limit or limitation bullets were found in RELEASE_NOTES.md."
    return "Known limits from RELEASE_NOTES.md:\n" + "\n".join(bullets)


def _version():
    release_notes = _read_doc("RELEASE_NOTES.md")
    version = nodebased.__version__
    return "\n\n".join((
        f"# Version {version}",
        _release_section(version, release_notes),
        _known_limits(release_notes),
    ))


def _live_state(dispatcher):
    if dispatcher is None:
        raise ValueError("live_state requires a live document")
    document = getattr(dispatcher, "document", None)
    if not isinstance(document, dict):
        raise ValueError("live_state requires a live document")
    nodes = document.get("nodes", {})
    state_nodes = {
        key: {
            "type": value.get("type"),
            "connections": value.get("inputs", {}),
        }
        for key, value in nodes.items()
        if isinstance(value, dict)
    }
    render_errors = None
    for name, value in vars(dispatcher).items():
        if "error" in name.lower() and not callable(value):
            render_errors = value
            break
    return json.dumps({"nodes": state_nodes, "render_errors": render_errors}, sort_keys=True, indent=2)


def topic(name, dispatcher=None):
    """Return the requested knowledge topic as plain markdown-ish text."""
    if name not in TOPICS:
        raise ValueError(f"Unknown knowledge topic: {name!r}")
    if name == "nodes":
        return _node_reference()
    if name == "limits":
        return _limits()
    if name == "version":
        return _version()
    if name == "live_state":
        return _live_state(dispatcher)
    return "\n\n".join(_read_doc(doc_name) for doc_name in _DOC_NAMES[name])
