#!/usr/bin/env python3
"""Score the measured NodeBased Roto UI against the sourced Nuke reference."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SECTIONS = ("toolbar", "tool_settings", "shape_list", "viewer_editing", "timing_animation")
WEIGHTS = {"match": 1.0, "differs": 0.5, "missing": 0.0}

def entries_for(section, sections):
    entries = []
    for item in sections[section]:
        if section == "toolbar":
            entries.extend(item.get("tools", []))
            if item.get("cycle"):
                entries.append(item["cycle"])
        elif section == "tool_settings":
            entries.extend(item.get("controls", []))
        elif section == "shape_list":
            if "commands" in item:
                entries.extend(item["commands"])
            else:
                entries.append(item)
        else:
            entries.append(item)
    return entries

def score(reference, measured):
    result = {}
    total = hits = 0.0
    for section in SECTIONS:
        rows = entries_for(section, measured)
        section_hits = sum(WEIGHTS[e["status"]] for e in rows)
        result[section] = {"score": round(100.0 * section_hits / len(rows), 1) if rows else 0.0,
                           "counts": {s: sum(e["status"] == s for e in rows) for s in WEIGHTS},
                           "entries": len(rows)}
        total += section_hits
        hits += len(rows)
    result["overall"] = round(100.0 * total / hits, 1) if hits else 0.0
    return result

def main():
    ref = json.loads((ROOT / "docs/parity/nuke_roto_ui.json").read_text())
    measured = json.loads((ROOT / "docs/parity/nodebased_roto_ui.json").read_text())
    print(json.dumps(score(ref, measured), indent=2))

if __name__ == "__main__":
    main()
