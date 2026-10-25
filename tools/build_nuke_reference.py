#!/usr/bin/env python3
"""Rebuild docs/parity/nuke_nodes.json from the Foundry Nuke 17.0 Reference Guide.

Facts only: node names, input sockets, knob names, labels, tabs, types, defaults and the page URL. No prose from the
guide is stored. Needs network the first time (pages are cached in --cache); the committed JSON is the artefact the
scorer reads, so this tool only runs when someone refreshes the reference.

    python tools/build_nuke_reference.py --cache /tmp/nuke-guide-cache --rows docs/PARITY_2D.md --out docs/parity/nuke_nodes.json
"""
from __future__ import annotations

import argparse
import html
import json
import re
import sys
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

BASE = "https://learn.foundry.com/nuke/17.0/content/reference_guide/"
GUIDE_GROUP = {"Image": "image", "Draw": "draw", "Time": "time", "Channel": "channel", "Color": "color", "Filter": "filter",
               "Keyer": "keyer", "Merge": "merge", "Transform": "transform", "Metadata": "metadata", "Other": "other"}

# Row cell name in docs/PARITY_2D.md -> guide page slug, where the plain lower-cased name is not the slug.
SLUG = {"UDIM Import": "udim_import", "Multiply": None, "Erode (filter)": "erode_filter", "Erode (fast)": "erode_fast",
        "LensDistortion (Undistort / Redistort)": "lensdistortion", "LightMixer (no Nuke node)": None, "CornerPin2D": "cornerpin2d"}
# Names that appear in several groups: (group, name) -> slug.
SLUG_BY_GROUP = {("Color", "Multiply"): "multiply_color", ("Merge", "Multiply"): "multiply_merge"}

# Pages whose knob tables hold many per-format option tabs. Only these tabs are scored; the rest are recorded as format options.
KEEP_TABS = {
    "Read": {"Read Tab", "Context", "Sequence Tab", "Metadata Tab"},
    "Write": {"Write Tab", "Context", None, "Metadata Export"},
    "Viewer": {"Viewer Tab"},
}
CHROME = {"channels": "channels", "maskChannelInput": "mask_channel", "maskChannelMask": "mask_channel", "inject": "mask_inject",
          "invert_mask": "mask_invert", "fringe": "mask_fringe", "unpremult": "unpremult", "invert_unpremult": "unpremult_invert",
          "mix": "mix", "bbox": "bbox"}
PYTHON_KNOBS = {"beforeRender", "beforeFrameRender", "afterFrameRender", "afterRender", "renderProgress", "onCreate", "onDestroy",
                "knobChanged", "updateUI", "autolabel", "onScriptLoad", "onScriptSave", "onScriptClose", "afterBackgroundRender",
                "afterBackgroundFrameRender", "beforeBackgroundRender", "afterRenderGroup", "afterFrameRenderGroup"}
GPU_KNOBS = {"gpuName", "useGPUIfAvailable", "disable_gpu"}


class _Tables(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tables, self.cur, self.row, self.cell, self.title, self.in_title = [], None, None, None, "", False

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "title":
            self.in_title = True
        elif tag == "table":
            self.cur = []
            self.tables.append(self.cur)
        elif tag == "tr" and self.cur is not None:
            self.row = []
            self.cur.append(self.row)
        elif tag in ("td", "th") and self.row is not None:
            self.cell = {"t": "", "cs": int(a.get("colspan", 1)), "rs": int(a.get("rowspan", 1))}
            self.row.append(self.cell)
        elif tag in ("p", "br", "li") and self.cell is not None:
            self.cell["t"] += "\n"

    def handle_endtag(self, tag):
        if tag == "title":
            self.in_title = False
        elif tag in ("td", "th"):
            self.cell = None
        elif tag == "tr":
            self.row = None
        elif tag == "table":
            self.cur = None

    def handle_data(self, data):
        if self.in_title:
            self.title += data
        if self.cell is not None:
            self.cell["t"] += data


def _clean(text):
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def _expand(rows):
    """Resolve rowspans/colspans into a rectangular grid of strings; keep the raw rows to spot tab-heading rows."""
    grid, pending = [], {}
    for row in rows:
        cells, out, col = list(row), [], 0
        while cells or any(c >= col for c in pending):
            if col in pending:
                text, left = pending[col]
                out.append(text)
                if left > 1:
                    pending[col] = (text, left - 1)
                else:
                    del pending[col]
                col += 1
                continue
            if not cells:
                break
            cell = cells.pop(0)
            for _ in range(cell[1]):
                out.append(cell[0])
                if cell[2] > 1:
                    pending[col] = (cell[0], cell[2] - 1)
                col += 1
        grid.append(out)
    return grid


def parse_page(text):
    p = _Tables()
    p.feed(text)
    inputs, knobs, tab = [], [], None
    for table in p.tables:
        rows = [[(_clean(c["t"]), c["cs"], c["rs"]) for c in r] for r in table]
        grid = _expand(rows)
        if not grid:
            continue
        head = [h.lower() for h in grid[0]]
        if head and head[0].startswith("connection type"):
            for r in grid[1:]:
                inputs.append({"type": r[0], "name": r[1] if len(r) > 1 else ""})
        elif head and head[0].startswith("control"):
            for raw, g in zip(rows[1:], grid[1:]):
                if len(raw) == 1 and raw[0][1] >= 3:
                    tab = raw[0][0]
                    continue
                if len(g) < 3:
                    continue
                ui, name, default = g[0], g[1], g[2]
                function = g[3] if len(g) > 3 else ""
                knobs.append({"tab": tab, "ui": ui, "name": name, "default": default, "function": function})
    return _clean(p.title), inputs, knobs


def parse_default(raw):
    s = raw.strip()
    if s in ("", "N/A", "-", "—", "n/a"):
        return None
    low = s.lower()
    if low == "enabled":
        return True
    if low == "disabled":
        return False
    if re.fullmatch(r"-?\d+(\.\d+)?(e-?\d+)?", s):
        v = float(s)
        return int(v) if v == int(v) and "." not in s and "e" not in low else v
    if re.match(r"(?i)^(dependent|dependant|depends|from auto|the first frame|\[)", s) or (s.endswith(".") and " " in s):
        return None
    parts = [x.strip() for x in s.split(",")]
    if len(parts) > 1 and all(re.fullmatch(r"-?\d+(\.\d+)?(e-?\d+)?", x) for x in parts):
        return [float(x) if "." in x else int(x) for x in parts]
    return s


def choices_of(function):
    found = [c.strip() for c in re.findall(r"•\s*([^•]+?)\s+[-–—]\s", function)]
    return [c for c in found if 0 < len(c) <= 40 and len(c.split()) <= 3]


def short_tab(tab):
    if tab is None:
        return None
    return re.split(r"\s+(?:\(|-\s|–\s)", tab)[0].strip()


def knob_type(name, ui, default, choices):
    if name == "channels" or ui.lower().endswith("channels"):
        return "channels"
    if isinstance(default, bool):
        return "bool"
    if isinstance(default, list):
        return "number_list"
    if isinstance(default, (int, float)):
        return "number"
    if choices:
        return "enum"
    return "unknown" if default is None else "string"


def build_node(name, slug, group, cache, fetch):
    path = cache / f"{GUIDE_GROUP[group]}__{slug}.html"
    url = f"{BASE}{GUIDE_GROUP[group]}_nodes/{slug}.html"
    if not path.exists():
        if not fetch:
            raise RuntimeError(f"missing cached page {path}")
        req = urllib.request.Request(url, headers={"User-Agent": "nodebased-parity-audit"})
        path.write_bytes(urllib.request.urlopen(req, timeout=40).read())
    title, inputs, raw = parse_page(path.read_text(encoding="utf-8", errors="replace"))
    first_tab = None
    knobs, seen = [], {}
    for k in raw:
        tab = short_tab(k["tab"])
        if first_tab is None and not knobs:
            first_tab = tab
        kname = re.sub(r"\s+", "", k["name"])
        default = parse_default(k["default"])
        ch = choices_of(k["function"]) if default is not None else []
        role, counted, why = "knob", True, None
        if kname in ("N/A", "") or kname.lower() == "n/a":
            role, counted, why = "ui_only", False, "enable checkbox or label without a scripting name"
        elif kname in CHROME:
            role = "chrome:" + CHROME[kname]
        elif kname in PYTHON_KNOBS or (k["tab"] or "").startswith("Python Tab"):
            role, counted, why = "python_callback", False, "Python callback knob"
        elif kname in GPU_KNOBS:
            role, counted, why = "gpu_option", False, "GPU selection option common to GPU nodes"
        if name in KEEP_TABS and tab not in KEEP_TABS[name] and counted and not role.startswith("chrome:"):
            role, counted, why = "format_option", False, "per-format or display option tab, not scored"
        entry = {"name": kname, "label": k["ui"], "tab": tab, "primary": None, "role": role, "counted": counted,
                 "type": knob_type(kname, k["ui"], default, ch), "default": default}
        if ch:
            entry["choices"] = ch
        if why:
            entry["reason"] = why
        if default is None and counted:
            entry["default_note"] = ("the guide gives no literal default (" + _clean(k["default"])[:60] + ")") if _clean(k["default"]) not in ("", "N/A", "-", "—") else "the guide gives no default"
        knobs.append(entry)
    # the first tab listed on the page is the node's main tab
    for e in knobs:
        e["primary"] = (e["tab"] == first_tab)
    # a duplicate scripting name keeps its first occurrence (the guide repeats e.g. invert for different controls)
    uniq = []
    for e in knobs:
        if e["name"] in seen and e["role"] not in ("ui_only",):
            e["name"] = f"{e['name']}#{seen[e['name']] + 1}"
            seen[e["name"].split("#")[0]] += 1
        else:
            seen.setdefault(e["name"], 1)
        uniq.append(e)
    socks = [{"name": i["name"], "mask": i["name"].lower() == "mask", "type": i["type"]} for i in inputs]
    return {"guide_title": title, "group": group, "page": url, "main_tab": first_tab, "inputs": socks, "knobs": uniq}


def guide_slugs(group, cache, fetch):
    """The node page slugs the guide's own group page lists."""
    path = cache / f"{GUIDE_GROUP[group]}__index.html"
    if not path.exists():
        if not fetch:
            return []
        url = f"{BASE}{GUIDE_GROUP[group]}_nodes/{GUIDE_GROUP[group]}_nodes.html"
        path.write_bytes(urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "nodebased-parity-audit"}), timeout=40).read())
    found = re.findall(r'href="([a-z0-9_]+)\.html"', path.read_text(encoding="utf-8", errors="replace"))
    return [f for f in dict.fromkeys(found) if f not in (f"{GUIDE_GROUP[group]}_nodes", "all_plugins")]


def dump(doc):
    """JSON with one line per knob, so the file diffs and reads."""
    out = ["{", f' "schema": {doc["schema"]},', ' "source": ' + json.dumps(doc["source"], ensure_ascii=False) + ",",
           ' "notes": ' + json.dumps(doc["notes"], ensure_ascii=False, indent=1).replace("\n", "\n ") + ",",
           ' "guide_index": ' + json.dumps(doc["guide_index"], ensure_ascii=False) + ",", ' "nodes": {']
    keys = list(doc["nodes"])
    for i, key in enumerate(keys):
        n = doc["nodes"][key]
        head = {k: v for k, v in n.items() if k not in ("inputs", "knobs")}
        out.append(f"  {json.dumps(key)}: {{")
        out.append("   " + json.dumps(head, ensure_ascii=False)[1:-1] + ",")
        out.append('   "inputs": ' + json.dumps(n["inputs"], ensure_ascii=False) + ",")
        out.append('   "knobs": [')
        for j, k in enumerate(n["knobs"]):
            out.append("    " + json.dumps(k, ensure_ascii=False) + ("," if j < len(n["knobs"]) - 1 else ""))
        out.append("   ]")
        out.append("  }" + ("," if i < len(keys) - 1 else ""))
    out += [" }", "}"]
    return "\n".join(out) + "\n"


def doc_nodes(rows_path):
    out, heading = [], ""
    for line in Path(rows_path).read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            heading = line[3:].strip()
        m = re.match(r"^\|\s*(\d+)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|", line)
        if m and m.group(3) in ("supported", "partial", "missing", "n/a") and heading in GUIDE_GROUP:
            out.append((heading, int(m.group(1)), m.group(2)))
    return out


def names_of(cell):
    cell = re.sub(r"\s*\(no Nuke node\)", "", cell)
    m = re.match(r"^LensDistortion \((.+)\)$", cell)
    if m:
        return ["LensDistortion"]
    return [n.strip() for n in cell.split(" / ")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-fetch", action="store_true")
    a = ap.parse_args()
    cache = Path(a.cache)
    cache.mkdir(parents=True, exist_ok=True)
    nodes, missing = {}, []
    for group, rank, cell in doc_nodes(a.rows):
        for n in names_of(cell):
            if n == "LightMixer":
                continue
            slug = SLUG_BY_GROUP.get((group, n)) or SLUG.get(n) or re.sub(r"[^a-z0-9]", "", n.lower())
            key = n if n not in nodes else f"{n} ({group})"
            if (group, n) == ("Merge", "Multiply"):
                key = "Multiply (Merge)"
            if (group, n) == ("Color", "Multiply"):
                key = "Multiply"
            try:
                nodes[key] = build_node(n, slug, group, cache, not a.no_fetch)
            except Exception as exc:  # noqa: BLE001
                missing.append((n, slug, str(exc)[:80]))
                nodes[key] = {"guide_title": None, "group": group, "page": None, "main_tab": None, "inputs": [], "knobs": []}
    doc = {"schema": 1,
           "source": {"name": "Foundry Nuke 17.0 Reference Guide", "base_url": BASE, "fetched": "2026-10-25"},
           "notes": ["Facts only: names, input sockets, labels, tabs, defaults. No guide prose is stored.",
                     "The guide has no type column: `type` is inferred from the default value and from the choices the guide lists.",
                     "`default: null` means the guide states no default (see `default_note`).",
                     "`primary` is true for the knobs on the first tab the guide lists for the node (its main tab).",
                     "`role` and `counted` are derived: chrome knobs (channels, mask channel, mask invert, unpremult, mix, bbox) are scored in "
                     "their own component; enable checkboxes, Python callbacks, GPU options and per-format option tabs are not counted."],
           "guide_index": {g: guide_slugs(g, cache, not a.no_fetch) for g in GUIDE_GROUP},
           "nodes": nodes}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    text = dump(doc)
    json.loads(text)
    Path(a.out).write_text(text, encoding="utf-8")
    print("nodes", len(nodes), "missing", missing)


if __name__ == "__main__":
    main()
