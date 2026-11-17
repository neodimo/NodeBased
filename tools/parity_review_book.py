#!/usr/bin/env python3
"""Write the Nuke parity review book: an index and one HTML page per scored node, for a person to judge by eye.

Each page has two halves. The first asks whether the panel looks like Nuke's: the Reference Guide's knob table per tab
(label, widget type, default; from docs/parity/nuke_nodes.json, with a link to Foundry's page), PNGs of NodeBased's real
Properties panel rendered offscreen through the app's own panel builder, and an alignment table (one row per Nuke knob:
the paired NodeBased param, or missing, or unsupported with the reason, and a mark for each mismatch in order, label,
widget type or default). The second half asks whether the node computes what Nuke computes: for every case of the
node in the fidelity audit, the input, NodeBased's output, the expectation the case states, the absolute difference,
the audit's verdict and the case's citation, rendered on readable 256 x 256 plates; then a "knobs that act" strip.

When docs/parity/nuke_renders/<case>.exr exists (rendered by Nuke from tools/nuke_script_export.py's scripts), the page
adds a "Nuke rendered" column and NodeBased's difference against it, and the index counts nodes verified against Nuke's
own pixels.

    python tools/parity_review_book.py --out docs/parity/review/out [--nodes Blur,Grade,...]

The book is build output (docs/parity/review/out is gitignored). See docs/parity/review/README.md.
"""
from __future__ import annotations

import argparse
import dataclasses
import html
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

NUKE_JSON = ROOT / "docs" / "parity" / "nuke_nodes.json"
MAP_JSON = ROOT / "docs" / "parity" / "nodebased_to_nuke.json"
FIDELITY_MD = ROOT / "docs" / "PARITY_2D_FIDELITY.md"
INTERACTIONS_JSON = ROOT / "docs" / "parity" / "nuke_knob_interactions.json"
RENDERS_DIR = ROOT / "docs" / "parity" / "nuke_renders"
TEN = ("Blur", "Grade", "Merge", "Transform", "ColorCorrect", "Keylight", "Crop", "Shuffle", "Reformat", "Defocus")
PLATE = 256
PANEL_WIDTH = 480
GUIDE_HOME = "https://learn.foundry.com/nuke/17.0/content/reference_guide/"

# Groups whose input plate is a ramp and colour bars; filters get a step edge and a white square, transforms a grid,
# keyers a screen-colour plate.
COLOUR_GROUPS = ("Color", "Channel", "Merge", "Image", "Draw", "Time", "Metadata", "Other")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def slug(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "node"


def esc(value):
    return html.escape(str(value), quote=True)


# ------------------------------------------------------------------------------------------------------- the audit data
def fidelity_rows(path=FIDELITY_MD):
    """{nuke node: (group, rank, nodebased, status, fidelity, math cell)} from the report's "Per node" tables."""
    rows, group, inside = {}, None, False
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            inside = line.strip() == "## Per node"
            continue
        if not inside:
            continue
        if line.startswith("### "):
            group = line[4:].strip()
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 12 and cells[0].isdigit():
            rows[cells[1]] = {"group": group, "rank": int(cells[0]), "nodebased": cells[2].strip("`"), "status": cells[3],
                              "fidelity": cells[4], "math": cells[11]}
    return rows


# --------------------------------------------------------------------------------------------------------------- images
def to_display(arr):
    """A float RGBA (or RGB) array as 8-bit RGB for the page: rgb clipped to 0..1 and given a 2.2 display gamma, over a
    dark checker where alpha is below one so a premultiplied hole reads as a hole."""
    a = np.asarray(arr, np.float64)
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, axis=2)
    rgb = np.clip(a[..., :3], 0.0, 1.0) ** (1 / 2.2)
    if a.shape[2] >= 4:
        h, w = a.shape[:2]
        yy, xx = np.mgrid[:h, :w]
        cell = max(1, min(h, w) // 16)
        checker = np.where(((yy // cell) + (xx // cell)) % 2 == 0, 0.18, 0.28)[..., None]
        alpha = np.clip(a[..., 3:4], 0.0, 1.0)
        # rgb is shown as stored (premultiplied or not); the checker shows through only where alpha is below one.
        rgb = rgb + checker * (1.0 - alpha) * (rgb.max(axis=2, keepdims=True) < 1e-6)
    return (np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8)


def enlarge(arr, size=PLATE):
    """Nearest-neighbour enlargement of a small array to about `size` pixels wide, so a 16 x 12 fixture can be seen."""
    a = np.asarray(arr)
    k = max(1, size // a.shape[1])
    return np.repeat(np.repeat(a, k, axis=0), k, axis=1)


def save_png(path, rgb8):
    from PySide6.QtGui import QImage
    rgb8 = np.ascontiguousarray(rgb8)
    h, w = rgb8.shape[:2]
    image = QImage(rgb8.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not image.save(str(path)):
        raise OSError(f"could not write {path}")
    return path


def diff_image(a, b, valid=None):
    """Absolute difference, the worst channel per pixel, shown x10 in a heat ramp (black = equal, white = 0.1 or more)."""
    err = np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))
    if valid is not None:
        err = np.where(valid, err, 0.0)
    worst = err.max(axis=2) if err.ndim == 3 else err
    t = np.clip(worst * 10.0, 0.0, 1.0)
    heat = np.stack([np.clip(t * 3, 0, 1), np.clip(t * 3 - 1, 0, 1), np.clip(t * 3 - 2, 0, 1)], axis=2)
    return (heat * 255 + 0.5).astype(np.uint8), float(err.max()) if err.size else 0.0


# --------------------------------------------------------------------------------------------------------------- plates
def _rgba(rgb, alpha=1.0):
    rgb = np.asarray(rgb, np.float32)
    a = np.broadcast_to(np.asarray(alpha, np.float32), rgb.shape[:2])[..., None]
    return np.concatenate([rgb, a], axis=2).astype(np.float32)


def plate_ramp_bars(n=PLATE):
    """Top half: a grey ramp from -0.1 to 1.5 (negative, mid and HDR values); bottom half: eight colour bars."""
    out = np.zeros((n, n, 3), np.float32)
    out[: n // 2] = np.linspace(-0.1, 1.5, n, dtype=np.float32)[None, :, None]
    bars = [(1, 1, 1), (1, 1, 0), (0, 1, 1), (0, 1, 0), (1, 0, 1), (1, 0, 0), (0, 0, 1), (0.18, 0.18, 0.18)]
    for i, c in enumerate(bars):
        out[n // 2:, i * n // 8:(i + 1) * n // 8] = np.asarray(c, np.float32) * 0.8
    alpha = np.ones((n, n), np.float32)
    alpha[n // 2:, :] = np.linspace(0.25, 1.0, n, dtype=np.float32)[None, :]     # partial alpha on the bars for premult nodes
    return _rgba(out, alpha)


def plate_bars_b(n=PLATE):
    """A second colour plate for two-input nodes: an orange disc over a blue-to-magenta gradient."""
    yy, xx = np.mgrid[:n, :n].astype(np.float32)
    out = np.stack([xx / n * 0.6, np.full_like(xx, 0.15), np.full_like(xx, 0.7)], axis=2)
    disc = (xx - n * 0.6) ** 2 + (yy - n * 0.45) ** 2 < (n * 0.25) ** 2
    out[disc] = (1.0, 0.55, 0.1)
    alpha = np.where(disc, 1.0, 0.6).astype(np.float32)
    return _rgba(out, alpha)


def plate_mask(n=PLATE):
    yy, xx = np.mgrid[:n, :n].astype(np.float32)
    r = np.sqrt((xx - n / 2) ** 2 + (yy - n / 2) ** 2) / (n / 2)
    m = np.clip(1.2 - r, 0, 1)
    return np.repeat(m[..., None], 4, axis=2).astype(np.float32)


def plate_step_square(n=PLATE):
    """A step edge (dark left, light right) with a white square in the middle: blurs and sharpens read at once."""
    out = np.full((n, n, 3), 0.1, np.float32)
    out[:, n // 2:] = 0.6
    out[n * 3 // 8:n * 5 // 8, n * 3 // 8:n * 5 // 8] = 1.0
    out[n // 8:n // 8 + 4, n // 8:n * 7 // 8] = (1.0, 0.3, 0.1)    # a thin coloured line for direction-dependent filters
    return _rgba(out, 1.0)


def plate_step(n=PLATE):
    out = np.full((n, n, 3), 0.0, np.float32)
    out[:, n // 2:] = 1.0
    return _rgba(out, 1.0)


def plate_square(n=PLATE):
    out = np.zeros((n, n, 3), np.float32)
    out[n * 3 // 8:n * 5 // 8, n * 3 // 8:n * 5 // 8] = 1.0
    return _rgba(out, np.where(out[..., 0] > 0, 1.0, 0.0))


def plate_grid(n=PLATE):
    """A grid with a coloured top-left corner and a marked centre: orientation and filtering both show."""
    yy, xx = np.mgrid[:n, :n]
    out = np.full((n, n, 3), 0.2, np.float32)
    lines = ((xx % 32) < 2) | ((yy % 32) < 2)
    out[lines] = 0.9
    out[: n // 4, : n // 4] *= np.asarray((1.0, 0.3, 0.2), np.float32)
    out[n // 2 - 6:n // 2 + 6, n // 2 - 6:n // 2 + 6] = (0.2, 0.9, 0.3)
    return _rgba(out, 1.0)


def plate_screen(screen, n=PLATE):
    """A screen-colour plate: the screen, a skin-tone disc with a soft edge, and a band of skin with green spill."""
    from nuke_fidelity_cases import KEY_SKIN, KEY_SPILL
    yy, xx = np.mgrid[:n, :n].astype(np.float32)
    r = np.sqrt((xx - n * 0.45) ** 2 + (yy - n * 0.45) ** 2)
    t = np.clip((n * 0.28 - r) / (n * 0.06), 0, 1)[..., None]             # 1 inside the disc, a soft edge
    out = np.asarray(screen, np.float32) * (1 - t) + np.asarray(KEY_SKIN, np.float32) * t
    out[int(n * 0.82):int(n * 0.92), int(n * 0.1):int(n * 0.9)] = KEY_SPILL
    return _rgba(out, 1.0)


def review_fixtures(group, small):
    """256 x 256 plates under the audit fixture's names, built for the node's group. Names without a dedicated plate
    keep the audit fixture's content, enlarged."""
    from nuke_fidelity_cases import KEY_BLUE_SCREEN, KEY_SCREEN
    if group == "Keyer":
        a = plate_screen(KEY_SCREEN)
    elif group == "Filter":
        a = plate_step_square()
    elif group == "Transform":
        a = plate_grid()
    else:
        a = plate_ramp_bars()
    built = {"A": a, "B": plate_bars_b(), "mask": plate_mask(), "C": plate_ramp_bars(), "S": plate_step(), "W": plate_square(),
             "E": plate_step(), "K": plate_screen(KEY_SCREEN), "KS": plate_screen(KEY_SCREEN), "KI": plate_screen(KEY_SCREEN),
             "KB": plate_screen(KEY_BLUE_SCREEN)}
    out = {}
    for name, arr in small.items():
        out[name] = built.get(name, enlarge(arr).astype(np.float32))
    return out


PLATE_NAMES = {"ramp": "a grey ramp (-0.1 to 1.5) over colour bars", "step": "a step edge and a white square",
               "grid": "a grid", "screen": "a screen-colour plate (screen, skin disc, spill band)"}


def plate_name(group):
    return PLATE_NAMES["screen" if group == "Keyer" else "step" if group == "Filter" else "grid" if group == "Transform" else "ramp"]


# ---------------------------------------------------------------------------------------------------------- the panels
class PanelRenderer:
    """NodeBased's real Properties panel, built offscreen by the app's own `Window.build_node_panel`. QSettings go to a
    private temporary directory so the developer's saved layout is neither read nor written."""

    def __init__(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtCore import QSettings
        self._settings = tempfile.TemporaryDirectory(prefix="nb-review-book-settings-")
        for fmt in (QSettings.Format.NativeFormat, QSettings.Format.IniFormat):
            QSettings.setPath(fmt, QSettings.Scope.UserScope, self._settings.name)
        QSettings.setDefaultFormat(QSettings.Format.IniFormat)
        from PySide6.QtWidgets import QApplication
        self.app = QApplication.instance() or QApplication([])
        from nodebased import app as nbapp
        self.app.setStyle("Fusion")
        self.app.setStyleSheet(nbapp.STYLE)
        nbapp.Preferences.workspace = lambda self: None          # never restore the developer's saved workspace
        self.window = nbapp.Window()
        self.window.show()
        self.app.processEvents()
        self._count = 0

    def render(self, kind, out_dir, stem):
        """[(tab name, png path)] for every tab of the node's panel."""
        from PySide6.QtWidgets import QTabWidget
        self._count += 1
        key = f"review_{self._count}"
        self.window.dispatcher.execute({"op": "create", "id": key, "type": kind, "pos": [0, 0]})
        panel = self.window.build_node_panel(key)
        panel.setParent(None)
        panel.resize(PANEL_WIDTH, 200)
        panel.show()
        self.app.processEvents()
        tabs = panel.findChild(QTabWidget, "node-tabs")
        names = [tabs.tabText(i) for i in range(tabs.count())] if tabs is not None else ["Knobs"]
        shots = []
        for i, name in enumerate(names):
            if tabs is not None:
                tabs.setCurrentIndex(i)
            self.app.processEvents()
            panel.resize(PANEL_WIDTH, max(200, panel.sizeHint().height()))
            self.app.processEvents()
            path = Path(out_dir) / f"{stem}_panel_{i}_{slug(name)}.png"
            if not panel.grab().save(str(path)):
                raise OSError(f"could not write {path}")
            shots.append((name, path))
        panel.close()
        panel.deleteLater()
        return shots

    def close(self):
        # Mark the document saved first: closing with unsaved edits opens the discard prompt, which would wait forever offscreen.
        self.window.saved_document = self.window.dispatcher.document
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()


# ------------------------------------------------------------------------------------------------------- the alignment
_WIDGETS = {"number": {"float", "float_slider", "int", "split", "row", "xy", "color"}, "bool": {"bool"},
            "string": {"string", "file_read", "file_write", "multiline", "enum"}, "enum": {"enum"},
            "channels": {"string", "enum"}, "number_list": {"xy", "xyz", "color", "split", "curve", "row"}}


def _norm_label(text):
    return re.sub(r"[^a-z0-9]+", "", str(text or "").lower())


def alignment(nuke_node, pair, kind):
    """One row per Nuke knob: the NodeBased pairing and its mismatch marks."""
    from audit_nuke_fidelity import _widget_kind, compare_default
    from nodebased import core
    from nodebased.knobs import knob_layout
    import nuke_fidelity_requirements as req
    entries = pair.get("knobs", [])
    by_nuke = {}
    for e in entries:
        for n in e.get("nuke", []) or []:
            by_nuke.setdefault(n, e)
    groups = list(knob_layout(kind)) if kind in core.SPECS else []
    position, group_of = {}, {}
    for i, g in enumerate(groups):
        for p in g.params:
            position.setdefault(p, i)
            group_of.setdefault(p, g)
    params = core.SPECS.get(kind, {}).get("params", {})
    # Order: the longest run of paired knobs already in Nuke's order is kept; the knobs outside it are out of order.
    seq = []
    for k in nuke_node["knobs"]:
        e = by_nuke.get(k["name"])
        spots = [position[p] for p in (e or {}).get("nodebased", []) if p in position] if e and e["kind"] in req.PAIRED else []
        seq.append(min(spots) if spots else None)
    placed = [(i, s) for i, s in enumerate(seq) if s is not None]
    keep = _lis_indices([s for _, s in placed])
    in_order = {placed[j][0] for j in keep}
    rows = []
    for i, k in enumerate(nuke_node["knobs"]):
        e = by_nuke.get(k["name"])
        row = {"tab": k["tab"], "name": k["name"], "label": k["label"], "type": k["type"], "default": k["default"],
               "primary": k["primary"], "status": "missing", "nodebased": "", "nb_label": "", "nb_widget": "", "nb_default": "",
               "marks": [], "reason": ""}
        if e is None:
            row["reason"] = "no pairing in nodebased_to_nuke.json"
        elif e["kind"] == "unsupported":
            row["status"], row["reason"] = "unsupported", e.get("reason", "")
        elif e["kind"] == "panel":
            row["status"], row["reason"] = "panel", e.get("reason", "") or "handled by the panel, not a param"
            row["nodebased"] = ", ".join(e.get("nodebased", []))
        else:
            row["status"] = e["kind"]
            nb = e.get("nodebased", [])
            row["nodebased"] = ", ".join(nb)
            first = nb[0] if nb else None
            g = group_of.get(first)
            if g is not None:
                row["nb_label"] = g.label or ""
                try:
                    row["nb_widget"] = _widget_kind(g, first)
                except Exception:  # noqa: BLE001 - a group without members still reads as its kind
                    row["nb_widget"] = g.kind
                if g.kind == "row":
                    row["nb_widget"] += f" (row: {', '.join(g.params)})"
                if seq[i] is not None and i not in in_order:
                    row["marks"].append("order")
                if len(nb) == 1 and _norm_label(g.label) not in (_norm_label(k["label"]), _norm_label(k["name"])) \
                        and _norm_label(k["label"]) not in ("na", ""):
                    row["marks"].append("label")
                ok_widgets = _WIDGETS.get(k["type"])
                base = (row["nb_widget"] or "").split(" ")[0]
                if ok_widgets and base not in ok_widgets:
                    row["marks"].append("widget")
            elif first is not None:
                row["marks"].append("not laid out")
            if first in params and len(nb) == 1 and k["type"] not in ("unknown",):
                row["nb_default"] = params[first]
                try:
                    same = compare_default(params[first], k["default"], pair.get("value_map"))
                except Exception:  # noqa: BLE001
                    same = None
                if same is False:
                    row["marks"].append("default")
        rows.append(row)
    return rows


def _lis_indices(seq):
    """Indices of one longest non-decreasing subsequence."""
    if not seq:
        return []
    best = [1] * len(seq)
    prev = [-1] * len(seq)
    for i in range(len(seq)):
        for j in range(i):
            if seq[j] <= seq[i] and best[j] + 1 > best[i]:
                best[i], prev[i] = best[j] + 1, j
    i = max(range(len(seq)), key=lambda x: best[x])
    out = []
    while i != -1:
        out.append(i)
        i = prev[i]
    return out[::-1]


# ------------------------------------------------------------------------------------------------------------- the cases
def run_on(case, fx):
    """(output, expected, valid) of a case on the given fixtures; raises when the case cannot run on them."""
    from nuke_fidelity_cases import run_nodebased
    out = run_nodebased(case, fx)
    if out is None:
        return None, None, None
    ref, valid = case.expected(fx)
    if np.shape(out) != np.shape(ref) or np.shape(out)[:2] != np.shape(fx["A"])[:2]:
        raise ValueError("shape differs from the plate")
    return np.asarray(out), np.asarray(ref), np.asarray(valid, bool)


def _special_document(case):
    return bool(case.nb_params) and any(k in case.nb_params for k in ("_remove", "_shapes", "_items", "_tracks", "_plate", "_stamp", "_marker_disc"))


def case_images(case, group, small_fx, big_fx, out_dir, stem):
    """Write the case's images; returns a dict for the page."""
    import audit_nuke_fidelity as audit
    res = audit.run_case(case)
    info = {"case": case, "audit": res, "plate": None, "images": {}, "plate_error": None, "nuke": None}
    first_input = case.nb_inputs[0] if case.nb_inputs else None
    attempts = []
    if case.nb_params is not None and not case.unscored and not _special_document(case):
        attempts.append(("review", big_fx))
    attempts.append(("audit", small_fx))
    for which, fx in attempts:
        if case.nb_params is None:
            break
        try:
            out, ref, valid = run_on(case, fx)
        except Exception as exc:  # noqa: BLE001 - a plate the case cannot run on falls back to the audit fixture
            info["plate_error"] = f"{type(exc).__name__}: {str(exc)[:100]}"
            continue
        if out is None:
            break
        grow = (lambda a: a) if which == "review" else enlarge
        tag = "" if info["plate"] is None else "fx_"
        target = info["images"] if info["plate"] is None else info.setdefault("fixture_images", {})
        if first_input:
            target["input"] = save_png(out_dir / f"{stem}_{slug(case.id)}_{tag}in.png", to_display(grow(fx[first_input])))
        target["output"] = save_png(out_dir / f"{stem}_{slug(case.id)}_{tag}out.png", to_display(grow(out)))
        target["expected"] = save_png(out_dir / f"{stem}_{slug(case.id)}_{tag}exp.png", to_display(grow(np.where(valid, ref, 0.0))))
        heat, worst = diff_image(out, ref, valid)
        target["diff"] = save_png(out_dir / f"{stem}_{slug(case.id)}_{tag}diff.png", grow(heat))
        if info["plate"] is not None:
            break
        info["plate"] = which
        info["plate_max_error"] = worst
        # The case's expectation was written for the audit fixture. When it does not carry over to the larger plate
        # although the scored case passes, show the fixture the case was scored on as well, and say so on the page.
        if which != "review" or worst <= 1e-4 or res["status"] != "pass":
            break
    # Nuke's own render of the scored case, when someone has rendered it (docs/parity/nuke_renders/<case>.exr).
    render = RENDERS_DIR / f"{case.id}.exr"
    if render.exists() and case.nb_params is not None:
        from make_nuke_reference_pack import compare_arrays, load_render
        from nuke_fidelity_cases import run_nodebased
        theirs = load_render(RENDERS_DIR, case.id)
        ours = run_nodebased(case, small_fx)
        cmp = compare_arrays(ours, theirs)
        info["nuke"] = cmp
        info["images"]["nuke"] = save_png(out_dir / f"{stem}_{slug(case.id)}_nuke.png", to_display(enlarge(theirs)))
        if ours.shape == theirs.shape:
            heat, _ = diff_image(ours, theirs)
            info["images"]["nuke_diff"] = save_png(out_dir / f"{stem}_{slug(case.id)}_nukediff.png", enlarge(heat))
    return info


def knob_strip(kind, nuke_node, pair, cases, group, big_fx, out_dir, stem):
    """The node at its defaults and with each paired main-tab knob moved, on the review plate."""
    import nuke_fidelity_requirements as req
    from nodebased import core
    template = next((c for c in cases if c.nb_params is not None and not _special_document(c) and c.nb_kind == kind), None)
    if template is None or kind not in core.SPECS:
        return [], "no case of this node runs on a plain image, so there is no plate to move knobs on"
    spec_params = core.SPECS[kind]["params"]
    primary = {k["name"] for k in nuke_node["knobs"] if k["primary"] and k["role"] == "knob"}
    shots = []
    base = dataclasses.replace(template, nb_params={k: v for k, v in template.nb_params.items() if k.startswith("_")})
    try:
        out, _, _ = run_on(dataclasses.replace(base, expected=lambda fx: (np.zeros_like(fx["A"]), np.ones(fx["A"].shape, bool))), big_fx)
    except Exception as exc:  # noqa: BLE001
        return [], f"the node did not run on the review plate ({type(exc).__name__})"
    shots.append(("defaults", save_png(out_dir / f"{stem}_knob_default.png", to_display(out))))
    seen = set()
    for e in pair.get("knobs", []):
        if e["kind"] not in req.PAIRED or not set(e.get("nuke", []) or []) & primary:
            continue
        for p in e.get("nodebased", []):
            if p in seen or p not in spec_params:
                continue
            seen.add(p)
            value = spec_params[p]
            moved = None
            for cand in req.candidates(p, value):
                if cand != value:
                    moved = cand
                    break
            if moved is None:
                continue
            case = dataclasses.replace(base, nb_params=dict(base.nb_params, **{p: moved}),
                                       expected=lambda fx: (np.zeros_like(fx["A"]), np.ones(fx["A"].shape, bool)))
            try:
                img, _, _ = run_on(case, big_fx)
            except Exception:  # noqa: BLE001 - a value the kernel rejects is skipped, not drawn
                continue
            acts = not np.allclose(img, out, atol=1e-6)
            caption = f"{', '.join(e['nuke'])} ({p}) = {moved!r}" + ("" if acts else " (no visible change)")
            shots.append((caption, save_png(out_dir / f"{stem}_knob_{slug(p)}.png", to_display(img))))
            if len(shots) > 9:
                break
    return shots, ""


# ---------------------------------------------------------------------------------------------------------------- pages
CSS = """body{font:14px/1.4 system-ui,sans-serif;margin:24px;background:#16181c;color:#d8dde3}a{color:#7fb6ff}
table{border-collapse:collapse;margin:8px 0}td,th{border:1px solid #333a43;padding:3px 6px;vertical-align:top;text-align:left}
th{background:#22262c}.mark{color:#ff9a6b;font-weight:600}.pass{color:#7fdc8a}.fail{color:#ff7b72}.ns{color:#9aa4ae}
.cols{display:flex;gap:24px;align-items:flex-start;flex-wrap:wrap}.cols>div{flex:1;min-width:420px}img{image-rendering:pixelated;max-width:256px}
.panel img{max-width:480px;image-rendering:auto}figure{display:inline-block;margin:4px}figcaption{font-size:12px;color:#9aa4ae;max-width:256px}"""


def _fmt_default(v):
    return "" if v == "" or v is None else esc(json.dumps(v) if not isinstance(v, str) else v)


def node_page(name, row, nuke_node, pair, kind, panels, align, case_infos, strip, strip_note, interactions):
    rel = lambda p: os.path.basename(str(p))  # noqa: E731 - images sit beside the page
    out = [f"<!doctype html><meta charset=utf-8><title>{esc(name)} review</title><style>{CSS}</style>",
           f"<p><a href='index.html'>&larr; index</a></p><h1>{esc(name)} <small>({esc(row.get('group', ''))})</small></h1>",
           f"<p>NodeBased node <code>{esc(kind)}</code>. Fidelity {esc(row.get('fidelity', '?'))}; behaviour cases {esc(row.get('math', '?'))}. "
           f"Nuke 17.0 Reference Guide: <a href='{esc(nuke_node.get('page', GUIDE_HOME))}'>{esc(nuke_node.get('guide_title', name))}</a>.</p>",
           "<h2>1. Does the panel look like Nuke's?</h2><div class=cols><div><h3>Nuke (Reference Guide knob table)</h3>"]
    tabs = {}
    for k in nuke_node["knobs"]:
        tabs.setdefault(k["tab"], []).append(k)
    for tab, knobs in tabs.items():
        out.append(f"<h4>{esc(tab)}</h4><table><tr><th>knob</th><th>label</th><th>widget</th><th>default</th></tr>")
        for k in knobs:
            out.append(f"<tr><td><code>{esc(k['name'])}</code></td><td>{esc(k['label'])}</td><td>{esc(k['type'])}</td>"
                       f"<td>{_fmt_default(k['default'])}</td></tr>")
        out.append("</table>")
    out.append("<p class=ns>Foundry's own screenshots are on the guide page linked above; they are not copied here.</p></div>")
    out.append("<div class=panel><h3>NodeBased (Properties panel, rendered offscreen)</h3>")
    for tab, path in panels:
        out.append(f"<h4>{esc(tab)}</h4><img alt='{esc(tab)} tab' src='{esc(rel(path))}'>")
    out.append("</div></div><h3>Alignment</h3><p>One row per Nuke knob. Marks: <span class=mark>order</span> (outside the longest run "
               "already in Nuke's order), <span class=mark>label</span>, <span class=mark>widget</span>, <span class=mark>default</span>. "
               "The guide does not say which knobs share a row, so rows are shown (NodeBased's row groups) but not marked.</p>"
               "<table><tr><th>tab</th><th>Nuke knob</th><th>label</th><th>type</th><th>default</th><th>NodeBased</th><th>label</th>"
               "<th>widget</th><th>default</th><th>marks</th></tr>")
    for r in align:
        if r["status"] in ("missing", "unsupported", "panel"):
            nb = f"<i>{esc(r['status'])}</i>" + (f": {esc(r['reason'])}" if r["reason"] else "")
        else:
            nb = f"<code>{esc(r['nodebased'])}</code> <span class=ns>({esc(r['status'])})</span>"
        out.append(f"<tr><td>{esc(r['tab'])}</td><td><code>{esc(r['name'])}</code></td><td>{esc(r['label'])}</td><td>{esc(r['type'])}</td>"
                   f"<td>{_fmt_default(r['default'])}</td><td>{nb}</td><td>{esc(r['nb_label'])}</td><td>{esc(r['nb_widget'])}</td>"
                   f"<td>{_fmt_default(r['nb_default'])}</td><td class=mark>{esc(' '.join(r['marks']))}</td></tr>")
    out.append("</table><h3>Per-widget behaviours</h3>")
    if interactions is None:
        out.append("<p class=ns>docs/parity/nuke_knob_interactions.json is not on main yet (lane 10 adds it); this section fills in when it is.</p>")
    else:
        items = interactions.get(name) or interactions.get("nodes", {}).get(name) or []
        out.append("<pre>" + esc(json.dumps(items, indent=1)[:4000]) + "</pre>" if items else "<p class=ns>No entries for this node.</p>")
    out.append("<h2>2. Does it compute what Nuke computes?</h2>")
    if not case_infos:
        out.append("<p class=ns>The audit has no behaviour case for this node.</p>")
    for info in case_infos:
        case, res = info["case"], info["audit"]
        status = res["status"]
        cls = "pass" if status == "pass" else "ns" if status in ("not scored", "inexpressible") else "fail"
        cite = "guide" if case.source == "guide" else "<i>convention</i>"
        out.append(f"<h3 id='{esc(case.id)}'><code>{esc(case.id)}</code> <span class={cls}>{esc(status)}</span></h3>"
                   f"<p>{esc(case.statement)} <span class=ns>[source: {cite}]</span></p>"
                   f"<p class=ns>Nuke knobs: {esc(', '.join(f'{k}={v}' for k, v in case.nuke_knobs) or 'defaults')}. "
                   f"The scored case ran on the audit's 16 x 12 fixture (max error {esc(res.get('max_abs_error'))}). ")
        if case.unscored:
            out.append(f"Not scored: {esc(case.unscored)}. ")
        if case.nb_params is None:
            out.append(f"NodeBased cannot express this setting: {esc(case.inexpressible)}.")
        if info["plate"] == "review":
            out.append(f"Shown on the 256 x 256 review plate: {esc(plate_name(row.get('group')))}; max error there "
                       f"{info.get('plate_max_error', 0):.3g}.</p>")
        elif info["plate"] == "audit":
            out.append("Shown on the audit fixture itself, enlarged" + (f" (the review plate failed: {esc(info['plate_error'])})" if info["plate_error"] else "") + ".</p>")
        else:
            out.append("</p>")
        figs = [("input", "input"), ("output", "NodeBased"), ("expected", "expected (the case's statement)"),
                ("diff", "|NodeBased - expected| x10"), ("nuke", "Nuke rendered"), ("nuke_diff", "|NodeBased - Nuke| x10")]
        for key, label in figs:
            if key in info["images"]:
                out.append(f"<figure><img src='{esc(rel(info['images'][key]))}' alt='{esc(label)}'><figcaption>{esc(label)}</figcaption></figure>")
        if info.get("fixture_images"):
            out.append("<p class=mark>The case's expectation was written for the audit fixture and does not carry over to the review "
                       "plate (the difference above), while the scored case passes. The fixture it was scored on, enlarged:</p>")
            for key, label in figs[:4]:
                if key in info["fixture_images"]:
                    out.append(f"<figure><img src='{esc(rel(info['fixture_images'][key]))}' alt='{esc(label)}'>"
                               f"<figcaption>{esc(label)} (audit fixture)</figcaption></figure>")
        if info["nuke"]:
            out.append(f"<p>Against Nuke's render: <span class={'pass' if info['nuke']['status'] == 'pass' else 'fail'}>{esc(info['nuke']['status'])}</span></p>")
    out.append("<h3>Knobs that act</h3>")
    if strip_note:
        out.append(f"<p class=ns>{esc(strip_note)}</p>")
    for caption, path in strip:
        out.append(f"<figure><img src='{esc(rel(path))}' alt='{esc(caption)}'><figcaption>{esc(caption)}</figcaption></figure>")
    return "\n".join(out) + "\n"


def index_page(entries, built, seconds):
    out = [f"<!doctype html><meta charset=utf-8><title>NodeBased Nuke parity review book</title><style>{CSS}</style>",
           "<h1>NodeBased: Nuke parity review book</h1>",
           f"<p>{len(built)} node pages, built in {seconds:.0f} s. Nodes verified against Nuke's own pixels: "
           f"{sum(1 for b in built.values() if b['nuke_verified'])} (renders read from docs/parity/nuke_renders/).</p>"]
    groups = {}
    for name, row in entries.items():
        groups.setdefault(row["group"], []).append((row["rank"], name, row))
    for group, items in groups.items():
        out.append(f"<h2>{esc(group)}</h2><table><tr><th>rank</th><th>Nuke node</th><th>NodeBased</th><th>fidelity</th>"
                   "<th>behaviour cases</th><th>verified against Nuke</th></tr>")
        for rank, name, row in sorted(items):
            b = built.get(name)
            link = f"<a href='{esc(b['page'])}'>{esc(name)}</a>" if b else esc(name)
            out.append(f"<tr><td>{rank}</td><td>{link}</td><td><code>{esc(row['nodebased'])}</code></td><td>{esc(row['fidelity'])}</td>"
                       f"<td>{esc(row['math'])}</td><td>{'yes' if b and b['nuke_verified'] else ''}</td></tr>")
        out.append("</table>")
    return "\n".join(out) + "\n"


# ----------------------------------------------------------------------------------------------------------------- build
def build(out_dir, nodes=None):
    """Write the book into `out_dir`. Returns {"pages": {node: page info}, "seconds": float, "index": path}."""
    started = time.time()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    nuke_doc, map_doc = load_json(NUKE_JSON), load_json(MAP_JSON)
    rows = fidelity_rows()
    interactions = load_json(INTERACTIONS_JSON) if INTERACTIONS_JSON.exists() else None
    from nuke_fidelity_cases import cases_for, make_fixtures
    small = make_fixtures()
    names = list(nodes) if nodes else list(rows)
    renderer = PanelRenderer()
    built = {}
    try:
        for name in names:
            row = rows.get(name, {"group": nuke_doc["nodes"].get(name, {}).get("group", "Other"), "rank": 0, "nodebased": "",
                                  "fidelity": "?", "math": "?"})
            pair = map_doc["pairs"].get(name)
            nuke_node = nuke_doc["nodes"].get(name)
            if pair is None or nuke_node is None:
                continue
            kind = pair.get("nodebased") or ""
            stem = slug(name)
            from nodebased import core
            panels = renderer.render(kind, out_dir, stem) if kind in core.SPECS else []
            align = alignment(nuke_node, pair, kind)
            cases = cases_for(name)
            big = review_fixtures(row["group"], small)
            infos = [case_images(c, row["group"], small, big, out_dir, stem) for c in cases]
            strip, note = knob_strip(kind, nuke_node, pair, cases, row["group"], big, out_dir, stem)
            page = f"{stem}.html"
            (out_dir / page).write_text(node_page(name, row, nuke_node, pair, kind, panels, align, infos, strip, note, interactions),
                                        encoding="utf-8")
            built[name] = {"page": page, "panels": [str(p) for _, p in panels], "cases": len(cases),
                           "marks": [(r["name"], r["marks"]) for r in align if r["marks"]],
                           "nuke_verified": any(i["nuke"] and i["nuke"]["status"] == "pass" for i in infos)}
    finally:
        renderer.close()
    seconds = time.time() - started
    index = out_dir / "index.html"
    index.write_text(index_page(rows, built, seconds), encoding="utf-8")
    return {"pages": built, "seconds": seconds, "index": index}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(ROOT / "docs" / "parity" / "review" / "out"))
    ap.add_argument("--nodes", help="comma-separated Nuke node names (default: every scored node)")
    ap.add_argument("--ten", action="store_true", help="the ten high-use nodes: " + ", ".join(TEN))
    args = ap.parse_args(argv)
    nodes = TEN if args.ten else (args.nodes.split(",") if args.nodes else None)
    res = build(args.out, nodes)
    print(f"wrote {len(res['pages'])} pages and {res['index']} in {res['seconds']:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
