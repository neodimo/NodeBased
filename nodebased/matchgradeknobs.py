"""MatchGrade's reference frames, CDL, locks, iterations and export (2D behaviour lane 2, step K finish).

Nuke's MatchGrade analyses the frames listed on its Ref Frames tab (Source and Target), fits a transform (a 3D LUT or a CDL) and shows the CDL
it found: slope, offset, power and saturation, each with a lock that keeps its value while Recalculate CDL fits the others, and an error. The
Export tab writes the transform as a LUT file or builds the nodes that apply it.

NodeBased reads this the following way. The statistical match NodeBased always had is the "3D LUT" transform. The "CDL" transform applies the
CDL the node holds (slope = gain, offset, power = 1 / gamma, then saturation on Rec.709 luma) and reads no target. Analyze Reference Frames pools
the pictures of the listed frames and bakes the per-channel gain and offset; Recalculate CDL fits slope, offset, power and saturation to the same
pictures by Levenberg-Marquardt, `iterations` steps, each step kept only if the error falls, so more iterations never fit worse. With the
source and target pixel-aligned (Match Graded Source) the fit is over the pixel pairs; for Match Different Clip it is over matching quantiles.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

LUMA = np.array([0.2126, 0.7152, 0.0722])           # Rec.709 luma, the weights the ASC CDL names for saturation
POWER_RANGE = (0.05, 10.0)
SATURATION_RANGE = (0.0, 4.0)
LOCKS = ("slope", "offset", "power", "saturation")
MAX_SAMPLES = 4096
QUANTILES = 256
CDL_NAMESPACE = "urn:ASC:CDL:v1.2"


# ------------------------------------------------------------------------------------------------------------- the frame lists
def parse_frames(text, fallback=None):
    """The frames a Ref Frames list names ("1, 5, 9-12"), sorted and without repeats; an empty list gives `fallback` (the current frame)."""
    frames = set()
    for part in re.split(r"[,;\s]+", str(text or "").strip()):
        found = re.fullmatch(r"(-?\d+)(?:-(-?\d+))?", part)
        if not found:
            continue
        first = int(found.group(1))
        last = int(found.group(2)) if found.group(2) is not None else first
        frames.update(range(min(first, last), max(first, last) + 1))
    if not frames and fallback is not None:
        return [int(fallback)]
    return sorted(frames)


def _text(frames):
    return ", ".join(str(f) for f in frames)


def add_frame(text, frame):
    """Nuke's Add button: the current frame joins the list."""
    return _text(sorted(set(parse_frames(text)) | {int(frame)}))


def delete_frame(text, frame):
    """Nuke's Delete button: the current frame leaves the list."""
    return _text([f for f in parse_frames(text) if f != int(frame)])


# ------------------------------------------------------------------------------------------------------------------- the CDL
def signed_power(x, power):
    """x ** power for either sign of x, the way MatchGrade's gamma already works on a negative value."""
    return np.sign(x) * np.power(np.abs(x), power)


def saturate(rgb, saturation):
    """The ASC CDL saturation: each colour moves away from (above 1) or towards (below 1) its Rec.709 luma."""
    luma = (rgb * LUMA.astype(rgb.dtype)).sum(axis=-1, keepdims=True)
    return luma + saturation * (rgb - luma)


def cdl_of(p):
    """(slope, offset, power, saturation) the node holds: gain, offset with lift added, 1 / gamma, saturation."""
    slope = np.array([float(p.get(f"grade_gain_{c}", 1.0)) for c in "rgb"])
    offset = np.array([float(p.get(f"grade_offset_{c}", 0.0)) + float(p.get(f"grade_lift_{c}", 0.0)) for c in "rgb"])
    power = np.array([1.0 / max(1e-6, float(p.get(f"grade_gamma_{c}", 1.0))) for c in "rgb"])
    return slope, offset, power, float(p.get("match_saturation", 1.0))


def apply_cdl(rgb, slope, offset, power, saturation):
    return saturate(signed_power(np.asarray(rgb, np.float64) * slope + offset, power), saturation)


# ------------------------------------------------------------------------------------------------------------------- the fits
def _wquantiles(values, weights, count=QUANTILES):
    """`count` evenly spaced weighted quantiles of `values`."""
    order = np.argsort(values, kind="stable")
    v, w = values[order], weights[order]
    positions = (np.cumsum(w) - 0.5 * w) / max(float(w.sum()), 1e-12)
    return np.interp((np.arange(count) + 0.5) / count, positions, v)


def _subsample(*arrays):
    count = arrays[0].shape[0]
    if count <= MAX_SAMPLES:
        return arrays
    index = np.linspace(0, count - 1, MAX_SAMPLES).astype(int)
    return tuple(a[index] for a in arrays)


def stat_bake(source, target, weights=None):
    """Analyze Reference Frames: the per-channel gain and offset that bring the source's mean and spread to the target's, as the `set` commands'
    values (grade_gain_*, grade_offset_*). `source` and `target` are the pooled pixels of the listed frames, (N, 3) each; `weights` weigh the source."""
    source, target = np.asarray(source, np.float64), np.asarray(target, np.float64)
    w_s = np.ones(len(source)) if weights is None else np.asarray(weights, np.float64)
    w_t = np.ones(len(target)) if weights is None or len(weights) != len(target) else w_s
    out = {}
    for c, channel in enumerate("rgb"):
        sm, ss = _mean_std(source[:, c], w_s)
        tm, ts = _mean_std(target[:, c], w_t)
        gain = ts / ss if ss > 1e-12 else 1.0
        out[f"grade_gain_{channel}"] = gain
        out[f"grade_offset_{channel}"] = tm - sm * gain
    return out


def _mean_std(values, weights):
    total = max(float(weights.sum()), 1e-12)
    mean = float((values * weights).sum() / total)
    return mean, float(np.sqrt((((values - mean) ** 2) * weights).sum() / total))


def fit_cdl(source, target, p, weights=None, aligned=True, iterations=None):
    """Recalculate CDL: slope, offset, power and saturation (with the error) that carry `source` to `target`, both (N, 3) pooled pixels.

    The values held in `p` stay where `p` has the matching lock on. The fit starts from the per-channel mean and spread match, then takes
    `iterations` Levenberg-Marquardt steps (default `match_iterations`); a step is kept only when it lowers the error, so the error never
    rises with more iterations. `aligned` fits the pixel pairs; otherwise the sources and targets are paired by their quantiles, and the
    saturation is the ratio of the two pictures' chroma. Returns slope, offset (the total, lift included), power, saturation and error."""
    source, target = np.asarray(source, np.float64), np.asarray(target, np.float64)
    weights = np.ones(len(source)) if weights is None else np.asarray(weights, np.float64)
    iterations = int(p.get("match_iterations", 6) if iterations is None else iterations)
    held_slope, held_offset, held_power, held_sat = cdl_of(p)
    lock = {name: bool(p.get(f"match_lock_{name}", 0)) for name in LOCKS}
    if aligned and len(source) == len(target):
        s, t, w = _subsample(source, target, weights)
        per_channel = False
    else:
        w_t = weights if len(weights) == len(target) else np.ones(len(target))
        s = np.stack([_wquantiles(source[:, c], weights) for c in range(3)], axis=1)
        t = np.stack([_wquantiles(target[:, c], w_t) for c in range(3)], axis=1)
        w = np.ones(len(s))
        per_channel = True
    slope, offset, power, saturation = held_slope.copy(), held_offset.copy(), held_power.copy(), held_sat
    # the start: the mean and spread match, honouring the locks
    for c in range(3):
        sm, ss = _mean_std(s[:, c], w)
        tm, ts = _mean_std(t[:, c], w)
        if not lock["slope"]:
            slope[c] = ts / ss if ss > 1e-12 else 1.0
        if not lock["offset"]:
            offset[c] = tm - sm * slope[c]
    if not lock["power"]:
        power[:] = 1.0
    if per_channel:
        free_sat = False
    else:
        free_sat = not lock["saturation"]
        if free_sat:
            saturation = 1.0
    layout = [("slope", c) for c in range(3) if not lock["slope"]] + [("offset", c) for c in range(3) if not lock["offset"]] \
        + [("power", c) for c in range(3) if not lock["power"]] + ([("saturation", 0)] if free_sat else [])
    root = np.sqrt(w)[:, None]

    def pack():
        values = {"slope": slope, "offset": offset, "power": power}
        return np.array([values[name][c] if name != "saturation" else saturation for name, c in layout], np.float64)

    def unpack(theta):
        values = {"slope": slope.copy(), "offset": offset.copy(), "power": power.copy()}
        sat = saturation
        for (name, c), v in zip(layout, theta):
            if name == "saturation":
                sat = float(np.clip(v, *SATURATION_RANGE))
            elif name == "power":
                values[name][c] = float(np.clip(v, *POWER_RANGE))
            else:
                values[name][c] = v
        return values["slope"], values["offset"], values["power"], sat

    def residual(theta):
        sl, off, pw, sat = unpack(theta)
        return ((apply_cdl(s, sl, off, pw, sat if not per_channel else 1.0) - t) * root).ravel()

    theta = pack()
    r = residual(theta)
    error = float(r @ r)
    damping = 1e-3
    for _ in range(iterations if layout else 0):
        jacobian = np.empty((len(r), len(theta)))
        for j in range(len(theta)):
            step = 1e-6 * max(1.0, abs(theta[j]))
            moved = theta.copy()
            moved[j] += step
            jacobian[:, j] = (residual(moved) - r) / step
        normal, gradient = jacobian.T @ jacobian, jacobian.T @ r
        for _attempt in range(10):
            try:
                delta = np.linalg.solve(normal + damping * np.diag(np.diag(normal) + 1e-12), -gradient)
            except np.linalg.LinAlgError:
                damping *= 4
                continue
            candidate = theta + delta
            r2 = residual(candidate)
            error2 = float(r2 @ r2)
            if error2 < error:
                theta, r, error = candidate, r2, error2
                damping = max(damping / 3.0, 1e-9)
                break
            damping *= 4
    slope, offset, power, saturation = unpack(theta)
    if per_channel and not lock["saturation"]:
        graded = apply_cdl(source, slope, offset, power, 1.0)
        chroma = lambda rgb: rgb - (rgb * LUMA).sum(axis=-1, keepdims=True)
        w_t = weights if len(weights) == len(target) else np.ones(len(target))
        have = float((chroma(graded) ** 2).sum(axis=1) @ weights / max(weights.sum(), 1e-12))
        want = float((chroma(target) ** 2).sum(axis=1) @ w_t / max(w_t.sum(), 1e-12))
        saturation = float(np.clip(np.sqrt(want / have), *SATURATION_RANGE)) if have > 1e-12 else 1.0
    final = residual(theta)
    return {"slope": slope, "offset": offset, "power": power, "saturation": saturation,
            "error": float(np.sqrt(final @ final / max(float(w.sum()) * 3, 1e-12)))}


def cdl_updates(fit, p):
    """The parameter values a fit sets: the gain, offset (less the lift the node adds on top), gamma, saturation and the error."""
    out = {"match_saturation": float(fit["saturation"]), "match_cdl_error": float(fit["error"])}
    for c, channel in enumerate("rgb"):
        out[f"grade_gain_{channel}"] = float(fit["slope"][c])
        out[f"grade_offset_{channel}"] = float(fit["offset"][c]) - float(p.get(f"grade_lift_{channel}", 0.0))
        out[f"grade_gamma_{channel}"] = 1.0 / float(fit["power"][c])
    return out


# ------------------------------------------------------------------------------------------------------------------- the export
def lut_values(p, size, axis=None):
    """The match the node holds (its CDL, with saturation; in `match_colorspace` when that is not RGB) sampled at a size**3 lattice, as a (size, size*size, 3) array in the lattice's order."""
    from .lutio import convert_colorspace, identity_lattice
    lattice = identity_lattice(size, axis)
    slope, offset, power, saturation = cdl_of(p)
    space = p.get("match_colorspace", "RGB")
    rgb = lattice[..., :3] if space == "RGB" else convert_colorspace(lattice[..., :3], space, "ACEScg")
    rgb = apply_cdl(rgb, slope, offset, power, saturation).astype(np.float32)
    return rgb if space == "RGB" else convert_colorspace(rgb, "ACEScg", space)


def prelut_style(p, file_type):
    """Pre LUT: Linear and Logarithmic as named; Auto Detect gives a .3dl its usual logarithmic shaper and a .cube (which has none) a linear one."""
    choice = p.get("match_prelut", "Auto Detect")
    if choice == "Auto Detect":
        return "logarithmic" if file_type == "3dl" else "linear"
    return choice.lower()


def write_lut(p, path):
    """Export > Write: the node's match as a .cube (a uniform lattice) or a .3dl (a lattice spaced by the Pre LUT), `match_lut_res` points a side."""
    from . import lutio
    path = str(path or "").strip()
    file_type = Path(path).suffix.lower().lstrip(".")
    if file_type not in ("cube", "3dl"):
        raise ValueError("MatchGrade: choose a LUT output file ending in .cube or .3dl")
    size = int(p.get("match_lut_res", 32))
    if file_type == "3dl":
        if size < 3 or (size - 1) & (size - 2):
            raise ValueError("MatchGrade: a .3dl needs a LUT resolution of 3, 5, 9, 17, 33, 65 or 129")
        style = prelut_style(p, "3dl")
        values = lut_values(p, size, lutio.prelut_inverse(style, np.linspace(0.0, 1.0, size)))
        return lutio.write_3dl(path, values, size, 1024, style, 10, 10, False)
    return lutio.write_cube(path, lut_values(p, size), size)


def cdl_xml(p, name="MatchGrade"):
    """The ASC CDL (.cc) the node's slope, offset, power and saturation make."""
    slope, offset, power, saturation = cdl_of(p)
    row = lambda values: " ".join("%.9g" % float(v) for v in values)
    return ('<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<ColorCorrectionCollection xmlns="{CDL_NAMESPACE}">\n  <ColorCorrection id="{name}">\n    <SOPNode>\n'
            f'      <Slope>{row(slope)}</Slope>\n      <Offset>{row(offset)}</Offset>\n      <Power>{row(power)}</Power>\n    </SOPNode>\n'
            f'    <SATNode>\n      <Saturation>{saturation:.9g}</Saturation>\n    </SATNode>\n  </ColorCorrection>\n</ColorCorrectionCollection>\n')


def write_cdl(p, path):
    path = Path(str(path or "").strip() or "match_grade.cc").expanduser()
    if path.suffix.lower() not in (".cc", ".xml"):
        path = path.with_suffix(".cc")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(cdl_xml(p), encoding="utf-8")
    return str(path)


def read_cdl(path):
    """(slope, offset, power, saturation) of the first ColorCorrection in an ASC CDL file."""
    root = ET.fromstring(Path(path).read_text(encoding="utf-8"))
    pick = lambda tag: next(e for e in root.iter() if e.tag.split("}")[-1] == tag).text.split()
    return (np.array(pick("Slope"), float), np.array(pick("Offset"), float), np.array(pick("Power"), float), float(pick("Saturation")[0]))


# ------------------------------------------------------------------------------------------------------- Align Target to Source
def align_commands(key, nodes, width, height, target=None):
    """Align Target to Source: Reformat node(s) on the Target input that give the target the Source's format (`width` x `height`).

    The target is stretched to the format, so nothing of it is lost. With Crop Target on, the target keeps its proportions instead: a Reformat
    fills the format (`target` is its (width, height)), a Transform centres it and a Crop takes the format-sized box out of what overflows.
    Returns the document commands (create, connect the target to the first node, connect the last to the MatchGrade's reference) and the last node's id."""
    node = nodes[key]
    reference = (node.get("inputs") or {}).get("reference")
    if not reference:
        raise ValueError("MatchGrade: connect a target to align")
    ident = f"{key}-align"
    pos = list(node.get("pos") or [0, 0])
    name = node.get("name", "MatchGrade")
    crop = bool(node["params"].get("match_crop_target", 0))
    if crop and not target:
        raise ValueError("MatchGrade: Crop Target needs the target's size")
    commands = [{"op": "create", "id": ident, "type": "Reformat", "name": f"{name} Align",
                 "params": {"reformat_type": "to_box", "width": int(width), "height": int(height), "resize_type": "fill" if crop else "distort"},
                 "pos": [pos[0] - 120, pos[1] - 40]},
                {"op": "connect", "id": ident, "input": "image", "source": reference}]
    last = ident
    if crop:
        scale = max(width / target[0], height / target[1])
        filled = (int(round(target[0] * scale)), int(round(target[1] * scale)))
        shift, last = f"{key}-align-shift", f"{key}-align-crop"
        commands += [{"op": "create", "id": shift, "type": "Transform", "name": f"{name} Centre",
                      "params": {"translate_x": -float((filled[0] - int(width)) // 2), "translate_y": float((filled[1] - int(height)) // 2)},
                      "pos": [pos[0] - 120, pos[1] + 20]},
                     {"op": "connect", "id": shift, "input": "image", "source": ident},
                     {"op": "create", "id": last, "type": "Crop", "name": f"{name} Crop",
                      "params": {"x": 0, "y": 0, "width": int(width), "height": int(height), "reformat": 1},
                      "pos": [pos[0] - 120, pos[1] + 80]},
                     {"op": "connect", "id": last, "input": "image", "source": shift}]
    commands.append({"op": "connect", "id": key, "input": "reference", "source": last})
    return commands, last
