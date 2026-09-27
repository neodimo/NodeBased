"""Small in-house .cube LUT reader, interpolator and graph exporter."""
from pathlib import Path

import numpy as np


def read_cube(path):
    path = Path(path).expanduser()
    size = None
    rows = []
    with open(path, "r", encoding="utf-8-sig") as handle:
        for number, raw in enumerate(handle, 1):
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            parts = line.split()
            if parts[0] in ("TITLE", "DOMAIN_MIN", "DOMAIN_MAX"):
                continue
            if parts[0] == "LUT_3D_SIZE":
                try:
                    size = int(parts[1])
                    if len(parts) != 2 or size < 2:
                        raise ValueError
                except (IndexError, ValueError):
                    raise ValueError(f"Malformed .cube file at line {number}: invalid LUT_3D_SIZE") from None
                continue
            try:
                values = [float(value) for value in parts]
                if len(values) != 3 or not np.isfinite(values).all():
                    raise ValueError
                rows.append(values)
            except ValueError:
                raise ValueError(f"Malformed .cube file at line {number}: expected three finite values") from None
    if size is None:
        raise ValueError("Malformed .cube file at line 1: missing LUT_3D_SIZE")
    if len(rows) != size ** 3:
        raise ValueError(f"Malformed .cube file at line {len(rows) + 1}: expected {size ** 3} entries, found {len(rows)}")
    # .cube ordering: red varies fastest, then green, then blue.
    return np.asarray(rows, dtype=np.float32).reshape((size, size, size, 3))


def _sample(lut, rgb, method="tetrahedral"):
    lut = np.asarray(lut, dtype=np.float32)
    rgb = np.asarray(rgb, dtype=np.float32)
    size = lut.shape[0]
    p = np.clip(rgb, 0.0, 1.0) * (size - 1)
    base = np.floor(p).astype(np.int32)
    base = np.minimum(base, size - 2)
    f = p - base
    r, g, b = base[..., 0], base[..., 1], base[..., 2]
    fr, fg, fb = f[..., 0:1], f[..., 1:2], f[..., 2:3]
    c000 = lut[b, g, r]
    c100 = lut[b, g, r + 1]
    c010 = lut[b, g + 1, r]
    c001 = lut[b + 1, g, r]
    c110 = lut[b, g + 1, r + 1]
    c101 = lut[b + 1, g, r + 1]
    c011 = lut[b + 1, g + 1, r]
    c111 = lut[b + 1, g + 1, r + 1]
    if method == "trilinear":
        return (c000 * (1-fr)*(1-fg)*(1-fb) + c100 * fr*(1-fg)*(1-fb) +
                c010 * (1-fr)*fg*(1-fb) + c001 * (1-fr)*(1-fg)*fb +
                c110 * fr*fg*(1-fb) + c101 * fr*(1-fg)*fb +
                c011 * (1-fr)*fg*fb + c111 * fr*fg*fb)
    # Six tetrahedra, selected by the ordering of the fractional coordinates.
    out = np.empty_like(c000)
    masks = [
        (fr >= fg) & (fg >= fb), (fr >= fb) & (fb > fg), (fb > fr) & (fr >= fg),
        (fg > fr) & (fr >= fb), (fg >= fb) & (fb > fr), (fb > fg) & (fg > fr)]
    vals = [
        c000 + fr*(c100-c000) + fg*(c110-c100) + fb*(c111-c110),
        c000 + fr*(c100-c000) + fb*(c101-c100) + fg*(c111-c101),
        c000 + fb*(c001-c000) + fr*(c101-c001) + fg*(c111-c101),
        c000 + fg*(c010-c000) + fr*(c110-c010) + fb*(c111-c110),
        c000 + fg*(c010-c000) + fb*(c011-c010) + fr*(c111-c011),
        c000 + fb*(c001-c000) + fg*(c011-c001) + fr*(c111-c011)]
    for mask, value in zip(masks, vals):
        out[mask[..., 0]] = value[mask[..., 0]]
    return out


def apply_cube(pixels, lut, interpolation="tetrahedral"):
    source = np.asarray(pixels, dtype=np.float32)
    out = source.copy()
    alpha = source[..., 3:4]
    straight = np.divide(source[..., :3], alpha, out=np.zeros_like(source[..., :3]), where=alpha != 0)
    out[..., :3] = _sample(lut, straight, interpolation) * alpha
    return out


def convert_colorspace(rgb, source, destination):
    from .color import INPUT_SPACES
    source = INPUT_SPACES.get(source, source)
    destination = INPUT_SPACES.get(destination, destination)
    if source == destination:
        return np.asarray(rgb, dtype=np.float32)
    from .color import config
    ocio = config()
    processor = ocio.getProcessor(source, destination).getDefaultCPUProcessor()
    values = np.asarray(rgb, dtype=np.float32).copy()
    shape = values.shape
    flat = values.reshape((-1, 3))
    processor.applyRGB(flat)
    return flat.reshape(shape)


def write_cube(path, values, size):
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"TITLE \"NodeBased LUT\"\nLUT_3D_SIZE {size}\nDOMAIN_MIN 0 0 0\nDOMAIN_MAX 1 1 1\n")
        for row in values.reshape((-1, 3)):
            handle.write("%.9g %.9g %.9g\n" % tuple(map(float, row)))
    return str(path)


def identity_lattice(size):
    axis = np.linspace(0.0, 1.0, size, dtype=np.float32)
    b, g, r = np.meshgrid(axis, axis, axis, indexing="ij")
    rgb = np.stack((r, g, b), axis=-1).reshape((size, size * size, 3))
    rgba = np.ones((size, size * size, 4), dtype=np.float32)
    rgba[..., :3] = rgb
    return rgba


def export_graph_cube(document, target_key, path, size, source_space="ACEScg", output_space="ACEScg"):
    """Evaluate the upstream graph on a synthetic identity lattice, replacing its image roots."""
    from copy import deepcopy
    path = str(path or "").strip()
    if Path(path).suffix.lower() != ".cube":
        raise ValueError("GenerateLUT: choose an output path ending in .cube")
    if size not in (17, 33, 65):
        raise ValueError("GenerateLUT size must be 17, 33 or 65")
    target = document["nodes"].get(target_key)
    if target is None or target["type"] != "GenerateLUT":
        raise ValueError("LUT export requires a GenerateLUT node")
    root = target["inputs"].get("source")
    if root not in document["nodes"]:
        raise ValueError("GenerateLUT: connect an upstream color graph")
    doc = deepcopy(document)
    nodes = doc["nodes"]
    ancestors, stack = set(), [root]
    while stack:
        key = stack.pop()
        if key in ancestors:
            continue
        ancestors.add(key)
        stack.extend(k for k in nodes[key]["inputs"].values() if k in nodes)
    pointwise = {"Read", "Constant", "Checker", "NoOp", "Dot", "Grade", "ColorCorrect", "Invert", "Clamp",
                 "Multiply", "Add", "Gamma", "Saturation", "Exposure", "HueCorrect", "ColorMatrix",
                 "Log2Lin", "PLogLin", "CrossTalk", "Toe", "Expression", "Shuffle", "ChannelShuffle",
                 "Premult", "Unpremult", "Vectorfield"}
    unsupported = sorted({nodes[key]["type"] for key in ancestors} - pointwise)
    if unsupported:
        raise ValueError("GenerateLUT supports pointwise colour graphs; unsupported upstream node(s): " +
                         ", ".join(unsupported))
    roots = [key for key in ancestors if not any(v in ancestors for v in nodes[key]["inputs"].values())]
    if not roots:
        raise ValueError("GenerateLUT: upstream graph has no image source")
    lattice = identity_lattice(size)
    lattice[..., :3] = convert_colorspace(lattice[..., :3], source_space, "ACEScg")
    from .raster import Raster
    raster = Raster.of(lattice)
    # Substitute source nodes with the same identity samples; multi-input graph roots thus remain aligned.
    # The identity lattice is deliberately different from any cached upstream frame.
    from .imaging import Evaluator
    evaluator = Evaluator()
    previous_roots = getattr(evaluator, "_lut_roots", {})
    evaluator._lut_roots = {key: raster for key in roots}
    try:
        sampled = evaluator.evaluate_raster(doc, root, frame=1, tier=1)
    finally:
        evaluator._lut_roots = previous_roots
    result = sampled.pixels if isinstance(sampled, Raster) else sampled
    out = result[..., :3]
    out = convert_colorspace(out, "ACEScg", output_space) if output_space != "ACEScg" else out
    return write_cube(path, out, size)
