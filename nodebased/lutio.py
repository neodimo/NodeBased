"""Small in-house .cube LUT reader, interpolator and graph exporter."""
from pathlib import Path
import math

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


def _integer_bit_depth(maximum):
    """Infer the common integer output depth used by Flame/Lustre 3DL files."""
    if maximum < 0:
        raise ValueError("negative code value")
    if maximum <= 511: return 8
    if maximum <= 2047: return 10
    if maximum <= 8191: return 12
    if maximum <= 32767: return 16
    if maximum <= 131071: return 16
    raise ValueError("code value exceeds the supported 16-bit range")


def _normalise_integer_lut(values, bit_depth):
    scale = float((1 << bit_depth) - 1)
    return np.asarray(values, dtype=np.float32) / scale


def read_3dl(path):
    """Read Flame/Lustre .3dl integer shaper + blue-fastest 3D mesh data.

    Header text is intentionally permissive for both common variants (`3DMESH`/`Mesh n depth`
    and Flame's bare numeric body), while malformed numeric rows identify their source line.
    Returns `(mesh, shaper)`; either format component may omit the shaper.
    """
    rows, shaper_values = [], None
    shaper_line = None
    declared_output_depth = None
    with Path(path).expanduser().open("r", encoding="utf-8-sig") as handle:
        for number, raw in enumerate(handle, 1):
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            parts = line.split()
            header = parts[0].lower()
            if header == "mesh":
                nums = []
                for token in parts[1:]:
                    try: nums.append(int(token))
                    except ValueError: break
                if len(nums) >= 2 and nums[-1] in (10, 12, 16):
                    declared_output_depth = nums[-1]
                continue
            if header in ("3dmesh", "lut8", "lut10", "lut12", "lut16", "gamma", "title"):
                continue
            try:
                values = [int(token, 10) for token in parts]
            except ValueError:
                raise ValueError(f"Malformed .3dl file at line {number}: expected integer LUT values or a known header") from None
            if any(value < 0 for value in values):
                raise ValueError(f"Malformed .3dl file at line {number}: code values must be non-negative")
            if len(values) > 3:
                if shaper_values is not None or rows:
                    raise ValueError(f"Malformed .3dl file at line {number}: unexpected or duplicate shaper row")
                shaper_values, shaper_line = values, number
            elif len(values) == 3:
                rows.append((values, number))
            else:
                raise ValueError(f"Malformed .3dl file at line {number}: expected three mesh values")
    if not rows:
        line = shaper_line or 1
        raise ValueError(f"Malformed .3dl file at line {line}: missing 3D mesh entries")
    edge = round(len(rows) ** (1.0 / 3.0))
    if edge < 2 or edge ** 3 != len(rows):
        raise ValueError(f"Malformed .3dl file at line {rows[-1][1]}: mesh entry count is not a complete cube")
    raw_mesh = [values for values, _line in rows]
    mesh_max = max(max(values) for values in raw_mesh)
    depth = declared_output_depth or _integer_bit_depth(mesh_max)
    if depth not in (8, 10, 12, 16):
        raise ValueError(f"Malformed .3dl file at line {rows[0][1]}: unsupported output bit depth {depth}")
    if mesh_max > 2 * ((1 << depth) - 1):
        raise ValueError(f"Malformed .3dl file at line {rows[-1][1]}: mesh values exceed {depth}-bit range")
    # 3DL traverses blue fastest. `_sample` consumes the cube representation used by .cube,
    # with red fastest, hence reverse the outer/inner axes once at load time.
    scale = float((1 << depth) - 1)
    mesh = _normalise_integer_lut(raw_mesh, depth).reshape((edge, edge, edge, 3)).transpose(2, 1, 0, 3)
    axis = np.linspace(0.0, 1.0, edge, dtype=np.float32)
    b, g, r = np.meshgrid(axis, axis, axis, indexing="ij")
    identity_mesh = np.stack((r, g, b), axis=-1)
    if np.max(np.abs(mesh - identity_mesh)) < 2.0 / scale:
        # Integer identity tables are rounded to their code depth. Removing that quantisation
        # avoids needlessly perturbing an image passed through an identity LUT.
        mesh = identity_mesh
    shaper = None
    if shaper_values is not None:
        shaper_depth = _integer_bit_depth(max(shaper_values))
        if shaper_depth not in (8, 10, 12, 16):
            raise ValueError(f"Malformed .3dl file at line {shaper_line}: unsupported shaper bit depth")
        shaper = _normalise_integer_lut(shaper_values, shaper_depth)
        ideal = np.linspace(0.0, 1.0, len(shaper), dtype=np.float32)
        if np.max(np.abs(shaper - ideal)) < 2.0 / float((1 << shaper_depth) - 1):
            shaper = None
    return mesh, shaper


def _apply_shaper(rgb, shaper):
    values = np.clip(np.asarray(rgb, dtype=np.float32), 0.0, 1.0)
    position = values * (len(shaper) - 1)
    low = np.floor(position).astype(np.int32)
    high = np.minimum(low + 1, len(shaper) - 1)
    weight = (position - low).astype(np.float32)
    return shaper[low] * (1.0 - weight) + shaper[high] * weight


def apply_3dl(pixels, lut, shaper=None, interpolation="tetrahedral"):
    source = np.asarray(pixels, dtype=np.float32)
    out = source.copy()
    alpha = source[..., 3:4]
    straight = np.divide(source[..., :3], alpha, out=np.zeros_like(source[..., :3]), where=alpha != 0)
    if shaper is not None:
        straight = _apply_shaper(straight, shaper)
    out[..., :3] = _sample(lut, straight, interpolation) * alpha
    return out


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
                 "Multiply", "Add", "Gamma", "Saturation", "Exposure", "HueCorrect", "ColorLookup", "ColorMatrix",
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
