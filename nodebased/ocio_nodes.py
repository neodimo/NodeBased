"""Per-node OpenColorIO transforms. Alpha is left alone; the colour is the stored colour (the older `divide_by_alpha` behaviour divides it by alpha first)."""
from pathlib import Path

import numpy as np

from . import ocioknobs as _knobs


def _config(name):
    import PyOpenColorIO as ocio
    name = str(name or "ocio://cg-config-v4.0.0_aces-v2.0_ocio-v2.5")
    try:
        if name.startswith("ocio://"):
            return ocio.Config.CreateFromBuiltinConfig(name)
        path = Path(name).expanduser()
        if not path.is_file():
            raise ValueError(f"OCIO config file does not exist: {path}")
        return ocio.Config.CreateFromFile(str(path))
    except Exception as error:
        raise ValueError(f"Could not load OCIO config '{name}': {error}") from None


def parameter_choices(name, parameter, display=""):
    """Return config-backed inspector choices, preserving custom configs and unknown values."""
    cfg = _config(name)
    if parameter in ("src", "dst"):
        return list(cfg.getColorSpaceNames())
    if parameter == "display":
        return list(cfg.getDisplays())
    if parameter == "view":
        displays = list(cfg.getDisplays())
        selected = display if display in displays else (displays[0] if displays else "")
        return list(cfg.getViews(selected)) if selected else []
    if parameter == "look":
        return [""] + list(cfg.getLookNames())
    return []


def _direction(ocio, value):
    return ocio.TRANSFORM_DIR_INVERSE if value == "inverse" else ocio.TRANSFORM_DIR_FORWARD


_PRIMARIES = {
    "Rec.709": ((.64, .33), (.30, .60), (.15, .06)),
    "Rec.2020": ((.708, .292), (.170, .797), (.131, .046)),
    "P3-D65": ((.680, .320), (.265, .690), (.150, .060)),
    "ACEScg": ((.713, .293), (.165, .830), (.128, .044)),
}
_WHITE = {"D65": (.3127, .3290), "D60": (.32168, .33767), "DCI": (.314, .351),
          # CIE standard illuminants, chromaticities as published by the CIE (2 degree observer)
          "A": (.44757, .40745), "B": (.34842, .35161), "C": (.31006, .31616), "D50": (.34567, .35850), "D55": (.33242, .34743),
          "D75": (.29902, .31485), "E": (1 / 3, 1 / 3)}
_BRADFORD = np.array([[.8951, .2664, -.1614], [-.7502, 1.7135, .0367], [.0389, -.0685, 1.0296]])


def _xy_xyz(xy):
    x, y = xy
    return np.array([x / y, 1.0, (1 - x - y) / y])


def _rgb_to_xyz(primaries, whitepoint):
    columns = np.stack([_xy_xyz(xy) for xy in _PRIMARIES[primaries]], axis=1)
    scale = np.linalg.solve(columns, _xy_xyz(_WHITE[whitepoint]))
    return columns * scale[None, :]


# Kodak's Cineon 10-bit printing density curve, the constants of Nuke's own Log2Lin defaults: black 95, white 685, 0.002 density per code value, gamma 0.6.
_CINEON_BLACK, _CINEON_WHITE, _CINEON_STEP, _CINEON_GAMMA = 95.0, 685.0, 0.002, 0.6
_CINEON_OFFSET = 10.0 ** ((_CINEON_BLACK - _CINEON_WHITE) * _CINEON_STEP / _CINEON_GAMMA)


def _cineon_decode(rgb):
    return (np.power(10.0, (rgb * 1023.0 - _CINEON_WHITE) * _CINEON_STEP / _CINEON_GAMMA) - _CINEON_OFFSET) / (1.0 - _CINEON_OFFSET)


def _cineon_encode(rgb):
    return (_CINEON_WHITE + np.log10(np.maximum(rgb * (1.0 - _CINEON_OFFSET) + _CINEON_OFFSET, 1e-10)) * _CINEON_GAMMA / _CINEON_STEP) / 1023.0


def _decode(rgb, transfer):
    if transfer == "Linear":
        return rgb
    if transfer == "Rec.709":     # ITU-R BT.709's camera transfer function, inverted
        return np.where(rgb < .081, rgb / 4.5, np.power((np.abs(rgb) + .099) / 1.099, 1 / .45))
    if transfer == "Cineon":
        return _cineon_decode(rgb)
    if transfer == "sRGB":
        return np.where(rgb <= .04045, rgb / 12.92, np.power((rgb + .055) / 1.055, 2.4))
    gamma = 2.2 if transfer == "Gamma 2.2" else 2.4
    return np.sign(rgb) * np.power(np.abs(rgb), gamma)


def _encode(rgb, transfer):
    if transfer == "Linear":
        return rgb
    if transfer == "Rec.709":
        return np.where(rgb < .018, rgb * 4.5, 1.099 * np.power(np.maximum(rgb, 0), .45) - .099)
    if transfer == "Cineon":
        return _cineon_encode(rgb)
    if transfer == "sRGB":
        return np.where(rgb <= .0031308, rgb * 12.92, 1.055 * np.power(np.maximum(rgb, 0), 1 / 2.4) - .055)
    gamma = 2.2 if transfer == "Gamma 2.2" else 2.4
    return np.sign(rgb) * np.power(np.abs(rgb), 1 / gamma)


def _colorspace(p, pixels):
    """Nuke's Colorspace on the stored colour: decode the input curve, go through CIE XYZ between the two sets of primaries (each built on its own
    illuminant, so the XYZ values stay the same), encode the output curve. `bradford_matrix` adds the Bradford illuminant adaptation when the two
    illuminants differ; `divide_by_alpha` is the older behaviour of dividing the colour by alpha first and multiplying it back (Nuke's own
    (un)premult by does the dividing when asked)."""
    if int(p.get("swap", 0)):      # swap input/output: the two sets of choices trade places
        p = dict(p)
        for name in ("primaries", "transfer", "whitepoint"):
            p[f"{name}_in"], p[f"{name}_out"] = p[f"{name}_out"], p[f"{name}_in"]
    src = np.asarray(pixels, dtype=np.float32)
    alpha = src[..., 3:4]
    divide = bool(p.get("divide_by_alpha", 0))
    rgb = np.divide(src[..., :3], alpha, out=np.zeros_like(src[..., :3]), where=alpha != 0) if divide else src[..., :3]
    try:
        a = _rgb_to_xyz(p["primaries_in"], p["whitepoint_in"])
        b = _rgb_to_xyz(p["primaries_out"], p["whitepoint_out"])
        xyz = _decode(rgb, p["transfer_in"]) @ a.T
        if p["whitepoint_in"] != p["whitepoint_out"] and p.get("bradford_matrix", 0):
            source_white = _BRADFORD @ _xy_xyz(_WHITE[p["whitepoint_in"]])
            target_white = _BRADFORD @ _xy_xyz(_WHITE[p["whitepoint_out"]])
            adapt = np.linalg.inv(_BRADFORD) @ np.diag(target_white / source_white) @ _BRADFORD
            xyz = xyz @ adapt.T
        converted = _encode(xyz @ np.linalg.inv(b).T, p["transfer_out"])
    except (KeyError, ValueError, np.linalg.LinAlgError) as error:
        raise ValueError(f"Colorspace: invalid primaries, transfer or white point: {error}") from None
    out = src.copy()
    out[..., :3] = converted * alpha if divide else converted
    return out


def transform(kind, p, pixels):
    if kind == "Colorspace":
        return _colorspace(p, pixels)
    import PyOpenColorIO as ocio
    cfg = _config(p.get("config"))
    direction = _direction(ocio, p.get("transform_direction", "forward"))
    if kind in ("OCIOColorspace", "OCIOLogConvert"):
        if kind == "OCIOLogConvert":
            src = cfg.getRoleColorSpace(ocio.ROLE_COMPOSITING_LOG)
            dst = cfg.getRoleColorSpace(ocio.ROLE_SCENE_LINEAR)
            if p["ocio_log_operation"] == "lin to log":
                src, dst = dst, src
        else:
            src, dst = str(p["src"]), str(p["dst"])
            if int(p.get("swap", 0)):
                src, dst = dst, src
        op = ocio.ColorSpaceTransform(src=src, dst=dst)
        op.setDirection(direction)
    elif kind == "OCIOLookTransform":
        op = ocio.LookTransform(src=str(p["src"]), dst=str(p["dst"]), looks=str(p["look"]))
        op.setDirection(direction)
    elif kind == "OCIODisplay":
        # The display transform starts from the input colorspace: a space of the config by that name, otherwise the config's scene-linear role
        # (Nuke's default input is "linear"; the built-in config's role is ACEScg, which is also what the node always used).
        wanted = str(p.get("src") or "")
        scene = cfg.getRoleColorSpace(ocio.ROLE_SCENE_LINEAR)
        start = wanted if cfg.getColorSpace(wanted) is not None and wanted else (scene or "ACEScg")
        op = ocio.DisplayViewTransform(src=start, display=str(p["display"]), view=str(p["view"]))
        op.setDirection(direction)
    elif kind == "OCIOFileTransform":
        path = Path(str(p.get("path", ""))).expanduser()
        if not str(p.get("path", "")).strip():
            return np.asarray(pixels, dtype=np.float32).copy()
        if not path.is_file():
            raise ValueError(f"OCIOFileTransform: file does not exist: {path}")
        op = ocio.FileTransform(src=str(path))
        op.setDirection(direction)
        interp = str(p.get("file_interpolation", "linear")).lower()
        names = {"nearest": ocio.INTERP_NEAREST, "linear": ocio.INTERP_LINEAR,
                 "tetrahedral": ocio.INTERP_TETRAHEDRAL, "best": ocio.INTERP_BEST}
        if interp not in names:
            raise ValueError(f"OCIOFileTransform: unsupported interpolation '{interp}'")
        op.setInterpolation(names[interp])
        # Working space: the picture goes from scene linear to that space, the file is applied there, and the result goes back to scene linear.
        working = str(p.get("working_space", "scene_linear") or "scene_linear")
        scene = cfg.getRoleColorSpace(ocio.ROLE_SCENE_LINEAR) or ""
        if working not in ("scene_linear", "linear", scene):
            group = ocio.GroupTransform()
            group.appendTransform(ocio.ColorSpaceTransform(src=scene or "ACEScg", dst=working))
            group.appendTransform(op)
            group.appendTransform(ocio.ColorSpaceTransform(src=working, dst=scene or "ACEScg"))
            op = group
    else:
        raise ValueError(f"Unknown OCIO operation: {kind}")
    try:
        if kind == "OCIODisplay" and p.get("look"):
            # DisplayViewTransform has no per-look override in OCIO 2.x; a look is only
            # applicable through the viewing pipeline, which composes it around the
            # display/view transform rather than the transform itself.
            pipeline = ocio.LegacyViewingPipeline()
            pipeline.setDisplayViewTransform(op)
            pipeline.setLooksOverrideEnabled(True)
            pipeline.setLooksOverride(str(p["look"]))
            context = _knobs.context_for(ocio, cfg, p)
            cpu = (pipeline.getProcessor(cfg, context) if context is not None else pipeline.getProcessor(cfg)).getDefaultCPUProcessor()
        else:
            context = _knobs.context_for(ocio, cfg, p)
            cpu = (cfg.getProcessor(context, op, ocio.TRANSFORM_DIR_FORWARD) if context is not None else cfg.getProcessor(op)).getDefaultCPUProcessor()
    except Exception as error:
        if kind == "OCIOLookTransform" and int(p.get("ignore_errors", 0)):
            # Ignore errors: a look the config cannot find is treated as a plain conversion from the input to the output colorspace.
            try:
                plain = ocio.ColorSpaceTransform(src=str(p["src"]), dst=str(p["dst"]))
                plain.setDirection(direction)
                context = _knobs.context_for(ocio, cfg, p)
                cpu = (cfg.getProcessor(context, plain, ocio.TRANSFORM_DIR_FORWARD) if context is not None else cfg.getProcessor(plain)).getDefaultCPUProcessor()
            except Exception as fallback_error:
                raise ValueError(f"{kind}: cannot build transform: {fallback_error}") from None
        else:
            raise ValueError(f"{kind}: cannot build transform: {error}") from None
    src = np.asarray(pixels, dtype=np.float32)
    alpha = src[..., 3:4]
    # Nuke transforms the stored colour (its own (un)premult by does any dividing); `divide_by_alpha` is the older behaviour of dividing first.
    divide = bool(p.get("divide_by_alpha", 0))
    rgb = (np.divide(src[..., :3], alpha, out=np.zeros_like(src[..., :3]), where=alpha != 0) if divide else src[..., :3]).copy()
    display = kind == "OCIODisplay"
    inverse = p.get("transform_direction", "forward") == "inverse"
    if display:
        rgb = (_knobs.display_undo_after(rgb, p) if inverse else _knobs.display_before(rgb, alpha, p)).astype(np.float32).copy()
    shape = rgb.shape
    flat = rgb.reshape((-1, 3))
    try:
        cpu.applyRGB(flat)
    except Exception as error:
        raise ValueError(f"{kind}: transform failed: {error}") from None
    converted = flat.reshape(shape)
    if display:
        converted = _knobs.display_undo_before(converted, p) if inverse else _knobs.display_after(converted, p)
    out = src.copy()
    out[..., :3] = converted * alpha if divide else converted
    if kind == "OCIOLogConvert":
        channels = p.get("channels", "rgb")
        selected = {"rgb": (0, 1, 2), "rgba": (0, 1, 2, 3), "alpha": (3,),
                    "none": ()}.get(channels)
        if selected is None:
            raise ValueError(f"OCIOLogConvert: unsupported channels '{channels}'")
        for channel in range(4):
            if channel not in selected:
                out[..., channel] = src[..., channel]
    return out
