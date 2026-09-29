"""Per-node OpenColorIO transforms. All operators preserve premultiplied alpha."""
from pathlib import Path

import numpy as np


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
_WHITE = {"D65": (.3127, .3290), "D60": (.32168, .33767), "DCI": (.314, .351)}
_BRADFORD = np.array([[.8951, .2664, -.1614], [-.7502, 1.7135, .0367], [.0389, -.0685, 1.0296]])


def _xy_xyz(xy):
    x, y = xy
    return np.array([x / y, 1.0, (1 - x - y) / y])


def _rgb_to_xyz(primaries, whitepoint):
    columns = np.stack([_xy_xyz(xy) for xy in _PRIMARIES[primaries]], axis=1)
    scale = np.linalg.solve(columns, _xy_xyz(_WHITE[whitepoint]))
    return columns * scale[None, :]


def _decode(rgb, transfer):
    if transfer == "Linear":
        return rgb
    if transfer == "sRGB":
        return np.where(rgb <= .04045, rgb / 12.92, np.power((rgb + .055) / 1.055, 2.4))
    gamma = 2.2 if transfer == "Gamma 2.2" else 2.4
    return np.sign(rgb) * np.power(np.abs(rgb), gamma)


def _encode(rgb, transfer):
    if transfer == "Linear":
        return rgb
    if transfer == "sRGB":
        return np.where(rgb <= .0031308, rgb * 12.92, 1.055 * np.power(np.maximum(rgb, 0), 1 / 2.4) - .055)
    gamma = 2.2 if transfer == "Gamma 2.2" else 2.4
    return np.sign(rgb) * np.power(np.abs(rgb), 1 / gamma)


def _colorspace(p, pixels):
    src = np.asarray(pixels, dtype=np.float32)
    alpha = src[..., 3:4]
    rgb = np.divide(src[..., :3], alpha, out=np.zeros_like(src[..., :3]), where=alpha != 0)
    try:
        a = _rgb_to_xyz(p["primaries_in"], p["whitepoint_in"])
        b = _rgb_to_xyz(p["primaries_out"], p["whitepoint_out"])
        xyz = _decode(rgb, p["transfer_in"]) @ a.T
        if p["whitepoint_in"] != p["whitepoint_out"]:
            source_white = _BRADFORD @ _xy_xyz(_WHITE[p["whitepoint_in"]])
            target_white = _BRADFORD @ _xy_xyz(_WHITE[p["whitepoint_out"]])
            adapt = np.linalg.inv(_BRADFORD) @ np.diag(target_white / source_white) @ _BRADFORD
            xyz = xyz @ adapt.T
        converted = _encode(xyz @ np.linalg.inv(b).T, p["transfer_out"])
    except (KeyError, ValueError, np.linalg.LinAlgError) as error:
        raise ValueError(f"Colorspace: invalid primaries, transfer or white point: {error}") from None
    out = src.copy()
    out[..., :3] = converted * alpha
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
        op = ocio.ColorSpaceTransform(src=src, dst=dst)
        op.setDirection(direction)
    elif kind == "OCIOLookTransform":
        op = ocio.LookTransform(src=str(p["src"]), dst=str(p["dst"]), looks=str(p["look"]))
        op.setDirection(direction)
    elif kind == "OCIODisplay":
        op = ocio.DisplayViewTransform(src="ACEScg", display=str(p["display"]), view=str(p["view"]))
        if p.get("look"):
            op.setLooks(str(p["look"]))
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
    else:
        raise ValueError(f"Unknown OCIO operation: {kind}")
    try:
        cpu = cfg.getProcessor(op).getDefaultCPUProcessor()
    except Exception as error:
        raise ValueError(f"{kind}: cannot build transform: {error}") from None
    src = np.asarray(pixels, dtype=np.float32)
    alpha = src[..., 3:4]
    rgb = np.divide(src[..., :3], alpha, out=np.zeros_like(src[..., :3]), where=alpha != 0).copy()
    shape = rgb.shape
    flat = rgb.reshape((-1, 3))
    try:
        cpu.applyRGB(flat)
    except Exception as error:
        raise ValueError(f"{kind}: transform failed: {error}") from None
    out = src.copy()
    out[..., :3] = flat.reshape(shape) * alpha
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
