"""The colour kernels exactly as they stood before the 2D behaviour lane 2 step B (copied from nodebased/imaging.py at aabf15e3).

Tests compare an older document's node with these, bit for bit. Shared helpers are the live ones: they were not changed.
"""
import numpy as np


class Legacy:
    @staticmethod
    def _log2lin(image, p):
        x = image.copy(); rgb = x[..., :3]
        black, white = float(p["black"]), float(p["white"])
        span = white - black
        if abs(span) < 1e-12: span = 1e-12
        if p["log_direction"] == "log to lin":
            rgb[:] = np.sign(rgb - black) * np.abs((rgb - black) / span) ** max(float(p["gamma"]), 1e-6)
        else:
            rgb[:] = black + np.sign(rgb) * np.abs(rgb) ** (1.0 / max(float(p["gamma"]), 1e-6)) * span
        return x

    @staticmethod
    def _ploglin(image, p):
        x = image.copy(); rgb = x[..., :3]
        lr, cr = float(p["linear_reference"]), float(p["log_reference"])
        d = max(float(p["density_per_code_value"]), 1e-9)
        gamma = max(float(p["negative_gamma"]), 1e-6)
        delta = (rgb - cr) * d
        mapped = lr * np.power(10.0, np.clip(delta, -30, 30))
        rgb[:] = np.where(rgb < 0, -lr * np.power(10.0, np.clip((np.abs(rgb) - cr) * d, -30, 30)) ** gamma, mapped)
        return x

    @staticmethod
    def _colorspace(image, p):
        """ocio_nodes._colorspace as it was: divides by alpha, always adapts between two illuminants with Bradford."""
        from nodebased.ocio_nodes import _BRADFORD, _WHITE, _decode, _encode, _rgb_to_xyz, _xy_xyz
        src = np.asarray(image, dtype=np.float32)
        alpha = src[..., 3:4]
        rgb = np.divide(src[..., :3], alpha, out=np.zeros_like(src[..., :3]), where=alpha != 0)
        a = _rgb_to_xyz(p["primaries_in"], p["whitepoint_in"])
        b = _rgb_to_xyz(p["primaries_out"], p["whitepoint_out"])
        xyz = _decode(rgb, p["transfer_in"]) @ a.T
        if p["whitepoint_in"] != p["whitepoint_out"]:
            source_white = _BRADFORD @ _xy_xyz(_WHITE[p["whitepoint_in"]])
            target_white = _BRADFORD @ _xy_xyz(_WHITE[p["whitepoint_out"]])
            adapt = np.linalg.inv(_BRADFORD) @ np.diag(target_white / source_white) @ _BRADFORD
            xyz = xyz @ adapt.T
        converted = _encode(xyz @ np.linalg.inv(b).T, p["transfer_out"])
        out = src.copy()
        out[..., :3] = converted * alpha
        return out

    @staticmethod
    def _color_lookup(image, p):
        from nodebased import colorcurves
        result = image.copy()
        master = colorcurves.decode(p["curve_master"])
        domain = max(1e-8, float(p.get("range", 1.0)))
        for index, name in enumerate(("red", "green", "blue", "alpha")):
            values = colorcurves.evaluate_array(master, result[..., index] / domain)
            result[..., index] = colorcurves.evaluate_array(colorcurves.decode(p[f"curve_{name}"]), values)
        return result
