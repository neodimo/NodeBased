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

    @staticmethod
    def _expression(image, p, second=None, frame=1):
        """ops2d_expression.evaluate_channels as it was (Python syntax, integer pixel indices from the top row)."""
        import ast
        from nodebased.expressions import CONSTANTS, MAX_EXPRESSION_LENGTH, ExpressionError
        from nodebased.ops2d_expression import VARS, VFUN, _eval
        h, w = image.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        env = {c: image[..., i] for i, c in enumerate("rgba")}
        env.update({f"second_{c}": (second[..., i] if second is not None else np.zeros((h, w), np.float32)) for i, c in enumerate("rgba")})
        env.update({f"{c}2": env[f"second_{c}"] for c in "rgba"})
        env.update(x=xx, y=yy, width=w, height=h, frame=float(frame or 0))
        out = image.copy()
        for i, c in enumerate("rgba"):
            text = str(p[f"expr_{c}"])
            tree = ast.parse(text, mode="eval")
            value = np.asarray(_eval(tree.body, env), dtype=np.float32)
            if value.ndim == 0:
                value = np.full((h, w), value, np.float32)
            if value.shape != (h, w) or not np.all(np.isfinite(value)):
                raise ExpressionError(f"{c}: expression produced invalid pixels")
            out[..., i] = value
        return out

    @staticmethod
    def _hist_eq(image, p):
        out = image.copy()
        if p.get("hist_eq_mode", "luminance") == "luminance":
            luma = np.sum(image[..., :3] * np.array([0.2126, 0.7152, 0.0722], np.float32), axis=-1)
            lo, hi = float(np.min(luma)), float(np.max(luma))
            if hi > lo:
                hist, edges = np.histogram(luma, bins=256, range=(lo, hi))
                cdf = np.cumsum(hist, dtype=np.float64)
                cdf = (cdf - cdf[0]) / max(float(cdf[-1] - cdf[0]), 1.0)
                mapped = np.interp(luma, edges[:-1], cdf).astype(np.float32)
                scale = np.divide(mapped, luma, out=np.ones_like(luma), where=np.abs(luma) > 1e-8)
                out[..., :3] *= scale[..., None]
        else:
            for c in range(3):
                values = image[..., c]
                lo, hi = float(np.min(values)), float(np.max(values))
                if hi > lo:
                    hist, edges = np.histogram(values, bins=256, range=(lo, hi))
                    cdf = np.cumsum(hist, dtype=np.float64)
                    cdf = (cdf - cdf[0]) / max(float(cdf[-1] - cdf[0]), 1.0)
                    out[..., c] = np.interp(values, edges[:-1], cdf)
        return out
