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
