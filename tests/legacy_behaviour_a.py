"""The colour kernels exactly as they stood before the 2D behaviour lane 2 step A (copied from nodebased/imaging.py at 5015cef7).

Tests compare an older document's node with these, bit for bit. Shared helpers are the live ones: they were not changed.
"""
import math

import numpy as np

from nodebased.imaging import Evaluator


class Legacy:
    @staticmethod
    def _color_correct(image, p):
        """Nuke's ColorCorrect, on unpremultiplied colour.

        Weights: the input's Rec.709 luminance (Nuke's 0.2125/0.7154/0.0721) looked up in the shadow
        and highlight curves, held flat past their end points; midtones = 1 - shadows - highlights.
        Each of the three ranges corrects the pixel with the master woven in knob by knob, in the order
        saturation, contrast, gamma, gain, offset: saturation, gamma and gain multiply (range x master),
        offset adds, contrast is the range's around its pivot (shadows 0.04, midtones 0.18, highlights
        0.65) then the master's around 0.18. Values at or below 0 skip contrast and gamma and are scaled
        by what 1 becomes through them. The three results are blended by the weights."""
        from nodebased.core import SPECS
        from nodebased import colorcurves
        q = {**SPECS["ColorCorrect"]["params"], **p}
        controls = ("saturation", "contrast", "gamma", "gain", "offset")
        if all(float(q[name]) == (0.0 if name == "offset" else 1.0) for name in controls) and all(
                float(q[f"{zone}_{name}"]) == (0.0 if name == "offset" else 1.0)
                for zone in ("shadows", "midtones", "highlights") for name in controls):
            return image
        frame = image
        alpha = frame[..., 3:4]
        straight = np.divide(frame[..., :3], alpha, out=np.zeros_like(frame[..., :3]), where=alpha > 1e-8)

        def luminance(rgb):
            return 0.2125 * rgb[..., 0:1] + 0.7154 * rgb[..., 1:2] + 0.0721 * rgb[..., 2:3]

        def weight(curve_text, lum):
            curve = colorcurves.decode(curve_text)
            xs = [point[0] for point in curve["points"]]
            # Hold endpoints in float64. Clipping into float32 can round an end
            # x just outside the curve, then evaluate_array extrapolates a tiny
            # negative weight; extreme HDR/gamma makes that tiny error visible.
            held = np.clip(np.asarray(lum, dtype=np.float64), xs[0], xs[-1])
            return colorcurves.evaluate_array(curve, held)

        lum = luminance(straight)
        shadows = weight(q["shadows_curve"], lum)
        highlights = weight(q["highlights_curve"], lum)
        weights = {"shadows": shadows, "midtones": 1.0 - shadows - highlights, "highlights": highlights}
        pivots = {"shadows": 0.04, "midtones": 0.18, "highlights": 0.65}
        master = {name: float(q[name]) for name in ("saturation", "contrast", "gamma", "gain", "offset")}
        # Ranges whose own knobs are equal correct the pixel identically, so the default case (only
        # the master moved) costs one pass, not three; their weights add up.
        passes = {}
        for zone, w in weights.items():
            knob = {name: float(q[f"{zone}_{name}"]) for name in master}
            pivot = pivots[zone] if knob["contrast"] != 1.0 else None
            key = (knob["saturation"], knob["contrast"], knob["gamma"], knob["gain"], knob["offset"], pivot)
            passes[key] = passes[key] + w if key in passes else w
        result = np.zeros_like(straight)
        for (z_sat, c_zone, z_gamma, z_gain, z_offset, pivot), w in passes.items():
            saturation = z_sat * master["saturation"]
            gamma = max(z_gamma * master["gamma"], 1e-6)
            gain = z_gain * master["gain"]
            offset = z_offset + master["offset"]
            c_master = master["contrast"]

            def tone(x):
                if pivot is not None:
                    x = (x / pivot) ** c_zone * pivot
                if c_master != 1.0:
                    x = (x / 0.18) ** c_master * 0.18
                return x ** (1.0 / gamma) if gamma != 1.0 else x

            x = lum + saturation * (straight - lum) if saturation != 1.0 else straight
            unit = float(tone(1.0))
            positive = x > 0.0
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                x = np.where(positive, tone(np.where(positive, x, 1.0)), x * unit)
            corrected = x * gain + offset if (gain != 1.0 or offset != 0.0) else x
            if len(passes) == 1:
                result = corrected      # the weights add up to 1; skip their rounding
                break
            result += w * corrected
        return np.concatenate([result * alpha, alpha], axis=2).astype(np.float32)

    @staticmethod
    def _saturation(image, p):
        # Unlike Multiply/Add/Gamma, Saturation has no channels knob in Nuke: it is inherently an
        # RGB-luma operation and always leaves alpha untouched.
        rgb = image[..., :3]
        luma = 0.2126 * rgb[..., 0:1] + 0.7152 * rgb[..., 1:2] + 0.0722 * rgb[..., 2:3]
        out_rgb = luma + (rgb - luma) * p.get("saturation", 1.0)
        return np.concatenate([out_rgb, image[..., 3:4]], axis=2).astype(np.float32)
