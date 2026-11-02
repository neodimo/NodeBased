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

    @staticmethod
    def _hue_correct(image, p):
        from nodebased import colorcurves
        rgb = image[..., :3]
        hue, sat = Evaluator._rgb_to_hue_sat(rgb)
        hue = hue[..., 0].astype(np.float64) % 360.0
        sat = sat[..., 0]
        threshold = max(0.0, float(p.get("sat_thrsh", 0.0)))
        if threshold > 0.0:
            # Nuke's threshold fades the correction in over the next threshold-width of
            # saturation, avoiding a hard edge at the cutoff.
            t = np.clip((sat - threshold) / threshold, 0.0, 1.0)
            sat_weight = t * t * (3.0 - 2.0 * t)
        else:
            sat_weight = np.ones_like(sat, dtype=np.float32)
        default_curve = '{"interpolation":"linear","points":[[0,1],[360,1]]}'
        legacy_names = ("red", "yellow", "green", "cyan", "blue", "magenta")
        for prefix in ("sat", "lum"):
            values = [float(p.get(f"{prefix}_{band}", 1.0)) for band in legacy_names]
            raw = p.get(f"curve_{prefix}", default_curve)
            if raw == default_curve and any(value != 1.0 for value in values):
                from nodebased.colorcurves import anchor_curve
                p = dict(p, **{f"curve_{prefix}": anchor_curve(values)})
        def sample(name, default=1.0):
            raw = p.get(f"curve_{name}")
            if raw is None:
                return np.full(hue.shape, default, np.float32)
            return colorcurves.evaluate_array(colorcurves.decode(raw), hue)
        s_mult, l_mult = sample("sat"), sample("lum")
        luma = 0.2126 * rgb[..., 0:1] + 0.7152 * rgb[..., 1:2] + 0.0722 * rgb[..., 2:3]
        out = rgb + (s_mult[..., None] - 1.0) * (rgb - luma) * sat_weight[..., None]
        out = out * (1.0 + (l_mult[..., None] - 1.0) * np.clip(sat[..., None], 0.0, 1.0) * sat_weight[..., None])
        for channel, name in enumerate(("red", "green", "blue")):
            out[..., channel] *= 1.0 + (sample(name) - 1.0) * sat_weight
        # Suppression is despill-style limiting, a separate output from the channel response above:
        # a channel is pulled toward the larger of the other two by the curve's value, so 1 limits
        # it to them and a channel already below them is left alone.
        pre = out.copy()
        for channel, name in enumerate("rgb"):
            others = np.maximum(pre[..., (channel + 1) % 3], pre[..., (channel + 2) % 3])
            out[..., channel] -= sample(name + "_sup", 0.0) * np.maximum(pre[..., channel] - others, 0.0) * sat_weight
        shift = float(p.get("hue_shift", 0.0))
        if shift % 360.0 != 0.0:
            a = math.radians(shift); c, s = math.cos(a), math.sin(a); k = 1.0 / math.sqrt(3.0)
            m = np.array([[c + (1-c)/3, (1-c)/3-s*k, (1-c)/3+s*k],
                          [(1-c)/3+s*k, c + (1-c)/3, (1-c)/3-s*k],
                          [(1-c)/3-s*k, (1-c)/3+s*k, c + (1-c)/3]], np.float32)
            out = out @ m.T
        return np.concatenate([out, image[..., 3:4]], axis=2).astype(np.float32)

    @staticmethod
    def _hsv_tool(image, p, range_masks=None):
        # HSVTool: RGB -> (hue in degrees, saturation, value), an adjustment weighted by how far
        # inside the hue, saturation and brightness ranges the pixel sits (product of the three
        # weights, each with its own linear rolloff), then back to RGB. Hue rotates by
        # hue_rotation degrees. Saturation and brightness scale by (1 + adjust), or, with the
        # matching "force" toggle, move to the adjust value itself (the definition of Nuke's
        # `saturation`/`brightness` adjustments is inferred, see docs/PARITY_2D.md). Alpha is
        # untouched unless output_alpha is on, when it becomes the combined range weight.
        # `color_replace` (Nuke's Color Replacement, `srccolor`/`dstcolor`) derives the rotation
        # and forced saturation/brightness from the two picked colours instead of the literal
        # rotation/adjust knobs: a pixel that lands at full weight and exactly matches srccolor's
        # hue, saturation and value becomes dstcolor exactly; the existing range/rolloff knobs
        # still gate which pixels that reaches, unchanged.
        rgb = image[..., :3].astype(np.float64)
        hue, sat, val = Evaluator._rgb_to_hsv(rgb)
        hue_weight = Evaluator._hsv_range_weight(hue, float(p.get("hue_range_min", 0.0)), float(p.get("hue_range_max", 360.0)),
                                              float(p.get("hue_rolloff", 0.0)), cyclic=True)
        sat_weight = Evaluator._hsv_range_weight(sat, float(p.get("saturation_range_min", 0.0)),
                                                float(p.get("saturation_range_max", 1.0)),
                                                float(p.get("saturation_rolloff", 0.0)))
        brt_weight = Evaluator._hsv_range_weight(val, float(p.get("brightness_range_min", 0.0)),
                                                float(p.get("brightness_range_max", 1.0)),
                                                float(p.get("brightness_rolloff", 0.0)))
        def mask_alpha(value):
            if value is None:
                return 1.0
            arr = np.asarray(value)
            return arr[..., 3] if arr.ndim == 3 else arr
        general, hm, sm, bm = (range_masks or (None, None, None, None))
        shared = mask_alpha(general)
        base_weights = (hue_weight, sat_weight, brt_weight)
        independent_masks = any(mask is not None for mask in (hm, sm, bm))
        hue_weight = base_weights[0] * shared * mask_alpha(hm)
        sat_weight = base_weights[1] * shared * mask_alpha(sm)
        brt_weight = base_weights[2] * shared * mask_alpha(bm)
        weight = base_weights[0] * base_weights[1] * base_weights[2] * shared
        sat_adjust, brt_adjust = float(p.get("sat_adjust", 0.0)), float(p.get("brt_adjust", 0.0))
        rotation = float(p.get("hue_rotation", 0.0))
        set_saturation, set_brightness = bool(p.get("set_saturation")), bool(p.get("set_brightness"))
        if p.get("color_replace"):
            src_rgb = np.array([p.get("srccolor_r", 0.0), p.get("srccolor_g", 0.0), p.get("srccolor_b", 0.0)], np.float64)
            dst_rgb = np.array([p.get("dstcolor_r", 0.0), p.get("dstcolor_g", 0.0), p.get("dstcolor_b", 0.0)], np.float64)
            src_hue, _src_sat, _src_val = Evaluator._rgb_to_hsv(src_rgb)
            dst_hue, dst_sat, dst_val = Evaluator._rgb_to_hsv(dst_rgb)
            rotation = ((float(dst_hue) - float(src_hue) + 180.0) % 360.0) - 180.0
            sat_adjust, brt_adjust = float(dst_sat), float(dst_val)
            set_saturation = set_brightness = True
        hue_effect = hue_weight if independent_masks else weight
        sat_effect = sat_weight if independent_masks else weight
        brt_effect = brt_weight if independent_masks else weight
        weight = weight * mask_alpha(hm) * mask_alpha(sm) * mask_alpha(bm)
        new_hue = (hue + rotation * hue_effect) % 360.0
        sat_target = sat_adjust if set_saturation else sat * (1.0 + sat_adjust)
        val_target = brt_adjust if set_brightness else val * (1.0 + brt_adjust)
        new_sat = np.clip(sat + (sat_target - sat) * sat_effect, 0.0, 1.0)
        new_val = np.maximum(val + (val_target - val) * brt_effect, 0.0)
        out_rgb = np.stack([new_val - new_val * new_sat * np.clip(np.minimum((n + new_hue / 60.0) % 6.0,
                                                                            4.0 - (n + new_hue / 60.0) % 6.0), 0.0, 1.0)
                            for n in (5.0, 3.0, 1.0)], axis=-1)
        untouched = (rotation == 0.0 and sat_adjust == 0.0 and brt_adjust == 0.0
                     and not set_saturation and not set_brightness)
        if untouched:
            out_rgb = rgb
        output_alpha = p.get("output_alpha", 0)
        # 0/1 are legacy serialized bools: preserve / combined range weight.
        output_alpha = {0: "preserve", 1: "combined_weight", False: "preserve", True: "combined_weight"}.get(output_alpha, output_alpha)
        alpha_values = {"combined_weight": weight, "hue_weight": hue_weight,
                        "saturation_weight": sat_weight, "brightness_weight": brt_weight,
                        "hue": hue, "saturation": sat, "value": val}
        alpha = (alpha_values[output_alpha][..., None] if output_alpha in alpha_values
                 else image[..., 3:4])
        return np.concatenate([out_rgb, alpha], axis=2).astype(np.float32)

    @staticmethod
    def _softclip(image, p):
        # Nuke's four conversions. "logarithmic compress" maps softclip_min..softclip_max onto
        # softclip_min..1 with a logarithmic curve of slope 1 at softclip_min (so values at or below
        # it are untouched and the join is smooth); values above softclip_max keep climbing past 1.
        # A max at or below 1 leaves nothing to compress. The two "preserve hue" modes only act on
        # pixels with a component above softclip_max: brightness is kept by desaturating toward the
        # luma, saturation is kept by scaling the whole pixel down. Alpha is untouched.
        mode = p.get("conversion", "none")
        # The limits are compared as float32, the pixel precision, so a pixel equal to softclip_min stays put.
        lo, hi = float(np.float32(p.get("softclip_min", 0.8))), float(np.float32(p.get("softclip_max", 1.0)))
        rgb = image[..., :3].astype(np.float64)
        if mode == "logarithmic compress":
            span, room = hi - lo, 1.0 - lo
            if span > room > 0.0:
                k = Evaluator._softclip_log_rate(span, room)
                t = np.maximum(rgb - lo, 0.0)
                out_rgb = np.where(rgb > lo, lo + room * np.log1p(k * t) / math.log1p(k * span), rgb)
            else:
                out_rgb = rgb
        elif mode in ("preserve hue and brightness", "preserve hue and saturation"):
            peak = rgb.max(axis=-1, keepdims=True)
            over = peak > hi
            if mode == "preserve hue and saturation":
                out_rgb = np.where(over, rgb * (hi / np.where(over, peak, 1.0)), rgb)
            else:
                luma = 0.2126 * rgb[..., 0:1] + 0.7152 * rgb[..., 1:2] + 0.0722 * rgb[..., 2:3]
                gap = np.where(peak > luma, peak - luma, 1.0)
                keep = np.clip((hi - luma) / gap, 0.0, 1.0)
                out_rgb = np.where(over, luma + (rgb - luma) * keep, rgb)
        else:
            out_rgb = rgb
        return np.concatenate([out_rgb, image[..., 3:4]], axis=2).astype(np.float32)
