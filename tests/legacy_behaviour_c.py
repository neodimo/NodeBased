"""The keyer and channel kernels exactly as they stood before the 2D behaviour lane 2 step C (copied from nodebased/imaging.py at ed80309c).

Tests compare an older document's node with these, bit for bit. Shared helpers are the live ones: they were not changed.
"""
import numpy as np

from nodebased import keyerknobs as _keyerknobs
from nodebased.imaging import Evaluator


class Legacy:
    @staticmethod
    def _hue_keyer(image, p):
        rgb = image[..., :3]
        hue, sat = Evaluator._rgb_to_hue_sat(rgb)
        center = float(p.get("hue_center", 0.0)) % 360.0
        raw_dist = np.abs(hue - center) % 360.0
        dist = np.minimum(raw_dist, 360.0 - raw_dist)
        hue_gate = Evaluator._edge_falloff(dist, float(p.get("hue_width", 30.0)) / 2.0,
                                           p.get("hue_softness", 15.0))
        sat_gate = ((sat >= p.get("sat_min", 0.0)) & (sat <= p.get("sat_max", 1.0))).astype(np.float32)
        alpha = (hue_gate * sat_gate).astype(np.float32)
        if p.get("invert"):
            alpha = 1.0 - alpha
        if "output_enabled" in p or "output_channel" in p:
            return _keyerknobs.hue_output(image, alpha, p)
        return np.concatenate([rgb, alpha], axis=2).astype(np.float32)

    @staticmethod
    def _channel_merge(a, b, p):
        index = {"r": 0, "g": 1, "b": 2, "a": 3}
        a_idx = index[p.get("a_channel", "A.a").split(".")[1]]
        b_idx = index[p.get("b_channel", "B.a").split(".")[1]]
        out_idx = {"R": 0, "G": 1, "B": 2, "A": 3}[p.get("out_channel", "A")]
        combined = Evaluator._channel_merge_op(p.get("operation", "over"),
                                               a[..., a_idx:a_idx + 1], b[..., b_idx:b_idx + 1])
        out = b.copy()
        out[..., out_idx:out_idx + 1] = combined
        return out
