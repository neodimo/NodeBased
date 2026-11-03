"""Flare, Glint, Sparkles and ScannedGrain exactly as they stood before the 2D behaviour lane 2 step H (copied from nodebased/imaging.py at 0b3564ac).

Tests compare an older document's node with these, bit for bit.
"""
import math

import numpy as np

from nodebased.imaging import Evaluator


class Legacy:
    @staticmethod
    def _flare(image, p, origin=(0, 0), frame_size=None):
        h, w = image.shape[:2]
        yy, xx = np.mgrid[:h, :w].astype(np.float32)
        x, y = xx + origin[0], yy + origin[1]
        cx, cy = float(p.get("position_x", w / 2)), float(p.get("position_y", h / 2))
        dx, dy = x-cx, y-cy
        r = np.sqrt(dx*dx + dy*dy)
        size = max(0.1, float(p.get("size", 18)))
        glow = np.exp(-(r / size) ** 2)
        # A pair of low-amplitude diffraction rings around the bright core.
        glow += 0.12 * np.exp(-np.abs(r - size * 2.0) / max(0.4, size * 0.06))
        glow += 0.06 * np.exp(-np.abs(r - size * 3.2) / max(0.4, size * 0.05))
        angle = math.radians(float(p.get("rotation", 0)))
        streaks = max(0, min(64, int(p.get("streaks", 6))))
        length = max(0.0, float(p.get("length", 180)))
        for i in range(streaks):
            a = angle + i * math.pi / max(1, streaks)
            along = np.abs(dx*math.cos(a) + dy*math.sin(a))
            across = np.abs(-dx*math.sin(a) + dy*math.cos(a))
            glow += np.exp(-along / max(1, length)) * np.exp(-across / max(0.5, size*0.12)) * 0.18
        ghosts = max(0, min(64, int(p.get("ghosts", 5))))
        # Optical ghosts lie on the axis through the flare and the frame centre.
        frame_width, frame_height = frame_size or (w, h)
        vx, vy = frame_width / 2 - cx, frame_height / 2 - cy
        norm = max(1.0, math.hypot(vx, vy))
        for i in range(1, ghosts + 1):
            t = -i * float(p.get("spread", 0.65))
            gx, gy = cx + vx*t, cy + vy*t
            gr = np.sqrt((x-gx)**2 + (y-gy)**2)
            gs = size * max(0.1, 1.0 - i/(ghosts+1))
            glow += 0.25 * np.exp(-(gr/gs)**2)
        shift = float(p.get("chromatic_shift", 2.0))
        color = np.array([p.get("red", .7), p.get("green", .85), p.get("blue", 1.)], np.float32)
        out = image.copy()
        gain = float(p.get("brightness", 1.0))
        for c in range(3):
            delta = shift * (c-1)
            channel_r = np.sqrt((x-(cx+delta))**2 + (y-cy)**2)
            out[..., c] += (gain * color[c] * (glow + .5*np.exp(-(channel_r/size)**2))).astype(np.float32)
        return out


    @staticmethod
    def _glint(image, p):
        # A line-kernel convolution of only the values above tolerance. Every chosen orientation
        # contributes two symmetric arms, so `rays` is the count of distinct star axes.
        bright = np.maximum(image[..., :3] - float(p.get("tolerance", 1)), 0.0)
        out = image.copy()
        count = max(1, min(32, int(p.get("rays", 4))))
        length = max(0.0, float(p.get("length", 12)))
        h, w = bright.shape[:2]
        glint = np.zeros_like(bright)
        falloff = max(0.0, float(p.get("falloff", .8)))
        for i in range(count):
            a = math.radians(float(p.get("rotation", 0))) + i*math.pi/count
            dx, dy = int(round(math.cos(a))), int(round(math.sin(a)))
            if dx == 0 and dy == 0: dx = 1
            for distance in range(1, min(1000, int(math.ceil(length))) + 1):
                ox, oy = dx * distance, dy * distance
                weight = math.exp(-falloff * distance / max(1.0, length)) / count
                for sx, sy in ((ox, oy), (-ox, -oy)):
                    x0, x1 = max(0, -sx), min(w, w-sx)
                    y0, y1 = max(0, -sy), min(h, h-sy)
                    if x1 > x0 and y1 > y0:
                        glint[y0+sy:y1+sy, x0+sx:x1+sx] += bright[y0:y1, x0:x1] * weight
        out[..., :3] += glint
        return out


    @staticmethod
    def _sparkles(image, p, frame=None, origin=(0, 0)):
        h, w = image.shape[:2]
        yy, xx = np.mgrid[:h, :w]
        # Hash per pixel, seed and frame; deterministic across tile boundaries.
        # Absolute-origin offset keeps tiles stable (coordinate hash rather than tile-local RNG).
        q = (xx + int(origin[0])) * 73856093 ^ (yy + int(origin[1])) * 19349663 ^ (int(p.get("seed", 1))+int(frame or 0))*83492791
        q = q.astype(np.uint32)
        q ^= q >> 13; q *= np.uint32(1274126177); q ^= q >> 16
        hits = ((q.astype(np.float64) / 4294967296.0) < float(p.get("density", .02))) & (np.max(image[..., :3], axis=2) > float(p.get("tolerance", 1)))
        # A cross-shaped glint, with a Gaussian centre.
        out = image.copy()
        size = max(1, min(64, int(round(float(p.get("size", 3))))))
        for dy in range(-size, size+1):
            for dx in range(-size, size+1):
                if dx and dy: continue
                weight = math.exp(-max(abs(dx), abs(dy))/max(1, size*.45))
                sy0, sy1 = max(0,-dy), min(h,h-dy); sx0,sx1=max(0,-dx),min(w,w-dx)
                if sy1>sy0 and sx1>sx0:
                    out[sy0+dy:sy1+dy,sx0+dx:sx1+dx,:3] += hits[sy0:sy1,sx0:sx1,None] * weight
        return out


    @staticmethod
    def _scanned_grain(image, plate, p, frame=None):
        if plate is None: raise ValueError("ScannedGrain: connect a grain plate")
        h, w = image.shape[:2]; ph, pw = plate.shape[:2]
        # Match each plate channel's scanned mean and variance, wrapping a translated plate per frame.
        seed, frame = int(p.get("seed", 1)), int(frame or 0)
        irregularity = float(p.get("irregularity", 0.0))
        jitter = int(round(irregularity * (seed * 1103515245 + frame * 12345 & 0x7fffffff) / 0x7fffffff * max(ph, pw)))
        sy, sx = (frame + jitter) % ph, (seed*17 + frame + jitter) % pw
        tile = np.tile(plate, (math.ceil((h+sy)/ph), math.ceil((w+sx)/pw), 1))[sy:sy+h, sx:sx+w]
        out = image.copy(); alpha = image[..., 3:4]
        luma = np.maximum(0.0, image[..., :3].mean(axis=2))
        response = np.maximum(np.power(luma, max(.01, float(p.get("response", 1.0)))),
                              float(p.get("minimum", 0.0)))
        preset_gain = {"neutral": 1.0, "35mm": 0.8, "16mm": 1.15, "8mm": 1.45, "reversal": 0.65}.get(p.get("preset", "neutral"), 1.0)
        alpha_gate = alpha if p.get("apply_through_alpha", 0) else 1.0
        for c in range(3):
            noise = tile[..., c] - float(plate[..., c].mean())
            out[..., c] += noise * float(p.get("amount", 1.0)) * response * preset_gain * alpha_gate[..., 0] if isinstance(alpha_gate, np.ndarray) else noise * float(p.get("amount", 1.0)) * response * preset_gain * alpha_gate
        out[..., 3:4] = alpha
        return out

