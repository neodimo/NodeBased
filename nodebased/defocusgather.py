"""The gather behind ZDefocus and Bokeh (`Evaluator._zdefocus`), vectorised.

Every output pixel averages the samples of its own filter window that lie at or in front of it (the occlusion rule), through the filter shape of
its own blur radius. The filter depends only on that radius, so pixels are grouped by radius: one kernel is built per distinct radius (a layered
ZDefocus has a handful, a continuous Bokeh ramp one per depth value), and a group is gathered as one (pixels x filter taps) block, in chunks that
bound the memory. Pixels whose window shows nothing in front of them keep the rule of the per-pixel loop (the pixel alone), and a filter under a
pixel across stays the pixel. `reference_gather` is that per-pixel loop, kept as the oracle the tests compare against.
"""
from __future__ import annotations

import numpy as np

# Elements (pixels x taps) of one gathered block; the colour gather holds four channels of it in float32.
CHUNK_TAPS = 1 << 21


def _centre_only(centre, kernel_grid, disk, sides_px, fill, image_kernel):
    """The result for a pixel whose window shows only its own sample (the loop's `visible[radius, radius] = True`), for n pixels at once.
    `centre` is (n, channels) float32; the formulas are the loop's, with a one-sample window."""
    if kernel_grid is not None:
        kc = np.asarray(kernel_grid[kernel_grid.shape[0] // 2, kernel_grid.shape[1] // 2], np.float64)
        return centre * (kc / np.maximum(kc, 1e-20))
    if not fill:
        return np.where((sides_px < 0)[:, None], centre / max(1, int(disk.sum())), centre)
    return centre


def gather(pad_img, pad_z, z, radii, radius, shape, p, kernel, override, fill, sides, sph_apply, dtype):
    """The filtered picture (h x w x channels, float32) of the loop below, grouped by blur radius.

    `shape(r, yy, xx, radius)` returns (disk, kernel_grid) for a radius; `sph_apply(disk, kernel_grid, r)` applies Bokeh's spherical aberration."""
    h, w = z.shape
    channels = pad_img.shape[2]
    out = np.empty((h, w, channels), dtype)
    yy, xx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    flat_r = np.ascontiguousarray(radii, np.float32).reshape(-1)
    values, inverse = np.unique(flat_r, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    ends = np.cumsum(np.bincount(inverse, minlength=len(values)))
    starts = ends - np.bincount(inverse, minlength=len(values))
    is_image = p.get("bokeh_shape") == "image"
    z_flat = z.reshape(-1)
    centre_z = z_flat + np.float32(1e-6)
    out_flat = out.reshape(-1, channels)
    pw = pad_z.shape[1]
    z_pad_flat = np.ascontiguousarray(pad_z).reshape(-1)
    img_flat = np.ascontiguousarray(pad_img).reshape(-1, channels)
    # A pixel's channels move as one 16-byte element (four float32) when gathering, which is several times cheaper than four indexed reads.
    packed = img_flat.view(np.complex128).reshape(-1) if channels == 4 and img_flat.dtype == np.float32 else None
    for gi, r in enumerate(values):
        r = float(r)
        index = order[starts[gi]:ends[gi]]
        ys, xs = np.divmod(index, w)
        if override is not None and r < 1.0 and not is_image:
            out_flat[index] = pad_img[ys + radius, xs + radius]       # a filter under a pixel across is the pixel itself
            continue
        disk, kernel_grid = shape(r, yy, xx, radius)
        if override is not None and override.get("sph") and not is_image and r > 0:
            kernel_grid = sph_apply(disk, kernel_grid, r)
        ty, tx = np.nonzero(disk)                                     # the taps: window rows and columns inside the filter
        taps = len(ty)
        weight = None if kernel_grid is None else np.asarray(kernel_grid)[ty, tx]
        sides_px = sides[ys, xs] if sides is not None else np.zeros(len(index), np.int64)
        tap_offset = (ty * pw + tx).astype(np.int64)
        step = max(1, CHUNK_TAPS // max(1, taps))
        for lo in range(0, len(index), step):
            sl = slice(lo, lo + step)
            gy, gx = ys[sl], xs[sl]
            flat = (gy * pw + gx)[:, None] + tap_offset[None, :]
            visible = z_pad_flat.take(flat) <= centre_z[index[sl]][:, None]
            count = visible.sum(axis=1)
            if packed is not None:
                samples = packed.take(flat).view(np.float32).reshape(flat.shape + (channels,))     # (n, taps, channels)
            else:
                samples = img_flat.take(flat, axis=0)
            if weight is None:
                total = np.einsum("nk,nkc->nc", visible.astype(np.float32), samples, dtype=np.float64)
                if not fill:
                    divisor = np.where(sides_px[sl] < 0, max(1, int(disk.sum())), np.maximum(count, 1))
                else:
                    divisor = np.maximum(count, 1)
                result = total / divisor[:, None]
            elif weight.ndim == 2:                                    # a filter image with a weight per channel: normalised per channel
                weights = visible[..., None] * weight[None, :, :]
                total = np.einsum("nkc,nkc->nc", weights, samples, dtype=np.float64)
                result = total / np.maximum(weights.sum(axis=1), 1e-20)
            else:
                weights = visible * weight[None, :]
                total = np.einsum("nk,nkc->nc", weights, samples, dtype=np.float64)
                result = total / np.maximum(weights.sum(axis=1), 1e-20)[:, None]
            none = count == 0
            if none.any():                                            # nothing in front: only the pixel's own sample is visible
                centre = pad_img[gy[none] + radius, gx[none] + radius]
                result[none] = _centre_only(centre, kernel_grid, disk, sides_px[sl][none], fill, is_image)
            out_flat[index[sl]] = result
    return out


def reference_gather(pad_img, pad_z, z, radii, radius, shape, p, kernel, override, fill, sides, sph_apply, dtype):
    """The per-pixel loop `gather` replaces, unchanged: the oracle for the tests (and the speed baseline of tools/benchmark_defocus_gather.py)."""
    h, w = z.shape
    out = np.empty((h, w, pad_img.shape[2]), dtype)
    yy, xx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    for y in range(h):
        for x in range(w):
            r = float(radii[y, x])
            if override is not None and r < 1.0 and p.get("bokeh_shape") != "image":
                out[y, x] = pad_img[y + radius, x + radius]
                continue
            disk, kernel_grid = shape(r, yy, xx, radius)
            if override is not None and override.get("sph") and p.get("bokeh_shape") != "image" and r > 0:
                kernel_grid = sph_apply(disk, kernel_grid, r)
            samples_z = pad_z[y:y + 2 * radius + 1, x:x + 2 * radius + 1]
            visible = disk & (samples_z <= z[y, x] + 1e-6)
            if not np.any(visible): visible[radius, radius] = True
            samples = pad_img[y:y + 2 * radius + 1, x:x + 2 * radius + 1]
            if p.get("bokeh_shape") == "image" and kernel is not None:
                if kernel_grid.ndim == 3:
                    weights = kernel_grid * visible[..., None]
                    out[y, x] = np.sum(samples * weights, axis=(0, 1)) / np.maximum(weights.sum(axis=(0, 1)), 1e-20)
                else:
                    weights = kernel_grid * visible
                    out[y, x] = np.sum(samples * weights[..., None], axis=(0, 1)) / max(float(weights.sum()), 1e-20)
            elif kernel_grid is not None:
                weights = kernel_grid * visible
                out[y, x] = np.sum(samples * weights[..., None], axis=(0, 1)) / max(float(weights.sum()), 1e-20)
            elif not fill and sides[y, x] < 0:
                out[y, x] = samples[visible].sum(axis=0) / max(1, int(disk.sum()))
            else:
                out[y, x] = samples[visible].mean(axis=0)
    return out
