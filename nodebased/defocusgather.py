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


# An FFT correlation replaces the gather for the pixels of one blur radius that also share one depth (the occlusion mask is then the same for all of
# them, so the group's sum is a plain correlation of the masked picture with the filter). It pays once the gather's work (pixels x taps) is this many
# times the FFT's (padded region size): measured in tools/benchmark_defocus_fft.py. Set to 0 to switch the FFT path off.
FFT_GAIN = 1.0
FFT_MIN_PIXELS = 256


def _good_size(n):
    """The next 2^a 3^b 5^c at or above n: sizes numpy's FFT does quickly."""
    best = 1 << (int(n) - 1).bit_length()
    f5 = 1
    while f5 < best:
        f3 = f5
        while f3 < best:
            f2 = f3
            while f2 < n:
                f2 *= 2
            best = min(best, f2)
            f3 *= 3
        f5 *= 5
    return best


def _fft_cost(count, taps, ys, xs, ty, tx):
    """Whether an FFT of this group is cheaper than gathering it: the gather costs count x taps, the FFT about 12 passes over the padded region."""
    hr = int(ys.max() - ys.min()) + int(ty.max() - ty.min()) + 1
    wr = int(xs.max() - xs.min()) + int(tx.max() - tx.min()) + 1
    return count * taps > FFT_GAIN * 12.0 * _good_size(hr) * _good_size(wr)


def _fft_group(pad_img, pad_z, ys, xs, ty, tx, disk, weight, centre_z, radius):
    """(total (n, channels), count (n,), weight sum (n,) or None) of the pixels (ys, xs), all sharing the depth `centre_z`, by FFT correlation.

    A pixel (y, x) sums the window samples at offsets (ty, tx) whose depth is at or in front of centre_z, with the filter weights; the mask depends
    only on the sample, so the whole group is one correlation of (picture x mask) with the filter. Float64 throughout."""
    t0, t1, u0, u1 = int(ty.min()), int(ty.max()), int(tx.min()), int(tx.max())
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
    hr, wr = y1 - y0 + t1 - t0 + 1, x1 - x0 + u1 - u0 + 1
    s = (_good_size(hr), _good_size(wr))
    rows = slice(y0 + t0, y0 + t0 + hr)
    cols = slice(x0 + u0, x0 + u0 + wr)
    mask = (pad_z[rows, cols] <= centre_z).astype(np.float64)
    picture = pad_img[rows, cols].astype(np.float64) * mask[..., None]
    shape_kernel = (t1 - t0 + 1, u1 - u0 + 1)
    unit = np.zeros(shape_kernel); unit[ty - t0, tx - u0] = 1.0
    spectrum = lambda grid, axes=(0, 1): np.fft.rfft2(grid, s, axes=axes)
    back = lambda f, axes=(0, 1): np.fft.irfft2(f, s, axes=axes)
    fm = spectrum(mask)
    sel = (slice(0, y1 - y0 + 1), slice(0, x1 - x0 + 1))
    py, px = ys - y0, xs - x0
    count = np.rint(back(fm * np.conj(spectrum(unit)))[sel][py, px]).astype(np.int64)
    fp = spectrum(picture, (0, 1))
    if weight is None:
        total = back(fp * np.conj(spectrum(unit))[..., None], (0, 1))[sel][py, px]
        return _denoise(total, picture, len(ty)), count, None
    grid = np.zeros(shape_kernel); grid[ty - t0, tx - u0] = weight
    fk = np.conj(spectrum(grid))
    total = back(fp * fk[..., None], (0, 1))[sel][py, px]
    return _denoise(total, picture, float(np.abs(weight).sum())), count, back(fm * fk)[sel][py, px]


def _denoise(total, picture, reach):
    """An FFT leaves about 1e-16 of the largest sum as noise where the exact sum is zero (an impulse's empty surroundings): sums under 1e-13 of
    the largest possible one become the exact zero the gather gives. The error this can cause is under 1e-13 x the brightest sample."""
    limit = 1e-13 * reach * np.abs(picture).max(axis=(0, 1))
    return np.where(np.abs(total) <= limit, 0.0, total)


def _centre_only(centre, kernel_grid, disk, sides_px, fill, image_kernel):
    """The result for a pixel whose window shows only its own sample (the loop's `visible[radius, radius] = True`), for n pixels at once.
    `centre` is (n, channels) float32; the formulas are the loop's, with a one-sample window."""
    if kernel_grid is not None:
        kc = np.asarray(kernel_grid[kernel_grid.shape[0] // 2, kernel_grid.shape[1] // 2], np.float64)
        return centre * (kc / np.maximum(kc, 1e-20))
    if not fill:
        return np.where((sides_px < 0)[:, None], centre / max(1, int(disk.sum())), centre)
    return centre


def _fft_subgroups(index, ys, xs, sides_px, z_flat, centre_z, pad_img, pad_z, ty, tx, disk, weight, fill, kernel_grid, is_image, out_flat, radius):
    """Write the pixels of one radius group that share a depth, and are numerous enough for an FFT to beat the gather, straight into `out_flat`.
    Returns (index, ys, xs, sides_px) of the pixels left for the gather."""
    taps = len(ty)
    zg = z_flat[index]
    if zg.min() == zg.max():
        groups = [np.arange(len(index))]
    else:
        _, inverse = np.unique(zg, return_inverse=True)
        sub_order = np.argsort(inverse, kind="stable")
        bounds = np.cumsum(np.bincount(inverse))
        groups = np.split(sub_order, bounds[:-1])
    keep = np.ones(len(index), bool)
    wsum = None if weight is None else float(weight.sum())
    for sub in groups:
        if len(sub) < FFT_MIN_PIXELS or not _fft_cost(len(sub), taps, ys[sub], xs[sub], ty, tx):
            continue
        gy, gx = ys[sub], xs[sub]
        total, count, den = _fft_group(pad_img, pad_z, gy, gx, ty, tx, disk, weight, centre_z[index[sub][0]], radius)
        if weight is None:
            divisor = np.where(sides_px[sub] < 0, max(1, int(disk.sum())), np.maximum(count, 1)) if not fill else np.maximum(count, 1)
            result = total / divisor[:, None]
            usable = np.ones(len(sub), bool)
        else:
            # A visible-weight sum that rounds to nothing cannot be told from zero by an FFT: those pixels go through the gather.
            usable = (count == 0) | (den > 1e-6 * wsum)
            result = total / np.maximum(den, 1e-20)[:, None]
        none = count == 0
        if none.any():
            centre = pad_img[gy[none] + radius, gx[none] + radius]
            result[none] = _centre_only(centre, kernel_grid, disk, sides_px[sub][none], fill, is_image)
        out_flat[index[sub][usable]] = result[usable]
        keep[sub[usable]] = False
    return index[keep], ys[keep], xs[keep], sides_px[keep]


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
        if FFT_GAIN > 0 and taps > 1 and len(index) >= FFT_MIN_PIXELS and not (weight is not None and weight.ndim == 2):
            index, ys, xs, sides_px = _fft_subgroups(index, ys, xs, sides_px, z_flat, centre_z, pad_img, pad_z, ty, tx, disk, weight, fill,
                                                     kernel_grid, is_image, out_flat, radius)
            if not len(index):
                continue
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
