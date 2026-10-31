"""Measurements of the fire shot's flame shape, read from the cached volume and from the emission render pass.

Everything here is deterministic given a volume (no sampling noise), except the image-row figures, which come from a
128-sample emission pass. Heights are cell slices counted up from the bottom of the solver's box.
"""
from __future__ import annotations

import numpy as np

from nodebased import volumerender

# ACEScg (AP1) luminance weights.
LUMA = np.array([0.2722287168, 0.6740817658, 0.0536895174])
LIVE_FRACTION = .05          # a slice or row counts as part of the flame when its peak is this fraction of the frame peak
TONGUE_FRACTION = .35        # a tongue is a connected region above this fraction of its own slice's peak
TONGUE_MIN_CELLS = 6         # smaller regions are speckle, not a tongue
SMOKE_THRESHOLD = .05        # density above which a cell holds smoke


def emission_luminance(volume, settings):
    """Per-voxel luminance of the radiance the renderer's fire emission gives each voxel (hot smoke plus flame glow).

    The same function the fire-light grid uses (`volumerender._fire_emission`), so the field matches what the renderers
    add before attenuation."""
    table = volumerender.fire_table(settings)
    temperature = None if volume.temperature is None else np.asarray(volume.temperature, np.float32)
    flame = None if volume.flame is None else np.asarray(volume.flame, np.float32)
    emission = volumerender._fire_emission(settings, table, np.asarray(volume.density, np.float32), temperature, flame)
    return emission @ LUMA


def label_regions(mask):
    """Number of 4-connected regions of at least `TONGUE_MIN_CELLS` cells in a 2D boolean mask (no SciPy here)."""
    mask = np.asarray(mask, bool)
    seen = np.zeros_like(mask)
    count = 0
    for start in zip(*np.nonzero(mask)):
        if seen[start]:
            continue
        stack, size = [start], 0
        seen[start] = True
        while stack:
            i, j = stack.pop()
            size += 1
            for ni, nj in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
                if 0 <= ni < mask.shape[0] and 0 <= nj < mask.shape[1] and mask[ni, nj] and not seen[ni, nj]:
                    seen[ni, nj] = True
                    stack.append((ni, nj))
        count += size >= TONGUE_MIN_CELLS
    return count


def runs_in_row(row, min_width):
    """Number of runs of True at least `min_width` long in a 1D boolean row."""
    padded = np.concatenate(([False], np.asarray(row, bool), [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return int(((edges[1::2] - edges[0::2]) >= min_width).sum())


def flame_span(peaks):
    """(first, last) slice index where the per-slice peak is at least LIVE_FRACTION of the frame's peak, or None."""
    peaks = np.asarray(peaks)
    if peaks.size == 0 or peaks.max() <= 0:
        return None
    live = np.flatnonzero(peaks >= LIVE_FRACTION * peaks.max())
    return (int(live[0]), int(live[-1])) if live.size else None


def above(field, volume, floor_height):
    """`field` (nx, ny, nz) with the slices below `floor_height` (world units above the volume's origin: the top of the fuel block,
    whose cells are hidden inside it) zeroed, so the measurements describe the flame the camera can see."""
    cut = int(np.ceil((floor_height - float(volume.origin[1])) / float(volume.voxel_size) - 1e-6))
    if cut <= 0:
        return field
    out = np.array(field, copy=True)
    out[:, :cut, :] = 0
    return out


def height_profile(luma):
    """Per-height-slice peak and mean-of-lit emission luminance, and where the peak sits in the flame's span.

    `luma` is (nx, ny, nz). `core_position` is 0 at the bottom of the flame's span and 1 at the top; the core is "in the
    lowest third" when it is at most 1/3. `top_over_bottom` is the brightest value in the top third of the span over the
    brightest in the bottom third; `top_over_bottom_mean` is the same with the mean of the slice peaks in each third (below 1
    means the flame cools with height; the mean does not hold at 1 just because the flame's last hot cell is in the top third)."""
    peaks = luma.reshape(luma.shape[0], luma.shape[1], -1).max(axis=(0, 2))
    span = flame_span(peaks)
    if span is None:
        return {'span': None, 'core_position': None, 'core_in_lowest_third': False, 'top_over_bottom': None,
                'top_over_bottom_mean': None, 'peaks': peaks.round(4).tolist()}
    lo, hi = span
    length = hi - lo + 1
    core = int(np.argmax(peaks[lo:hi + 1]))
    third = max(1, length // 3)
    bottom, top = float(peaks[lo:lo + third].max()), float(peaks[hi + 1 - third:hi + 1].max())
    bottom_mean, top_mean = float(peaks[lo:lo + third].mean()), float(peaks[hi + 1 - third:hi + 1].mean())
    return {'span': [lo, hi], 'core_position': round(core / max(length - 1, 1), 3),
            'core_in_lowest_third': core / max(length - 1, 1) <= 1 / 3 + 1e-9,
            'top_over_bottom': round(top / bottom, 4) if bottom > 0 else None,
            'top_over_bottom_mean': round(top_mean / bottom_mean, 4) if bottom_mean > 0 else None,
            'peaks': peaks.round(4).tolist()}


def tongue_counts(flame):
    """Separate bright regions per horizontal slice of the flame (burn-rate) field: the mean over the slices of the flame's span
    (and the maximum). A region is a 4-connected set of at least `TONGUE_MIN_CELLS` cells above `TONGUE_FRACTION` of its
    slice's peak, so unburnt smoke specks do not count."""
    flame = np.maximum(np.asarray(flame, np.float32), 0)
    peaks = flame.reshape(flame.shape[0], flame.shape[1], -1).max(axis=(0, 2))
    span = flame_span(peaks)
    if span is None:
        return {'mean': 0.0, 'max': 0, 'slices': 0}
    counts = []
    for y in range(span[0], span[1] + 1):
        plane = flame[:, y, :]
        counts.append(label_regions(plane >= TONGUE_FRACTION * plane.max()))
    return {'mean': round(float(np.mean(counts)), 3), 'max': int(max(counts)), 'slices': len(counts)}


def smoke_eddies(volume):
    """Smoke eddy energy over the smoke that is not burning: mean vorticity magnitude and high-pass density contrast.

    The cells are those with density above `SMOKE_THRESHOLD` and a flame (burn rate) below `LIVE_FRACTION` of the frame's peak
    flame, so the fast flame zone does not count. `vorticity_mean` is the mean |curl v| (1/s) over them. `density_contrast` is the rms
    of (density - its 5-cell box blur) over them, divided by their mean density: how much of the haze is structure rather than a
    smooth column."""
    density = np.asarray(volume.density, np.float32)
    smoke = density > SMOKE_THRESHOLD
    if volume.flame is not None:
        flame = np.asarray(volume.flame, np.float32)
        smoke &= flame < LIVE_FRACTION * max(float(flame.max()), 1e-9)
    if not smoke.any():
        return {'vorticity_mean': 0.0, 'density_contrast': 0.0}
    out = {}
    if volume.velocity is not None:
        h = float(volume.voxel_size)
        v = np.asarray(volume.velocity, np.float32)
        grad = [[np.gradient(v[..., c], h, axis=a) for a in range(3)] for c in range(3)]
        wx, wy, wz = grad[2][1] - grad[1][2], grad[0][2] - grad[2][0], grad[1][0] - grad[0][1]
        out['vorticity_mean'] = float(np.sqrt(wx * wx + wy * wy + wz * wz)[smoke].mean())
    blur = density.copy()
    for axis in range(3):
        padded = np.pad(blur, [(2, 2) if a == axis else (0, 0) for a in range(3)], mode='edge')
        blur = sum(np.take(padded, np.arange(k, k + density.shape[axis]), axis=axis) for k in range(5)) / 5
    out['density_contrast'] = float(np.sqrt(((density - blur)[smoke] ** 2).mean()) / density[smoke].mean())
    return out


def column_sway(volume, threshold_fraction=.05):
    """Lateral drift of the flame's centroid across heights (cells): the standard deviation of the x and z centroids
    over the height slices, as the larger of the two. Same measure as `render_evidence.flame_occupancy`."""
    flame = np.asarray(volume.flame, np.float32)
    if flame.max() <= 0:
        return 0.0
    centroids = []
    xi, zi = np.indices(flame.shape[::2])
    for y in range(flame.shape[1]):
        plane = flame[:, y, :]
        if plane.max() > threshold_fraction * flame.max():
            total = float(plane.sum())
            centroids.append(((plane * xi).sum() / total, (plane * zi).sum() / total))
    return round(float(np.asarray(centroids).std(axis=0).max()), 3) if len(centroids) > 1 else 0.0


def image_rows(emission):
    """The same two measures on a rendered emission pass (H, W, 4): per image row, the peak luminance and the number of
    separate bright runs (at least 4 pixels wide, above TONGUE_FRACTION of the row's peak).

    Rows count as the flame's span where their peak is above LIVE_FRACTION of the image peak. Row 0 is the top of the picture,
    so the profile is flipped to run from the bottom up like the field profile."""
    luma = np.asarray(emission[..., :3], np.float64) @ LUMA
    peaks = luma.max(axis=1)[::-1]
    span = flame_span(peaks)
    if span is None:
        return {'span': None}
    lo, hi = span
    length = hi - lo + 1
    core = int(np.argmax(peaks[lo:hi + 1]))
    third = max(1, length // 3)
    rows = luma[::-1]
    counts = [runs_in_row(rows[y] >= TONGUE_FRACTION * rows[y].max(), 4) for y in range(lo, hi + 1)]
    return {'span_rows': [lo, hi], 'core_position': round(core / max(length - 1, 1), 3),
            'top_over_bottom': round(float(peaks[hi + 1 - third:hi + 1].max() / peaks[lo:lo + third].max()), 4),
            'tongues_mean': round(float(np.mean(counts)), 3), 'peak': round(float(peaks.max()), 4)}


def frame_metrics(volume, settings, emission_pixels=None, floor_height=0.0):
    """All the flame-shape numbers for one frame. `floor_height` is the top of whatever hides the flame's lowest cells."""
    luma = above(emission_luminance(volume, settings), volume, floor_height)
    profile = height_profile(luma)
    entry = {'profile': {k: v for k, v in profile.items() if k != 'peaks'},
             'tongues': tongue_counts(above(np.asarray(volume.flame, np.float32), volume, floor_height)),
             'eddies': smoke_eddies(volume), 'sway_cells': column_sway(volume),
             'peak_luminance': round(float(luma.max()), 4), 'height_peaks': profile['peaks']}
    if emission_pixels is not None:
        entry['image'] = image_rows(emission_pixels)
    return entry


def summarise(frames):
    """Averages over the per-frame entries (a dict frame -> `frame_metrics`)."""
    entries = list(frames.values())

    def mean(getter):
        values = [getter(e) for e in entries]
        values = [v for v in values if v is not None]
        return round(float(np.mean(values)), 4) if values else None

    return {'core_in_lowest_third_frames': sum(bool(e['profile']['core_in_lowest_third']) for e in entries),
            'frames': len(entries),
            'core_position_mean': mean(lambda e: e['profile']['core_position']),
            'top_over_bottom_mean': mean(lambda e: e['profile']['top_over_bottom']),
            'top_over_bottom_of_means': mean(lambda e: e['profile']['top_over_bottom_mean']),
            'tongues_per_slice_mean': mean(lambda e: e['tongues']['mean']),
            'vorticity_mean': mean(lambda e: e['eddies'].get('vorticity_mean')),
            'density_contrast_mean': mean(lambda e: e['eddies']['density_contrast']),
            'sway_cells_mean': mean(lambda e: e['sway_cells'])}


def colour_by_height(volume, settings, floor_height=0.0):
    """Emission colour per height slice: green over red of the summed emission (1 is yellow-white, near 0 is deep red),
    for the bottom and the top third of the flame's span, and the summed luminance share of the bottom third.

    The field's own radiance (`volumerender._fire_emission`), before attenuation by smoke in front."""
    table = volumerender.fire_table(settings)
    temperature = None if volume.temperature is None else np.asarray(volume.temperature, np.float32)
    flame = None if volume.flame is None else np.asarray(volume.flame, np.float32)
    emission = volumerender._fire_emission(settings, table, np.asarray(volume.density, np.float32), temperature, flame)
    emission = above(emission, volume, floor_height)
    per_slice = emission.sum(axis=(0, 2))                         # (ny, 3)
    peaks = (emission @ LUMA).reshape(emission.shape[0], emission.shape[1], -1).max(axis=(0, 2))
    span = flame_span(peaks)
    if span is None:
        return {'bottom_g_over_r': None, 'top_g_over_r': None, 'bottom_luminance_share': None}
    lo, hi = span
    third = max(1, (hi - lo + 1) // 3)
    bottom, top = per_slice[lo:lo + third].sum(axis=0), per_slice[hi + 1 - third:hi + 1].sum(axis=0)
    total = (per_slice[lo:hi + 1] @ LUMA).sum()
    return {'bottom_g_over_r': round(float(bottom[1] / max(bottom[0], 1e-9)), 4),
            'top_g_over_r': round(float(top[1] / max(top[0], 1e-9)), 4),
            'bottom_luminance_share': round(float(bottom @ LUMA / max(total, 1e-9)), 4)}
