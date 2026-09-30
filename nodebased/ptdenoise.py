"""Edge-aware denoiser for the path tracer's beauty (lane L4, step R4): the spatial half of SVGF on an a-trous
wavelet, guided by albedo, normal and depth and steered by a per-pixel variance.

What it does. The beauty is divided by the first-hit albedo (so the filter never blurs texture, which the albedo
puts back afterwards), then filtered by `iterations` passes of the 5 x 5 B3-spline kernel with holes: pass `i`
reads taps `2**i` pixels apart, so four passes reach a 61 x 61 neighbourhood for the price of four small filters.
Every tap is weighted by how well it agrees with the pixel:

* normal: `max(0, n_p . n_q) ** phi_normal`, so a crease keeps its edge;
* depth: `exp(-|z_p - z_q| / (phi_depth * |grad z . (p - q)| + eps))`, the plane distance a neighbour should
  have if the surface were flat, so a slanted floor is smoothed along itself and a silhouette is not crossed;
* albedo: a Gaussian in the albedo difference, so a colour change between two materials at one depth keeps its edge;
* radiance: `exp(-|l_p - l_q| / (phi_color * sqrt(var_p) + eps))`, `var_p` the variance of the pixel's own
  estimate (the tracer's sample variance of the mean, blurred by 3 x 3 and carried through the passes as
  `sum(w^2 var) / (sum w)^2`), so a pixel that is noisy accepts far neighbours, a converged one keeps its detail
  and a real edge in the picture stops the filter where the guides cannot see it (a shadow boundary).

The weights sum to one at every pixel, so the filter conserves the mean of a flat region; the radiance weight
depends on the noisy value itself, which pulls a lone bright sample slightly less than a dim one (SVGF's known
small bias), and the tests measure the mean shift. Coverage (alpha) is left as the tracer estimated it, so a
silhouette keeps its exact edge. A pixel with no albedo (a background, an emitter with none) is filtered as it is.

The same function serves any backend: the GPU path tracer gives no variance, so a local estimate from the image
stands in. `guides` in the EXR (`albedo`, `normals`, `depth` next to the raw beauty) are what an external denoiser
such as OpenImageDenoise reads; see docs/3D_FOUNDATION.md, "Denoising", for what was checked about it.
"""
import numpy as np

LUMA = np.array((0.2126, 0.7152, 0.0722))
KERNEL = np.array((1 / 16, 1 / 4, 3 / 8, 1 / 4, 1 / 16))
ITERATIONS = 4
PHI_COLOR = 2.0
PHI_NORMAL = 64.0
PHI_DEPTH = 1.0
SIGMA_ALBEDO = 0.25
ALBEDO_FLOOR = 1e-3
TEMPORAL_BLEND = 0.85  # weight given to the reprojected previous frame where its motion vector matched


def _tap(a, dy, dx):
    """Entry (y, x) of the result is `a[clamp(y + dy), clamp(x + dx)]`."""
    h, w = a.shape[:2]
    ys = np.clip(np.arange(h) + dy, 0, h - 1)
    xs = np.clip(np.arange(w) + dx, 0, w - 1)
    return a[ys][:, xs]


def _box3(a):
    """3 x 3 mean with repeated edges."""
    return sum(_tap(a, dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)) / 9.0


def local_variance(luma, radius=3):
    """A stand-in for the variance of the mean when the renderer gives none: the spread of `luma` in a
    (2 radius + 1)-square window (an upper bound, edges included)."""
    total, squares, count = 0.0, 0.0, 0
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            v = _tap(luma, dy, dx)
            total, squares, count = total + v, squares + v * v, count + 1
    mean = total / count
    return np.maximum(squares / count - mean * mean, 0.0)


def temporal_blend(current, previous, vectors, blend=TEMPORAL_BLEND):
    """`current` (H, W, 4), this frame's filtered result, with `previous` (H, W, 4), the last frame's filtered
    result, blended in at pixels the surface moved into: `vectors` (H, W, 4) is `motionblur.motion_vectors`'
    screen-space displacement in pixels (red x, green y) from this frame's pixel back to where it was on the
    previous frame, alpha 1 where that surface matched. A pixel with no match (alpha 0) or whose source pixel
    falls outside the frame (a disocclusion, or the first frame with no history) keeps this frame's own value,
    so a newly revealed surface never smears in a stale colour; RGB only blends, coverage (alpha) stays this
    frame's own, since a moving silhouette must keep its own edge."""
    current = np.asarray(current, np.float64)
    previous = np.asarray(previous, np.float64)
    vectors = np.asarray(vectors, np.float64)
    h, w = current.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    sx, sy = xs + vectors[..., 0], ys + vectors[..., 1]
    ix, iy = np.round(sx).astype(np.int64), np.round(sy).astype(np.int64)
    valid = (vectors[..., 3] > 0) & (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
    reprojected = previous[np.clip(iy, 0, h - 1), np.clip(ix, 0, w - 1)]
    weight = np.where(valid, blend, 0.0)[..., None]
    out = current.copy()
    out[..., :3] = weight * reprojected[..., :3] + (1 - weight) * current[..., :3]
    return out.astype(np.float32)


def denoise(color, albedo, normal, depth, variance=None, *, iterations=ITERATIONS, phi_color=PHI_COLOR,
            phi_normal=PHI_NORMAL, phi_depth=PHI_DEPTH, sigma_albedo=SIGMA_ALBEDO, strength=1.0, history=None):
    """Filter a premultiplied RGBA beauty `color` (H, W, 4) with its guides: `albedo` (H, W, 3, premultiplied like
    the beauty), `normal` (H, W, 3) world normals, `depth` (H, W) view depth (any value where nothing is hit) and,
    optionally, `variance` (H, W) of the luminance of `color`'s mean. `strength` 0 skips the filter entirely and
    returns `color` unchanged (the knob's contract: 0 is the raw beauty); 1 is the filter as computed. `history`,
    when given, is `(previous, vectors)` for `temporal_blend`, applied to the filtered result before `strength`
    mixes it back toward the raw beauty. Returns float32 (H, W, 4)."""
    strength = float(strength)
    if strength <= 0.0:
        return np.asarray(color, np.float32).copy()
    color = np.asarray(color, np.float64)
    albedo = np.asarray(albedo, np.float64)[..., :3]
    normal = np.asarray(normal, np.float64)[..., :3]
    depth = np.asarray(depth, np.float64)
    if depth.ndim == 3:
        depth = depth[..., 0]
    rgb, alpha = color[..., :3], color[..., 3]
    h, w = alpha.shape
    covered = alpha > 1e-4
    a_max = albedo.max(axis=-1)
    demod = a_max > ALBEDO_FLOOR
    denominator = np.where(demod[..., None], np.maximum(albedo, ALBEDO_FLOOR), 1.0)
    e = rgb / denominator
    a_lum = np.maximum(albedo @ LUMA, ALBEDO_FLOOR)
    if variance is None:
        var = local_variance(e @ LUMA) * 0.25
    else:
        var = np.asarray(variance, np.float64) / np.where(demod, a_lum, 1.0) ** 2
    var = _box3(var)
    a_unit = np.where(demod[..., None], albedo / np.maximum(alpha, 1e-6)[..., None], 0.0)
    z = np.where(covered, depth, 0.0)
    gzy, gzx = np.gradient(z)
    for i in range(int(iterations)):
        step = 2 ** i
        l_center = e @ LUMA
        sigma_l = phi_color * np.sqrt(np.maximum(_box3(var), 0.0)) + 1e-4
        e_sum, v_sum, w_sum = np.zeros_like(e), np.zeros_like(var), np.zeros_like(var)
        for ky in range(-2, 3):
            for kx in range(-2, 3):
                dy, dx = ky * step, kx * step
                h_k = KERNEL[ky + 2] * KERNEL[kx + 2]
                if dy == 0 and dx == 0:
                    weight, e_q, v_q = np.full((h, w), h_k), e, var
                else:
                    n_q, z_q = _tap(normal, dy, dx), _tap(z, dy, dx)
                    e_q, v_q, a_q, c_q = _tap(e, dy, dx), _tap(var, dy, dx), _tap(a_unit, dy, dx), _tap(covered, dy, dx)
                    w_n = np.maximum(0.0, np.sum(normal * n_q, axis=-1)) ** phi_normal
                    plane = np.abs(gzx * dx) + np.abs(gzy * dy)
                    w_z = np.exp(-np.abs(z - z_q) / (phi_depth * plane + 1e-3 * np.maximum(z, 1e-3)))
                    w_l = np.exp(-np.abs(l_center - e_q @ LUMA) / sigma_l)
                    w_a = np.exp(-np.sum((a_unit - a_q) ** 2, axis=-1) / (2 * sigma_albedo ** 2))
                    weight = h_k * w_n * w_z * w_l * w_a * (covered == c_q)
                e_sum, v_sum, w_sum = e_sum + weight[..., None] * e_q, v_sum + weight * weight * v_q, w_sum + weight
        e = e_sum / w_sum[..., None]
        var = v_sum / (w_sum * w_sum)
    out = np.empty_like(color)
    out[..., :3] = e * denominator
    out[..., 3] = alpha
    out = out.astype(np.float32)
    if history is not None:
        previous, vectors = history
        out = temporal_blend(out, previous, vectors)
    if strength < 1.0:
        out = out.astype(np.float64)
        out[..., :3] = color[..., :3] * (1.0 - strength) + out[..., :3] * strength
        out = out.astype(np.float32)
    return out
