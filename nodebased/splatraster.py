"""Deterministic NumPy EWA splat layer, in linear premultiplied colour.

Visibility uses centre depth against opaque meshes. Transparent meshes are not
sorted against splats. Three-sigma square bounds truncate each projected Gaussian.
The 400M pair budget counts clipped bounding-box pixels, not tile padding;
working compositing arrays are bounded to 256 pixels by 64 splats.
"""
import numpy as np

from .cancellation import Cancelled
from .splats import eval_sh, to_linear_color

SPLAT_WORK_BUDGET = 400_000_000
_DEFAULT_BUDGET = SPLAT_WORK_BUDGET


def render_splats(instances, camera, width, height, mesh_depth=None, *,
                  cancel=None, budget=SPLAT_WORK_BUDGET):
    """Return float32 (RGB premultiplied, alpha), using stable front-to-back order.

    Instances are (cloud, column-vector world matrix) pairs. Mesh depth is positive
    view depth, inf for empty pixels; only strictly nearer splat centres contribute.
    Cancellation is checked during preparation and between tiles/chunks.
    """
    from .scene3d import _view_basis

    def check():
        if cancel is not None and cancel.is_set():
            raise Cancelled()

    check()
    width, height = int(width), int(height)
    if width <= 0 or height <= 0:
        raise ValueError('Render dimensions must be positive')
    if mesh_depth is not None and np.shape(mesh_depth) != (height, width):
        raise ValueError('mesh_depth must have shape (height, width)')
    if budget == _DEFAULT_BUDGET:
        budget = SPLAT_WORK_BUDGET
    eye, view = _view_basis(camera)
    # Jacobian coordinates are right, up, positive forward depth (not view -Z).
    basis = view.astype(np.float64).copy()
    basis[2] *= -1
    focal = 1 / np.tan(np.deg2rad(camera.fov) / 2)
    fx, fy = focal * width / (2 * (width / height)), focal * height / 2
    batches, work = [], 0
    for cloud, matrix in instances:
        check()
        if not len(cloud):
            continue
        cloud = cloud.transformed(matrix)
        local = (cloud.positions.astype(np.float64) - eye) @ basis.T
        valid = (local[:, 2] > camera.near) & (local[:, 2] < camera.far)
        if not valid.any():
            continue
        x, y, z = local[valid].T
        centres = np.column_stack((width/2 + fx*x/z, height/2 - fy*y/z))
        jac = np.zeros((len(z), 2, 3))
        jac[:, 0, 0], jac[:, 0, 2] = fx/z, -fx*x/z**2
        jac[:, 1, 1], jac[:, 1, 2] = -fy/z, fy*y/z**2
        cov = basis @ cloud.covariance()[valid] @ basis.T
        cov = jac @ cov @ jac.transpose(0, 2, 1) + .3*np.eye(2)
        radius = 3*np.sqrt(np.linalg.eigvalsh(cov)[:, 1])
        # Half-open integer boxes; clip in float before converting to avoid overflow.
        lo = np.floor(np.clip(centres-radius[:, None], 0, (width, height))).astype(np.int64)
        hi = np.ceil(np.clip(centres+radius[:, None], 0, (width, height))).astype(np.int64)
        work += int(np.sum(np.prod(hi-lo, axis=1)))
        if work > budget:
            raise ValueError(f'Splat render exceeds the CPU reference budget: {work:,} splat-pixel pairs > {budget:,}')
        dirs = cloud.positions.astype(np.float64)-eye
        dirs /= np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-30)
        colors = to_linear_color(eval_sh(cloud, dirs), cloud.colorspace)[valid]
        keep = np.all(hi > lo, axis=1)
        batches.append((z[keep], centres[keep], np.linalg.inv(cov[keep]),
                        cloud.opacity[valid][keep], colors[keep], lo[keep], hi[keep]))
    rgb = np.zeros((height, width, 3), np.float32)
    alpha = np.zeros((height, width), np.float32)
    if not batches:
        return rgb, alpha
    z, centres, conic, opacity, colors, lo, hi = [np.concatenate(a) for a in zip(*batches)]
    order = np.argsort(z, kind='stable')
    nx, ny = (width+15)//16, (height+15)//16
    bins = [[] for _ in range(nx*ny)]
    for count, i in enumerate(order):
        if count % 4096 == 0:
            check()
        for ty in range(lo[i, 1]//16, (hi[i, 1]-1)//16+1):
            for tx in range(lo[i, 0]//16, (hi[i, 0]-1)//16+1):
                bins[ty*nx+tx].append(i)
    for tile, ids in enumerate(bins):
        check()
        if not ids:
            continue
        tx, ty = tile % nx * 16, tile // nx * 16
        yy, xx = np.mgrid[ty:min(ty+16, height), tx:min(tx+16, width)]
        pixels = np.column_stack((xx.ravel()+.5, yy.ravel()+.5))
        trans = np.ones(len(pixels))
        color = np.zeros((len(pixels), 3))
        mesh = mesh_depth[yy, xx].ravel() if mesh_depth is not None else None
        for start in range(0, len(ids), 64):
            check()
            if np.all(trans < 1e-4):
                break
            ix = np.asarray(ids[start:start+64])
            d = pixels[:, None, :] - centres[ix]
            q = np.einsum('pki,kij,pkj->pk', d, conic[ix], d)
            a = np.minimum(.99, opacity[ix]*np.exp(-.5*q))
            inside = np.all((pixels[:, None, :] >= lo[ix]) & (pixels[:, None, :] < hi[ix]), axis=2)
            a[(a < 1/255) | ~inside] = 0
            if mesh is not None:
                a[z[ix][None, :] >= mesh[:, None]] = 0
            before = np.concatenate((np.ones((len(pixels), 1)), np.cumprod(1-a[:, :-1], axis=1)), axis=1)*trans[:, None]
            a[before < 1e-4] = 0
            weights = before*a
            color += weights @ colors[ix]
            trans *= np.prod(1-a, axis=1)
        rgb[yy, xx] = color.reshape(*xx.shape, 3)
        alpha[yy, xx] = (1-trans).reshape(xx.shape)
    return rgb, alpha
