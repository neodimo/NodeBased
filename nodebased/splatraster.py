"""Deterministic NumPy EWA splat layer, in linear premultiplied colour.

Visibility uses per-pixel ray/plane depth. The legacy opaque path retains
centre-sorted splat accumulation; mesh layers use stable per-pixel event sorting.
The 400M pair budget counts clipped bounding-box pixels, not tile padding.
"""
from dataclasses import dataclass
import numpy as np

from .cancellation import Cancelled
from .splats import eval_sh, to_linear_color

SPLAT_AOV_OPACITY = 0.5

SPLAT_WORK_BUDGET = 400_000_000
_DEFAULT_BUDGET = SPLAT_WORK_BUDGET


def _perspective_jacobian(x, y, z, fx, fy, tan_fovx, tan_fovy):
    # Limit covariance projection only; screen centres retain the authored position.
    # Preserve in-range coordinates exactly (divide/multiply can round differently).
    tx = np.where(abs(x/z) > 1.3*tan_fovx,
                  np.clip(x/z, -1.3*tan_fovx, 1.3*tan_fovx)*z, x)
    ty = np.where(abs(y/z) > 1.3*tan_fovy,
                  np.clip(y/z, -1.3*tan_fovy, 1.3*tan_fovy)*z, y)
    jac = np.zeros((len(z), 2, 3))
    jac[:, 0, 0], jac[:, 0, 2] = fx/z, -fx*tx/z**2
    jac[:, 1, 1], jac[:, 1, 2] = -fy/z, fy*ty/z**2
    return jac


@dataclass(frozen=True)
class PreparedSplats:
    """Opaque full-frame preparation; owned NumPy arrays are read-only."""
    width: int
    height: int
    output: str
    cancel: object
    frame: tuple
    splats: tuple
    plane_depth: np.ndarray
    sort_order: np.ndarray
    tile_offsets: np.ndarray
    tile_indices: np.ndarray


def render_splats(instances, camera, width, height, mesh_depth=None, *,
                  cancel=None, budget=SPLAT_WORK_BUDGET, mesh_layers=None, background_rgba=None,
                  output="rgba", object_id_offset=0, hit_depth=None, rows=None):
    """Return float32 (RGB premultiplied, alpha), using stable front-to-back order.

    Instances are SplatInstance values or legacy (cloud, world matrix) pairs. Mesh depth is positive
    view depth, inf for empty pixels; only strictly nearer splat fragments contribute.
    mesh_layers is (depth HxWxM, premultiplied RGB HxWxMx3, alpha HxWxM),
    sorted front to back, with alpha=0/depth=inf unused entries. With layers,
    return the full composite over background_rgba (premultiplied HxWx4).
    Output "splats" excludes mesh colour and counts only attenuated splat alpha.
    Data outputs use mesh RGB as hit values and binary mesh coverage, selecting
    the first mesh or the splat crossing SPLAT_AOV_OPACITY; hit_depth is optional
    HxW storage for the selected view depth.
    rows=(y0, y1) selects half-open full-frame pixel rows. Projection and
    work budget use the full frame; returned arrays and all mesh/background/
    hit_depth arrays have band height y1-y0, with local row indexing.
    Exact depth ties place meshes before splats, then preserve input order.
    Cancellation is checked during preparation and between tiles/chunks.
    """
    prepared = prepare_splats(instances, camera, width, height, cancel=cancel,
                              budget=budget, output=output, object_id_offset=object_id_offset)
    return accumulate_splats(prepared, mesh_depth=mesh_depth, cancel=cancel,
                             mesh_layers=mesh_layers, background_rgba=background_rgba,
                             output=output, hit_depth=hit_depth, rows=rows)


def prepare_splats(instances, camera, width, height, *, cancel=None,
                   budget=SPLAT_WORK_BUDGET, output="rgba", object_id_offset=0, lighting=None):
    """Project, shade, budget-check and bin splats once for the full frame."""
    from .scene3d import _view_basis, DATA_OUTPUTS
    data_output = output in DATA_OUTPUTS

    def check():
        if cancel is not None and cancel.is_set():
            raise Cancelled()

    check()
    width, height = int(width), int(height)
    if width <= 0 or height <= 0:
        raise ValueError('Render dimensions must be positive')
    if budget == _DEFAULT_BUDGET:
        budget = SPLAT_WORK_BUDGET
    eye, view = _view_basis(camera)
    # Jacobian coordinates are right, up, positive forward depth (not view -Z).
    basis = view.astype(np.float64).copy()
    basis[2] *= -1
    tan_fovy = np.tan(np.deg2rad(camera.fov) / 2)
    tan_fovx = tan_fovy * (width / height)
    focal = 1 / tan_fovy
    fx, fy = focal * width / (2 * (width / height)), focal * height / 2
    batches, work = [], 0
    for instance_index, instance in enumerate(instances):
        if hasattr(instance, 'cloud'):
            cloud, matrix = instance.cloud, instance.matrix
            degree, opacity_scale, scale_scale = instance.sh_degree, instance.opacity_scale, instance.scale_scale
        else:
            cloud, matrix = instance
            degree, opacity_scale, scale_scale = None, 1.0, 1.0
        check()
        if not len(cloud):
            continue
        authored_normals = cloud.normals() if data_output else None
        cloud = cloud.transformed(matrix)
        world_normals = cloud.normals()
        if data_output:
            linear = np.asarray(matrix)[:3, :3]
            if np.linalg.det(linear) != 0:
                world_normals = authored_normals @ np.linalg.inv(linear)
                world_normals /= np.maximum(np.linalg.norm(world_normals, axis=1, keepdims=True), 1e-30)
        local = (cloud.positions.astype(np.float64) - eye) @ basis.T
        valid = (local[:, 2] > camera.near) & (local[:, 2] < camera.far)
        if not valid.any():
            continue
        x, y, z = local[valid].T
        centres = np.column_stack((width/2 + fx*x/z, height/2 - fy*y/z))
        jac = _perspective_jacobian(x, y, z, fx, fy,
                                    tan_fovx, tan_fovy)
        cov = basis @ cloud.covariance()[valid] @ basis.T
        if scale_scale != 1.0:
            cov *= scale_scale**2
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
        degree = cloud.sh_degree if degree is None else max(0, min(int(degree), cloud.sh_degree))
        colors = to_linear_color(eval_sh(cloud.sh[:, :(degree+1)**2], dirs), cloud.colorspace)[valid]
        if lighting is not None and not data_output and getattr(instance, 'relight', 0) > 0:
            from .splatshade import splat_albedo, normal_confidence, shade_splats
            lights, ambient = lighting
            colors = shade_splats(colors, splat_albedo(cloud)[valid], cloud.positions[valid],
                                  world_normals[valid], normal_confidence(cloud.scales[valid]),
                                  eye, lights, ambient, instance.relight)
        keep = np.all(hi > lo, axis=1)
        sorted_scales = np.sort(cloud.scales[valid], axis=1)
        low_confidence = sorted_scales[:, 0] > .8*sorted_scales[:, 1]
        batches.append((z[keep], centres[keep], np.linalg.inv(cov[keep]),
                        np.clip(cloud.opacity[valid][keep]*opacity_scale, 0, 1), colors[keep], lo[keep], hi[keep],
                        (cloud.normals()[valid] @ basis.T)[keep],
                        np.max(cloud.scales[valid], axis=1)[keep]*scale_scale, local[valid][keep],
                        world_normals[valid][keep], np.full(keep.sum(), object_id_offset+1+instance_index), low_confidence[keep], radius[keep]))
    if batches:
        z, centres, conic, opacity, colors, lo, hi, normals, scales, local, world_normals, object_ids, low_confidence, radii = [np.concatenate(a) for a in zip(*batches)]
    else:
        z = np.empty(0)
        centres = conic = opacity = colors = lo = hi = normals = scales = local = world_normals = object_ids = low_confidence = radii = z
    check()
    order = np.argsort(z, kind='stable')
    nx, ny = (width+15)//16, (height+15)//16
    # CSR bins: expand rectangles in stable centre-depth order, then stable
    # group by tile. No Python object is allocated per splat or tile entry.
    if len(order):
        tile_lo = lo[order]//16
        tile_size = (hi[order]-1)//16 + 1 - tile_lo
        counts = np.prod(tile_size, axis=1)
        starts = np.cumsum(counts)-counts
        owner = np.repeat(np.arange(len(order)), counts)
        offset = np.arange(len(owner))-np.repeat(starts, counts)
        tiles = ((tile_lo[owner, 1]+offset//tile_size[owner, 0])*nx
                 + tile_lo[owner, 0]+offset%tile_size[owner, 0])
        check()
        grouped = np.argsort(tiles, kind='stable')
        tile_indices = order[owner[grouped]]
        tile_offsets = np.concatenate(([0], np.cumsum(np.bincount(tiles, minlength=nx*ny))))
    else:
        tile_indices = np.empty(0, np.int64)
        tile_offsets = np.zeros(nx*ny+1, np.int64)
    plane_depth = np.einsum('ij,ij->i', normals, local) if len(z) else np.empty(0)
    frame = (eye, basis, fx, fy)
    arrays = (z, centres, conic, opacity, colors, lo, hi, normals, scales, local, world_normals, object_ids, low_confidence, radii)
    for array in (*arrays, eye, basis, plane_depth, order, tile_offsets, tile_indices):
        array.flags.writeable = False
    check()
    return PreparedSplats(width, height, output, cancel, frame, arrays,
                          plane_depth, order, tile_offsets, tile_indices)


def accumulate_splats(prepared, rows=None, *, mesh_depth=None, cancel=None,
                      mesh_layers=None, background_rgba=None, output=None, hit_depth=None):
    """Accumulate only tiles intersecting rows, with band-local mesh buffers.

    Preparation must use the same output pass; cancellation defaults to the
    preparation's token and is checked at entry and between tiles/chunks.
    """
    from .scene3d import DATA_OUTPUTS
    output = prepared.output if output is None else output
    if output != prepared.output:
        raise ValueError('output must match the prepared output pass')
    cancel = prepared.cancel if cancel is None else cancel
    def check():
        if cancel is not None and cancel.is_set():
            raise Cancelled()
    check()
    width, height = prepared.width, prepared.height
    data_output = output in DATA_OUTPUTS
    y0, y1 = (0, height) if rows is None else rows
    if not (0 <= y0 < y1 <= height):
        raise ValueError("rows must satisfy 0 <= y0 < y1 <= height")
    band_height = y1-y0
    if mesh_depth is not None and np.shape(mesh_depth) != (band_height, width):
        raise ValueError('mesh_depth must have shape (band_height, width)')
    if mesh_layers is not None:
        md, mc, ma = map(np.asarray, mesh_layers)
        if md.ndim != 3 or md.shape[:2] != (band_height, width) or ma.shape != md.shape or mc.shape != (*md.shape, 3):
            raise ValueError('mesh_layers must have shapes (band_height,W,M), (band_height,W,M,3), (band_height,W,M)')
        if mesh_depth is not None:
            raise ValueError('mesh_layers and mesh_depth are mutually exclusive')
    if data_output and mesh_layers is None:
        # Reuse first-hit event selection with no mesh events: the caller keeps
        # its raster attributes, and mesh_depth clips all splat contributions.
        md = np.empty((band_height, width, 0), np.float32)
        mc = np.empty((band_height, width, 0, 3), np.float32)
        ma = np.empty_like(md)
    if background_rgba is not None and np.shape(background_rgba) != (band_height, width, 4):
        raise ValueError('background_rgba must have shape (band_height,W,4)')
    if hit_depth is not None and np.shape(hit_depth) != (band_height, width):
        raise ValueError("hit_depth must have shape (band_height, width)")
    eye, basis, fx, fy = prepared.frame
    z, centres, conic, opacity, colors, lo, hi, normals, scales, local, world_normals, object_ids, low_confidence, radii = prepared.splats
    plane_depth = prepared.plane_depth
    nx = (width+15)//16
    rgb = np.zeros((band_height, width, 3), np.float32)
    alpha = np.zeros((band_height, width), np.float32)
    def fragments(pixels, ix):
        d = pixels[:, None, :] - centres[ix]
        q = np.einsum('pki,kij,pkj->pk', d, conic[ix], d)
        a = np.minimum(.99, opacity[ix]*np.exp(-.5*q))
        inside = np.all((pixels[:, None, :] >= lo[ix]) & (pixels[:, None, :] < hi[ix]), axis=2)
        a[(a < 1/255) | ~inside] = 0
        rays = np.column_stack(((pixels[:, 0]-width/2)/fx,
                                (height/2-pixels[:, 1])/fy, np.ones(len(pixels))))
        den = rays @ normals[ix].T
        numerator = plane_depth[ix]
        zp = np.divide(numerator[None, :], den, out=np.broadcast_to(z[ix], den.shape).copy(), where=den != 0)
        # Grazing planes (unit-ray dot < .05) and intersections more than
        # three world max-scales from centre depth fall back to centre depth.
        # Near-isotropic scales (s_min/s_mid > .8) have arbitrary normals;
        # use centre depth in beauty and data passes for these too.
        fallback = low_confidence[ix][None, :] | (abs(den)/np.linalg.norm(rays, axis=1)[:, None] < .05) | (abs(zp-z[ix]) > 3*scales[ix])
        zp[fallback] = np.broadcast_to(z[ix], zp.shape)[fallback]
        return a, zp
    for tile in range((y0//16)*nx, ((y1+15)//16)*nx):
        ids = prepared.tile_indices[prepared.tile_offsets[tile]:prepared.tile_offsets[tile+1]]
        check()
        if not len(ids) and mesh_layers is None:
            continue
        tx, ty = tile % nx * 16, (tile // nx) * 16
        yy, xx = np.mgrid[max(ty, y0):min(ty+16, y1), tx:min(tx+16, width)]
        pixels = np.column_stack((xx.ravel()+.5, yy.ravel()+.5))
        yy = yy-y0
        if mesh_layers is not None or data_output:
            # Bound scratch to ~16k events, except a single pixel's event list.
            block = max(1, 16384 // max(1, len(ids)+md.shape[2]))
            # Restore authored order before stable per-pixel sorting (plane
            # depths can tie even when the centre depths differ).
            ix = np.sort(ids)
            for start in range(0, len(pixels), block):
                check()
                stop = min(start+block, len(pixels))
                py, px = yy.ravel()[start:stop], xx.ravel()[start:stop]
                if len(ix):
                    a, zp = fragments(pixels[start:stop], ix)
                    if mesh_depth is not None:
                        a[zp >= mesh_depth[py, px, None]] = 0
                    depths = np.concatenate((md[py, px], zp), axis=1)
                    alphas = np.concatenate((ma[py, px], a), axis=1)
                    sources = np.concatenate((mc[py, px], a[..., None]*colors[ix]), axis=1)
                else:
                    depths, alphas, sources = md[py, px], ma[py, px], mc[py, px]
                if data_output:
                    # Only the first covered mesh is needed: it terminates the search.
                    order_pixel = np.argsort(depths, axis=1, kind='stable')
                    sorted_alpha = np.take_along_axis(alphas, order_pixel, axis=1)
                    is_mesh = order_pixel < md.shape[2]
                    splat_alpha = np.where(is_mesh, 0, sorted_alpha)
                    opacity_hit = 1-np.cumprod(1-splat_alpha, axis=1) >= SPLAT_AOV_OPACITY
                    hit = (is_mesh & (sorted_alpha > 0)) | opacity_hit
                    covered = hit.any(axis=1)
                    rows = np.flatnonzero(covered)
                    if len(rows):
                        event = order_pixel[rows, hit[rows].argmax(axis=1)]
                        values = sources[rows, event].copy()
                        splat_hit = event >= md.shape[2]
                        rr = rows[splat_hit]
                        if len(rr):
                            splat = ix[event[splat_hit]-md.shape[2]]
                            zz = depths[rr, event[splat_hit]]
                            pp = pixels[start:stop][rr]
                            rays = np.column_stack(((pp[:, 0]-width/2)/fx,
                                                    (height/2-pp[:, 1])/fy, np.ones(len(rr))))
                            world_rays = rays @ np.linalg.inv(basis).T
                            if output == 'depth':
                                value = np.repeat(zz[:, None], 3, axis=1)
                            elif output == 'position':
                                value = eye + world_rays*zz[:, None]
                            elif output == 'normals':
                                value = world_normals[splat].copy()
                                value[np.einsum('ij,ij->i', value, world_rays) > 0] *= -1
                            elif output == 'object_id':
                                value = np.column_stack((object_ids[splat], np.zeros((len(rr), 2))))
                            else:
                                value = np.zeros((len(rr), 3))
                            values[splat_hit] = value
                        rgb[py[rows], px[rows]] = values
                        alpha[py[rows], px[rows]] = 1
                        if hit_depth is not None:
                            hit_depth[py[rows], px[rows]] = depths[rows, event]
                    continue
                order_pixel = np.argsort(depths, axis=1, kind='stable')
                alphas = np.take_along_axis(alphas, order_pixel, axis=1)
                sources = np.take_along_axis(sources, order_pixel[..., None], axis=1)
                before = np.concatenate((np.ones((stop-start, 1)), np.cumprod(1-alphas, axis=1)), axis=1)
                if output == 'splats':
                    splat_events = order_pixel >= md.shape[2]
                    sources = np.where(splat_events[..., None], sources, 0)
                result = np.sum(before[:, :-1, None]*sources, axis=1)
                remaining = before[:, -1]
                coverage = (np.sum(before[:, :-1]*alphas*splat_events, axis=1)
                            if output == "splats" else 1-remaining)
                if background_rgba is not None:
                    result += remaining[:, None]*background_rgba[py, px, :3]
                    coverage += remaining*background_rgba[py, px, 3]
                rgb[py, px], alpha[py, px] = result, coverage
            continue
        trans = np.ones(len(pixels))
        color = np.zeros((len(pixels), 3))
        mesh = mesh_depth[yy, xx].ravel() if mesh_depth is not None else None
        for start in range(0, len(ids), 64):
            check()
            if np.all(trans < 1e-4):
                break
            ix = np.asarray(ids[start:start+64])
            a, zp = fragments(pixels, ix)
            if mesh is not None:
                a[zp >= mesh[:, None]] = 0
            before = np.concatenate((np.ones((len(pixels), 1)), np.cumprod(1-a[:, :-1], axis=1)), axis=1)*trans[:, None]
            a[before < 1e-4] = 0
            weights = before*a
            color += weights @ colors[ix]
            trans *= np.prod(1-a, axis=1)
        rgb[yy, xx] = color.reshape(*xx.shape, 3)
        alpha[yy, xx] = (1-trans).reshape(xx.shape)
    return rgb, alpha
