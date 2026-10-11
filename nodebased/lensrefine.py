"""Focal length and lens distortion from a monocular camera solve's own tracked points.

A learned SLAM (providers/camera_solve_dpvo.py) is given a focal length and assumes a pinhole; it never
estimates either. It does return what it tracked: a sparse set of points and, for each, where the network
found it in the other frames. This module refines that result as a bundle adjustment. The unknowns are the
focal length (optionally), the radial terms `k1` (and `k2`), every solved keyframe pose and every tracked
point; the data are the tracked pixel positions. It is a sparse Levenberg-Marquardt written with numpy (the
points are eliminated by a Schur complement, so the system solved is the size of the poses), the same
damped reprojection least squares as `conditioning_verify._fit_camera` taken to many cameras at once.

The lens is `lensmodel.py`'s: film-back units about the format centre (1.0 is half the format height) and the
Brown-Conrady radial terms, so a recovered `k1` can be typed straight into `Camera3D.lens_k1`. The principal
point stays at the image centre, as the solver assumes; tangential terms and `k3` are not estimated.

The refinement is deterministic given the solve it starts from (no random choices: points and frames are picked
by count and spacing). It inherits whatever the solver's run-to-run differences are.

Observation arrays (all of one length):
    frame   index into the solved poses (video frame number minus the first solved frame)
    point   index into the solve's point array
    xy      pixel position of the point in that frame, top-left origin, on the plate as it is (distorted)
    weight  confidence in [0, 1], the solver's own weight for a tracked position, 1 for the position a point was born at
"""
from __future__ import annotations

import math

import numpy as np

DISTORTIONS = ("none", "k1", "k1 k2")
MIN_VIEWS = 3              # a point seen on fewer frames says little about the lens
MIN_WEIGHT = 0.05          # weights below this are the solver giving up on a position
MAX_FRAMES = 120           # frames used by the refinement (evenly spaced over the solve)
MAX_POINTS = 3000          # tracked points used (those seen the most)
HUBER_PX = 1.5             # residual at which the loss turns from squared to linear, at 960 px wide
GAUGE = 1e12               # stiffness that pins the first pose


def distortion_terms(distortion):
    if distortion not in DISTORTIONS:
        raise ValueError(f"distortion must be one of {', '.join(DISTORTIONS)} (got {distortion!r})")
    return {"none": 0, "k1": 1, "k1 k2": 2}[distortion]


# ----------------------------------------------------------------------------- geometry

def _exp_rotation(w):
    """Rotation matrices of rotation vectors `w` (N, 3)."""
    w = np.asarray(w, np.float64).reshape(-1, 3)
    angle = np.linalg.norm(w, axis=1)
    small = angle < 1e-12
    safe = np.where(small, 1.0, angle)
    axis = w / safe[:, None]
    k = np.zeros((len(w), 3, 3))
    k[:, 0, 1], k[:, 0, 2], k[:, 1, 0] = -axis[:, 2], axis[:, 1], axis[:, 2]
    k[:, 1, 2], k[:, 2, 0], k[:, 2, 1] = -axis[:, 0], -axis[:, 1], axis[:, 0]
    s, c = np.sin(angle)[:, None, None], np.cos(angle)[:, None, None]
    rotation = np.eye(3) + s * k + (1.0 - c) * (k @ k)
    rotation[small] = np.eye(3) + _skew(w[small])
    return rotation


def _skew(v):
    v = np.asarray(v, np.float64).reshape(-1, 3)
    out = np.zeros((len(v), 3, 3))
    out[:, 0, 1], out[:, 0, 2], out[:, 1, 0] = -v[:, 2], v[:, 1], v[:, 2]
    out[:, 1, 2], out[:, 2, 0], out[:, 2, 1] = -v[:, 0], -v[:, 1], v[:, 0]
    return out


def world_to_camera(c2w):
    """(R, t) with X_camera = R X_world + t of camera-to-world poses (N, 4, 4)."""
    c2w = np.asarray(c2w, np.float64)
    rotation = np.transpose(c2w[:, :3, :3], (0, 2, 1))
    return rotation, -np.einsum("nij,nj->ni", rotation, c2w[:, :3, 3])


def camera_to_world(rotation, translation):
    out = np.tile(np.eye(4), (len(rotation), 1, 1))
    out[:, :3, :3] = np.transpose(rotation, (0, 2, 1))
    out[:, :3, 3] = -np.einsum("nij,nj->ni", out[:, :3, :3], translation)
    return out


def camera_pixels(camera_points, focal, k1, k2, width, height):
    """Pixel positions on the plate of points in camera space (OpenCV axes), with the radial lens.

    The same function as `lensmodel.distort_pixels` of the ideal pixel `f x/z + c`: the radius is in
    film-back units (half the height is 1.0)."""
    z = camera_points[:, 2]
    z = np.where(np.abs(z) < 1e-9, 1e-9, z)
    xn, yn = camera_points[:, 0] / z, camera_points[:, 1] / z
    half = height / 2.0
    r2 = (focal * xn / half) ** 2 + (focal * yn / half) ** 2
    radial = 1.0 + r2 * (k1 + r2 * k2)
    return np.stack((width / 2.0 + focal * xn * radial, height / 2.0 + focal * yn * radial), axis=1)


def _project(rotation, translation, frame, points, point, intrinsics, width, height):
    camera = np.einsum("nij,nj->ni", rotation[frame], points[point]) + translation[frame]
    return camera_pixels(camera, *intrinsics, width, height), camera


# ----------------------------------------------------------------------------- selection

def select_observations(frame, point, xy, weight, point_count, frame_count, *, max_frames=MAX_FRAMES,
                        max_points=MAX_POINTS):
    """Choose what the refinement uses; returns indices into the observation arrays, the frames kept (sorted)
    and the points kept (sorted). Deterministic."""
    frame, point = np.asarray(frame, np.int64), np.asarray(point, np.int64)
    xy, weight = np.asarray(xy, np.float64), np.asarray(weight, np.float64)
    keep = ((point >= 0) & (point < point_count) & (frame >= 0) & (frame < frame_count)
            & np.isfinite(xy).all(axis=1) & np.isfinite(weight) & (weight >= MIN_WEIGHT))
    frames = np.unique(frame[keep])
    if len(frames) > max_frames:
        frames = frames[np.unique(np.round(np.linspace(0, len(frames) - 1, max_frames)).astype(int))]
    keep &= np.isin(frame, frames)
    # a point seen twice on one frame (two edges) still counts as one view
    pairs = np.unique(np.stack((point[keep], frame[keep]), axis=1), axis=0)
    views = np.bincount(pairs[:, 0], minlength=point_count) if len(pairs) else np.zeros(point_count, int)
    candidates = np.nonzero(views >= MIN_VIEWS)[0]
    if len(candidates) > max_points:
        order = np.lexsort((candidates, -views[candidates]))      # most views first, ties by index
        candidates = np.sort(candidates[order[:max_points]])
    keep &= np.isin(point, candidates)
    return np.nonzero(keep)[0], frames, candidates


# ----------------------------------------------------------------------------- the adjustment

class _Linear:
    """One linearisation: the normal equations of poses and lens with the points still in (to be eliminated)."""


def _intrinsics(q, base, free):
    """(focal, k1, k2) from the free parameters q = [log f?, k1?, k2?] on top of the base values."""
    focal, k1, k2 = base
    index = 0
    if free[0]:
        focal = base[0] * math.exp(q[index]); index += 1
    if free[1]:
        k1 = base[1] + q[index]; index += 1
    if free[2]:
        k2 = base[2] + q[index]
    return focal, k1, k2


def _robust(residual, weight, huber):
    """IRLS weights and the Huber cost of residual pairs (N, 2)."""
    norm = np.maximum(np.linalg.norm(residual, axis=1), 1e-12)
    irls = weight * np.minimum(1.0, huber / norm)
    loss = np.where(norm <= huber, 0.5 * norm * norm, huber * (norm - 0.5 * huber))
    return irls, float(np.sum(weight * loss))


class _Problem:
    """Observations, the lens parametrisation and the arithmetic of one adjustment."""

    def __init__(self, frames, points, obs_frame, obs_point, obs_xy, obs_weight, width, height, base, free, huber, fixed):
        self.frames, self.points = frames, points
        self.f, self.p, self.xy, self.w = obs_frame, obs_point, obs_xy, obs_weight
        self.width, self.height, self.base, self.free = width, height, base, free
        self.huber, self.fixed = huber, fixed
        self.m = int(sum(bool(v) for v in free))
        self.order = np.argsort(obs_point, kind="stable")
        self.starts = np.searchsorted(obs_point[self.order], np.arange(points + 1))

    def residual(self, rotation, translation, points, q):
        camera = np.einsum("nij,nj->ni", rotation[self.f], points[self.p]) + translation[self.f]
        out = camera_pixels(camera, *_intrinsics(q, self.base, self.free), self.width, self.height) - self.xy
        out[camera[:, 2] <= 1e-6] = 1e4           # behind the camera: far outside every loss
        return out

    def linearise(self, rotation, translation, points, q):
        n, m, frames = len(self.f), self.m, self.frames
        camera = np.einsum("nij,nj->ni", rotation[self.f], points[self.p]) + translation[self.f]
        intr = _intrinsics(q, self.base, self.free)
        pixels = camera_pixels(camera, *intr, self.width, self.height)
        d_camera = np.empty((n, 2, 3))
        for axis in range(3):
            moved = camera.copy()
            step = 1e-6 * np.maximum(1.0, np.abs(camera[:, axis]))
            moved[:, axis] += step
            d_camera[:, :, axis] = (camera_pixels(moved, *intr, self.width, self.height) - pixels) / step[:, None]
        d_lens = np.zeros((n, 2, m))
        for column in range(m):
            nudged = q.copy()
            nudged[column] += 1e-6
            d_lens[:, :, column] = (camera_pixels(camera, *_intrinsics(nudged, self.base, self.free), self.width,
                                                  self.height) - pixels) / 1e-6
        j_pose = np.concatenate((-(d_camera @ _skew(camera - translation[self.f])), d_camera), axis=2)   # (n, 2, 6)
        j_point = d_camera @ rotation[self.f]
        residual = pixels - self.xy
        residual[camera[:, 2] <= 1e-6] = 1e4
        irls, cost = _robust(residual, self.w, self.huber)
        lin = _Linear()
        lin.j_pose, lin.j_point, lin.j_lens, lin.residual, lin.irls, lin.cost = j_pose, j_point, d_lens, residual, irls, cost
        w = irls[:, None, None]
        size = 6 * frames + m
        h = np.zeros((size, size))
        g = np.zeros(size)
        jt_pose = np.transpose(j_pose, (0, 2, 1))
        weighted = irls[:, None] * residual
        blocks = np.zeros((frames, 6, 6))
        np.add.at(blocks, self.f, jt_pose @ (w * j_pose))
        for k in range(frames):
            h[6 * k:6 * k + 6, 6 * k:6 * k + 6] = blocks[k]
        g_pose = np.zeros((frames, 6))
        np.add.at(g_pose, self.f, np.einsum("nki,ni->nk", jt_pose, weighted))
        g[:6 * frames] = g_pose.reshape(-1)
        a_lens = np.zeros((n, 0, 3))
        if m:
            jt_lens = np.transpose(d_lens, (0, 2, 1))
            h[6 * frames:, 6 * frames:] = np.sum(jt_lens @ (w * d_lens), axis=0)
            g[6 * frames:] = np.einsum("nki,ni->k", jt_lens, weighted)
            cross = np.zeros((frames, 6, m))
            np.add.at(cross, self.f, jt_pose @ (w * d_lens))
            for k in range(frames):
                h[6 * k:6 * k + 6, 6 * frames:] = cross[k]
                h[6 * frames:, 6 * k:6 * k + 6] = cross[k].T
            a_lens = jt_lens @ (w * j_point)
        jt_point = np.transpose(j_point, (0, 2, 1))
        lin.v = np.zeros((self.points, 3, 3))
        np.add.at(lin.v, self.p, jt_point @ (w * j_point))
        lin.g_point = np.zeros((self.points, 3))
        np.add.at(lin.g_point, self.p, np.einsum("nki,ni->nk", jt_point, weighted))
        lin.a_pose = jt_pose @ (w * j_point)                                 # (n, 6, 3)
        lin.a_lens = a_lens
        k = self.fixed
        h[6 * k:6 * k + 6, 6 * k:6 * k + 6] += GAUGE * np.eye(6)
        lin.h, lin.g = h, g
        return lin

    def reduced(self, lin, lam):
        """The Schur complement over the points: (S, rhs, inverse point blocks) at damping `lam`."""
        frames, m = self.frames, self.m
        damped = lin.h + lam * np.diag(np.maximum(np.diag(lin.h), 1e-9))
        diag = np.maximum(np.einsum("pii->pi", lin.v), 1e-9)
        v_inv = np.linalg.inv(lin.v + lam * diag[:, :, None] * np.eye(3))
        rhs = -lin.g.copy()
        reduced = damped
        for p in range(self.points):
            ids = self.order[self.starts[p]:self.starts[p + 1]]
            if not len(ids):
                continue
            cams, inverse = np.unique(self.f[ids], return_inverse=True)
            stacked = np.zeros((len(cams), 6, 3))
            np.add.at(stacked, inverse, lin.a_pose[ids])
            rows = stacked.reshape(-1, 3)
            index = (6 * cams[:, None] + np.arange(6)[None, :]).reshape(-1)
            if m:
                rows = np.concatenate((rows, lin.a_lens[ids].sum(axis=0)))
                index = np.concatenate((index, 6 * frames + np.arange(m)))
            wv = rows @ v_inv[p]
            reduced[np.ix_(index, index)] -= wv @ rows.T
            rhs[index] += wv @ lin.g_point[p]
        return reduced, rhs, v_inv

    def step(self, lin, lam):
        """One damped Gauss-Newton step: (pose/lens delta, point delta)."""
        frames = self.frames
        reduced, rhs, v_inv = self.reduced(lin, lam)
        delta = np.linalg.solve(reduced, rhs)
        poses = delta[:6 * frames].reshape(frames, 6)
        moved = np.einsum("nij,nj->ni", lin.j_pose, poses[self.f])
        if self.m:
            moved = moved + np.einsum("nij,j->ni", lin.j_lens, delta[6 * frames:])
        pull = np.zeros_like(lin.g_point)
        np.add.at(pull, self.p, np.einsum("nki,ni->nk", np.transpose(lin.j_point, (0, 2, 1)), lin.irls[:, None] * moved))
        return delta, -np.einsum("pij,pj->pi", v_inv, lin.g_point + pull)


def bundle_adjust(rotation, translation, points, obs_frame, obs_point, obs_xy, obs_weight, width, height, base,
                  free=(False, False, False), *, iterations=40, huber=None, fixed_frame=0):
    """Levenberg-Marquardt over the poses, the points and the free lens parameters.

    `rotation`, `translation`: world-to-camera pose per frame (N, 3, 3) and (N, 3); `points` (P, 3); `base` is
    (focal px, k1, k2), the lens the solve was made with; `free` says which of focal (a log), k1 and k2 are
    unknowns. The frame `fixed_frame` keeps its pose. Returns the refined state, the lens, the cost history and
    the formal 1-sigma of the free lens parameters (a lower bound: the tracked positions are not independent)."""
    rotation, translation = np.array(rotation, np.float64), np.array(translation, np.float64)
    points = np.array(points, np.float64)
    huber = (HUBER_PX * width / 960.0) if huber is None else float(huber)
    problem = _Problem(len(rotation), len(points), np.asarray(obs_frame, np.int64), np.asarray(obs_point, np.int64),
                       np.asarray(obs_xy, np.float64), np.asarray(obs_weight, np.float64), width, height, base,
                       tuple(bool(v) for v in free), huber, fixed_frame)
    q = np.zeros(problem.m)
    lam, accepted, history = 1e-3, 0, []
    lin = problem.linearise(rotation, translation, points, q)
    history.append(lin.cost)
    for _ in range(iterations):
        improved = False
        for _attempt in range(12):
            delta, delta_point = problem.step(lin, lam)
            poses = delta[:6 * problem.frames].reshape(problem.frames, 6)
            trial = (_exp_rotation(poses[:, :3]) @ rotation, translation + poses[:, 3:], points + delta_point,
                     q + delta[6 * problem.frames:])
            _, trial_cost = _robust(problem.residual(*trial), problem.w, huber)
            if trial_cost < lin.cost:
                gain = (lin.cost - trial_cost) / max(lin.cost, 1e-12)
                rotation, translation, points, q = trial
                lam = max(lam / 3.0, 1e-9)
                accepted += 1
                improved = True
                break
            lam *= 5.0
            if lam > 1e9:
                break
        if not improved:
            break
        lin = problem.linearise(rotation, translation, points, q)
        history.append(lin.cost)
        if gain < 1e-9:
            break
    sigma = None
    if problem.m:
        reduced, _rhs, _inv = problem.reduced(lin, 1e-9)
        count = len(problem.f)
        rms = float(np.sqrt(np.mean(np.sum(lin.residual ** 2, axis=1) / 2.0)))
        sigma = _formal_sigma(reduced, problem.m, rms, count, 6 * (problem.frames - 1) + 3 * problem.points + problem.m)
    return {"rotation": rotation, "translation": translation, "points": points, "q": q,
            "intrinsics": _intrinsics(q, base, problem.free), "cost": history, "iterations": accepted, "sigma": sigma,
            "residual": lin.residual}


def _formal_sigma(reduced, m, rms, count, unknowns):
    dof = 2 * count - unknowns
    if dof <= 0 or m == 0:
        return None
    try:
        inverse = np.linalg.pinv(reduced, rcond=1e-13)
    except np.linalg.LinAlgError:
        return None
    variance = rms * rms * 2 * count / dof
    return np.sqrt(np.maximum(np.diag(inverse)[-m:], 0.0) * variance)


# ----------------------------------------------------------------------------- the whole estimate

def estimate_lens(c2w, points, tracks, width, height, focal_px, *, estimate_focal=True, distortion="none",
                  max_frames=MAX_FRAMES, max_points=MAX_POINTS, iterations=40):
    """Refine a monocular solve's poses and points, and estimate the lens it was made without.

    `c2w` (N, 4, 4) and `points` (M, 3) are the solve in its own units; `tracks` holds `frame`, `point`, `xy` and
    `weight` (see the module text). `focal_px` is the focal length the solve assumed, in pixels of a `height`
    pixel tall plate; with `estimate_focal` False it is held, and `distortion` ("none", "k1", "k1 k2") picks the
    radial terms to estimate. Returns a dict: `focal_px`, `k1`, `k2`, their 1-sigma, `rms_px_before` and
    `rms_px_after` (root mean square of the pixel distance between a tracked position and its reprojection,
    inliers of the final fit), the counts, `poses` (c2w for every solved frame), `points` and `point_index`
    (the refined points and which of the solve's points they are), or None when there is nothing to estimate."""
    terms = distortion_terms(distortion)
    if not estimate_focal and not terms:
        return None
    c2w = np.asarray(c2w, np.float64)
    points = np.asarray(points, np.float64)
    keep, frames, chosen = select_observations(tracks["frame"], tracks["point"], tracks["xy"], tracks["weight"],
                                               len(points), len(c2w), max_frames=max_frames, max_points=max_points)
    if len(frames) < 3 or len(chosen) < 12:
        raise ValueError(f"lens estimation needs tracked points on three frames or more "
                         f"(found {len(chosen)} points on {len(frames)} frames)")
    rotation_all, translation_all = world_to_camera(c2w)
    frame_index = {int(f): i for i, f in enumerate(frames)}
    point_index = {int(p): i for i, p in enumerate(chosen)}
    obs_frame = np.array([frame_index[int(f)] for f in np.asarray(tracks["frame"])[keep]])
    obs_point = np.array([point_index[int(p)] for p in np.asarray(tracks["point"])[keep]])
    obs_xy = np.asarray(tracks["xy"], np.float64)[keep]
    obs_weight = np.asarray(tracks["weight"], np.float64)[keep]
    rotation, translation, cloud = rotation_all[frames], translation_all[frames], points[chosen]
    # points behind their own camera cannot be repaired by a step: drop them up front
    camera = np.einsum("nij,nj->ni", rotation[obs_frame], cloud[obs_point]) + translation[obs_frame]
    behind = np.bincount(obs_point[camera[:, 2] <= 1e-6], minlength=len(cloud)) > 0
    if behind.any():
        alive = ~behind[obs_point]
        remap = np.cumsum(~behind) - 1
        obs_frame, obs_xy, obs_weight = obs_frame[alive], obs_xy[alive], obs_weight[alive]
        obs_point = remap[obs_point[alive]]
        cloud, chosen = cloud[~behind], chosen[~behind]
    base = (float(focal_px), 0.0, 0.0)
    free = (bool(estimate_focal), terms >= 1, terms >= 2)
    residual_before = _Problem(len(rotation), len(cloud), obs_frame, obs_point, obs_xy, obs_weight, width, height, base,
                               free, 1.0, 0).residual(rotation, translation, cloud, np.zeros(sum(free)))
    before = _rms(residual_before, width)
    fit = _staged_adjust(rotation, translation, cloud, obs_frame, obs_point, obs_xy, obs_weight, width, height, base, free,
                         iterations)
    focal, k1, k2 = fit["intrinsics"]
    sigma = dict(zip([n for n, on in zip(("focal", "k1", "k2"), free) if on], fit["sigma"] if fit["sigma"] is not None else []))
    out = {"focal_px": focal, "focal_px_start": float(focal_px), "k1": k1, "k2": k2,
           "focal_px_sigma": focal * sigma["focal"] if "focal" in sigma else None,
           "k1_sigma": sigma.get("k1"), "k2_sigma": sigma.get("k2"),
           "rms_px_before": before, "rms_px_after": _rms(fit["residual"], width),
           "median_px_before": float(np.median(np.linalg.norm(residual_before, axis=1))),
           "median_px_after": float(np.median(np.linalg.norm(fit["residual"], axis=1))),
           "inliers_before": _inlier_fraction(residual_before, width), "inliers_after": _inlier_fraction(fit["residual"], width),
           "observations": int(len(obs_frame)), "points_used": int(len(cloud)), "frames_used": int(len(frames)),
           "iterations": int(fit["iterations"]), "point_index": chosen, "points": fit["points"]}
    refined = camera_to_world(fit["rotation"], fit["translation"])
    refined, cloud, scale = _restore_scale(c2w[frames], refined, fit["points"])
    out["poses"] = _carry_unused_frames(c2w, frames, refined)
    out["points"], out["path_scale"] = cloud, scale
    return out


def _staged_adjust(rotation, translation, cloud, obs_frame, obs_point, obs_xy, obs_weight, width, height, base, free,
                   iterations):
    """The adjustment with a loss that tightens: a start whose lens is off has residuals of several pixels, which the
    final Huber threshold would treat as outliers and crawl through. Each stage starts from the one before."""
    scale = width / 960.0
    fit, accepted = None, 0
    for threshold in (24.0 * scale, 6.0 * scale, HUBER_PX * scale):
        fit = bundle_adjust(rotation, translation, cloud, obs_frame, obs_point, obs_xy, obs_weight, width, height, base,
                            free, iterations=iterations, huber=threshold)
        rotation, translation, cloud, base = fit["rotation"], fit["translation"], fit["points"], fit["intrinsics"]
        accepted += fit["iterations"]
    fit["iterations"] = accepted
    return fit


def _inlier_threshold(width):
    return 4.0 * HUBER_PX * width / 960.0


def _rms(residual, width):
    """Root mean square pixel distance of the inliers (those within 6 px at 960 wide, the same bound before and after)."""
    distance = np.linalg.norm(residual, axis=1)
    inside = distance < _inlier_threshold(width)
    return float(np.sqrt(np.mean(distance[inside] ** 2))) if inside.any() else float("nan")


def _inlier_fraction(residual, width):
    return float(np.mean(np.linalg.norm(residual, axis=1) < _inlier_threshold(width)))


def _carry_unused_frames(c2w, used, refined):
    """Poses for every frame: the refined ones for the frames used; any other frame keeps its motion relative to the
    nearest earlier used frame (the nearest later one before the first)."""
    out = c2w.copy()
    out[used] = refined
    used = np.asarray(used)
    known = set(used.tolist())
    for index in range(len(c2w)):
        if index in known:
            continue
        earlier = used[used < index]
        anchor = int(earlier[-1]) if len(earlier) else int(used[0])
        relative = np.linalg.inv(c2w[anchor]) @ c2w[index]
        out[index] = out[anchor] @ relative
    return out


def _restore_scale(before, after, cloud):
    """Scale the refined frames and points about the fixed first pose so the path between the used frames is as long
    as the solver's was: the adjustment has no scale of its own, and a unit scale then means what it did."""
    length = lambda poses: float(np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1).sum())
    if length(before) <= 0 or length(after) <= 0:
        return after, cloud, 1.0
    s = length(before) / length(after)
    origin = after[0, :3, 3].copy()
    after = after.copy()
    after[:, :3, 3] = origin + (after[:, :3, 3] - origin) * s
    return after, origin + (cloud - origin) * s, s
