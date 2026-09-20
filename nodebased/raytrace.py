"""NumPy CPU ray queries over a deterministic, primitive-agnostic flat BVH.

Triangle and Gaussian splat sets supply bounds and leaf callbacks.
"""
from dataclasses import dataclass
import numpy as np


def _cancel(event):
    if event is not None and event.is_set():
        from .cancellation import Cancelled
        raise Cancelled()


@dataclass
class Bvh:
    """Root is node 0 (no nodes for empty input). Interior counts are zero;
    leaves have left/right=-1 and own prim_order[offset:offset+count].
    """
    node_lo: np.ndarray
    node_hi: np.ndarray
    left: np.ndarray
    right: np.ndarray
    prim_offset: np.ndarray
    prim_count: np.ndarray
    prim_order: np.ndarray

    @classmethod
    def build(cls, lo, hi, leaf_size=4, *, cancel=None):
        """Stable median splits; iterative construction, padded float64 bounds."""
        lo, hi = np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)
        if lo.shape != hi.shape or lo.ndim != 2 or lo.shape[1] != 3:
            raise ValueError('bounds must have matching (N, 3) shapes')
        if leaf_size < 1 or not np.isfinite(lo).all() or not np.isfinite(hi).all() or np.any(lo > hi):
            raise ValueError('invalid bounds or leaf_size')
        _cancel(cancel)
        n = len(lo)
        order = np.arange(n, dtype=np.int32)
        pad = 8 * np.finfo(np.float32).eps * np.maximum(1, np.maximum(hi-lo, np.maximum(abs(lo), abs(hi))))
        lower, upper = lo-pad, hi+pad
        centers = lo*.5 + hi*.5
        nodes = []
        pending = [(0, n, -1, 0)] if n else []
        while pending:
            _cancel(cancel)
            start, end, parent, side = pending.pop()
            index = len(nodes)
            ids = order[start:end]
            nodes.append([lower[ids].min(0), upper[ids].max(0), -1, -1, start, end-start])
            if parent >= 0:
                nodes[parent][2+side] = index
            if end-start > leaf_size:
                axis = np.argmax(np.ptp(centers[ids], axis=0))
                order[start:end] = ids[np.argsort(centers[ids, axis], kind='stable')]
                mid = (start+end)//2
                nodes[index][5] = 0
                pending.extend(((mid, end, index, 1), (start, mid, index, 0)))
        return cls(np.array([x[0] for x in nodes], dtype=np.float64).reshape(-1, 3),
                   np.array([x[1] for x in nodes], dtype=np.float64).reshape(-1, 3),
                   *(np.array([x[i] for x in nodes], dtype=np.int32) for i in range(2, 6)), order)


def traverse(bvh, origins, dirs, tmax, leaf_callback, chunk=1024, *, cancel=None, stats=None, pair_chunk=None, near_first=False, active=None, tmin=None):
    """Visit (ray, primitive) pairs, once each, in bounded ray chunks.

    tmax may be a mutable per-ray array: callbacks can shorten it for pruning.
    stats is an optional accumulating dict of node_tests and primitive_tests.
    Pair storage is O(chunk * node count), independent of the total ray count.
    pair_chunk bounds frontier batches with depth-first scheduling, for queries
    that need a strict memory bound even when every bounding box overlaps.
    Slabs include boundaries, parallel rays, inside origins and negative t.
    tmin (optional per-ray array) additionally prunes boxes lying entirely before it (exit t below
    tmin, with a roundoff allowance). That is exact for queries whose primitives can only contribute
    inside their own bounding box at t >= tmin (splat shadows: a 3-sigma ellipsoid that ends before
    tmin cannot contain the clamped closest point), and it skips the neighbours around a ray origin.
    """
    if chunk < 1 or (pair_chunk is not None and pair_chunk < 1):
        raise ValueError('chunk sizes must be positive')
    origins, dirs = np.asarray(origins), np.asarray(dirs)
    limits = np.broadcast_to(tmax, (len(origins),))
    lowers = None if tmin is None else np.broadcast_to(tmin, (len(origins),))
    stats = {} if stats is None else stats
    stats.setdefault('node_tests', 0)
    stats.setdefault('primitive_tests', 0)
    _cancel(cancel)
    if not len(bvh.left):
        return stats
    for start in range(0, len(origins), chunk):
        _cancel(cancel)
        rays = np.arange(start, min(start+chunk, len(origins)))
        nodes = np.zeros(len(rays), dtype=np.int32)
        pending = [(rays, nodes)]
        while pending:
            rays, nodes = pending.pop()
            _cancel(cancel)
            if active is not None:
                keep = active[rays]
                rays, nodes = rays[keep], nodes[keep]
                if not len(rays):
                    continue
            stats['node_tests'] += len(nodes)
            o, d = origins[rays], dirs[rays]
            lo, hi = bvh.node_lo[nodes], bvh.node_hi[nodes]
            parallel = d == 0
            a = np.divide(lo-o, d, out=np.full_like(lo, -np.inf), where=~parallel)
            b = np.divide(hi-o, d, out=np.full_like(hi, np.inf), where=~parallel)
            near, far = np.minimum(a, b).max(1), np.maximum(a, b).min(1)
            hit = (near <= np.minimum(far, limits[rays])) & ~np.any(parallel & ((o < lo) | (o > hi)), axis=1)
            if lowers is not None:
                floor = lowers[rays]
                hit &= far >= floor - 1e-9*(1+np.abs(floor))
            rays, nodes = rays[hit], nodes[hit]
            counts = bvh.prim_count[nodes]
            leaf = counts > 0
            if leaf.any():
                lr, ln, lc = rays[leaf], nodes[leaf], counts[leaf]
                rr = np.repeat(lr, lc)
                offsets = np.repeat(bvh.prim_offset[ln], lc)
                local = np.arange(len(rr)) - np.repeat(np.cumsum(lc)-lc, lc)
                pp = bvh.prim_order[offsets+local]
                stats['primitive_tests'] += len(pp)
                leaf_callback(rr, pp)
            inner = nodes[~leaf]
            if near_first:
                rr = rays[~leaf]
                children = np.column_stack((bvh.left[inner], bvh.right[inner]))
                o, d = origins[rr, None], dirs[rr, None]
                lo, hi = bvh.node_lo[children], bvh.node_hi[children]
                a = np.divide(lo-o, d, out=np.full_like(lo, -np.inf), where=d != 0)
                b = np.divide(hi-o, d, out=np.full_like(hi, np.inf), where=d != 0)
                entry = np.minimum(a, b).max(2)
                swap = entry[:, 1] < entry[:, 0]
                children[swap] = children[swap, ::-1]
                # Visit each ray's near child before its far child, allowing the
                # callback's running limit to prune the deferred far subtree.
                step = pair_chunk or max(1, len(rr))
                for side in (1, 0):
                    for first in reversed(range(0, len(rr), step)):
                        pending.append((rr[first:first+step], children[first:first+step, side]))
                continue
            else:
                rays = np.repeat(rays[~leaf], 2)
                nodes = np.column_stack((bvh.left[inner], bvh.right[inner])).ravel()
            step = pair_chunk or max(1, len(nodes))
            for first in reversed(range(0, len(nodes), step)):
                pending.append((rays[first:first+step], nodes[first:first+step]))
    return stats


class TriangleSet:
    def __init__(self, v0, e1, e2, alpha):
        self.v0, self.e1, self.e2 = (np.asarray(a) for a in (v0, e1, e2))
        if self.v0.ndim != 2 or self.v0.shape[1] != 3 or any(a.shape != self.v0.shape for a in (self.e1, self.e2)):
            raise ValueError('triangles must have matching (N, 3) shapes')
        self.alpha = np.broadcast_to(np.asarray(alpha, dtype=np.float32), (len(self.v0),))

    def aabbs(self):
        # Reconstruct in double precision so edge subtraction cannot shrink boxes.
        v = self.v0.astype(np.float64)
        vertices = np.stack((v, v+self.e1, v+self.e2))
        return vertices.min(0), vertices.max(0)

    def _intersect(self, origins, dirs, r, p, tmin, tmax, edge_epsilon=0.):
        h = np.cross(dirs[r], self.e2[p])
        det = np.einsum('ij,ij->i', h, self.e1[p])
        valid = abs(det) > 1e-10
        inv = np.divide(1., det, out=np.zeros_like(det), where=valid)
        delta = origins[r]-self.v0[p]
        u = np.einsum('ij,ij->i', delta, h)*inv
        q = np.cross(delta, self.e1[p])
        v = np.einsum('ij,ij->i', dirs[r], q)*inv
        t = np.einsum('ij,ij->i', self.e2[p], q)*inv
        hit = valid & (u >= -edge_epsilon) & (v >= -edge_epsilon) & (u+v <= 1+edge_epsilon) & (t > tmin[r]) & (t < tmax[r])
        return hit, t, u, v

    def closest_hit(self, bvh, origins, dirs, tmin=0., tmax=np.inf, **kwargs):
        """Two-sided hits; exact equal-t ties choose the lowest primitive index."""
        origins, dirs = np.asarray(origins), np.asarray(dirs)
        n = len(origins)
        lower, upper = np.broadcast_to(tmin, (n,)), np.broadcast_to(tmax, (n,))
        best = np.full(n, np.inf)
        limits = upper.astype(np.float64).copy()
        prim = np.full(n, -1, dtype=np.int32)
        us, vs = np.zeros(n), np.zeros(n)
        def leaf(r, p):
            hit, t, u, v = self._intersect(origins, dirs, r, p, lower, upper)
            ids = np.flatnonzero(hit)
            ids = ids[np.lexsort((p[ids], t[ids], r[ids]))]
            ids = ids[np.r_[True, np.diff(r[ids]) != 0]] if len(ids) else ids
            rr, pp, tt = r[ids], p[ids], t[ids]
            take = (tt < best[rr]) | ((tt == best[rr]) & ((prim[rr] < 0) | (pp < prim[rr])))
            ids, rr = ids[take], rr[take]
            best[rr], prim[rr], us[rr], vs[rr] = t[ids], p[ids], u[ids], v[ids]
            limits[rr] = best[rr]
        traverse(bvh, origins, dirs, limits, leaf, **kwargs)
        return best, prim, us, vs

    def nearest_hits(self, bvh, origins, dirs, tmin, tmax, k, *, after_t=None,
                     after_primitive=None, chunk=1024, cancel=None):
        """Return at most k hits per ray ordered by (t, primitive).

        Float64 intersections and the inclusive edge band match all_hits. The
        mutable K-th distance prunes deferred nodes; equal distances remain
        eligible so primitive indices break ties independently of traversal.
        An optional per-ray (after_t, after_primitive) cursor excludes pairs
        at or before it; tmin and tmax remain strict intersection bounds.
        """
        if chunk < 1 or k < 1:
            raise ValueError('chunk and k must be positive')
        origins, dirs = np.asarray(origins, dtype=np.float64), np.asarray(dirs, dtype=np.float64)
        n = len(origins)
        lower, upper = np.broadcast_to(tmin, (n,)), np.broadcast_to(tmax, (n,))
        if (after_t is None) != (after_primitive is None):
            raise ValueError('after_t and after_primitive must be supplied together')
        cursor_t = None if after_t is None else np.broadcast_to(after_t, (n,))
        cursor_p = None if after_primitive is None else np.broadcast_to(after_primitive, (n,))
        dtype = np.dtype([('t', 'f8'), ('primitive', 'i4'), ('u', 'f8'), ('v', 'f8')])
        result = []
        _cancel(cancel)
        for start in range(0, n, chunk):
            _cancel(cancel)
            stop = min(start+chunk, n)
            o, d = origins[start:stop], dirs[start:stop]
            lo, hi = lower[start:stop], upper[start:stop]
            limits = hi.astype(float).copy()
            best = np.empty((len(o), k), dtype=dtype)
            best['t'] = np.inf
            best['primitive'] = np.iinfo(np.int32).max
            def leaf(r, p):
                hit, t, u, v = self._intersect(o, d, r, p, lo, hi, 32*np.finfo(float).eps)
                if cursor_t is not None:
                    ct, cp = cursor_t[start:stop][r], cursor_p[start:stop][r]
                    hit &= (t > ct) | ((t == ct) & (p > cp))
                r, p = r[hit], p[hit]
                if not len(r):
                    return
                batch = np.empty(len(r), dtype=dtype)
                batch['t'], batch['primitive'] = t[hit], p
                batch['u'], batch['v'] = u[hit], v[hit]
                # Merge all leaf candidates together, retaining only K per ray.
                rr = np.unique(r)
                merged = np.concatenate((best[rr].ravel(), batch))
                rays = np.concatenate((np.repeat(rr, k), r))
                order = np.lexsort((merged['primitive'], merged['t'], rays))
                rays, merged = rays[order], merged[order]
                first = np.r_[0, np.flatnonzero(np.diff(rays))+1]
                ranks = np.arange(len(rays))-np.repeat(first, np.diff(np.r_[first, len(rays)]))
                keep = ranks < k
                best[rays[keep], ranks[keep]] = merged[keep]
                limits[rr] = np.minimum(hi[rr], best['t'][rr, -1])
            traverse(bvh, o, d, limits, leaf, chunk=chunk, cancel=cancel,
                     pair_chunk=len(o), near_first=True)
            _cancel(cancel)
            result.extend(h[np.isfinite(h['t'])] for h in best)
        return result

    def all_hits(self, bvh, origins, dirs, tmin=0., tmax=np.inf, *,
                 chunk=1024, max_hits=64, cancel=None):
        """Per-ray structured hit arrays sorted by (t, primitive), including ties.

        Fields are t, primitive, u, v. Storage is bounded by max_hits per ray;
        the limit counts triangle intersections, including shared-edge duplicates.
        Traversal and collection check cancellation between bounded chunks.
        """
        if chunk < 1 or max_hits < 1:
            raise ValueError('chunk and max_hits must be positive')
        origins, dirs = np.asarray(origins, dtype=np.float64), np.asarray(dirs, dtype=np.float64)
        n = len(origins)
        lower, upper = np.broadcast_to(tmin, (n,)), np.broadcast_to(tmax, (n,))
        dtype = np.dtype([('t', 'f8'), ('primitive', 'i4'), ('u', 'f8'), ('v', 'f8')])
        result = []
        _cancel(cancel)
        for start in range(0, n, chunk):
            _cancel(cancel)
            stop = min(start+chunk, n)
            o, d = origins[start:stop], dirs[start:stop]
            lo, hi = lower[start:stop], upper[start:stop]
            counts = np.zeros(len(o), dtype=np.int32)
            batches, ray_batches = [], []
            def leaf(r, p):
                # Double precision and a roundoff-sized inclusive edge band prevent
                # shared-edge cracks. Primary shading resolves duplicate edge hits.
                hit, t, u, v = self._intersect(o, d, r, p, lo, hi, 32*np.finfo(float).eps)
                r, p = r[hit], p[hit]
                counts[:] += np.bincount(r, minlength=len(o))
                if np.any(counts > max_hits):
                    raise ValueError(f'Ray-traced render exceeds MAX_HITS_PER_RAY ({max_hits})')
                if len(r):
                    batch = np.empty(len(r), dtype=dtype)
                    batch['t'], batch['primitive'] = t[hit], p
                    batch['u'], batch['v'] = u[hit], v[hit]
                    batches.append(batch)
                    ray_batches.append(r)
            traverse(bvh, o, d, hi, leaf, chunk=chunk, cancel=cancel, pair_chunk=chunk)
            _cancel(cancel)
            if batches:
                hits, rays = np.concatenate(batches), np.concatenate(ray_batches)
                order = np.lexsort((hits['primitive'], hits['t'], rays))
                hits = hits[order]
            else:
                hits = np.empty(0, dtype=dtype)
            result.extend(np.split(hits, np.cumsum(counts)[:-1]))
        return result

    def any_hit(self, bvh, origins, dirs, tmin=0., tmax=np.inf, **kwargs):
        return self.closest_hit(bvh, origins, dirs, tmin, tmax, **kwargs)[1] >= 0

    def transmittance(self, bvh, origins, dirs, tmin=0., tmax=np.inf, **kwargs):
        """Product over ALL hits, including duplicate/shared-edge triangles.

        Pass bvh=None to build an acceleration structure for this query.
        """
        if bvh is None:
            bvh = Bvh.build(*self.aabbs(), cancel=kwargs.get('cancel'))
        origins, dirs = np.asarray(origins), np.asarray(dirs)
        n = len(origins)
        lower, upper = np.broadcast_to(tmin, (n,)), np.broadcast_to(tmax, (n,))
        result = np.ones(n, dtype=np.float64)
        def leaf(r, p):
            hit, _, _, _ = self._intersect(origins, dirs, r, p, lower, upper)
            np.multiply.at(result, r[hit], 1-self.alpha[p[hit]])
        traverse(bvh, origins, dirs, upper, leaf, **kwargs)
        return result.astype(np.float32)

    def brute_transmittance(self, origins, dirs, tmin=0., tmax=np.inf, *, chunk=128, triangle_chunk=512, cancel=None):
        """Original broadcast shadow kernel, independent of BVH traversal."""
        origins, dirs = np.asarray(origins), np.asarray(dirs)
        n = len(origins)
        lower, upper = np.broadcast_to(tmin, (n,)), np.broadcast_to(tmax, (n,))
        result = np.ones(n, np.float32)
        _cancel(cancel)
        for start in range(0, n, chunk):
            stop = min(start+chunk, n)
            for first in range(0, len(self.v0), triangle_chunk):
                _cancel(cancel)
                sl = slice(first, first+triangle_chunk)
                h = np.cross(dirs[start:stop, None], self.e2[sl])
                det = np.einsum('rtj,tj->rt', h, self.e1[sl])
                valid = abs(det) > 1e-10
                inv = np.divide(1., det, out=np.zeros_like(det), where=valid)
                delta = origins[start:stop, None]-self.v0[sl]
                u = np.einsum('rtj,rtj->rt', delta, h)*inv
                q = np.cross(delta, self.e1[sl])
                v = np.einsum('rj,rtj->rt', dirs[start:stop], q)*inv
                t = np.einsum('tj,rtj->rt', self.e2[sl], q)*inv
                hit = valid & (u >= 0) & (v >= 0) & (u+v <= 1) & (t > lower[start:stop, None]) & (t < upper[start:stop, None])
                result[start:stop] *= np.prod(np.where(hit, 1-self.alpha[sl], 1), axis=1)
        return result


def brute_transmittance(triangles, origins, dirs, tmin=0., tmax=np.inf, **kwargs):
    return triangles.brute_transmittance(origins, dirs, tmin, tmax, **kwargs)


class SplatSet:
    """World-space Gaussian shadow primitives; rotation columns are local axes.

    Density is sampled at the closest point on the bounded ray, truncated at
    three sigma. Zero scales use a tiny positive width for whitening.
    """
    def __init__(self, positions, rotations_matrix, scales, opacity):
        self.positions, self.rotations_matrix, self.scales, self.opacity = (
            np.asarray(a, dtype=np.float64) for a in
            (positions, rotations_matrix, scales, opacity))
        n = len(self.positions)
        for a, shape in zip((self.positions, self.rotations_matrix, self.scales, self.opacity),
                            ((n, 3), (n, 3, 3), (n, 3), (n,))):
            if a.shape != shape or not np.isfinite(a).all():
                raise ValueError('invalid splat arrays')
        if np.any(self.scales < 0) or np.any((self.opacity < 0) | (self.opacity > 1)):
            raise ValueError('invalid splat scales or opacity')

    def covariance(self):
        rs = self.rotations_matrix * self.scales[:, None, :]
        return rs @ rs.transpose(0, 2, 1)

    def aabbs(self, k=3.0):
        from .splats import SplatCloud
        return SplatCloud.aabbs(self, k)

    def _factors(self, origins, dirs, r, p, lower, upper, exclude):
        inv = 1 / np.maximum(self.scales[p], 1e-30)
        o = np.einsum('nij,ni->nj', self.rotations_matrix[p], origins[r]-self.positions[p])*inv
        d = np.einsum('nij,ni->nj', self.rotations_matrix[p], dirs[r])*inv
        dd = np.sum(d*d, axis=1)
        t = np.divide(-np.sum(o*d, axis=1), dd, out=np.zeros(len(r)), where=dd > 0)
        t = np.clip(t, lower[r], upper[r])
        d2 = np.sum((o+t[:, None]*d)**2, axis=1)
        valid = (d2 <= 9) & (lower[r] <= upper[r])
        if exclude is not None:
            valid &= p != exclude[r]
        return np.where(valid, 1-np.minimum(.99, self.opacity[p]*np.exp(-.5*d2)), 1.)

    def transmittance(self, bvh, origins, dirs, tmin=0., tmax=np.inf, *,
                      exclude=None, cancel=None, chunk=1024, cutoff=0.0, prune_tmin=True):
        """Exact by default; positive cutoff drops rays below it and returns zero.

        The absolute transmittance error is bounded by cutoff.
        """
        if not np.isfinite(cutoff) or not 0 <= cutoff <= 1:
            raise ValueError("cutoff must be finite and between zero and one")
        origins, dirs = np.asarray(origins, dtype=float), np.asarray(dirs, dtype=float)
        n = len(origins)
        lower, upper = np.broadcast_to(tmin, (n,)), np.broadcast_to(tmax, (n,))
        exclude = None if exclude is None else np.broadcast_to(exclude, (n,))
        result = np.ones(n)
        active = np.ones(n, dtype=bool) if cutoff else None
        def leaf(r, p):
            np.multiply.at(result, r, self._factors(origins, dirs, r, p, lower, upper, exclude))
            if active is not None:
                active[r] = result[r] >= cutoff
        traverse(bvh, origins, dirs, upper, leaf, chunk=chunk, pair_chunk=chunk, cancel=cancel, active=active,
                 tmin=lower if prune_tmin else None)
        if active is not None:
            result[~active] = 0
        return result

    def brute_transmittance(self, origins, dirs, tmin=0., tmax=np.inf, *,
                            exclude=None, cancel=None, chunk=128, splat_chunk=256):
        if chunk < 1 or splat_chunk < 1:
            raise ValueError('chunk sizes must be positive')
        origins, dirs = np.asarray(origins, dtype=float), np.asarray(dirs, dtype=float)
        n = len(origins)
        lower, upper = np.broadcast_to(tmin, (n,)), np.broadcast_to(tmax, (n,))
        exclude = None if exclude is None else np.broadcast_to(exclude, (n,))
        result = np.ones(n)
        _cancel(cancel)
        for start in range(0, n, chunk):
            for first in range(0, len(self.positions), splat_chunk):
                _cancel(cancel)
                rr = np.arange(start, min(n, start+chunk))
                pp = np.arange(first, min(len(self.positions), first+splat_chunk))
                r, p = np.repeat(rr, len(pp)), np.tile(pp, len(rr))
                np.multiply.at(result, r, self._factors(origins, dirs, r, p, lower, upper, exclude))
        return result
