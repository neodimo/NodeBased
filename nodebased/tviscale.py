"""TVIScale: Nuke's edge-preserving two-times enlargement.

Nuke's TVIScale always enlarges by 2 and uses Total Variational Inpainting to smooth noise while keeping
edges. `Iterations` (default 5) is how long the iterative filter runs, `Lambda` (default 5) is how far the
result may stray from the input (larger keeps more of the input's noise) and `A` (default 0.0001) is a
regularizer that must stay above zero. The reference guide states no formula and we have no Nuke render to
check against, so this is a deterministic, bounded total-variation solver in the same family, not a claim of
Nuke pixel equivalence:

* the data target `f` is the plain `filter` (bilinear by default) 2x enlargement of the input;
* `iterations` Jacobi steps of  u <- (lambda f + sum_n w_n u_n) / (lambda + sum_n w_n)  follow, with the
  four neighbours n of each pixel and the edge-aware weights  w_n = 1 / sqrt((u_n - u)^2 + A^2);
* zero iterations returns `f`, the bilinear target.

A small difference to a neighbour has a large weight, so noise is averaged away; a large difference (an edge)
has a small weight, so the edge is left alone. Every pixel step reads the previous step only (Jacobi), each
channel is solved on its own, and the picture is solved whole, which is what keeps a tile boundary from
changing the result: the tile executor solves the node once and slices it.

Nodes saved before this behaviour (a signed `power` count of doublings) carry `legacy_scale` = 1 and keep
their old resize exactly (see core.py's schema migration and `imaging.Evaluator`'s TVIScale branch).
"""
from __future__ import annotations

import numpy as np

from . import ops2d_chrome as _chrome

A_FLOOR = 1e-6          # `A` is clamped to at least this so the weights never divide by zero
FACTOR = 2              # Nuke's TVIScale always enlarges by two


def tv_smooth(base, iterations, lam, a, channels=None):
    """`iterations` Jacobi steps of edge-preserving total-variation smoothing of `base` (h, w, c) float32.

    `channels` is an optional boolean mask over the last axis: only those channels are solved and the others
    are returned untouched. Zero iterations (or no channel) returns `base` unchanged."""
    iterations = max(0, int(iterations))
    out = np.asarray(base, dtype=np.float32)
    if iterations == 0:
        return out
    sel = np.ones(out.shape[-1], dtype=bool) if channels is None else np.asarray(channels, dtype=bool)
    if not sel.any():
        return out
    f = np.ascontiguousarray(out[..., sel], dtype=np.float32)
    lam = np.float32(max(0.0, float(lam)))
    a2 = np.float32(max(float(a), A_FLOOR)) ** 2
    u = f.copy()
    for _ in range(iterations):
        padded = np.pad(u, ((1, 1), (1, 1), (0, 0)), mode="edge")
        total = f * lam
        weights = np.full_like(u, lam)
        for neighbour in (padded[:-2, 1:-1], padded[2:, 1:-1], padded[1:-1, :-2], padded[1:-1, 2:]):
            d = neighbour - u
            w = np.float32(1.0) / np.sqrt(d * d + a2)
            total += w * neighbour
            weights += w
        u = (total / weights).astype(np.float32)
    result = out.copy()
    result[..., sel] = u
    return result


def legacy(p):
    """True for a node that still runs the old signed power-of-two resize."""
    return bool(p.get("legacy_scale", 0))


def needs_full_frame(p):
    """A node whose chrome writes or reads a named layer, or injects the mask, rides on the whole-image path:
    the tile executor solves TVIScale once and slices pixels only."""
    if legacy(p):
        return False
    cfg = _chrome.filter_config("TVIScale", p)
    if cfg.inject or cfg.channel_set.layers:
        return True
    return cfg.mask_channel != "none" and not _chrome._is_rgba_token(cfg.mask_channel)


def enlarge(raster, filter, resample, Region, Raster):
    """`raster` at twice the size with its edge pixels carried outward (no dark border), as a Raster whose data
    window is the input's data window doubled. `resample` is `imaging.Evaluator._resample`."""
    d = raster.data
    box = Region(d.x * FACTOR, d.y * FACTOR, d.width * FACTOR, d.height * FACTOR)
    sx = ((np.arange(box.width, dtype=np.float32) + 0.5) / FACTOR - 0.5)[None, :].repeat(box.height, axis=0)
    sy = ((np.arange(box.height, dtype=np.float32) + 0.5) / FACTOR - 0.5)[:, None].repeat(box.width, axis=1)
    pixels = resample(np.asarray(raster.pixels, dtype=np.float32), sx, sy, filter, clamp=True).astype(np.float32)
    return Raster(pixels, box, Region(0, 0, raster.display.width * FACTOR, raster.display.height * FACTOR))


def run(p, source, mask, resample, Region, Raster):
    """The TVIScale kernel on Rasters: enlarge by two, smooth, then the shared mask strip (channels, mask
    channel, invert, inject, fringe, mix). `resample` is `imaging.Evaluator._resample`."""
    if mask is not None and mask.display != source.display:
        raise ValueError(
            f"Mask display window {mask.display} does not match source {source.display}; "
            "no silent resampling is performed")
    cfg = _chrome.filter_config("TVIScale", p)
    big = lambda raster: enlarge(raster, p["filter"], resample, Region, Raster)
    scaled = big(source)
    box, display, base = scaled.data, scaled.display, scaled.pixels
    layers = {name: big(layer) for name, layer in (source.layers or {}).items()}
    selected = cfg.channel_set
    solve = lambda pixels, sel: tv_smooth(pixels, p["tvi_iterations"], p["tvi_lambda"], p["tvi_a"], sel)
    result = solve(base, np.array(selected.rgba, dtype=bool))      # channels outside the selection keep the plain enlargement
    for token in selected.layers:
        name, component = token, None
        if name not in layers and "." in token:
            name, component = token.rsplit(".", 1)
        if name not in layers:
            have = ", ".join(sorted(layers))
            raise ValueError(f"TVIScale: channels names {token!r}, but the input has no such layer"
                             + (f" (it has {have})" if have else " (it carries no named layers)"))
        plain = layers[name]
        sel = None
        if component is not None:
            which = _chrome._COMPONENT.get(component.upper(), -1)
            if not 0 <= which < plain.pixels.shape[-1]:
                raise ValueError(f"TVIScale: {token!r} is not a channel")
            sel = np.zeros(plain.pixels.shape[-1], dtype=bool)
            sel[which] = True
        layers[name] = Raster(solve(plain.pixels, sel), plain.data, plain.display)
    matte = None
    if mask is not None or cfg.mask_channel != "none":
        holder = big(mask) if mask is not None else scaled
        if cfg.mask_channel == "none":
            channel = holder.fit(box)[..., 3:4]
        else:
            holder_layers = {name: big(layer) for name, layer in (mask.layers or {}).items()} if mask is not None else layers
            channel = _chrome.resolve_channel(
                cfg.mask_channel, holder.fit(box),
                lambda n: holder_layers[n].fit(box) if n in holder_layers else None)
        matte = _chrome.matte_of(channel, cfg.invert_mask)
    pixels = _chrome.gate(base, result, _chrome.gate_matte(matte, cfg), cfg.mix)
    if cfg.inject and matte is not None:
        layers["mask"] = Raster(_chrome.injected_layer(matte), box, display)
    return Raster(pixels, box, display, layers or None, source.meta)
