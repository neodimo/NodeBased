"""Cryptomatte ID mattes out of Render3D (docs/PARITY_2D.md Keyer row 8 is the reader side;
`nodebased/cryptomatte.py` holds the hashing and matte maths both sides share).

Three sets: CryptoObject (the node that created the surface, plus an instance id for Instance3D
copies), CryptoMaterial (the geometry's `material` string) and CryptoAsset (the outermost
Scene3D/Axis3D parent's name, `scene3d.scene_from_node` stamps it; an unparented object is its own
asset). Ids reuse `cryptomatte.name_to_bits`, the exact function the Cryptomatte node hashes names
with, so a Render3D EXR and a name typed into the Cryptomatte node always agree.

Coverage rides on the existing `object_id` render pass (the scene's 1-based geometry index, then
splats offset by `len(scene.geometries)`, `scene3d.render`'s own convention) supersampled by this
module rather than by `render()` itself, which forces `samples=1` for every data output: a subsample
carries exactly one winning id, never a blend, so averaging IDs across subsamples would invent ids
that hit nothing. Instead every subsample's id is kept and binned per output pixel into ranked
(id, coverage) pairs, coverage = subsample count / total subsamples.

Limits: volumes hold no id (the `object_id` pass never raymarches them; see the Render3D known
limit in docs/PARITY_2D.md) and their coverage is not represented in any of the three sets. Splats
get a name from their ReadSplat3D node but share one `CryptoMaterial` entry ("splat"): a splat
cloud carries no material name, only PBR scalars. `MergeGeo3D` collapses its inputs into one
Geometry, so a merged object is one Cryptomatte id, not one per original input (the same limit
`merge_geometry` already documents for colour and material).
"""
from __future__ import annotations

import numpy as np

from . import cryptomatte
from .scene3d import resolve_instances, render as render3d

_SETS = (("CryptoObject", "object"), ("CryptoMaterial", "material"), ("CryptoAsset", "asset"))


def _names(scene, kind):
    """`[name, ...]`, index 0 (background/no hit) is `""`, then one entry per `scene.geometries`
    followed by one per `scene.splats`, in `object_id`'s own index order."""
    names = [""]
    for i, g in enumerate(scene.geometries, 1):
        if kind == "material":
            names.append(g.material or "standard")
        elif kind == "asset":
            names.append(g.asset or g.name or f"object{i}")
        else:
            names.append(g.name or f"object{i}")
    for j, sp in enumerate(scene.splats, 1):
        if kind == "material":
            names.append("splat")
        elif kind == "asset":
            names.append(sp.name or f"splat{j}")
        else:
            names.append(sp.name or f"splat{j}")
    return names


def _rank_coverage(id_blocks, levels, total_samples):
    """`(rank_ids, rank_covs)`, both `(height, width, levels)`: the `levels` most-covered ids per
    pixel, best first, coverage clamped to what actually hit (fewer objects than `levels` pads with
    id 0 / coverage 0). `id_blocks` is `(height, width, samples**2)` uint32, one winning id per
    subsample, 0 for a miss."""
    height, width, k = id_blocks.shape
    flat = id_blocks.reshape(-1, k)
    count = flat.shape[0]
    rank_ids = np.zeros((count, levels), np.uint32)
    rank_covs = np.zeros((count, levels), np.float32)
    for i in range(count):
        row = flat[i]
        row = row[row != 0]
        if not row.size:
            continue
        uniq, counts = np.unique(row, return_counts=True)
        order = np.argsort(-counts, kind="stable")[:levels]
        uniq, counts = uniq[order], counts[order]
        rank_ids[i, :len(uniq)] = uniq
        rank_covs[i, :len(uniq)] = counts.astype(np.float32) / total_samples
    return rank_ids.reshape(height, width, levels), rank_covs.reshape(height, width, levels)


def render_cryptomatte(scene, camera, width, height, *, samples=1, levels=6, mode="raster", cancel=None):
    """`({layer name: HxWx4 float32}, {set name: {"key": ..., "manifest": {name: hex id}}})` for
    the three Cryptomatte sets, at `width` x `height`. `samples` supersamples per axis exactly like
    Render3D's own antialiasing knob (coverage accumulates over the `samples**2` subsamples);
    `levels` is the rank count (default 6: three `<Set>00`, `<Set>01`, `<Set>02` channel groups).
    """
    samples = max(1, int(samples))
    levels = max(2, (int(levels) // 2) * 2)
    width, height = int(width), int(height)
    scene = resolve_instances(scene)
    big = render3d(scene, camera, width * samples, height * samples, output="object_id",
                   cancel=cancel, mode=mode)
    index_grid = np.where(big[..., 3] > 0, np.round(big[..., 0]), 0).astype(np.int64)
    # (height, width, samples*samples): every subsample's winning scene index for this pixel.
    blocks = (index_grid.reshape(height, samples, width, samples)
              .transpose(0, 2, 1, 3).reshape(height, width, samples * samples))
    total_samples = samples * samples
    layers, metadata = {}, {}
    for set_name, kind in _SETS:
        names = _names(scene, kind)
        id_bits = np.zeros(len(names), np.uint32)
        for i, name in enumerate(names):
            if i and name:
                id_bits[i] = cryptomatte.name_to_bits(name)
        id_blocks = id_bits[blocks]
        rank_ids, rank_covs = _rank_coverage(id_blocks, levels, total_samples)
        for group in range(levels // 2):
            r_id, r_cov = rank_ids[..., 2 * group], rank_covs[..., 2 * group]
            g_id, g_cov = rank_ids[..., 2 * group + 1], rank_covs[..., 2 * group + 1]
            layer = np.zeros((height, width, 4), np.float32)
            layer[..., 0] = r_id.view(np.float32)
            layer[..., 1] = r_cov
            layer[..., 2] = g_id.view(np.float32)
            layer[..., 3] = g_cov
            layers[f"{set_name}{group:02d}"] = layer
        manifest = {}
        for name, bits in zip(names[1:], id_bits[1:]):
            if name:
                manifest[name] = format(int(bits), "08x")
        metadata[set_name] = {"key": cryptomatte.set_key(set_name), "manifest": manifest}
    return layers, metadata


def cryptomatte_header(metadata):
    """`{header attribute: string}` for `write_exr`'s `metadata=` argument, from
    `render_cryptomatte`'s per-set `{"key", "manifest"}`."""
    import json
    header = {}
    for set_name, entry in metadata.items():
        key = entry["key"]
        header[f"cryptomatte/{key}/name"] = set_name
        header[f"cryptomatte/{key}/hash"] = "MurmurHash3_32"
        header[f"cryptomatte/{key}/conversion"] = "uint32_to_float32"
        header[f"cryptomatte/{key}/manifest"] = json.dumps(entry["manifest"], sort_keys=True)
    return header
