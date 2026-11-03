"""Cryptomatte ID mattes (Psyop, 2015; specification version 1.x).

A Cryptomatte EXR carries one or more *layer sets*. A set named `crypto_object` is a run of
channel groups `crypto_object00`, `crypto_object01`, ... each holding two (id, coverage) pairs in
its R,G and B,A channels, best coverage first, so N groups store 2N ranks per pixel. An id is the
MurmurHash3_x86_32 of the object's UTF-8 name, reinterpreted as a float32 with its exponent kept
away from 0 and 255. The name-to-id table (the manifest) is JSON in the EXR header under
`cryptomatte/<key>/manifest`, next to `name`, `hash` and `conversion`; `<key>` is the first seven
hex digits of the set name's own id.

The matte for a list of ids is the sum, over every rank, of the coverage where the rank's id is in
the list. Ids are compared as 32-bit patterns, never as floats. Everything here is pure numpy on
the `Raster.layers` the EXR reader already builds; the node in `imaging.py` only wires it up.
Not covered: sidecar manifests (`manifest_file`), Encryptomatte, and half-float files (the format
requires 32-bit ids; a half-float file cannot hold them, and the reader is not asked to guess).
"""
from __future__ import annotations

import json
import re

import numpy as np

_MASK = 0xFFFFFFFF
_SET_LAYER = re.compile(r"^(?P<base>.+?)(?P<rank>\d{2,})$")
METADATA_PREFIXES = ("exr/cryptomatte/", "cryptomatte/")


def murmur3_32(data, seed=0):
    """MurmurHash3_x86_32 of `data` (bytes or str) as an unsigned 32-bit integer."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    c1, c2 = 0xCC9E2D51, 0x1B873593
    h = seed & _MASK
    blocks = len(data) // 4
    for i in range(blocks):
        k = int.from_bytes(data[4 * i:4 * i + 4], "little")
        k = (k * c1) & _MASK
        k = ((k << 15) | (k >> 17)) & _MASK
        k = (k * c2) & _MASK
        h ^= k
        h = ((h << 13) | (h >> 19)) & _MASK
        h = (h * 5 + 0xE6546B64) & _MASK
    tail = data[4 * blocks:]
    k = 0
    if len(tail) >= 3:
        k ^= tail[2] << 16
    if len(tail) >= 2:
        k ^= tail[1] << 8
    if tail:
        k ^= tail[0]
        k = (k * c1) & _MASK
        k = ((k << 15) | (k >> 17)) & _MASK
        k = (k * c2) & _MASK
        h ^= k
    h ^= len(data)
    h ^= h >> 16
    h = (h * 0x85EBCA6B) & _MASK
    h ^= h >> 13
    h = (h * 0xC2B2AE35) & _MASK
    h ^= h >> 16
    return h


def name_to_bits(name):
    """The 32-bit pattern of a name's Cryptomatte id: the hash with a zero or all-ones exponent
    nudged (bit 23 flipped) so the float is never a denormal, infinity or NaN."""
    h = murmur3_32(name)
    if (h >> 23) & 255 in (0, 255):
        h ^= 1 << 23
    return h


def bits_to_float(bits):
    return float(np.array([bits & _MASK], np.uint32).view(np.float32)[0])


def float_to_bits(value):
    return int(np.array([value], np.float32).view(np.uint32)[0])


def name_to_float(name):
    return bits_to_float(name_to_bits(name))


def set_key(name):
    """The header key of a layer set: the first seven hex digits of the set name's id."""
    return ("%08x" % name_to_bits(name))[:7]


def id_token(bits):
    """How a raw id is written in a matte list: `<value>`, twelve significant digits (enough to
    round-trip a float32)."""
    return "<{:.12g}>".format(bits_to_float(bits))


def split_matte_list(text):
    """Comma-separated entries, `\\,` keeping a comma inside a name; empty entries dropped."""
    entries, current, escaped = [], [], False
    for char in text or "":
        if escaped:
            current.append(char)
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == ",":
            entries.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    entries.append("".join(current).strip())
    return [entry for entry in entries if entry]


def escape_name(name):
    return name.replace("\\", "\\\\").replace(",", "\\,")


def parse_matte_list(text, manifest=None):
    """The 32-bit ids a matte list names, as a set. An entry is a name (its manifest id when the
    manifest lists it, otherwise the hash of the name) or a raw id in angle brackets, written as a
    float (`<6.07056271024e-17>`) or as hex bits (`<0x13851ff6>`)."""
    manifest = manifest or {}
    ids = set()
    for entry in split_matte_list(text):
        if entry.startswith("<") and entry.endswith(">"):
            body = entry[1:-1].strip()
            try:
                if body.lower().startswith("0x"):
                    ids.add(int(body, 16) & _MASK)
                else:
                    ids.add(float_to_bits(float(body)))
            except ValueError:
                raise ValueError(f"Cryptomatte: cannot read the raw id {entry!r} in the matte list")
        else:
            ids.add(manifest[entry] if entry in manifest else name_to_bits(entry))
    return ids


def crypto_metadata(attributes):
    """Cryptomatte header entries by set key: `{key: {"name": ..., "hash": ..., "manifest": ...}}`.
    `attributes` maps header attribute names to strings (both the `exr/cryptomatte/` and the
    plain `cryptomatte/` prefixes occur in the wild)."""
    found = {}
    for attribute, value in (attributes or {}).items():
        for prefix in METADATA_PREFIXES:
            if attribute.startswith(prefix):
                key, _, field = attribute[len(prefix):].partition("/")
                if key and field:
                    found.setdefault(key, {})[field] = value
                break
    return found


def parse_manifest(text):
    """`{name: 32-bit id}` from the manifest JSON (values are the id's bits as hex)."""
    if not text:
        return {}
    try:
        table = json.loads(text)
    except ValueError as exc:
        raise ValueError(f"Cryptomatte: the manifest in the file header is not valid JSON ({exc})")
    return {str(name): int(str(value), 16) & _MASK for name, value in table.items()}


def read_sidecar(path):
    """`{name: 32-bit id}` from a sidecar manifest file: the JSON text a header's `manifest` entry holds, kept in its own file."""
    from pathlib import Path
    path = str(path or "").strip()
    if not path:
        raise ValueError("Cryptomatte: Manifest Source is Sidecar but no sidecar file is set")
    file = Path(path).expanduser()
    if not file.is_file():
        raise ValueError(f"Cryptomatte: the sidecar manifest does not exist: {file}")
    return parse_manifest(file.read_text(encoding="utf-8"))


def layer_sets(raster):
    """The Cryptomatte sets on a raster: `{set name: [layer names in rank order]}`. A set is a run
    of layers `<name>NN` that each hold R, G, B and A (the EXR reader stores a group without its
    own alpha as A = 1, so a set is recognised by the set name as well: `crypto...` or a header
    entry naming it)."""
    layers = getattr(raster, "layers", None) or {}
    named = {entry.get("name") for entry in crypto_metadata(getattr(raster, "meta", None)).values()}
    sets = {}
    for layer in layers:
        match = _SET_LAYER.match(layer)
        if match:
            base = match.group("base")
            if base in named or base.lower().startswith("crypto"):
                sets.setdefault(base, []).append(layer)
    return {base: sorted(members, key=lambda layer: int(_SET_LAYER.match(layer).group("rank")))
            for base, members in sorted(sets.items())}


def choose_set(raster, wanted=""):
    """(set name, its layer names) for `wanted`, or the first set when it is empty."""
    sets = layer_sets(raster)
    if not sets:
        raise ValueError("Cryptomatte: the input has no Cryptomatte layers (crypto_object00, ...); "
                         "wire a multichannel EXR Read")
    if not wanted:
        name = next(iter(sets))
        return name, sets[name]
    if wanted in sets:
        return wanted, sets[wanted]
    raise ValueError(f"Cryptomatte: no layer set {wanted!r} on the input; available: {', '.join(sets)}")


def manifest_for(raster, set_name):
    """The name-to-id table of a set from the raster's header entries ({} when there is none)."""
    for entry in crypto_metadata(getattr(raster, "meta", None)).values():
        if entry.get("name") == set_name:
            return parse_manifest(entry.get("manifest", ""))
    return {}


def ranks(raster, members):
    """Yield (id bits, coverage) array pairs for every rank of a set, best first."""
    for layer in members:
        pixels = raster.layers[layer].pixels
        yield np.ascontiguousarray(pixels[..., 0]).view(np.uint32), pixels[..., 1]
        yield np.ascontiguousarray(pixels[..., 2]).view(np.uint32), pixels[..., 3]


def matte(raster, members, ids):
    """The coverage of `ids` (a set of 32-bit ids): the sum over ranks, clamped to 0..1."""
    height, width = raster.layers[members[0]].pixels.shape[:2]
    total = np.zeros((height, width), np.float32)
    wanted = np.array(sorted(ids), np.uint32)
    if wanted.size:
        for bits, coverage in ranks(raster, members):
            total += np.where(np.isin(bits, wanted), coverage, np.float32(0))
    return np.clip(total, 0.0, 1.0)


def _mix(bits):
    """The 32-bit finaliser of MurmurHash3, vectorised: spreads neighbouring ids apart."""
    bits = bits.astype(np.uint32)
    bits ^= bits >> np.uint32(16)
    bits *= np.uint32(0x85EBCA6B)
    bits ^= bits >> np.uint32(13)
    bits *= np.uint32(0xC2B2AE35)
    bits ^= bits >> np.uint32(16)
    return bits


def id_colors(bits):
    """A stable preview colour per id: three bytes of the mixed id, lifted to 0.25..1."""
    mixed = _mix(bits)
    channels = [((mixed >> np.uint32(shift)) & np.uint32(255)).astype(np.float32) / 255.0
                for shift in (0, 8, 16)]
    return np.stack(channels, axis=-1) * 0.75 + 0.25


def preview(raster, members):
    """Every id painted its own colour, weighted by coverage over all ranks (the picture a
    renderer's `crypto_object` preview layer holds)."""
    height, width = raster.layers[members[0]].pixels.shape[:2]
    out = np.zeros((height, width, 3), np.float32)
    for bits, coverage in ranks(raster, members):
        live = coverage > 0
        out += np.where(live[..., None], id_colors(bits) * coverage[..., None], np.float32(0))
    return np.clip(out, 0.0, 1.0)


def pick(raster, members, x, y):
    """The id (32-bit pattern) with the most coverage at pixel (x, y) of the layer arrays, or None
    where nothing is stored."""
    best, best_cover = None, 0.0
    for bits, coverage in ranks(raster, members):
        if coverage[y, x] > best_cover:
            best, best_cover = int(bits[y, x]), float(coverage[y, x])
    return best


def add_token(text, token):
    """`text` with the matte-list entry `token` appended (unchanged when it is already listed)."""
    if split_matte_list(token)[0] in split_matte_list(text):
        return text
    return f"{text.strip()}, {token}" if text.strip() else token


def pick_token(raster, wanted, x, y):
    """The matte-list entry for the object at pixel (x, y) of `raster`'s data window, or None when
    the pixel is outside it or holds no object. `wanted` is the layer set ("" = the first)."""
    name, members = choose_set(raster, wanted)
    x, y = x - raster.data.x, y - raster.data.y
    height, width = raster.layers[members[0]].pixels.shape[:2]
    if not (0 <= x < width and 0 <= y < height):
        return None
    bits = pick(raster, members, x, y)
    return None if bits is None else token_for(raster, name, bits)


def token_for(raster, set_name, bits):
    """The matte-list entry that selects `bits`: its manifest name when the header has one,
    otherwise the raw id."""
    for name, value in manifest_for(raster, set_name).items():
        if value == bits:
            return escape_name(name)
    return id_token(bits)


MERGE_OPERATIONS = ("over", "under", "plus", "replace")


def matte_names(params, sources=None):
    """The eight matte names Encryptomatte writes: with `matte_name_type` auto, the name of the node each matte comes from (`sources`, the
    walk's `_matte_names`; Nuke: "the name of the matte source node is used. For example, Roto1"); with manual, the names typed in `id0`..`id7`."""
    typed = [str(params.get(f"id{i}") or "").strip() for i in range(8)]
    if params.get("matte_name_type", "manual") != "auto":
        return typed
    named = params.get("_matte_names") if sources is None else sources
    if named is None:
        return typed
    return [str(name or "").strip() for name in named] + [""] * (8 - len(named))


def set_layer_names(layer_names, set_name):
    """The rank layers of the set `set_name` among `layer_names` (`<set_name>NN`), in rank order."""
    import re
    found = [(int(m.group(1)), name) for name in layer_names for m in [re.fullmatch(re.escape(set_name) + r"(\d+)", name)] if m]
    return [name for _, name in sorted(found)]


def rank_arrays(layers):
    """(ids, coverage) of the rank layers `layers` (arrays of shape (H, W, 4), best first): ids (H, W, K) uint32, coverage (H, W, K) float32."""
    ids, coverage = [], []
    for layer in layers:
        layer = np.asarray(layer)
        for slot in (0, 2):
            ids.append(np.ascontiguousarray(layer[..., slot]).view(np.uint32))
            coverage.append(layer[..., slot + 1])
    return np.stack(ids, axis=-1), np.stack(coverage, axis=-1).astype(np.float32)


def write_set(entries, existing=None, operation="replace"):
    """The rank layers (a list of (H, W, 4) float32 arrays, group 0 first) of a Cryptomatte set made from `entries`, a list of
    (id bits, coverage array). With `existing` (the rank layers already on the image under this set's name) and an operation other than
    `replace`, the new mattes join the set the way a Merge joins its A input to its B: `over` takes the old coverage down by the new
    mattes' total (B * (1 - a)), `under` takes the new coverage down by the old total (A * (1 - b)), `plus` keeps both. `replace` writes the
    entries alone (the set that was there is gone), padded to an even number of ranks."""
    new_ids = np.broadcast_to(np.array([bits for bits, _ in entries], np.uint32), (*entries[0][1].shape, len(entries)))
    new_cov = np.stack([cov for _, cov in entries], axis=-1).astype(np.float32)
    merging = bool(existing) and operation != "replace"
    if merging:
        old_ids, old_cov = rank_arrays(existing)
        if operation == "over":
            old_cov = old_cov * (1.0 - np.clip(new_cov.sum(axis=-1, keepdims=True), 0.0, 1.0))
        elif operation == "under":
            new_cov = new_cov * (1.0 - np.clip(old_cov.sum(axis=-1, keepdims=True), 0.0, 1.0))
        ids, cov = np.concatenate([old_ids, new_ids], axis=-1), np.concatenate([old_cov, new_cov], axis=-1)
    else:
        ids, cov = new_ids, new_cov
    order = np.argsort(-cov, axis=-1, kind="stable")
    ids, cov = np.take_along_axis(ids, order, axis=-1), np.take_along_axis(cov, order, axis=-1)
    if merging:     # ranks that hold nothing anywhere are dropped, so adding mattes over and over does not grow the set for ever
        keep = max(1, int(np.flatnonzero((cov > 0).any(axis=(0, 1)))[-1]) + 1 if (cov > 0).any() else 1)
        ids, cov = ids[..., :keep], cov[..., :keep]
    levels = max(2, -(-cov.shape[-1] // 2) * 2)     # even, at least 2: one group per two ranks
    pad = levels - cov.shape[-1]
    if pad:
        ids, cov = np.pad(ids, ((0, 0), (0, 0), (0, pad))), np.pad(cov, ((0, 0), (0, 0), (0, pad)))
    layers = []
    for group in range(levels // 2):
        layer = np.zeros((*cov.shape[:2], 4), np.float32)
        layer[..., 0] = np.ascontiguousarray(ids[..., 2 * group]).view(np.float32)
        layer[..., 1] = cov[..., 2 * group]
        layer[..., 2] = np.ascontiguousarray(ids[..., 2 * group + 1]).view(np.float32)
        layer[..., 3] = cov[..., 2 * group + 1]
        layers.append(layer)
    return layers
