"""Read glTF 2.0 meshes (.gltf and .glb) as a scene3d.Scene.

Pure numpy and the standard library, so no new dependency: an in-house reader like alembicio.
Pixal3D, WorldSculpt and SAM 3D all write GLB, which is why this exists. As usdio does, world
transforms are baked into the vertices. A material contributes its base colour (factor and texture)
and nothing else: scene3d has no metallic, roughness, normal or emissive maps to give them to.

glTF is right-handed, +Y up, metres, with counter-clockwise front faces, the same as scene3d, so no
axis conversion is needed. Its UV origin is the top-left of the image where scene3d's is the
bottom-left, so V is flipped once here.
"""
from __future__ import annotations

import base64
import json
import os
import struct
import tempfile
from functools import lru_cache
from pathlib import Path
from urllib.parse import unquote

import numpy as np

from . import scene3d as s

# Longest side kept for a base colour texture. A 4096^2 texture is 268 MB as float32 RGBA, and
# Pixal3D writes them; anything larger is box-filtered down by an integer factor.
MAX_TEXTURE = 2048

_MAGIC = b'glTF'
_JSON_CHUNK, _BIN_CHUNK = 0x4E4F534A, 0x004E4942
_DTYPES = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32,
           5126: np.float32}
_WIDTHS = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4, 'MAT2': 4, 'MAT3': 9, 'MAT4': 16}
_TRIANGLES, _STRIP, _FAN = 4, 5, 6
# Required extensions this reader satisfies. Any other entry in extensionsRequired is refused by
# name rather than silently misread (Draco and meshopt compression, KTX2 textures).
_HANDLED_REQUIRED = frozenset(('KHR_mesh_quantization', 'KHR_texture_transform',
                               'KHR_materials_pbrSpecularGlossiness'))
_SUFFIXES = {'image/png': '.png', 'image/jpeg': '.jpg'}


def fingerprint(path) -> list:
    """Identity of the file, plus any external buffers and images a .gltf refers to."""
    if not path:
        raise ValueError('ReadGLTF3D: choose a glTF or GLB file')
    file = Path(path)
    try:
        stat = file.stat()
    except OSError:
        raise ValueError(f'ReadGLTF3D: cannot read {path}') from None
    out = [str(file.resolve()), stat.st_size, stat.st_mtime_ns]
    if file.suffix.lower() == '.gltf':
        try:
            doc = json.loads(file.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            return out  # load_scene reports the real problem
        for item in list(doc.get('buffers', ())) + list(doc.get('images', ())):
            uri = item.get('uri') if isinstance(item, dict) else None
            if uri and not uri.startswith('data:'):
                try:
                    external = (file.parent / unquote(uri)).stat()
                    out += [uri, external.st_size, external.st_mtime_ns]
                except OSError:
                    out += [uri, None]
    return out


def load_scene(path, root='') -> s.Scene:
    """Meshes of the default scene as world-space geometry; `root` names a node to load instead.

    Points and lines are skipped. Cameras, lights, skins, morph targets, animations and vertex
    colours are not loaded. A material's base colour factor and texture become the geometry's
    colour and texture; alpha is used only when the material's alphaMode is BLEND or MASK.
    """
    return _load(tuple(fingerprint(path)), root or '')


@lru_cache(maxsize=4)
def _load(identity, root):
    path = Path(identity[0])
    name = path.name
    try:
        doc, binary = _document(path)
        buffers = _buffers(doc, binary, path.parent, name)
    except (OSError, ValueError, struct.error, KeyError, TypeError) as error:
        raise ValueError(f'{name}: {error}') from None
    required = set(doc.get('extensionsRequired', ())) - _HANDLED_REQUIRED
    if required:
        raise ValueError(f'{name}: requires unsupported glTF extensions: {", ".join(sorted(required))}')
    materials = {}
    textures = {}
    geometries = []
    total = 0

    def material(index):
        if index not in materials:
            materials[index] = _material(doc, buffers, path.parent, index, textures)
        return materials[index]

    matched = False
    for node_index, world in _walk(doc, root, name):
        matched = True
        node = doc['nodes'][node_index]
        if 'mesh' not in node:
            continue
        for primitive in doc['meshes'][node['mesh']].get('primitives', ()):
            try:
                geometry = _geometry(doc, buffers, primitive, world, material, s.MAX_TRIANGLES - total)
            except (ValueError, KeyError, TypeError, struct.error) as error:
                raise ValueError(f'{name}: mesh {node["mesh"]}: {error}') from None
            if geometry is not None:
                geometries.append(geometry)
                total += len(geometry.triangles)
    if root and not matched:
        raise ValueError(f'{name}: no node named {root!r}')
    return s.Scene(tuple(geometries))


# --- file structure ------------------------------------------------------------------------------

def _document(path):
    data = path.read_bytes()
    if data[:4] != _MAGIC:
        try:
            return json.loads(data.decode('utf-8')), None
        except ValueError:
            raise ValueError('not a glTF JSON document or a GLB file') from None
    if len(data) < 12:
        raise ValueError('truncated GLB header')
    version, length = struct.unpack_from('<II', data, 4)
    if version != 2:
        raise ValueError(f'GLB version {version} is not supported (2 only)')
    doc = binary = None
    offset = 12
    while offset + 8 <= min(length, len(data)):
        size, kind = struct.unpack_from('<II', data, offset)
        offset += 8
        chunk = data[offset:offset + size]
        if len(chunk) < size:
            raise ValueError('truncated GLB chunk')
        offset += size
        if kind == _JSON_CHUNK and doc is None:
            doc = json.loads(chunk.decode('utf-8'))
        elif kind == _BIN_CHUNK and binary is None:
            binary = chunk
    if doc is None:
        raise ValueError('GLB has no JSON chunk')
    return doc, binary


def _uri_bytes(uri, folder):
    if uri.startswith('data:'):
        header, _, payload = uri.partition(',')
        if not header.endswith(';base64'):
            raise ValueError('only base64 data URIs are supported')
        return base64.b64decode(payload)
    if '://' in uri:
        raise ValueError(f'remote URIs are not supported: {uri}')
    try:
        return (folder / unquote(uri)).read_bytes()
    except OSError:
        raise ValueError(f'cannot read {uri}') from None


def _buffers(doc, binary, folder, name):
    out = []
    for index, buffer in enumerate(doc.get('buffers', ())):
        uri = buffer.get('uri')
        if uri is None:
            if binary is None and int(buffer.get('byteLength', 0)):
                raise ValueError(f'buffer {index} has no uri and the file has no BIN chunk')
            blob = binary or b''
        else:
            blob = _uri_bytes(uri, folder)
        if len(blob) < int(buffer.get('byteLength', 0)):
            raise ValueError(f'buffer {index} is shorter than its byteLength')
        out.append(blob)
    return out


def _accessor(doc, buffers, index):
    accessor = doc['accessors'][index]
    if 'sparse' in accessor:
        raise ValueError(f'accessor {index}: sparse accessors are not supported')
    dtype = np.dtype(_DTYPES[accessor['componentType']])
    width = _WIDTHS[accessor['type']]
    count = int(accessor['count'])
    if 'bufferView' not in accessor:
        return np.zeros((count, width), dtype)  # the specification's all-zeros accessor
    view = doc['bufferViews'][accessor['bufferView']]
    blob = buffers[view['buffer']]
    element = dtype.itemsize * width
    stride = int(view.get('byteStride', element))
    if stride < element:
        raise ValueError(f'accessor {index}: byteStride {stride} is smaller than its element')
    view_start = int(view.get('byteOffset', 0))
    start = view_start + int(accessor.get('byteOffset', 0))
    span = stride * (count - 1) + element if count else 0
    if start + span > view_start + int(view['byteLength']) or start + span > len(blob):
        raise ValueError(f'accessor {index} overruns its buffer view')
    if count == 0:
        return np.zeros((0, width), dtype)
    if stride == element:
        array = np.frombuffer(blob, dtype, count * width, start).reshape(count, width)
    else:
        raw = np.frombuffer(blob, np.uint8, span, start)
        rows = np.lib.stride_tricks.as_strided(raw, shape=(count, element), strides=(stride, 1))
        array = np.ascontiguousarray(rows).view(dtype).reshape(count, width)
    if accessor.get('normalized') and dtype.kind in 'iu':
        array = np.maximum(array.astype(np.float32) / np.iinfo(dtype).max, -1.0)
    return array


# --- node hierarchy ------------------------------------------------------------------------------

def _local(node):
    if 'matrix' in node:
        # Column-major in the file; transposed into the column-vector matrix scene3d uses.
        return np.asarray(node['matrix'], np.float64).reshape(4, 4).T
    x, y, z, w = (float(v) for v in node.get('rotation', (0.0, 0.0, 0.0, 1.0)))
    rotation = np.array((
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y))), np.float64)
    matrix = np.eye(4)
    matrix[:3, :3] = rotation @ np.diag([float(v) for v in node.get('scale', (1.0, 1.0, 1.0))])
    matrix[:3, 3] = [float(v) for v in node.get('translation', (0.0, 0.0, 0.0))]
    return matrix


def _roots(doc):
    nodes = doc.get('nodes', ())
    scenes = doc.get('scenes')
    if scenes:
        return list(scenes[int(doc.get('scene', 0))].get('nodes', ()))
    children = {child for node in nodes for child in node.get('children', ())}
    return [index for index in range(len(nodes)) if index not in children]


def _walk(doc, root, name):
    """Yield (node index, world matrix) for every node to load, depth first."""
    nodes = doc.get('nodes', ())
    stack = [(index, np.eye(4), not root, ()) for index in reversed(_roots(doc))]
    while stack:
        index, parent, selected, path = stack.pop()
        if index in path:
            raise ValueError(f'{name}: node {index} is its own ancestor')
        if not 0 <= index < len(nodes):
            raise ValueError(f'{name}: node {index} does not exist')
        node = nodes[index]
        world = parent @ _local(node)
        selected = selected or node.get('name') == root
        if selected:
            yield index, world
        for child in reversed(node.get('children', ())):
            stack.append((child, world, selected, path + (index,)))


# --- meshes --------------------------------------------------------------------------------------

def _geometry(doc, buffers, primitive, world, material, remaining):
    mode = int(primitive.get('mode', _TRIANGLES))
    if mode not in (_TRIANGLES, _STRIP, _FAN):
        return None  # points and lines have no surface to render
    attributes = primitive['attributes']
    if 'POSITION' not in attributes:
        raise ValueError('a primitive has no POSITION')
    positions = np.asarray(_accessor(doc, buffers, attributes['POSITION']), np.float32)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError('POSITION is not VEC3')
    if not np.isfinite(positions).all():
        raise ValueError('non-finite vertex positions')
    count = len(positions)
    if 'indices' in primitive:
        indices = np.asarray(_accessor(doc, buffers, primitive['indices'])).reshape(-1).astype(np.int64)
    else:
        indices = np.arange(count, dtype=np.int64)
    if len(indices) < 3:
        return None
    if mode == _TRIANGLES:
        triangles = indices[:len(indices) - len(indices) % 3].reshape(-1, 3)
    elif mode == _STRIP:
        first = np.arange(len(indices) - 2)
        odd = first % 2 == 1
        triangles = np.stack((indices[first], indices[first + 1], indices[first + 2]), axis=1)
        triangles[odd, 0], triangles[odd, 1] = triangles[odd, 1].copy(), triangles[odd, 0].copy()
    else:
        first = np.arange(1, len(indices) - 1)
        triangles = np.stack((np.full(len(first), indices[0]), indices[first], indices[first + 1]), axis=1)
    if len(triangles) and (triangles.min() < 0 or triangles.max() >= count):
        raise ValueError('a triangle references a missing vertex (index out of range)')
    color, texture, texcoord, uv_transform = material(primitive.get('material'))
    uvs = None
    uv_source = attributes.get(f'TEXCOORD_{texcoord}')
    if uv_source is not None:
        uvs = np.asarray(_accessor(doc, buffers, uv_source), np.float32)
        if uvs.ndim != 2 or uvs.shape[1] != 2 or len(uvs) != count:
            raise ValueError('TEXCOORD does not match POSITION')
        if uv_transform:
            uvs = _transform_uvs(uvs, uv_transform)
        uvs = np.ascontiguousarray(np.stack((uvs[:, 0], 1.0 - uvs[:, 1]), axis=1), np.float32)
    elif texture is not None:
        texture = None  # a texture with no coordinates to sample it by
    normals = None
    if 'NORMAL' in attributes:
        normals = np.asarray(_accessor(doc, buffers, attributes['NORMAL']), np.float32)
        if normals.ndim != 2 or normals.shape[1] != 3 or len(normals) != count:
            raise ValueError('NORMAL does not match POSITION')
    vertices, normals = _world(world, positions, normals)
    if np.linalg.det(world[:3, :3]) < 0:
        triangles = triangles[:, (0, 2, 1)]  # a mirroring transform reverses the winding
    corners = vertices[triangles]
    area = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    triangles = triangles[np.any(area != 0, axis=1)]
    if not len(triangles):
        return None
    if len(triangles) > remaining:
        raise ValueError(f'more than {s.MAX_TRIANGLES} triangles; the CPU reference renderer refuses meshes this large')
    return s.Geometry(vertices, np.ascontiguousarray(triangles, np.int32), color,
                      uvs=uvs, normals=normals, texture=texture)


def _world(matrix, vertices, normals):
    linear = matrix[:3, :3]
    vertices = (vertices.astype(np.float64) @ linear.T + matrix[:3, 3]).astype(np.float32)
    if normals is not None:
        try:
            normals = normals.astype(np.float64) @ np.linalg.inv(linear)
        except np.linalg.LinAlgError:
            raise ValueError('cannot transform normals with a singular node transform') from None
        normals = (normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8)).astype(np.float32)
    return vertices, normals


def _transform_uvs(uvs, transform):
    # KHR_texture_transform: uv' = T * R * S * uv, in glTF's top-left UV space.
    offset = np.asarray(transform.get('offset', (0.0, 0.0)), np.float64)
    scale = np.asarray(transform.get('scale', (1.0, 1.0)), np.float64)
    angle = float(transform.get('rotation', 0.0))
    scaled = uvs.astype(np.float64) * scale
    c, sn = np.cos(angle), np.sin(angle)
    rotated = np.stack((c * scaled[:, 0] + sn * scaled[:, 1], -sn * scaled[:, 0] + c * scaled[:, 1]), axis=1)
    return (rotated + offset).astype(np.float32)


# --- materials -----------------------------------------------------------------------------------

def _material(doc, buffers, folder, index, textures):
    """(colour, texture, texcoord set, KHR_texture_transform) of a material; None is the default."""
    from .color import to_working
    factor = [1.0, 1.0, 1.0, 1.0]
    reference = None
    alpha_mode = 'OPAQUE'
    if index is not None:
        material = doc['materials'][index]
        pbr = material.get('pbrMetallicRoughness')
        glossy = material.get('extensions', {}).get('KHR_materials_pbrSpecularGlossiness')
        if pbr is not None or glossy is None:
            pbr = pbr or {}
            factor = list(pbr.get('baseColorFactor', factor))
            reference = pbr.get('baseColorTexture')
        else:
            factor = list(glossy.get('diffuseFactor', factor))
            reference = glossy.get('diffuseTexture')
        alpha_mode = material.get('alphaMode', 'OPAQUE')
    if len(factor) != 4:
        raise ValueError(f'material {index}: base colour factor is not RGBA')
    # Factors are linear in the sRGB primaries; scene3d colours live in the working space.
    rgb = to_working(np.array([[[*factor[:3], 1.0]]], np.float32), 'Linear Rec.709')[0, 0, :3]
    alpha = float(factor[3]) if alpha_mode != 'OPAQUE' else 1.0
    color = (float(rgb[0]), float(rgb[1]), float(rgb[2]), alpha)
    texture, texcoord, transform = None, 0, None
    if reference is not None:
        texcoord = int(reference.get('texCoord', 0))
        transform = reference.get('extensions', {}).get('KHR_texture_transform')
        if transform and 'texCoord' in transform:
            texcoord = int(transform['texCoord'])
        source = doc['textures'][int(reference['index'])].get('source')
        if source is not None:
            key = (int(source), alpha_mode == 'OPAQUE')
            if key not in textures:
                textures[key] = _image(doc, buffers, folder, key[0], key[1])
            texture = textures[key]
    return color, texture, texcoord, transform


def _image(doc, buffers, folder, index, opaque):
    image = doc['images'][index]
    try:
        if 'bufferView' in image:
            view = doc['bufferViews'][image['bufferView']]
            start = int(view.get('byteOffset', 0))
            data = bytes(buffers[view['buffer']][start:start + int(view['byteLength'])])
            uri = ''
        else:
            uri = image.get('uri', '')
            data = _uri_bytes(uri, folder)
        mime = image.get('mimeType')
        if not mime and uri.startswith('data:'):
            mime = uri[5:].partition(';')[0]
        suffix = (_SUFFIXES.get(mime)
                  or (Path(unquote(uri)).suffix.lower() if not uri.startswith('data:') else '') or '.png')
        rgba = _decode(data, suffix)
    except (ValueError, KeyError, TypeError) as error:
        raise ValueError(f'image {index}: {error}') from None
    if opaque:
        rgba[..., 3] = 1.0
    from .color import to_working
    return to_working(_shrink(rgba), 'sRGB', associated=False)


def _decode(data, suffix):
    """Straight float32 RGBA, row 0 at the top, from PNG or JPEG bytes via OpenImageIO."""
    import OpenImageIO as oiio
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        handle.write(data)
        temporary = handle.name
    try:
        options = oiio.ImageSpec()
        options.attribute('oiio:UnassociatedAlpha', 1)
        source = oiio.ImageInput.open(temporary, options)
        if source is None:
            raise ValueError(f'cannot decode the texture: {oiio.geterror()}')
        try:
            spec = source.spec()
            pixels = source.read_image('float')
        finally:
            source.close()
    finally:
        os.unlink(temporary)
    if pixels is None:
        raise ValueError(f'cannot decode the texture: {oiio.geterror()}')
    channels = spec.nchannels
    pixels = np.asarray(pixels, np.float32).reshape(spec.height, spec.width, channels)
    rgba = np.ones((spec.height, spec.width, 4), np.float32)
    if channels >= 3:
        rgba[..., :3] = pixels[..., :3]
        if channels >= 4:
            rgba[..., 3] = pixels[..., 3]
    else:
        rgba[..., :3] = pixels[..., :1]
        if channels == 2:
            rgba[..., 3] = pixels[..., 1]
    return rgba


def _shrink(rgba):
    height, width = rgba.shape[:2]
    factor = -(-max(height, width) // MAX_TEXTURE)
    if factor <= 1:
        return rgba
    h, w = max(height // factor, 1), max(width // factor, 1)
    pooled = rgba[:h * factor, :w * factor].reshape(h, factor, w, factor, 4).mean(axis=(1, 3))
    return np.ascontiguousarray(pooled, np.float32)
