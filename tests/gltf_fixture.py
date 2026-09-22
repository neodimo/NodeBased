"""A small glTF 2.0 writer for the reader's tests: enough of the format to build real files."""
import base64
import json
import struct
from pathlib import Path

import numpy as np

COMPONENT = {np.dtype(np.int8): 5120, np.dtype(np.uint8): 5121, np.dtype(np.int16): 5122,
             np.dtype(np.uint16): 5123, np.dtype(np.uint32): 5125, np.dtype(np.float32): 5126}
TYPE = {1: 'SCALAR', 2: 'VEC2', 3: 'VEC3', 4: 'VEC4', 16: 'MAT4'}


def png_bytes(rgba8):
    """Encode an (H, W, 4) uint8 array as PNG through OpenImageIO."""
    import tempfile, os
    import OpenImageIO as oiio
    h, w = rgba8.shape[:2]
    buf = oiio.ImageBuf(oiio.ImageSpec(w, h, 4, 'uint8'))
    buf.set_pixels(oiio.ROI(0, w, 0, h), np.ascontiguousarray(rgba8, np.uint8))
    handle = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    handle.close()
    try:
        assert buf.write(handle.name), oiio.geterror()
        return Path(handle.name).read_bytes()
    finally:
        os.unlink(handle.name)


class Builder:
    def __init__(self):
        self.doc = {'asset': {'version': '2.0'}, 'bufferViews': [], 'accessors': [], 'meshes': [],
                    'nodes': [], 'scenes': [{'nodes': []}], 'scene': 0}
        self.blob = bytearray()

    # --- binary data ---
    def view(self, data, stride=None):
        while len(self.blob) % 4:
            self.blob.append(0)
        entry = {'buffer': 0, 'byteOffset': len(self.blob), 'byteLength': len(data)}
        if stride:
            entry['byteStride'] = stride
        self.blob += data
        self.doc['bufferViews'].append(entry)
        return len(self.doc['bufferViews']) - 1

    def accessor(self, array, normalized=False, view=None, offset=0, stride=None):
        array = np.asarray(array)
        width = 1 if array.ndim == 1 else array.shape[1]
        if view is None:
            view = self.view(np.ascontiguousarray(array).tobytes(), stride)
        entry = {'bufferView': view, 'byteOffset': offset, 'componentType': COMPONENT[array.dtype],
                 'count': len(array), 'type': TYPE[width]}
        if normalized:
            entry['normalized'] = True
        if width == 3 and array.dtype == np.float32:
            entry['min'] = array.min(0).tolist(); entry['max'] = array.max(0).tolist()
        self.doc['accessors'].append(entry)
        return len(self.doc['accessors']) - 1

    # --- materials ---
    def image(self, data, mime='image/png', uri=None):
        images = self.doc.setdefault('images', [])
        if uri is None:
            images.append({'bufferView': self.view(data), 'mimeType': mime})
        else:
            images.append({'uri': uri, 'mimeType': mime} if mime else {'uri': uri})
        textures = self.doc.setdefault('textures', [])
        textures.append({'source': len(images) - 1})
        return len(textures) - 1

    def material(self, factor=None, texture=None, alpha_mode=None, texcoord=None, transform=None,
                 glossy=False):
        entry = {}
        reference = None
        if texture is not None:
            reference = {'index': texture}
            if texcoord is not None:
                reference['texCoord'] = texcoord
            if transform is not None:
                reference['extensions'] = {'KHR_texture_transform': transform}
        if glossy:
            block = {}
            if factor is not None:
                block['diffuseFactor'] = list(factor)
            if reference is not None:
                block['diffuseTexture'] = reference
            entry['extensions'] = {'KHR_materials_pbrSpecularGlossiness': block}
        else:
            pbr = {}
            if factor is not None:
                pbr['baseColorFactor'] = list(factor)
            if reference is not None:
                pbr['baseColorTexture'] = reference
            entry['pbrMetallicRoughness'] = pbr
        if alpha_mode is not None:
            entry['alphaMode'] = alpha_mode
        self.doc.setdefault('materials', []).append(entry)
        return len(self.doc['materials']) - 1

    # --- meshes and nodes ---
    def primitive(self, positions, indices=None, normals=None, uvs=None, material=None, mode=None,
                  uvs1=None, position_accessor=None):
        attributes = {'POSITION': position_accessor if position_accessor is not None
                      else self.accessor(np.asarray(positions, np.float32))}
        if normals is not None:
            attributes['NORMAL'] = self.accessor(np.asarray(normals, np.float32))
        if uvs is not None:
            attributes['TEXCOORD_0'] = uvs if isinstance(uvs, int) else self.accessor(np.asarray(uvs, np.float32))
        if uvs1 is not None:
            attributes['TEXCOORD_1'] = self.accessor(np.asarray(uvs1, np.float32))
        entry = {'attributes': attributes}
        if indices is not None:
            entry['indices'] = indices if isinstance(indices, int) else self.accessor(np.asarray(indices))
        if material is not None:
            entry['material'] = material
        if mode is not None:
            entry['mode'] = mode
        return entry

    def mesh(self, *primitives, name=None):
        entry = {'primitives': list(primitives)}
        if name:
            entry['name'] = name
        self.doc['meshes'].append(entry)
        return len(self.doc['meshes']) - 1

    def node(self, mesh=None, children=(), name=None, translation=None, rotation=None, scale=None,
             matrix=None, root=True):
        entry = {}
        if mesh is not None:
            entry['mesh'] = mesh
        if children:
            entry['children'] = list(children)
        if name:
            entry['name'] = name
        for key, value in (('translation', translation), ('rotation', rotation), ('scale', scale),
                           ('matrix', matrix)):
            if value is not None:
                entry[key] = list(value)
        self.doc['nodes'].append(entry)
        index = len(self.doc['nodes']) - 1
        if root:
            self.doc['scenes'][0]['nodes'].append(index)
        return index

    # --- output ---
    def glb(self):
        doc = dict(self.doc)
        doc['buffers'] = [{'byteLength': len(self.blob)}]
        text = json.dumps(doc).encode()
        text += b' ' * (-len(text) % 4)
        binary = bytes(self.blob) + b'\0' * (-len(self.blob) % 4)
        body = struct.pack('<II', len(text), 0x4E4F534A) + text
        if binary:
            body += struct.pack('<II', len(binary), 0x004E4942) + binary
        return b'glTF' + struct.pack('<II', 2, 12 + len(body)) + body

    def write_glb(self, path):
        Path(path).write_bytes(self.glb())
        return Path(path)

    def write_gltf(self, path, external='buffer.bin'):
        """A .gltf beside its buffer, or with the buffer inlined as a data URI when external is None."""
        doc = dict(self.doc)
        if external:
            Path(path).parent.joinpath(external).write_bytes(bytes(self.blob))
            doc['buffers'] = [{'byteLength': len(self.blob), 'uri': external}]
        else:
            doc['buffers'] = [{'byteLength': len(self.blob),
                               'uri': 'data:application/octet-stream;base64,' + base64.b64encode(bytes(self.blob)).decode()}]
        Path(path).write_text(json.dumps(doc), encoding='utf-8')
        return Path(path)


QUAD = np.array([(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)], np.float32)
QUAD_INDICES = np.array([0, 1, 2, 0, 2, 3], np.uint16)
QUAD_UVS = np.array([(0, 1), (1, 1), (1, 0), (0, 0)], np.float32)  # glTF: V grows downwards
QUAD_NORMALS = np.array([(0, 0, 1)] * 4, np.float32)


def quad_builder(**material):
    """One unit quad facing +Z on a single root node, optionally with a material."""
    b = Builder()
    index = b.material(**material) if material else None
    b.node(mesh=b.mesh(b.primitive(QUAD, QUAD_INDICES, QUAD_NORMALS, QUAD_UVS, material=index)))
    return b
