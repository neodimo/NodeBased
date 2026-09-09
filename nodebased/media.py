"""OpenImageIO ingest/EXR output. RGBA/layer selection onto the display window."""
from pathlib import Path
import os
import tempfile
import numpy as np
from .color import to_working


def read_media(path, colorspace='Auto', alpha_mode='Auto', layer='', subimage=0):
    import OpenImageIO as oiio
    if not path:
        raise ValueError('Choose an EXR, PNG, JPEG or TIFF file in Read properties')
    ext = Path(path).suffix.lower()
    if ext not in {'.exr', '.png', '.jpg', '.jpeg', '.tif', '.tiff'}:
        raise ValueError('Supported images: EXR, PNG, JPEG and TIFF')
    options = oiio.ImageSpec()
    options.attribute('oiio:UnassociatedAlpha', 1)
    source = oiio.ImageInput.open(str(path), options)
    if source is None:
        raise ValueError(f'Unable to read image: {oiio.geterror()}')
    try:
        if not source.seek_subimage(subimage, 0):
            raise ValueError(f'No subimage/part {subimage} in this file')
        spec = source.spec()
        if spec.deep:
            raise ValueError('Deep EXR is not supported yet; flatten it before reading')
        w, h = spec.full_width or spec.width, spec.full_height or spec.height
        if spec.depth > 1 or spec.full_depth > 1:
            raise ValueError('Volume images are not supported by this 2D reader')
        if any(v <= 0 or v > 8192 for v in (w, h, spec.width, spec.height)):
            raise ValueError('Full-frame reader dimensions must be between 1 and 8192')
        names = list(spec.channelnames)
        prefix = layer + '.' if layer else ''
        rgb = [names.index(prefix + c) if prefix + c in names else None for c in 'RGB']
        # Support luminance/single-channel data as an explicitly selected image.
        if all(index is None for index in rgb):
            mono = next((n for n in (layer, prefix + 'Y', 'Y' if not layer else '') if n and n in names), None)
            if mono is None and not layer and len(names) == 1:
                mono = names[0]
            if mono:
                rgb = [names.index(mono)] * 3
            else:
                raise ValueError('No RGB channels for this layer. Available: ' + ', '.join(names[:32]))
        if any(index is None for index in rgb):
            raise ValueError('Selected layer has incomplete RGB channels')
        alpha = names.index(prefix + 'A') if prefix + 'A' in names else None
        selected = [*rgb, *([alpha] if alpha is not None else [])]
        first, last = min(selected), max(selected) + 1
        if spec.width * spec.height * (last - first) > 128 * 1024 * 1024:
            raise ValueError('Selected channel span exceeds the 512 MiB decode limit')
        pixels = source.read_image(subimage, 0, first, last, oiio.FLOAT)
        if pixels is None:
            raise ValueError('Image decode failed: ' + source.geterror())
        rgba = np.ones((spec.height, spec.width, 4), dtype=np.float32)
        rgba[..., :3] = pixels[..., [index - first for index in rgb]]
        if alpha is not None:
            rgba[..., 3] = pixels[..., alpha - first]
        space = ('Linear Rec.709' if ext == '.exr' else 'sRGB') if colorspace == 'Auto' else colorspace
        associated = (ext == '.exr') if alpha_mode == 'Auto' else alpha_mode == 'Premultiplied'
        rgba = to_working(rgba, space, associated)
        # Clip the data window against the display window, honoring negative origins.
        canvas = np.zeros((h, w, 4), dtype=np.float32)
        x, y = spec.x - spec.full_x, spec.y - spec.full_y
        left, right, top, bottom = max(0, x), min(w, x + spec.width), max(0, y), min(h, y + spec.height)
        if left < right and top < bottom:
            canvas[top:bottom, left:right] = rgba[top-y:bottom-y, left-x:right-x]
        # Respect standard raster orientation metadata. EXR camera-space conventions
        # stay explicit; do not reinterpret EXR coordinate systems here.
        orientation = spec.get_int_attribute('Orientation', 1) if ext != '.exr' else 1
        transforms = {2: lambda a: a[:, ::-1], 3: lambda a: a[::-1, ::-1],
                      4: lambda a: a[::-1], 5: lambda a: a.transpose(1, 0, 2),
                      6: lambda a: np.rot90(a, -1), 7: lambda a: a.transpose(1, 0, 2)[::-1, ::-1],
                      8: lambda a: np.rot90(a, 1)}
        return np.ascontiguousarray(transforms.get(orientation, lambda a: a)(canvas))
    finally:
        source.close()


def selftest():
    """Prove EXR and OCIO binaries actually work here, not merely that they import.

    Packaged builds are the real risk: PyInstaller can ship the Python modules while
    missing their shared libraries or the built-in color configs.
    """
    from .color import CONFIG, config, display_rgb
    import OpenImageIO as oiio
    source = np.array([[[8.0, -0.25, 0.18, 1.0]]], np.float32)
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / 'selftest.exr'
        write_exr(path, source)
        roundtrip = read_media(str(path))
    srgb = display_rgb(np.array([[[0.18, 0.18, 0.18]]], np.float32), 'sRGB')
    return {'oiio': oiio.__version__, 'ocio': config().getName() == CONFIG.removeprefix('ocio://'),
            'exr_roundtrip': bool(np.array_equal(roundtrip, source)),
            'display_transform': bool(abs(float(srgb[0, 0, 0]) - 0.4613) < 2e-3)}


def write_exr(path, frame):
    import OpenImageIO as oiio
    path = Path(path).expanduser().resolve()
    if path.suffix.lower() != '.exr':
        raise ValueError('EXR output path must end in .exr')
    fd, temporary = tempfile.mkstemp(prefix='.' + path.stem, suffix='.exr', dir=path.parent)
    os.close(fd)
    writer = None
    try:
        spec = oiio.ImageSpec(frame.shape[1], frame.shape[0], 4, oiio.FLOAT)
        spec.channelnames = ['R', 'G', 'B', 'A']
        spec.attribute('oiio:ColorSpace', 'Linear Rec.709 (sRGB)')
        spec.attribute('compression', 'zip')
        writer = oiio.ImageOutput.create(temporary)
        if writer is None or not writer.open(temporary, spec):
            raise ValueError('Cannot open EXR output: ' + oiio.geterror())
        if not writer.write_image(np.ascontiguousarray(frame, dtype=np.float32)):
            raise ValueError('Cannot write EXR: ' + writer.geterror())
        if not writer.close():
            raise ValueError('Cannot finalize EXR: ' + writer.geterror())
        writer = None
        os.replace(temporary, path)
    finally:
        if writer:
            writer.close()
        if os.path.exists(temporary):
            os.unlink(temporary)
