"""OpenImageIO ingest/EXR output. RGBA/layer selection onto the display window."""
from pathlib import Path
import os
import re
import tempfile
import numpy as np
from .color import to_working, WORKING

# Files that name their own colour space get taken at their word. OIIO normalises whatever
# string is in the file to a colour-interop ID before we see it ('ACEScg' and 'scene_linear'
# both arrive as 'lin_ap1_scene'), so those IDs are the real keys; the spelled-out config
# names are kept as a fallback in case a future OIIO stops normalising. Values are
# `color.INPUT_SPACES` keys. Anything unlisted falls back to the extension rule.
_TAGGED_SPACES = {'lin_ap1_scene': 'ACEScg', 'acescg': 'ACEScg',
                  'lin_ap0_scene': 'ACES2065-1', 'aces2065-1': 'ACES2065-1',
                  'lin_rec709_scene': 'Linear Rec.709', 'linear rec.709 (srgb)': 'Linear Rec.709',
                  'srgb_rec709_scene': 'sRGB', 'srgb encoded rec.709 (srgb)': 'sRGB', 'srgb': 'sRGB',
                  'data': 'Raw', 'raw': 'Raw'}


def auto_space(ext, spec):
    """Resolve `colorspace='Auto'` for one decoded file.

    An EXR is *not* assumed to already be in the working space. Untagged EXRs are read as
    linear Rec.709 — the long-standing convention for untagged scene-linear files, and what
    NodeBased itself wrote before the working space moved to ACEScg — and converted on
    ingest like any other input.
    """
    tag = (spec.get_string_attribute('oiio:ColorSpace') or '').strip().lower()
    if tag in _TAGGED_SPACES:
        return _TAGGED_SPACES[tag]
    return 'Linear Rec.709' if ext == '.exr' else 'sRGB'


# A sequence path carries exactly one frame token: printf padding (%04d, %d) or a run of hashes
# (####). Everything else is a still. See docs/TIME_MODEL.md — Read maps timeline frame to source
# frame itself, so this module only has to turn (pattern, source frame) into a concrete path.
_PRINTF_TOKEN = re.compile(r'%(0(\d+))?d')
_HASH_TOKEN = re.compile(r'#+')


def parse_sequence(path):
    """Split a sequence pattern into (prefix, padding, suffix), or None when the path is a still."""
    text = str(path)
    tokens = [*_PRINTF_TOKEN.finditer(text), *_HASH_TOKEN.finditer(text)]
    if not tokens:
        return None
    if len(tokens) != 1:
        raise ValueError(f'Sequence path has more than one frame token: {text}')
    match = tokens[0]
    if match.re is _PRINTF_TOKEN:
        padding = int(match.group(2)) if match.group(2) else 1
    else:
        padding = len(match.group(0))
    if padding > 20:
        raise ValueError('Sequence frame padding must be at most 20 digits')
    return text[:match.start()], padding, text[match.end():]


def is_sequence(path):
    return bool(path) and parse_sequence(path) is not None


def sequence_path(path, frame):
    """Concrete file for one source frame. Stills ignore the frame entirely."""
    parsed = parse_sequence(path)
    if parsed is None:
        return str(path)
    prefix, padding, suffix = parsed
    # Negative frames keep their sign outside the zero padding, matching printf's %04d.
    digits = f'{abs(int(frame)):0{padding}d}'
    return f'{prefix}{"-" if frame < 0 else ""}{digits}{suffix}'


def scan_sequence(path):
    """Frames that actually exist on disk for a pattern, ascending. Empty for stills or no matches."""
    parsed = parse_sequence(path)
    if parsed is None:
        return []
    prefix, padding, suffix = parsed
    directory = Path(prefix).parent if Path(prefix).parent != Path('') else Path('.')
    if not directory.is_dir():
        return []
    stem = Path(prefix).name
    # Accept over-padded frames (frame 100000 in a %04d sequence) the way shotgun-era tools do.
    matcher = re.compile(f'{re.escape(stem)}(-?\\d{{{padding},}}){re.escape(suffix)}$')
    frames = []
    for entry in directory.iterdir():
        found = matcher.match(entry.name)
        if found:
            frames.append(int(found.group(1)))
    return sorted(frames)


def nearest_sequence_path(path, frame):
    """Existing sequence member nearest to `frame`, with earlier winning an equal-distance tie."""
    available = scan_sequence(path)
    if not available:
        return None
    nearest = min(available, key=lambda value: (abs(value - frame), value))
    return sequence_path(path, nearest)


def resolve_source_path(path, frame, missing='error'):
    """Map an already-offset source frame onto a real file, applying the missing-frame policy.

    Returns (path, exists). A "black" miss returns (None, False) so the caller can synthesize.
    """
    if not is_sequence(path):
        return str(path), Path(path).is_file() if path else False
    candidate = sequence_path(path, frame)
    if Path(candidate).is_file():
        return candidate, True
    if missing == 'hold':
        nearest = nearest_sequence_path(path, frame)
        if nearest:
            return nearest, True
        raise ValueError(f'No frames found for sequence {path}')
    if missing == 'black':
        return None, False
    raise ValueError(f'Missing frame {frame}: {candidate}')


def read_media(path, colorspace='Auto', alpha_mode='Auto', layer='', subimage=0):
    """The display-window array. Overscan, if the file carries any, is dropped here.

    Kept as the shape-compatible entry point for callers that genuinely want a frame. Anything
    that must preserve a data window larger than the frame calls `read_media_raster` instead —
    see docs/BOUNDING_BOX.md.
    """
    return read_media_raster(path, colorspace, alpha_mode, layer, subimage).to_display()


def read_media_raster(path, colorspace='Auto', alpha_mode='Auto', layer='', subimage=0):
    """Decode into a `Raster`, preserving the file's data window even when it exceeds the frame.

    OpenEXR stores `dataWindow` and `displayWindow` separately precisely so a render can carry
    margin outside the frame. Clipping that away at ingest — which this function used to do —
    means no node downstream can ever recover it, so a Transform that pans the plate reveals black
    instead of the overscan that was rendered for exactly that purpose.
    """
    import OpenImageIO as oiio
    from .raster import Raster
    from .tiers import Region
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
        space = auto_space(ext, spec) if colorspace == 'Auto' else colorspace
        associated = (ext == '.exr') if alpha_mode == 'Auto' else alpha_mode == 'Premultiplied'
        rgba = to_working(rgba, space, associated)
        # Both windows in display-window-relative coordinates: the display window is rebased to
        # (0, 0) and the data window keeps its offset, which may be negative or reach past w/h.
        display = Region(0, 0, w, h)
        data = Region(spec.x - spec.full_x, spec.y - spec.full_y, spec.width, spec.height)
        # Respect standard raster orientation metadata. EXR camera-space conventions
        # stay explicit; do not reinterpret EXR coordinate systems here.
        orientation = spec.get_int_attribute('Orientation', 1) if ext != '.exr' else 1
        transforms = {2: lambda a: a[:, ::-1], 3: lambda a: a[::-1, ::-1],
                      4: lambda a: a[::-1], 5: lambda a: a.transpose(1, 0, 2),
                      6: lambda a: np.rot90(a, -1), 7: lambda a: a.transpose(1, 0, 2)[::-1, ::-1],
                      8: lambda a: np.rot90(a, 1)}
        if orientation != 1:
            # Rotating or transposing a data window that differs from its display window has no
            # single right answer, and no format that carries orientation also carries overscan.
            # Flatten to the frame first, then orient — the historical behaviour, kept exactly.
            oriented = np.ascontiguousarray(
                transforms.get(orientation, lambda a: a)(Raster(rgba, data, display).fit(display)))
            square = Region(0, 0, oriented.shape[1], oriented.shape[0])
            return Raster(oriented, square, square)
        return Raster(np.ascontiguousarray(rgba), data, display)
    finally:
        source.close()


def read_media_region(path, region, colorspace='Auto', alpha_mode='Auto', layer='', subimage=0):
    """Read only the requested display-relative rectangle from an image.

    This is deliberately a separate API from :func:`read_media_raster`: callers that need an
    entire image retain the simple path, while the tile executor can prove that it acquired only
    the source scanlines intersecting its requested data-window rectangle.  Pixels outside a
    file's data window are transparent black; no edge extension or display-window clipping occurs.
    """
    import OpenImageIO as oiio
    from .raster import Raster
    from .tiers import Region
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
        if spec.deep or spec.depth > 1 or spec.full_depth > 1:
            raise ValueError('Deep or volume images are not supported by this 2D reader')
        display = Region(0, 0, spec.full_width or spec.width, spec.full_height or spec.height)
        data = Region(spec.x - spec.full_x, spec.y - spec.full_y, spec.width, spec.height)
        requested = Region(int(region.x), int(region.y), int(region.width), int(region.height))
        out = np.zeros((requested.height, requested.width, 4), np.float32)
        overlap = requested.intersect(data)
        if overlap.is_empty:
            return Raster(out, requested, display)
        names = list(spec.channelnames)
        prefix = layer + '.' if layer else ''
        rgb = [names.index(prefix + c) if prefix + c in names else None for c in 'RGB']
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
        # OIIO image coordinates include full-window origin. `read_scanlines` is bounded I/O;
        # this is the essential difference from `read_image` in the full-frame path.
        y0 = overlap.y + spec.full_y
        y1 = overlap.bottom + spec.full_y
        pixels = source.read_scanlines(subimage, 0, y0, y1, 0, first, last, oiio.FLOAT)
        if pixels is None:
            raise ValueError('Region decode failed: ' + source.geterror())
        # ImageInput scanline reads are vertically bounded but span the file's data-window width.
        # Crop horizontally before channel conversion; tiled EXRs use `read_tiles` in future for
        # true two-dimensional I/O, but this still prevents a full-frame scanline decode.
        x0 = overlap.x - data.x
        pixels = pixels[:, x0:x0 + overlap.width]
        rgba = np.ones((overlap.height, overlap.width, 4), np.float32)
        rgba[..., :3] = pixels[..., [index - first for index in rgb]]
        if alpha is not None:
            rgba[..., 3] = pixels[..., alpha - first]
        space = auto_space(ext, spec) if colorspace == 'Auto' else colorspace
        associated = (ext == '.exr') if alpha_mode == 'Auto' else alpha_mode == 'Premultiplied'
        rgba = to_working(rgba, space, associated)
        oy, ox = overlap.y - requested.y, overlap.x - requested.x
        out[oy:oy + overlap.height, ox:ox + overlap.width] = rgba
        return Raster(out, requested, display)
    finally:
        source.close()


def read_media_bounds(path, subimage=0):
    """Return (data_window, display_window) without decoding source pixels."""
    import OpenImageIO as oiio
    from .tiers import Region
    source = oiio.ImageInput.open(str(path))
    if source is None:
        raise ValueError(f'Unable to read image: {oiio.geterror()}')
    try:
        if not source.seek_subimage(subimage, 0):
            raise ValueError(f'No subimage/part {subimage} in this file')
        spec = source.spec()
        return (Region(spec.x - spec.full_x, spec.y - spec.full_y, spec.width, spec.height),
                Region(0, 0, spec.full_width or spec.width, spec.full_height or spec.height))
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
        # Tag what is actually in the buffer: working-space scene-linear. Reading one of our
        # own EXRs back is then a true no-op instead of a silent Rec.709 -> ACEScg conversion.
        spec.attribute('oiio:ColorSpace', WORKING)
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
