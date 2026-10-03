"""Read generated frames against a conditioning bundle and expose its control layers.

Generated pictures are centre-cropped to the shot aspect ratio, then resized to the
manifest's pixel size. This keeps the image filling frame without stretching it.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile
import zipfile
import re

import numpy as np

from .control_bundle import read_control_bundle
from .color import WORKING, to_working
from .media import auto_space, sequence_path


@dataclass(frozen=True)
class ConditionedFrame:
    frame: int
    beauty: np.ndarray
    layers: dict[str, np.ndarray]
    layer_metadata: dict


def _resize_center_cover(image, width, height):
    """Centre-crop to the destination aspect ratio and bilinearly resize."""
    src = np.asarray(image, dtype=np.float32)
    sh, sw = src.shape[:2]
    scale = max(width / sw, height / sh)
    cw, ch = width / scale, height / scale
    x0, y0 = (sw - cw) * .5, (sh - ch) * .5
    xs = np.clip(x0 + (np.arange(width) + .5) * cw / width - .5, 0, sw - 1)
    ys = np.clip(y0 + (np.arange(height) + .5) * ch / height - .5, 0, sh - 1)
    xlo, ylo = np.floor(xs).astype(int), np.floor(ys).astype(int)
    xhi, yhi = np.minimum(xlo + 1, sw - 1), np.minimum(ylo + 1, sh - 1)
    fx, fy = (xs - xlo)[None, :, None], (ys - ylo)[:, None, None]
    return ((1 - fy) * ((1 - fx) * src[ylo[:, None], xlo[None, :]] + fx * src[ylo[:, None], xhi[None, :]])
            + fy * ((1 - fx) * src[yhi[:, None], xlo[None, :]] + fx * src[yhi[:, None], xhi[None, :]])).astype(np.float32)


def _read_rgba(path, expected_space=None):
    import OpenImageIO as oiio
    image = oiio.ImageInput.open(str(path))
    if image is None:
        raise ValueError(f"ConditionedRead cannot open generated frame: {path}")
    try:
        spec = image.spec()
        names = list(spec.channelnames)
        raw = np.asarray(image.read_image(oiio.FLOAT), dtype=np.float32).reshape(spec.height, spec.width, spec.nchannels)
        indices = {name: i for i, name in enumerate(names)}
        rgb_names = [indices[name] for name in ("R", "G", "B") if name in indices]
        if len(rgb_names) != 3:
            raise ValueError(f"Generated frame {path} must contain R, G and B channels")
        alpha = raw[..., indices["A"]:indices["A"] + 1] if "A" in indices else np.ones((*raw.shape[:2], 1), np.float32)
        rgba = np.concatenate((raw[..., rgb_names], alpha), axis=-1)
        space = auto_space(Path(path).suffix.lower(), spec)
        tag = (spec.get_string_attribute("oiio:ColorSpace") or "").strip()
        if expected_space and space != expected_space:
            raise ValueError(f"Generated frame colour-space tag {tag!r} resolves to {space}, expected {expected_space}")
        return to_working(rgba, space, associated=False), space
    finally:
        image.close()


def read_conditioned_sequence(generated_pattern, manifest_path, scene_state_path, *, frame_offset=0):
    """Read a generated sequence aligned to bundle frames; return beauty and all controls.

    `frame_offset` maps a shot frame to generated frame `shot + frame_offset`. The
    bundle reader verifies the paired SceneState and every exported layer before return.
    """
    if re.fullmatch(r"[0-9a-f]{64}", str(generated_pattern)):
        from .artifacts import ArtifactStore
        try: payload = ArtifactStore().get(str(generated_pattern))
        except ValueError as exc: raise ValueError(f"ConditionedRead missing artifact id {generated_pattern}") from exc
        with tempfile.TemporaryDirectory(prefix="nodebased-artifact-") as folder:
            archive = Path(folder) / "sequence.zip"
            archive.write_bytes(payload)
            with zipfile.ZipFile(archive) as zf: zf.extractall(folder)
            names = [Path(name).name for name in zipfile.ZipFile(archive).namelist()]
            # Preserve frame numbering from the archived generated paths.
            if names:
                filename = Path(names[0]); match = re.search(r"\d+$", filename.stem)
                if not match: raise ValueError(f"ConditionedRead artifact {generated_pattern} has an invalid frame name")
                pattern_name = filename.stem[:match.start()] + "%04d" + filename.suffix
                pattern = str(Path(folder) / pattern_name)
            else: pattern = ""
            if names:
                return read_conditioned_sequence(pattern, manifest_path, scene_state_path, frame_offset=frame_offset)
            raise ValueError(f"ConditionedRead artifact {generated_pattern} contains no frames")
    controls = read_control_bundle(manifest_path, scene_state_path)
    import json
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    records = {int(row["frame"]): row for row in manifest["frames"]}
    output = []
    for control in controls:
        source_frame = control.frame + int(frame_offset)
        path = sequence_path(generated_pattern, source_frame)
        manifest_space = records[control.frame]["layers"]["beauty"]["color_space"]
        expected_space = manifest_space.rsplit("(", 1)[-1].rstrip(")") if "(" in manifest_space else None
        rgba, source_space = _read_rgba(path, expected_space)
        width, height = map(int, records[control.frame]["pixel_size"])
        rgba = _resize_center_cover(rgba, width, height)
        layers = {name: getattr(control, name).values for name in records[control.frame]["layers"]}
        metadata = {name: records[control.frame]["layers"][name] for name in layers}
        metadata["generated_beauty"] = {"color_space": source_space, "working_space": WORKING,
                                        "resize_rule": "centre-crop-to-fill, then bilinear resize"}
        output.append(ConditionedFrame(control.frame, rgba, layers, metadata))
    return output
