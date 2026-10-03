"""Capability-checked local generative providers for ControlBundle sequences."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .control_bundle import read_control_bundle
from .media import sequence_path


CONTROL_NAMES = ("depth", "normals", "motion", "ids", "camera_pose", "text", "reference_frames")
PROVIDER_DIR = Path(__file__).with_name("providers")


@dataclass(frozen=True)
class ProviderDescription:
    name: str
    version: str
    accepts: dict
    max_resolution: tuple[int, int]
    max_frames: int
    color_spaces: tuple[str, ...]
    locality: str
    returns: dict

    @classmethod
    def load(cls, source):
        path = PROVIDER_DIR / f"{source}.json" if isinstance(source, str) and not Path(source).exists() else Path(source)
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("schema") != 1:
            raise ValueError("Unsupported provider description schema")
        if set(data["accepts"]) != set(CONTROL_NAMES):
            raise ValueError("Provider description must declare every conditioning control")
        declared = list(data["returns"].get("honoured", []))
        actual = [name for name in CONTROL_NAMES if ProviderDescription._accepted(data["accepts"][name])]
        if declared != actual:
            raise ValueError("Provider honoured controls must match its accepted capabilities")
        if data.get("locality") not in ("local", "remote"):
            raise ValueError("Provider locality must be local or remote")
        return cls(data["name"], data["version"], data["accepts"],
                   tuple(data["limits"]["max_resolution"]), int(data["limits"]["max_frames"]),
                   tuple(data["color_spaces"]), data["locality"], data["returns"])

    def accepted(self, control):
        return self._accepted(self.accepts[control])

    @staticmethod
    def _accepted(value):
        return bool(value.get("accepted")) if isinstance(value, dict) else bool(value)

    def panel_lines(self):
        return tuple(f"{name.replace('_', ' ').title()}: {'used' if self.accepted(name) else 'ignored'}"
                     for name in CONTROL_NAMES)


def _write_frame(path, rgba, color_space):
    import OpenImageIO as oiio
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.asarray(rgba, dtype=np.float32)
    spec = oiio.ImageSpec(pixels.shape[1], pixels.shape[0], 4, oiio.FLOAT)
    spec.channelnames = ["R", "G", "B", "A"]
    spec.attribute("oiio:ColorSpace", color_space)
    output = oiio.ImageOutput.create(str(path))
    if output is None or not output.open(str(path), spec):
        raise ValueError(f"Cannot write generated frame: {path}")
    try:
        if not output.write_image(np.ascontiguousarray(pixels)):
            raise ValueError(f"Cannot write generated frame: {path}")
    finally:
        output.close()


def _reproject(frame):
    """Forward-splat beauty using pixel motion; nearest depth wins each destination pixel."""
    src = np.asarray(frame.beauty.values, dtype=np.float32)
    h, w = src.shape[:2]
    motion = np.asarray(frame.motion_forward.values[..., :2], dtype=np.float32)
    depth = np.asarray(frame.depth.values[..., 0], dtype=np.float32)
    yy, xx = np.mgrid[:h, :w]
    tx = np.rint(xx + motion[..., 0]).astype(np.int32)
    ty = np.rint(yy + motion[..., 1]).astype(np.int32)
    inside = (tx >= 0) & (tx < w) & (ty >= 0) & (ty < h) & np.isfinite(depth) & (depth > 0)
    out = np.zeros_like(src)
    zbuf = np.full((h, w), np.inf, np.float32)
    # Stable ordering makes ties deterministic; camera pose is bundled and checked by reader.
    for y, x in zip(*np.nonzero(inside)):
        dy, dx = ty[y, x], tx[y, x]
        if depth[y, x] < zbuf[dy, dx]:
            zbuf[dy, dx] = depth[y, x]
            out[dy, dx] = src[y, x]
    # Disocclusions are explicit transparent black.
    return out


def _bundle_color_space(frame):
    label = frame.beauty.color_space
    if "(" in label and label.endswith(")"):
        return label.rsplit("(", 1)[1][:-1]
    return label


def generate(manifest_path, scene_state_path, output_pattern, provider="reproject", *, description=None,
             text=None, reference_frames=None):
    """Validate capabilities before work, then emit a sequence and honoured-control manifest."""
    desc = ProviderDescription.load(description if description is not None else provider)
    controls = read_control_bundle(manifest_path, scene_state_path)
    spaces = {_bundle_color_space(frame) for frame in controls}
    unsupported_spaces = sorted(spaces - set(desc.color_spaces))
    if unsupported_spaces:
        raise ValueError(f"Provider {desc.name} does not accept colour spaces: {', '.join(unsupported_spaces)}")
    if len(controls) > desc.max_frames:
        raise ValueError(f"Provider {desc.name} does not support {len(controls)} frames (limit {desc.max_frames})")
    too_large = [(c.frame, c.beauty.values.shape[1], c.beauty.values.shape[0]) for c in controls
                 if c.beauty.values.shape[1] > desc.max_resolution[0] or c.beauty.values.shape[0] > desc.max_resolution[1]]
    if too_large:
        raise ValueError(f"Provider {desc.name} resolution limit {desc.max_resolution} exceeded")
    # Bundle layers are always available; unsupported ones remain explicitly visible as ignored.
    # Fail before work only when the artist actively supplies an unsupported optional control.
    supplied = {"text": text is not None, "reference_frames": reference_frames is not None}
    ignored = [name for name, active in supplied.items() if active and not desc.accepted(name)]
    if ignored:
        raise ValueError(f"Provider {desc.name} cannot honour controls: {', '.join(ignored)}")
    if desc.name not in ("reproject", "null"):
        raise ValueError(f"No local implementation for provider {desc.name}")
    outputs = []
    for frame in controls:
        if desc.name == "null":
            pixels = frame.beauty.values
        else:
            pixels = _reproject(frame)
        path = Path(sequence_path(output_pattern, frame.frame))
        _write_frame(path, pixels, _bundle_color_space(frame))
        outputs.append(str(path))
    result = {"provider": desc.name, "version": desc.version,
              "frames": [c.frame for c in controls],
              "honoured": [name for name in CONTROL_NAMES if desc.accepted(name)],
              "outputs": outputs}
    manifest_output = Path(str(output_pattern).replace("####", "provider").replace("%04d", "provider")
                           .replace("{frame:04d}", "provider")).with_suffix(".json")
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
