"""Capability-checked local generative providers for ControlBundle sequences."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from .control_bundle import read_control_bundle
from .media import sequence_path


CONDITIONING_NAMES = ("depth", "normals", "motion", "ids", "camera_pose", "text", "reference_frames")
# `seed` and `sampler_settings` are controls like the rest: a description says whether it honours
# them. Descriptions written before they existed may omit them, which reads as "ignored".
SAMPLING_NAMES = ("seed", "sampler_settings")
CONTROL_NAMES = CONDITIONING_NAMES + SAMPLING_NAMES
MAX_SEED = 2**63 - 1
MAX_SAMPLER_SETTINGS_BYTES = 4096


def normalize_seed(value):
    """Return `value` as a non-negative integer seed (None means the default, 0)."""
    if value is None:
        return 0
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= MAX_SEED:
        raise ValueError(f"seed must be an integer from 0 to {MAX_SEED}")
    return value


def normalize_sampler_settings(value):
    """Return `value` as a small JSON object (None or empty means no settings)."""
    if value in (None, ""):
        return {}
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError as error:
            raise ValueError(f"sampler_settings is not valid JSON: {error}") from error
    if not isinstance(value, dict):
        raise ValueError("sampler_settings must be a JSON object")
    try:
        text = json.dumps(value, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"sampler_settings must be plain JSON: {error}") from error
    if len(text.encode("utf-8")) > MAX_SAMPLER_SETTINGS_BYTES:
        raise ValueError(f"sampler_settings is larger than {MAX_SAMPLER_SETTINGS_BYTES} bytes")
    return json.loads(text)
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
    # A provider whose subprocess drives the GPU (CUDA reserves terabytes of address space): no worker memory cap.
    local_gpu: bool = False

    @classmethod
    def load(cls, source):
        path = PROVIDER_DIR / f"{source}.json" if isinstance(source, str) and not Path(source).exists() else Path(source)
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("schema") != 1:
            raise ValueError("Unsupported provider description schema")
        declared_names = set(data["accepts"])
        if not set(CONDITIONING_NAMES) <= declared_names <= set(CONTROL_NAMES):
            raise ValueError("Provider description must declare every conditioning control")
        declared = list(data["returns"].get("honoured", []))
        actual = [name for name in CONTROL_NAMES if ProviderDescription._accepted(data["accepts"].get(name, False))]
        if declared != actual:
            raise ValueError("Provider honoured controls must match its accepted capabilities")
        if data.get("locality") not in ("local", "remote"):
            raise ValueError("Provider locality must be local or remote")
        return cls(data["name"], data["version"], data["accepts"],
                   tuple(data["limits"]["max_resolution"]), int(data["limits"]["max_frames"]),
                   tuple(data["color_spaces"]), data["locality"], data["returns"], bool(data.get("local_gpu", False)))

    def accepted(self, control):
        return self._accepted(self.accepts.get(control, False))

    @staticmethod
    def _accepted(value):
        return bool(value.get("accepted")) if isinstance(value, dict) else bool(value)

    def control_report(self):
        """Every control as "used" or "ignored", for the panel, the MCP tools and the docs."""
        return {name: "used" if self.accepted(name) else "ignored" for name in CONTROL_NAMES}

    def panel_lines(self):
        return tuple(f"{name.replace('_', ' ').title()}: {'used' if self.accepted(name) else 'ignored'}"
                     for name in CONTROL_NAMES)


GENERATION_MANIFEST = "generation.json"
GENERATED_FRAME_DATETIME = "1980:01:01 00:00:00"


def _write_frame(path, rgba, color_space):
    import OpenImageIO as oiio
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.asarray(rgba, dtype=np.float32)
    spec = oiio.ImageSpec(pixels.shape[1], pixels.shape[0], 4, oiio.FLOAT)
    spec.channelnames = ["R", "G", "B", "A"]
    spec.attribute("oiio:ColorSpace", color_space)
    # OpenImageIO stamps the wall-clock time into every EXR header unless DateTime is given. The
    # sequence's artifact id is the hash of these files, so the same frames written a second apart
    # got different ids: the worker-versus-inline check failed on GitHub's slower Linux runner
    # (10/3). A fixed stamp keeps generated output content-addressed.
    spec.attribute("DateTime", GENERATED_FRAME_DATETIME)
    output = oiio.ImageOutput.create(str(path))
    if output is None or not output.open(str(path), spec):
        raise ValueError(f"Cannot write generated frame: {path}")
    try:
        if not output.write_image(np.ascontiguousarray(pixels)):
            raise ValueError(f"Cannot write generated frame: {path}: {output.geterror()}")
    finally:
        closed = output.close()
    if not closed:
        raise ValueError(f"Cannot write generated frame: {path}: {output.geterror()}")
    _check_frame_readable(path, pixels.shape[1], pixels.shape[0])


def _check_frame_readable(path, width, height):
    """Read the last scanline back: a short write must fail here, not in the next stage.

    Before this check an out-of-memory EXR write left a header-only file and reported success
    (10/10); the verifier then failed with "cannot reshape array of size 1".
    """
    import OpenImageIO as oiio
    image = oiio.ImageInput.open(str(path))
    if image is None:
        raise ValueError(f"Generated frame is unreadable after writing: {path}")
    try:
        spec = image.spec()
        tail = image.read_scanlines(0, 0, height - 1, height, 0, spec.nchannels, oiio.FLOAT)
        if (spec.width, spec.height) != (width, height) or tail is None:
            raise ValueError(f"Generated frame is incomplete after writing: {path}")
    finally:
        image.close()


def _reproject(frame, previous=None):
    """Carry the previous frame's beauty into this frame with its forward motion; nearest depth wins.

    Forward motion points at the next frame, so the pixels that belong on frame N are frame N-1's
    beauty moved by frame N-1's forward vectors. The first frame (or one that does not follow a frame
    of the sequence) has nothing before it and is copied unchanged.
    """
    source = frame if previous is None else previous
    src = np.asarray(source.beauty.values, dtype=np.float32)
    h, w = src.shape[:2]
    if previous is None:
        return src.copy()
    motion = np.asarray(source.motion_forward.values[..., :2], dtype=np.float32)
    depth = np.asarray(source.depth.values[..., 0], dtype=np.float32)
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
             text=None, reference_frames=None, seed=0, sampler_settings=None, progress=None,
             artifact_store=None):
    """Validate capabilities before work, then emit a sequence and honoured-control manifest.

    The sequence's artifact id covers a `generation.json` manifest stored beside the frames:
    provider name and version, seed, sampler settings, the SceneState and ControlBundle ids and
    every option. Two runs that differ in any of them get different ids even when a provider
    that ignores the difference writes identical frames.
    """
    desc = ProviderDescription.load(description if description is not None else provider)
    seed = normalize_seed(seed)
    sampler_settings = normalize_sampler_settings(sampler_settings)
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
    # Bundle layers are always available; unsupported ones remain explicitly visible as ignored. The text
    # brief is advisory: a provider that does not honour it still runs the shot, and the call records the
    # text as ignored (in the manifest and the artifact's provenance). A non-default seed or sampler
    # setting and reference frames change what the provider is asked to compute, so those are still
    # refused before work, as are inputs the provider cannot physically take (colour space, frame count,
    # size) above.
    supplied = {"reference_frames": reference_frames is not None, "seed": seed != 0,
                "sampler_settings": bool(sampler_settings)}
    refused = [name for name, active in supplied.items() if active and not desc.accepted(name)]
    if refused:
        raise ValueError(f"Provider {desc.name} cannot honour controls: {', '.join(refused)}")
    ignored = ["text"] if text is not None and not desc.accepted("text") else []
    if desc.name == "sdxl_controlnet":
        from .providers import sdxl_controlnet
        sdxl_controlnet.resolve_settings(sampler_settings)  # an unknown key or a bad value refuses before any work
    if desc.name not in ("reproject", "null", "sdxl_controlnet"):
        raise ValueError(f"No local implementation for provider {desc.name}")
    pictures, run_info = None, None
    if desc.name == "sdxl_controlnet":
        # The model runs in its own interpreter (see providers/sdxl_controlnet.py); every frame is made from that
        # frame's own depth and normals with the same seed. The recorded settings are the resolved ones, so a
        # change of default shows in the artifact id.
        from .providers import sdxl_controlnet
        sampler_settings = sdxl_controlnet.resolve_settings(sampler_settings)
        import tempfile
        Path(output_pattern).parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="sdxl-controlnet-", dir=Path(output_pattern).parent) as work:
            pictures, run_info = sdxl_controlnet.run_frames(
                controls, text=text, seed=seed, sampler_settings=sampler_settings, workdir=work, progress=progress)
    outputs = []
    for index, frame in enumerate(controls):
        if progress:
            progress((index / max(1, len(controls)), f"Frame {frame.frame}: preparing"))
        if pictures is not None:
            pixels = pictures[index]
        elif desc.name == "null":
            pixels = frame.beauty.values
        else:
            before = controls[index - 1] if index and controls[index - 1].frame == frame.frame - 1 else None
            pixels = _reproject(frame, before)
        path = Path(sequence_path(output_pattern, frame.frame))
        _write_frame(path, pixels, _bundle_color_space(frame))
        outputs.append(str(path))
        if progress:
            progress(((index + 1) / max(1, len(controls)), f"Frame {frame.frame}: complete"))
    result = {"provider": desc.name, "version": desc.version, "seed": seed,
              "sampler_settings": sampler_settings,
              "frames": [c.frame for c in controls],
              "honoured": [name for name in CONTROL_NAMES if desc.accepted(name)],
              "ignored": ignored, "outputs": outputs}
    manifest_output = Path(str(output_pattern).replace("####", "provider").replace("%04d", "provider")
                           .replace("{frame:04d}", "provider")).with_suffix(".json")
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    from .artifacts import ArtifactStore
    store = artifact_store or ArtifactStore()
    bundle_sidecar = Path(manifest_path).with_suffix(".artifact.json")
    bundle_id = (json.loads(bundle_sidecar.read_text())["artifact_id"] if bundle_sidecar.exists()
                 else store.put(manifest_path, "control_bundle", {"producer":"ControlBundle", "version":1,
                      "inputs":[], "document":str(scene_state_path), "frame_range":result["frames"], "time":__import__("time").time()}))
    provider_path = PROVIDER_DIR / f"{desc.name}.json"
    provider_id = store.put(provider_path if provider_path.exists() else json.dumps({"name":desc.name,"version":desc.version}).encode(),
        "provider_description", {"producer":"ProviderDescription", "version":desc.version, "inputs":[],
        "document":str(scene_state_path), "frame_range":result["frames"], "time":__import__("time").time()})
    scene_sidecar = Path(str(scene_state_path) + ".artifact.json")
    scene_id = json.loads(scene_sidecar.read_text())["artifact_id"] if scene_sidecar.exists() else None
    generation = {"schema": 1, "provider": desc.name, "version": desc.version,
                  "provider_description": provider_id, "seed": seed, "sampler_settings": sampler_settings,
                  "seed_honoured": desc.accepted("seed"),
                  "sampler_settings_honoured": desc.accepted("sampler_settings"),
                  "ignored_controls": ignored,
                  "honoured": result["honoured"],
                  "scene_state": scene_id, "control_bundle": bundle_id, "frames": result["frames"],
                  "options": {"text": text,
                              "reference_frames": [hashlib.sha256(Path(item).read_bytes()).hexdigest()
                                                   for item in (reference_frames or [])]}}
    sequence_files = {Path(output).name: output for output in outputs}
    sequence_files[GENERATION_MANIFEST] = (json.dumps(generation, sort_keys=True, indent=2) + "\n").encode("utf-8")
    sequence_id = store.put_files(sequence_files, "generated_sequence", {
        "producer":"Generate", "version":desc.version, "inputs":[{"id":bundle_id},{"id":provider_id}],
        "document":str(scene_state_path), "frame_range":result["frames"], "time":__import__("time").time(),
        "pattern":str(output_pattern), "provider":desc.name, "provider_version":desc.version,
        "seed":seed, "sampler_settings":sampler_settings, "seed_honoured":desc.accepted("seed"),
        "ignored_controls":ignored, "scene_state":scene_id, "control_bundle":bundle_id})
    result["artifact_id"] = sequence_id
    if run_info is not None:
        result["run"] = run_info  # timings and peak VRAM: reported, never part of the artifact
    manifest_output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
