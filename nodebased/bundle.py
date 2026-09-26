"""The conditioning bundle: a multichannel EXR plus a JSON manifest that says what is in it.

A fluid, particle or 3D render exports intrinsic data (motion vectors, depth, normals, density...)
as named EXR layers. A diffusion or transform model that consumes those layers needs to know what
each one means (units, direction, range), which frame and camera they belong to, and which state of
the document made them. `Write` with `bundle` on writes that description next to every frame; the
`ReadBundle` node reads a model's output for the same frame back into the graph and refuses it when
the manifest says it belongs to another frame or another format. The format is documented in
docs/3D_FOUNDATION.md ("The conditioning bundle").
"""
from __future__ import annotations

import json
import os
from pathlib import Path

BUNDLE_FORMAT = "nodebased-bundle"
BUNDLE_VERSION = 1
MANIFEST_SUFFIX = ".bundle.json"

# What each known layer means. Anything not listed is written as "unspecified" rather than guessed.
# `direction` says which way values point or grow; `range` is the span the values normally take.
_UNKNOWN = {"units": "unspecified", "direction": "unspecified", "range": "unspecified", "space": "unspecified"}
CONVENTIONS = {
    "normals": {"units": "unit vector", "direction": "points away from the surface",
                "range": [-1.0, 1.0], "space": "world"},
    "depth": {"units": "scene units", "direction": "distance along the view ray, grows away from the camera",
              "range": "0 to the camera far plane", "space": "view"},
    "position": {"units": "scene units", "direction": "x, y, z of the first surface hit",
                 "range": "unbounded", "space": "world"},
    "uv": {"units": "normalised texture coordinates", "direction": "u to the right, v up (the STMap convention)",
           "range": [0.0, 1.0], "space": "texture"},
    "motion": {"units": "pixels per frame",
               "direction": "forward: where the pixel moves by the next frame; x to the right, y down",
               "range": "unbounded", "space": "image"},
    "density": {"units": "as exported by the simulation", "direction": "scalar", "range": "non-negative",
                "space": "image"},
    "temperature": {"units": "as exported by the simulation", "direction": "scalar", "range": "unbounded",
                    "space": "image"},
    "vorticity": {"units": "as exported by the simulation", "direction": "scalar", "range": "unbounded",
                  "space": "image"},
}
RELIGHT_CONVENTION = {"units": "unitless response", "direction": "scalar",
                      "range": "0 to 1 before light colour and intensity", "space": "image"}


def layer_convention(name):
    """The convention record for one layer name (a copy; safe to mutate)."""
    if name in CONVENTIONS:
        return dict(CONVENTIONS[name])
    if name.startswith("relight_") or name in ("albedo", "diffuse", "specular", "emission") \
            or name.startswith(("diffuse_L", "specular_L")):
        return dict(RELIGHT_CONVENTION)
    return dict(_UNKNOWN)


def manifest_path(image_path):
    """The manifest that sits next to one written frame: `shot.0012.exr` -> `shot.0012.bundle.json`."""
    path = Path(image_path)
    return path.with_name(path.stem + MANIFEST_SUFFIX)


def _upstream_render(document, key, frame):
    """The first Render3D reached from `key` by following inputs, and the camera node that fed it."""
    from .animation import resolve_params
    from .core import LIMITS, SPECS
    nodes = document["nodes"]
    seen, queue = set(), [key]
    while queue:
        current = queue.pop(0)
        if current in seen or current not in nodes:
            continue
        seen.add(current)
        node = nodes[current]
        if node["type"] == "Render3D":
            camera_key = node["inputs"].get("camera")
            render = resolve_params(node, document.get("animation", {}).get("curves", {}).get(current),
                                    frame, SPECS["Render3D"]["params"], LIMITS)
            return current, render, camera_key
        queue.extend(source for source in node["inputs"].values() if source is not None)
    return None, None, None


def camera_record(document, key, frame):
    """The camera of the Render3D that produced this image, or None when no Render3D is upstream."""
    from .animation import resolve_params
    from .core import LIMITS, SPECS
    from . import scene3d
    render_key, render, camera_key = _upstream_render(document, key, frame)
    if render_key is None:
        return None
    record = {"render_node": document["nodes"][render_key]["name"],
              "width": int(render["width"]), "height": int(render["height"])}
    node = document["nodes"].get(camera_key) if camera_key else None
    if node is None:
        return {**record, "type": None}
    record.update(node=node["name"], type=node["type"])
    if node["type"] == "Camera3D":
        p = resolve_params(node, document.get("animation", {}).get("curves", {}).get(camera_key), frame,
                           SPECS["Camera3D"]["params"], LIMITS)
        camera = scene3d.camera_from_node({"params": p})
        record.update(position=[p["tx"], p["ty"], p["tz"]], target=[p["target_x"], p["target_y"], p["target_z"]],
                      roll=float(p["roll"]), fov=float(camera.fov), focal=float(p["focal"]),
                      haperture=float(p["haperture"]), vaperture=float(p["vaperture"]),
                      near=float(p["near"]), far=float(p["far"]))
    else:
        record["note"] = "camera comes from a file; only its node is recorded"
    return record


def build_manifest(document, key, raster, frame, image_path, bits, fingerprint):
    """The manifest for one written frame of `key`'s image."""
    from .media import layer_channels
    layers = []
    for name in (raster.layers or {}):
        layers.append({"name": name, "channels": [f"{name}.{channel}" for channel in layer_channels(name)],
                       "convention": layer_convention(name)})
    display = raster.display
    return {
        "format": BUNDLE_FORMAT, "version": BUNDLE_VERSION,
        "frame": int(frame),
        "image": Path(image_path).name, "file_type": "exr", "bit_depth": bits,
        "width": display.width, "height": display.height,
        "beauty": {"channels": ["R", "G", "B", "A"], "space": "scene-linear working space, premultiplied"},
        "layers": layers,
        "camera": camera_record(document, key, frame),
        "write_node": document["nodes"][key]["name"],
        "fingerprint": fingerprint,
    }


def write_manifest(image_path, manifest):
    """Write the manifest next to `image_path` atomically and return its path."""
    target = manifest_path(image_path)
    temporary = target.with_name("." + target.name + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, target)
    return target


def read_manifest(path):
    """Load and shape-check a manifest; raises ValueError naming what is wrong."""
    try:
        manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"Cannot read bundle manifest {path}: {error}") from None
    except ValueError as error:
        raise ValueError(f"Bundle manifest {path} is not valid JSON: {error}") from None
    if not isinstance(manifest, dict) or manifest.get("format") != BUNDLE_FORMAT:
        raise ValueError(f"{path} is not a NodeBased bundle manifest")
    if manifest.get("version") != BUNDLE_VERSION:
        raise ValueError(f"Bundle manifest {path} is version {manifest.get('version')!r}; "
                         f"this build reads version {BUNDLE_VERSION}")
    for field in ("frame", "width", "height", "layers"):
        if field not in manifest:
            raise ValueError(f"Bundle manifest {path} is missing '{field}'")
    return manifest


def check_manifest(manifest, frame, width=None, height=None, where="bundle"):
    """Refuse a manifest that belongs to another frame, or an image of another size."""
    if int(manifest["frame"]) != int(frame):
        raise ValueError(f"{where}: the manifest is for frame {manifest['frame']} but the graph is at "
                         f"frame {int(frame)}; a model output must be read at the frame it was made for")
    if width is not None and (int(width), int(height)) != (int(manifest["width"]), int(manifest["height"])):
        raise ValueError(f"{where}: the image is {width}x{height} but the bundle was written at "
                         f"{manifest['width']}x{manifest['height']}")
    return manifest


def resolve_manifest(pattern, frame):
    """The manifest file for `frame`: a padded pattern (render.%04d.bundle.json) or a single file."""
    from .media import sequence_path
    if not pattern:
        raise ValueError("ReadBundle: choose the bundle manifest (.bundle.json) written by Write")
    return sequence_path(pattern, frame)


def write_frame(evaluator, document, key, frame, target, bits):
    """Evaluate `key` at `frame`, write the multichannel EXR to `target` and its manifest beside it.

    The manifest's fingerprint is the evaluator's own cache digest of the node at that frame, so it
    changes exactly when anything that feeds the image changes. Returns (raster, manifest path).
    """
    from .media import raster_layer_arrays, write_exr
    raster, digest = evaluator.evaluate_raster(document, key, frame=frame, tier=1, return_digest=True)
    write_exr(target, raster.to_display(), bits=bits, layers=raster_layer_arrays(raster))
    manifest = build_manifest(document, key, raster, frame, target, bits, digest)
    return raster, write_manifest(target, manifest)


def read_bundle_raster(path, bundle, colorspace="Auto", alpha_mode="Auto", frame=None):
    """A model's output image for `frame`, read only if its bundle manifest matches that frame."""
    from .media import read_media_raster, resolve_source_path
    frame = int(frame if frame is not None else 0)
    manifest_file = resolve_manifest(bundle, frame)
    manifest = check_manifest(read_manifest(manifest_file), frame, where="ReadBundle")
    if not path:
        raise ValueError("ReadBundle: choose the model's output image")
    resolved, exists = resolve_source_path(path, frame, "error")
    raster = read_media_raster(resolved, colorspace, alpha_mode)
    check_manifest(manifest, frame, raster.display.width, raster.display.height, where="ReadBundle")
    return raster


def fingerprint(params, frame):
    """What the cache digest of a ReadBundle must include: both files' identity at this frame."""
    from .media import resolve_source_path
    manifest_file = resolve_manifest(params["bundle"], int(frame))
    if not params["path"]:
        raise ValueError("ReadBundle: choose the model's output image")
    resolved, _ = resolve_source_path(params["path"], int(frame), "error")
    parts = []
    for candidate in (manifest_file, resolved):
        if candidate is None or not Path(candidate).is_file():
            raise ValueError(f"ReadBundle: missing file {candidate}")
        stat = Path(candidate).stat()
        parts.extend([str(Path(candidate).resolve()), stat.st_size, stat.st_mtime_ns])
    return parts
