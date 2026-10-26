"""Nuke's CurveTool knobs of fidelity step H: the Curve Type (`curve_type`, Nuke's `operation`), the Go! button, the number of frames
for the exposure base average (`avg_frames`, Nuke's `avgframes`) and the `channels` selector.

CurveTool measures the frames of its range and writes the results as animation keys on its own result knobs. Step H lets the
Curve Type choose which analysis the Go! button runs, and so which result knobs it keys:

* `Auto Crop`: `crop_x`, `crop_y`, `crop_width`, `crop_height`.
* `Avg Intensities`: `average_r`, `average_g`, `average_b`, `average_a`, `average_luminance`.
* `Exposure Difference`: `exposure_diff`, the average luminance of each frame less the base: the previous frame's, or with
  `avg_frames` above 0 the mean of the first `avg_frames` frames' average luminance (Nuke's "# frames for base average"; the guide
  gives no more than that, so this reading of the base is NodeBased's). `avg_frames` 0, the default, is the previous frame.
* `Max Luma Pixel`: `max_x`, `max_y`, `max_value` and the dimmest pixel's `min_x`, `min_y`, `min_value`.

`channels` (default `rgba`) restricts what is measured to those channels: an unselected channel reads as zero, the luminance is the
mean of the selected colour channels, and with `alpha` unselected the alpha-mode auto crop finds nothing. The default measures
what the node always measured.

The Go! button is the window's Analyze button; `analyze` below is what it runs, so a test and the window measure the same way.
`Auto Crop`'s colour (`autocrop_color_*`) and Intensity Range (`autocrop_tolerance`) are the knobs the node already had.
"""
from __future__ import annotations

PARAMS = {"curve_type": "Avg Intensities", "avg_frames": 0, "channels": "rgba"}
KINDS = ("CurveTool",)
CURVE_TYPES = ("Auto Crop", "Avg Intensities", "Exposure Difference", "Max Luma Pixel")
RESULTS = {
    "Auto Crop": ("crop_x", "crop_y", "crop_width", "crop_height"),
    "Avg Intensities": ("average_r", "average_g", "average_b", "average_a", "average_luminance"),
    "Exposure Difference": ("exposure_diff",),
    "Max Luma Pixel": ("max_x", "max_y", "max_value", "min_x", "min_y", "min_value"),
}


def new_params(kind):
    return dict(PARAMS) if kind in KINDS else {}


def upgrade_params(kind):
    return dict(PARAMS) if kind in KINDS else {}


def result_names(params):
    """The result knobs the Go! button keys for this node's Curve Type."""
    return RESULTS.get(str(params.get("curve_type", "Avg Intensities")), RESULTS["Avg Intensities"])


def metrics_for(pixels, frame_params, previous_luminance=None):
    """One frame's measurements for the node's parameters at that frame (box, auto crop colour and range, channels)."""
    from .imaging import curve_tool_metrics
    p = frame_params
    return curve_tool_metrics(pixels, (p["box_x"], p["box_y"], p["box_width"], p["box_height"]),
                              previous_luminance=previous_luminance, autocrop_mode=p["autocrop_mode"],
                              autocrop_color=(p["autocrop_color_r"], p["autocrop_color_g"], p["autocrop_color_b"]),
                              autocrop_tolerance=p["autocrop_tolerance"], channels=str(p.get("channels", "rgba")))


def key_commands(key, frame, measured, names):
    """The set_key commands that write `names` of one frame's measurements."""
    return [{"op": "set_key", "id": key, "param": name, "frame": frame, "value": float(measured[name]),
             "interpolation": "linear"} for name in names]


def exposure_base(luminances, avg_frames):
    """The base the exposure difference is taken from: the mean of the first `avg_frames` frames, or None for the previous frame."""
    count = int(avg_frames)
    if count <= 0 or not luminances:
        return None
    first = luminances[:count]
    return sum(first) / len(first)


def analyze(document, evaluator, key, progress=None, cancel=None):
    """The Go! button: measure every frame of the node's range and return the commands that key the Curve Type's results, or None
    when `cancel()` says stop. `progress(done, total)` is called after each frame.

    Raises ValueError with a message for a missing input or a range that ends before it starts."""
    from . import analysisregion
    node = document["nodes"][key]
    source = node["inputs"].get("image")
    if not source:
        raise ValueError("Connect an image input first.")
    params = node["params"]
    first, last = int(params["frame_start"]), int(params["frame_end"])
    if last < first:
        raise ValueError("The end frame must be at or after the start frame.")
    names = result_names(params)
    rows, previous = [], None
    for index, frame in enumerate(range(first, last + 1)):
        if cancel is not None and cancel():
            return None
        raster = evaluator.evaluate_raster(document, target=source, frame=frame).to_display()
        frame_params = analysisregion.region_params_at_frame(document, key, frame)
        frame_params = {**frame_params, "channels": params.get("channels", "rgba")}
        measured = metrics_for(raster, frame_params, previous)
        previous = measured["average_luminance"]
        rows.append((frame, measured))
        if progress is not None:
            progress(index + 1, last - first + 1)
    base = exposure_base([row[1]["average_luminance"] for row in rows], params.get("avg_frames", 0))
    commands = []
    for frame, measured in rows:
        if base is not None:
            measured = {**measured, "exposure_diff": measured["average_luminance"] - base}
        commands.extend(key_commands(key, frame, measured, names))
    return commands
