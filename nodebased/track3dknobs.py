"""PointsTo3D and Reconcile3D as the Nuke 17.0 Reference Guide describes them (lane 9, 2D behaviour step N).

PointsTo3D ("calculate the 3D location of a 2D point in an image sequence using the parallax between two or three tracked points"):

* `img` is "the image sequence that has a 2D point whose 3D position you want to calculate". The points are pixels of that image, so when it is wired its
  display window (the format) is the plate the points and the camera's film back are read against; otherwise `image_width` and `image_height` are.
* Camera type (`camera_mov_type`): "free move" turns the 2D point into a point in 3D from its position in two or three frames (the triangulation
  the node always did, so a node saved before reads as free move); "nodal" "ignores Point B and Point C. Instead, it just puts Point A a fixed distance
  in front of the camera, using the Camera's focal length to decide the distance". The guide gives no unit; this reads the distance as the focal
  length in millimetres taken as scene units, along the ray through Point A at Point A's frame. A nodal camera does not move, so the position is
  the same whichever frame is current, and no parallax (so no minimum ray angle) is needed.
* point3D and point2D, the Output tab's results: "the calculated X, Y, and Z axis location for the 3D point" and "the current X and Y axis
  location of the point2d marker", the 3D point "converted back to screen space". A render never writes a parameter, so they are read from the evaluated
  node: `Evaluator.track3d_diagnostics[key]` holds `position` (point3D) and `point2d` (point2D, through the camera at the current frame).

Reconcile3D ("turns a point in 3D space into an x, y position in the camera"):

* `img`: "Output resolution is determined from this image": its display window replaces `image_width` and `image_height`.
* `output` (XY output) and `outputxyz` (XYZ output, "the point in 3D space, transformed by the axis input"): the evaluated track holds them for every
  frame, and the current frame's pair is `output` and `outputxyz` on the node's value.
* The 2D transform (`track`, `offset`, `stabilize`): "Normally the transformation moves the track point to the projected 3D point, thus moving a plate
  to match a 3D scene. If you check this [stabilize], the transformation is reversed". The node here outputs a track, not the moved picture, so the
  transform is its translation per frame: T = (projected point - track) + offset, and T reversed (negated, offset included) with stabilize. It is the
  value's `transform` curves and each sample's `translate_x` and `translate_y`. The XY output is the projected point and does not change.
* calc_output is always on (the node evaluates live). filter, clamp, black_outside, motionblur, shutter, shutter offset and its custom offset resample or
  blur the moved picture, which this node does not output; they are listed `unsupported` in the mapping.
"""
from __future__ import annotations

import numpy as np

CAMERA_TYPES = ("free move", "nodal")
POINTS_NEW = {"camera_mov_type": "free move"}
RECONCILE_NEW = {"track_x": 0.0, "track_y": 0.0, "offset_x": 0.0, "offset_y": 0.0, "stabilize": 0}
LIMITS = {"stabilize": (0, 1), **{name: (-1000000.0, 1000000.0) for name in ("track_x", "track_y", "offset_x", "offset_y")}}


def plate_size(img, tier, width, height):
    """(width, height) of the plate: the wired `img`'s display window at full size, else the node's own pair."""
    if img is None:
        return int(width), int(height)
    display = getattr(img, "display", None)
    if display is None or display.width <= 0 or display.height <= 0:
        return int(width), int(height)
    return int(display.width * tier), int(display.height * tier)


def nodal_point(camera, xy, width, height):
    """Nodal camera type: Point A on its ray through `xy`, at the focal length (millimetres as scene units) from the camera."""
    from . import filmback, track3d
    eye, direction = track3d.pixel_ray(camera, width, height, xy)
    focal = filmback.focal_from_fov(camera.fov, camera.vaperture)
    return eye + direction * focal


def translation(projected, p):
    """Reconcile3D's 2D transform for a projected point `(x, y)`: the translation that moves the track point to it (plus offset), or the reverse."""
    t = np.asarray((projected[0] - float(p["track_x"]) + float(p["offset_x"]),
                    projected[1] - float(p["track_y"]) + float(p["offset_y"])), np.float64)
    return -t if int(p["stabilize"]) else t
