"""Small, Qt-free geometry helpers for the Viewer 2D Transform handle."""

import math


def axes_flipped(params):
    """Whether the Transform node counts its vertical axis upwards, as Nuke does (a missing flag is an older node)."""
    return not int(params.get("legacy_transform_axes", 1))


def flip_axes(values, flipped):
    """The handle geometry below draws the picture on screen: y down and a positive rotate clockwise. A node that follows Nuke
    stores translate Y and rotate with the opposite sign; this flips those two (it is its own inverse). Other keys pass through."""
    out = dict(values)
    if flipped:
        for name in ("translate_y", "rotate", "skew_x", "skew_y"):
            if name in out:
                out[name] = -out[name]
    return out


def scale_pair(scale):
    """(x, y) factors of Transform's scale, which is one value or two (Nuke's "one or two values")."""
    if isinstance(scale, (list, tuple)):
        return float(scale[0]), float(scale[1])
    return float(scale), float(scale)


def linear_terms(rotate, scale, skew_x=0.0, skew_y=0.0, skew_order="XY"):
    """The 2x2 part of the Transform's map as (a, b, c, d): rotate . skew . scale, the order `Evaluator` uses."""
    sx, sy = scale_pair(scale)
    if skew_order == "XY":      # skew_y @ skew_x
        k = (1.0, skew_x, skew_y, skew_y * skew_x + 1.0)
    else:                       # skew_x @ skew_y
        k = (1.0 + skew_x * skew_y, skew_x, skew_y, 1.0)
    theta = math.radians(rotate)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    r = (cos_t * k[0] - sin_t * k[2], cos_t * k[1] - sin_t * k[3], sin_t * k[0] + cos_t * k[2], sin_t * k[1] + cos_t * k[3])
    return r[0] * sx, r[1] * sy, r[2] * sx, r[3] * sy


def forward_point(x, y, translate_x, translate_y, rotate, scale, center_x, center_y, skew_x=0.0, skew_y=0.0, skew_order="XY"):
    """Map a data-space point through the Transform node's forward affine map (scale one value or two, optional skew)."""
    a, b, c, d = linear_terms(rotate, scale, skew_x, skew_y, skew_order)
    u, v = x - center_x, y - center_y
    return a * u + b * v + center_x + translate_x, c * u + d * v + center_y + translate_y


def pivot_point(translate_x, translate_y, center_x, center_y):
    """Return the Transform pivot in output/displayed coordinates."""
    return center_x + translate_x, center_y + translate_y


def box_corners(width, height, translate_x, translate_y, rotate, scale, center_x, center_y, **skew):
    """Return the forward-mapped corners of a width by height source rectangle."""
    return [
        forward_point(x, y, translate_x, translate_y, rotate, scale, center_x, center_y, **skew)
        for x, y in ((0, 0), (width, 0), (width, height), (0, height))
    ]


def edge_midpoints(corners):
    """Return the midpoints of the four consecutive quadrilateral edges."""
    return [
        ((corners[index][0] + corners[(index + 1) % 4][0]) / 2,
         (corners[index][1] + corners[(index + 1) % 4][1]) / 2)
        for index in range(4)
    ]


def point_in_quad(x, y, corners):
    """Return whether a point is inside (or on an edge of) a quadrilateral."""
    inside = False
    for index, (x1, y1) in enumerate(corners):
        x2, y2 = corners[(index + 1) % len(corners)]
        # Treat the boundary as part of the handle body, avoiding a dead one-pixel seam.
        cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
        if (abs(cross) < 1e-9 and min(x1, x2) - 1e-9 <= x <= max(x1, x2) + 1e-9
                and min(y1, y2) - 1e-9 <= y <= max(y1, y2) + 1e-9):
            return True
        if (y1 > y) != (y2 > y):
            crossing_x = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < crossing_x:
                inside = not inside
    return inside


def pivot_drag_result(translate_x, translate_y, rotate, scale, center_x, center_y, dx, dy):
    """Move the pivot by ``(dx, dy)`` while leaving the rendered image unchanged."""
    sx, sy = scale_pair(scale)
    if sx != sy:
        a, b, c, d = linear_terms(rotate, (sx, sy))
        rx, ry = a * dx + b * dy, c * dx + d * dy
        return center_x + dx, center_y + dy, translate_x - dx + rx, translate_y - dy + ry
    scale = sx
    theta = math.radians(rotate)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    # dst(p) = R*k*p + (c - R*k*c) + t.  With c' = c + dc, preserving
    # dst requires t' = t - dc + k*R*dc.
    rx = scale * (cos_t * dx - sin_t * dy)
    ry = scale * (sin_t * dx + cos_t * dy)
    return (
        center_x + dx,
        center_y + dy,
        translate_x - dx + rx,
        translate_y - dy + ry,
    )


def scale_from_drag(scale_start, pivot, start_point, current_point, minimum=0.001):
    """Return scale proportional to the pointer's distance from the pivot."""
    d0 = math.hypot(start_point[0] - pivot[0], start_point[1] - pivot[1])
    if d0 < 1e-6:
        return scale_start
    d1 = math.hypot(current_point[0] - pivot[0], current_point[1] - pivot[1])
    return max(minimum, scale_start * d1 / d0)


def rotate_from_drag(rotate_start, pivot, start_point, current_point):
    """Return rotation after applying the pointer's signed angular delta."""
    a0 = math.atan2(start_point[1] - pivot[1], start_point[0] - pivot[0])
    a1 = math.atan2(current_point[1] - pivot[1], current_point[0] - pivot[0])
    return rotate_start + math.degrees(a1 - a0)


def axis_scale_from_drag(scale_start, pivot, rotate, axis, start_point, current_point, minimum=0.001):
    """One factor of a split scale from a drag on the jack's left/right (axis 0) or top/bottom (axis 1) point: the pointer's
    distance from the pivot along that rotated axis, in proportion to where the drag began."""
    theta = math.radians(rotate)
    direction = (math.cos(theta), math.sin(theta)) if axis == 0 else (-math.sin(theta), math.cos(theta))
    d0 = (start_point[0] - pivot[0]) * direction[0] + (start_point[1] - pivot[1]) * direction[1]
    if abs(d0) < 1e-6:
        return scale_start
    d1 = (current_point[0] - pivot[0]) * direction[0] + (current_point[1] - pivot[1]) * direction[1]
    return max(minimum, scale_start * d1 / d0)


def skew_bar_end(pivot, rotate, skew_x, length):
    """The top of the jack's vertical bar: `length` above the pivot along the image's skewed vertical axis (screen y down)."""
    theta = math.radians(rotate)
    ux, uy = -skew_x, -1.0               # the image's upward axis after skew x, before rotation
    norm = math.hypot(ux, uy)
    ux, uy = ux / norm * length, uy / norm * length
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    return pivot[0] + ux * cos_t - uy * sin_t, pivot[1] + ux * sin_t + uy * cos_t


def skew_from_drag(pivot, rotate, current_point, minimum_height=1e-3):
    """Skew x (screen convention) that tilts the vertical bar through the pointer: the pointer's sideways offset over its height
    above the pivot, both measured in the unrotated frame."""
    theta = math.radians(-rotate)
    dx, dy = current_point[0] - pivot[0], current_point[1] - pivot[1]
    lx = dx * math.cos(theta) - dy * math.sin(theta)
    ly = dx * math.sin(theta) + dy * math.cos(theta)
    if ly > -minimum_height:
        ly = -minimum_height
    return lx / ly


ROTATE_SNAP_STEP = 45.0
ROTATE_SNAP_TOLERANCE = 2.0


def snap_rotate(value, step=ROTATE_SNAP_STEP, tolerance=ROTATE_SNAP_TOLERANCE):
    """Nuke 17.0 User Guide, Transforming in 2D: "The transform overlay snaps to typical values." The guide names no values;
    NodeBased snaps to the nearest multiple of 45 degrees when the drag ends within 2 degrees of it."""
    nearest = round(value / step) * step
    return float(nearest) if abs(value - nearest) <= tolerance else value


# Crop's and Text's on-screen box (lane 10 step T3 finish). Nuke 17.0 Reference Guide, Crop: "You can also adjust the crop box in
# the Viewer by dragging its edges."; Text: "You can also move and resize the box in the Viewer." and its center "You can also
# move it using Ctrl/Cmd +drag." Knobs per node: (x, y, width, height) of the box in data coordinates, rows counting down.
BOX_NODES = {"Crop": ("x", "y", "width", "height"), "Text": ("box_x", "box_y", "box_width", "box_height"),
             "ColorTransfer": ("roi_x", "roi_y", "roi_r", "roi_t")}
# Nodes whose box knobs are corners counted from the bottom left (Nuke's x, y, r, t), not (x, y, width, height) rows down.
# ColorTransfer: Nuke 17.0 Reference Guide, ColorTransfer, ROI: "You can also adjust the ROI in the Viewer by dragging its edges."
# All four zero is the guide's default ("set to cover the entire input"), so that box is drawn over the whole frame.
CORNER_BOX_NODES = ("ColorTransfer",)
GRIPS = ("nw", "n", "ne", "e", "se", "s", "sw", "w")


def box_grips(x, y, w, h):
    """{grip: (x, y)} for the box's four corners and four edge midpoints."""
    return {"nw": (x, y), "n": (x + w / 2, y), "ne": (x + w, y), "e": (x + w, y + h / 2),
            "se": (x + w, y + h), "s": (x + w / 2, y + h), "sw": (x, y + h), "w": (x, y + h / 2)}


def box_hit(point, x, y, w, h, radius, body=True):
    """The part of the box under `point`: a grip, an edge (its compass side), "body" inside, else None."""
    px, py = point
    for grip, (gx, gy) in box_grips(x, y, w, h).items():
        if math.hypot(px - gx, py - gy) <= radius:
            return grip
    inside_x = x - radius <= px <= x + w + radius
    inside_y = y - radius <= py <= y + h + radius
    if inside_y and abs(px - x) <= radius:
        return "w"
    if inside_y and abs(px - (x + w)) <= radius:
        return "e"
    if inside_x and abs(py - y) <= radius:
        return "n"
    if inside_x and abs(py - (y + h)) <= radius:
        return "s"
    if body and x <= px <= x + w and y <= py <= y + h:
        return "body"
    return None


def box_drag(start, part, dx, dy, minimum=1.0):
    """(x, y, w, h) after dragging `part` of the box `start` by (dx, dy): the opposite side stays put; "body" moves it."""
    x, y, w, h = start
    if part == "body":
        return x + dx, y + dy, w, h
    if "w" in part:
        dx = min(dx, w - minimum); x += dx; w -= dx
    if "e" in part:
        w = max(minimum, w + dx)
    if "n" in part:
        dy = min(dy, h - minimum); y += dy; h -= dy
    if "s" in part:
        h = max(minimum, h + dy)
    return x, y, w, h


def box_from_knobs(kind, values, width, height):
    """The box (x, y, w, h) in data coordinates, rows counting down, that `kind`'s box knobs in `values` describe."""
    names = BOX_NODES[kind]
    a, b, c, d = (float(values[n]) for n in names)
    if kind not in CORNER_BOX_NODES:
        return a, b, c, d
    if c <= a or d <= b:                       # the default (all zero) or an empty box reads the whole input
        return 0.0, 0.0, float(width), float(height)
    return a, float(height) - d, c - a, d - b


def knobs_from_box(kind, box, height):
    """{knob: value} for `kind` from a box (x, y, w, h) in data coordinates, rows counting down."""
    names = BOX_NODES[kind]
    x, y, w, h = box
    if kind not in CORNER_BOX_NODES:
        return dict(zip(names, (x, y, w, h)))
    return dict(zip(names, (x, float(height) - (y + h), x + w, float(height) - y)))
