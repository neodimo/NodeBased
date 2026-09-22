"""Small, Qt-free geometry helpers for the Viewer 2D Transform handle."""

import math


def forward_point(x, y, translate_x, translate_y, rotate, scale, center_x, center_y):
    """Map a data-space point through the Transform node's forward affine map."""
    theta = math.radians(rotate)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    u = (x - center_x) * scale
    v = (y - center_y) * scale
    return (
        u * cos_t - v * sin_t + center_x + translate_x,
        u * sin_t + v * cos_t + center_y + translate_y,
    )


def pivot_point(translate_x, translate_y, center_x, center_y):
    """Return the Transform pivot in output/displayed coordinates."""
    return center_x + translate_x, center_y + translate_y


def box_corners(width, height, translate_x, translate_y, rotate, scale, center_x, center_y):
    """Return the forward-mapped corners of a width by height source rectangle."""
    return [
        forward_point(x, y, translate_x, translate_y, rotate, scale, center_x, center_y)
        for x, y in ((0, 0), (width, 0), (width, height), (0, height))
    ]


def edge_midpoints(corners):
    """Return the midpoints of the four consecutive quadrilateral edges."""
    return [
        ((corners[index][0] + corners[(index + 1) % 4][0]) / 2,
         (corners[index][1] + corners[(index + 1) % 4][1]) / 2)
        for index in range(4)
    ]


def pivot_drag_result(translate_x, translate_y, rotate, scale, center_x, center_y, dx, dy):
    """Move the pivot by ``(dx, dy)`` while leaving the rendered image unchanged."""
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
