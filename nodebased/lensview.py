"""The viewer's "view undistorted / view plate" switch for a graph with an Undistort node (Rendering 13 step L3).

Pure document logic, no Qt: the strip button in `viewerstrip.py` and the tests both ask these functions. An Undistort node
grows its frame past the format (the overscan); the viewer shows that whole frame for the node (`shows_overscan`), so the
artist can check that the overscan edge holds the picture a Redistort will read, and "view plate" returns to the node's
input, the plate as it was filmed.
"""
from __future__ import annotations

UNDISTORTED, PLATE = "undistorted", "plate"
LABELS = {UNDISTORTED: "View undistorted", PLATE: "View plate"}
TIPS = {UNDISTORTED: "Show the plate straightened by the Undistort node, with the overscan the lens adds beyond the format",
        PLATE: "Show the plate as filmed, the Undistort node's input"}


def undistort_nodes(document):
    """Ids of the Undistort nodes that have a plate wired in, in the document's order."""
    return [key for key, node in document["nodes"].items()
            if node["type"] == "Undistort" and not node.get("disabled") and node["inputs"].get("image") in document["nodes"]]


def lens_node(document):
    """The Undistort node the switch works on: the viewed one (or the one the viewed plate feeds), else the first."""
    nodes = undistort_nodes(document)
    if not nodes:
        return None
    view = document.get("view")
    for key in nodes:
        if view in (key, document["nodes"][key]["inputs"]["image"]):
            return key
    return nodes[0]


def plate_of(document, undistort):
    return document["nodes"][undistort]["inputs"]["image"]


def state(document):
    """What the viewer shows, as far as the switch is concerned: "undistorted", "plate" or None (neither, or no Undistort)."""
    key = lens_node(document)
    if key is None:
        return None
    view = document.get("view")
    if view == key:
        return UNDISTORTED
    return PLATE if view == plate_of(document, key) else None


def next_view(document):
    """What the button does next: (the mode it switches to, the node to view), or None without an Undistort node.

    From the undistorted view it goes to the plate; from anywhere else it goes to the undistorted frame."""
    key = lens_node(document)
    if key is None:
        return None
    if state(document) == UNDISTORTED:
        return PLATE, plate_of(document, key)
    return UNDISTORTED, key


def shows_overscan(document, target):
    """True when the viewer shows `target` as its whole grown frame, not only the format: an Undistort node."""
    node = document["nodes"].get(target) if target else None
    return bool(node and node["type"] == "Undistort")
