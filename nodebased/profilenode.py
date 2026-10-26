"""Nuke's Profile node knobs (fidelity step G): the frame range it profiles, its channels, and the profile panel.

Profile passes its input through untouched and records, for each evaluation, the frame, the wall time and the cache hits of the
subgraph above it (`Evaluator.profile_log`). Its knobs:

* `frame_range` is the range *Profile frame range* (in the panel) runs: Nuke's range text (nodebased/framerange.py); empty is the
  project's range. Running it evaluates the Profile node at every frame of the range, one row per frame.
* `channels` is the channel set the profile is taken for. NodeBased evaluates all four channels of a picture together, so it
  labels each recorded row (the `channels` column) and does not change the picture, which passes through as it does in Nuke.
* `open_profile_panel` opens the profile table in a window of its own (the node's properties panel holds the same table).
"""
from __future__ import annotations

from . import framerange

NEW = {"frame_range": "", "channels": "rgba"}
KINDS = ("Profile",)


def new_params(kind):
    return dict(NEW) if kind in KINDS else {}


def upgrade_params(kind):
    return dict(NEW) if kind in KINDS else {}


def frames_to_profile(params, time):
    """The timeline frames Profile frame range runs: the node's range text, or the project's range when it is empty."""
    project = list(range(int(time["first"]), int(time["last"]) + 1))
    return framerange.parse(params.get("frame_range", ""), project)
