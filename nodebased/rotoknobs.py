"""Nuke's Roto knobs that live on the node itself (fidelity step F): the output mask channel and the global motion blur.

The per-shape knobs (feather on and falloff, the transform's skew and extra matrix, motion blur samples/shutter/offset, lifetime and
the open-spline end caps) are fields of each shape in the node's payload; `nodebased/shapes.py` validates and resolves them and
`nodebased/roto.py` draws them. This module holds the node-level parameters, which an older document gets at the values that leave
its Roto as it was.

* `outputMask`: the name of a layer. The shapes' matte (after the node's opacity) is also written to that layer; "none" writes none.
* `shape_motionblur_on`: when on (the default), a shape's own motion blur settings count.
* `global_motionblur_on`: applies the global samples, shutter and offset to every shape that has no motion blur of its own.
"""
from __future__ import annotations

PARAMS = {"outputMask": "none", "shape_motionblur_on": 1, "global_motionblur_on": 0, "global_motionblur": 1,
          "global_motionblur_shutter": 0.5, "global_motionblur_shutter_offset_type": "start",
          "global_motionblur_shutter_offset": 0.0}
KINDS = ("Roto",)


def new_params(kind):
    return dict(PARAMS) if kind in KINDS else {}


def upgrade_params(kind):
    return dict(PARAMS) if kind in KINDS else {}
