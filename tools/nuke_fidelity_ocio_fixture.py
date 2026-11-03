"""A small OpenColorIO config whose transforms are stated by hand, for the OCIO nodes' behaviour cases (2D behaviour lane 2, step F).

The OCIO nodes run OpenColorIO itself, so the question is whether NodeBased hands OCIO the right request (the right spaces, direction, look,
display, view and file). The expected pixels are therefore written from the transforms below and never taken from OCIO:

* `lin` is the scene-linear space and the reference space; `compositing_log` is the role of `cmplog`.
* `gain2`: lin to gain2 multiplies the colour by 2 (a matrix, 2 on the diagonal of R, G and B, alpha untouched).
* `encoded`: a power-law space; its colour is the linear colour to the power of 1 / 2.2 (the config states the other way: linear = encoded ** 2.2).
* `cmplog` is the same power law (the compositing_log role), so OCIOLogConvert "log to lin" raises to 2.2 and "lin to log" takes the 2.2th root.
* The look `double` multiplies the colour by 2 in `lin`.
* The display `Mon` has the views `Plain` (the `encoded` space) and `Bright` (the `gain2` space); the display `Alt` has the one view `Plain`, the `gain2` space.
* `bent.cube` is a 1D LUT with three entries per channel: 0, M, 1 with M = (0.75, 0.25, 0.5) for R, G, B, so the curve rises from 0 to M over the
  first half of 0 to 1 and from M to 1 over the second, straight between the entries.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

GAIN = 2.0
GAMMA = 2.2
LUT_MID = (0.75, 0.25, 0.5)

_MATRIX = "[%s, 0, 0, 0, 0, %s, 0, 0, 0, 0, %s, 0, 0, 0, 0, 1]" % ((GAIN,) * 3)
CONFIG = f"""ocio_profile_version: 2

environment: {{}}
search_path: ""
strictparsing: true
luma: [0.2126, 0.7152, 0.0722]
name: nodebased-fixture
description: Stated transforms for the OCIO behaviour cases

roles:
  scene_linear: lin
  compositing_log: cmplog
  default: lin
  reference: lin

file_rules:
  - !<Rule> {{name: Default, colorspace: lin}}

displays:
  Mon:
    - !<View> {{name: Plain, colorspace: encoded}}
    - !<View> {{name: Bright, colorspace: gain2}}
  Alt:
    - !<View> {{name: Plain, colorspace: gain2}}

active_displays: [Mon, Alt]
active_views: [Plain, Bright]

looks:
  - !<Look>
    name: double
    process_space: lin
    transform: !<MatrixTransform> {{matrix: {_MATRIX}}}

colorspaces:
  - !<ColorSpace>
    name: lin
    isdata: false
    allocation: uniform

  - !<ColorSpace>
    name: gain2
    isdata: false
    from_scene_reference: !<MatrixTransform> {{matrix: {_MATRIX}}}

  - !<ColorSpace>
    name: encoded
    isdata: false
    to_scene_reference: !<ExponentTransform> {{value: [{GAMMA}, {GAMMA}, {GAMMA}, 1]}}

  - !<ColorSpace>
    name: cmplog
    isdata: false
    to_scene_reference: !<ExponentTransform> {{value: [{GAMMA}, {GAMMA}, {GAMMA}, 1]}}
"""

CUBE = "LUT_1D_SIZE 3\n" + "".join("0 0 0\n" if i == 0 else "1 1 1\n" if i == 2 else " ".join(repr(m) for m in LUT_MID) + "\n" for i in range(3))


def write(directory):
    """Write the config and the LUT into `directory` (made if it is missing); returns (config path, cube path) as strings."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    config, cube = directory / "fixture-config.ocio", directory / "bent.cube"
    config.write_text(CONFIG, encoding="utf-8")
    cube.write_text(CUBE, encoding="utf-8")
    return str(config), str(cube)


# ---------------------------------------------------------------------------------------------- the stated transforms, in float64
def gain(x):
    return np.asarray(x, np.float64) * GAIN


def encode(x):
    """lin to `encoded` (and to `cmplog`): the 2.2th root. Defined for x >= 0; OCIO clamps a negative base to 0 for a power law."""
    return np.power(np.maximum(np.asarray(x, np.float64), 0.0), 1.0 / GAMMA)


def decode(x):
    return np.power(np.maximum(np.asarray(x, np.float64), 0.0), GAMMA)


def lut(x):
    """The bent 1D LUT per channel (the last axis is R, G, B); the input is clamped to 0..1, as a 1D LUT's domain is."""
    x = np.clip(np.asarray(x, np.float64), 0.0, 1.0)
    m = np.array(LUT_MID, np.float64)
    return np.where(x <= 0.5, 2.0 * x * m, m + (x - 0.5) * 2.0 * (1.0 - m))


def lut_inverse(y):
    y = np.clip(np.asarray(y, np.float64), 0.0, 1.0)
    m = np.array(LUT_MID, np.float64)
    return np.where(y <= m, y / (2.0 * m), 0.5 + (y - m) / (2.0 * (1.0 - m)))
