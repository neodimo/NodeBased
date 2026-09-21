"""Parity tolerance for adapters that cannot blend in float32.

``gpu3d`` renders the beauty and splat layers to ``rgba32float`` when the adapter
offers ``float32-blendable`` and to ``rgba16float`` when it does not. The splat
parity tests compare those layers against a CPU reference computed in float32, so
on a half-float target the comparison measures the target's precision rather than
the shader. Half float carries about 11 mantissa bits, and the error accumulates
over every splat blended into a pixel: a forced half-float run on an RTX 3080 Ti
reached 8.5e-3, and GitHub's Windows runner (Microsoft Basic Render Driver, which
has no ``float32-blendable``) reached 1.4e-2 against a 2e-3 tolerance.

Widening by ten on those adapters keeps every float32 adapter at the original
tolerance and still fails a shader that is actually wrong, since a wrong shader
misses by whole tenths rather than hundredths.
"""
from nodebased import gpu3d

FACTOR = 10.0


def half_float_target():
    """True when this adapter blends the rgba layers in half precision."""
    return gpu3d.available() and gpu3d._state()['format'] == 'rgba16float'


def tolerance(base):
    """``base`` on a float32 adapter, ten times ``base`` on a half-float one."""
    return base * FACTOR if half_float_target() else base


_announced = False


def note(test):
    """Print once why the tolerances were widened, so the CI log carries the reason."""
    global _announced
    if half_float_target() and not _announced:
        _announced = True
        print(f'\nsplat parity tolerances x{FACTOR:g} from {test} on: this adapter lacks '
              'float32-blendable, so the rgba layers blend as rgba16float', flush=True)
