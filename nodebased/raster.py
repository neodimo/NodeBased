"""Pixels plus the two rectangles OpenEXR has always carried: data window and display window.

See `docs/BOUNDING_BOX.md` for the contract. In one line: an array on its own cannot say *where*
it lives, so an image whose real pixels extend past the frame — overscan kept for stabilisation,
retiming or motion blur — had nowhere to go and was silently clipped at ingest.

Coordinates match the evaluator's array layout: x to the right, y down, origin at the display
window's top-left corner. The display window is therefore always rebased to (0, 0) by the reader;
the data window may sit anywhere, including at negative coordinates and extending beyond the
display window's extent.

`Region` (from `tiers.py`) is reused as the rectangle type rather than a second, parallel geometry
class — the ROI scheduler and the bounding box are talking about the same kind of rectangle, and
two implementations would eventually disagree.
"""
from __future__ import annotations

import numpy as np

from .tiers import Region


class Raster:
    """An RGBA float32 array together with its data window and display window.

    Invariant: ``pixels.shape[:2] == (data.height, data.width)``. It is checked on construction
    because every downstream alignment silently produces garbage once it drifts.
    """

    __slots__ = ("pixels", "data", "display")

    def __init__(self, pixels, data: Region | None = None, display: Region | None = None):
        pixels = np.asarray(pixels, dtype=np.float32)
        if pixels.ndim != 3 or pixels.shape[2] != 4:
            raise ValueError(f"Raster expects an HxWx4 array, got {pixels.shape}")
        height, width = pixels.shape[:2]
        if data is None:
            data = Region(0, 0, width, height)
        if (data.width, data.height) != (width, height):
            raise ValueError(
                f"Data window {data} does not match pixel extent {(width, height)}")
        self.pixels = pixels
        self.data = data
        self.display = data if display is None else display

    # -- construction helpers ---------------------------------------------------------------

    @staticmethod
    def of(pixels, display: Region | None = None) -> "Raster":
        """A raster whose data window is exactly its array, anchored at the display origin."""
        return Raster(pixels, None, display)

    def with_pixels(self, pixels, data: Region | None = None) -> "Raster":
        """Same windows, new pixels. `data` defaults to this raster's own data window."""
        return Raster(pixels, self.data if data is None else data, self.display)

    # -- geometry ---------------------------------------------------------------------------

    @property
    def nbytes(self) -> int:
        return self.pixels.nbytes

    @property
    def shape(self):
        return self.pixels.shape

    @property
    def has_overscan(self) -> bool:
        """True when the data window reaches outside the display window on any side."""
        return self.data.union(self.display) != self.display

    def fit(self, box: Region) -> np.ndarray:
        """This raster's pixels laid into `box`, zero-filled where the data window does not reach.

        Zero — not edge extension — because a premultiplied RGBA zero *is* "no pixel here". Padding
        with edge values would invent coverage that the source never claimed.
        """
        if box == self.data:
            return self.pixels
        out = np.zeros((box.height, box.width, 4), dtype=np.float32)
        overlap = box.intersect(self.data)
        if not overlap.is_empty:
            out[overlap.y - box.y:overlap.bottom - box.y,
                overlap.x - box.x:overlap.right - box.x] = self.pixels[
                    overlap.y - self.data.y:overlap.bottom - self.data.y,
                    overlap.x - self.data.x:overlap.right - self.data.x]
        return out

    def aligned(self, box: Region) -> "Raster":
        return Raster(self.fit(box), box, self.display)

    def to_display(self) -> np.ndarray:
        """The array a caller asking for "the frame" expects: exactly the display window.

        This is the boundary where overscan is dropped, and it is the *only* place that should
        drop it. Everything upstream keeps it.
        """
        return self.fit(self.display)

    def cropped_to_display(self) -> "Raster":
        """Discard overscan but keep the raster type. Used by export, which writes a frame."""
        return Raster(self.to_display(), self.display, self.display)

    def __eq__(self, other):
        return (isinstance(other, Raster) and self.data == other.data
                and self.display == other.display
                and np.array_equal(self.pixels, other.pixels))

    def __repr__(self):
        tag = " overscan" if self.has_overscan else ""
        return f"<Raster data={self.data} display={self.display}{tag}>"


def as_raster(value, display: Region | None = None) -> Raster:
    """Accept either a bare array (legacy call sites) or a Raster."""
    return value if isinstance(value, Raster) else Raster.of(value, display)


def as_array(value) -> np.ndarray:
    """The display-window array for a Raster; the value itself for a bare array."""
    return value.to_display() if isinstance(value, Raster) else value


def scale_window(box: Region, tier: int, width: int, height: int) -> Region:
    """The data window of a decimated raster.

    The origin floors into tier space and the extent is taken from the array that decimation
    actually produced, rather than recomputing it from the untiered rectangle: `Region.scaled`
    rounds both edges outward and can land one pixel wider than `ceil(n / tier)` when the origin
    is not a multiple of the tier. A window that disagrees with its own pixels by one row is the
    kind of bug that only shows up on a plate with an odd overscan margin.
    """
    tier = int(tier)
    if tier == 1:
        return box
    return Region(box.x // tier, box.y // tier, int(width), int(height))
