from __future__ import annotations

from enum import Enum

from . import functional


class InterpolationMode(str, Enum):
    NEAREST = "nearest"
    NEAREST_EXACT = "nearest-exact"
    BOX = "box"
    BILINEAR = "bilinear"
    HAMMING = "hamming"
    BICUBIC = "bicubic"
    LANCZOS = "lanczos"


__all__ = ["InterpolationMode", "functional"]
