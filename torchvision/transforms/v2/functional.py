from __future__ import annotations

from enum import Enum

from PIL import Image


class InterpolationMode(str, Enum):
    NEAREST = "nearest"
    NEAREST_EXACT = "nearest-exact"
    BOX = "box"
    BILINEAR = "bilinear"
    HAMMING = "hamming"
    BICUBIC = "bicubic"
    LANCZOS = "lanczos"


def pil_to_tensor(image: Image.Image):
    from ..functional import pil_to_tensor as _pil_to_tensor

    return _pil_to_tensor(image)


def resize(image, size, interpolation=None, antialias=None):
    _ = antialias
    if isinstance(size, int):
        size = (size, size)
    if hasattr(image, "resize"):
        return image.resize((size[1], size[0]), resample=Image.Resampling.BICUBIC)
    raise TypeError(f"unsupported image type for resize: {type(image)!r}")


def center_crop(image, output_size):
    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    if hasattr(image, "crop"):
        width, height = image.size
        crop_h, crop_w = output_size
        left = max(0, (width - crop_w) // 2)
        top = max(0, (height - crop_h) // 2)
        return image.crop((left, top, left + crop_w, top + crop_h))
    raise TypeError(f"unsupported image type for center_crop: {type(image)!r}")
