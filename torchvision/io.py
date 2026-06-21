from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any


class ImageReadMode(str, Enum):
    RGB = "RGB"
    UNCHANGED = "UNCHANGED"


def decode_image(data: Any, mode: ImageReadMode = ImageReadMode.RGB):
    import io

    import torch
    from PIL import Image

    if isinstance(data, (str, Path)):
        with Image.open(data) as image:
            return _pil_to_tensor(image.convert("RGB" if mode == ImageReadMode.RGB else image.mode))

    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().contiguous().numpy().tobytes()
    elif isinstance(data, (bytes, bytearray, memoryview)):
        data = bytes(data)
    else:
        raise TypeError(f"unsupported image buffer type: {type(data)!r}")

    with Image.open(io.BytesIO(data)) as image:
        return _pil_to_tensor(image.convert("RGB" if mode == ImageReadMode.RGB else image.mode))


def _pil_to_tensor(image):
    import torch
    import numpy as np

    array = np.asarray(image, dtype=np.uint8)
    if array.ndim == 2:
        array = array[:, :, None]
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()
