from __future__ import annotations

from PIL import Image


def pil_to_tensor(image: Image.Image):
    import numpy as np
    import torch

    array = np.asarray(image, dtype=np.uint8)
    if array.ndim == 2:
        array = array[:, :, None]
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()
