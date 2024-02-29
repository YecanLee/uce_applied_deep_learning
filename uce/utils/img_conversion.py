from typing import List

import numpy as np
import torch
from numpy.typing import NDArray


def batched_tensor_to_img_list(
        batched_img_tensor: torch.Tensor) -> List[NDArray[np.uint8]]:
    batched_img_tensor = (batched_img_tensor / 2 + 0.5).clamp(0., 1.)
    batched_img_array = batched_img_tensor.detach().cpu().permute(0, 2, 3, 1).numpy()
    batched_img_array = np.clip(
        batched_img_array * 255, a_min=0, a_max=255).astype('uint8')
    img_list = [x for x in batched_img_array]
    return img_list
