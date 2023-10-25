import string
from abc import ABCMeta, abstractmethod
from typing import Any, Dict, Tuple

import numpy as np
from diffusers import StableDiffusionPipeline

from ..utils import Device


class BaseEditor(metaclass=ABCMeta):

    def __init__(
        self,
        stable_diffusion: Dict,
        device: Device = 'cuda:0',
    ) -> None:
        self.sd_model = StableDiffusionPipeline.from_pretrained(
            **stable_diffusion).to(device)
        self.device = device
        self._id_code = self.generate_id_code()

    @staticmethod
    def generate_id_code() -> str:
        rng = np.random.default_rng()
        return ''.join(rng.choice(list(string.ascii_lowercase + string.digits), 5))

    @property
    def id_code(self) -> str:
        return self._id_code

    @id_code.setter
    def id_code(self, id_code: str) -> None:
        self._id_code = id_code

    @abstractmethod
    def state_dict(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        pass
