import os.path as osp
from copy import deepcopy
from typing import Any, Dict, List

import pandas as pd
import torch
from diffusers import StableDiffusionPipeline
from mmengine import ProgressBar, mkdir_or_exist
from PIL import Image

from ..utils import Device, manual_seed, setup_logger
from .builder import GENERATORS


@GENERATORS.register_module()
class StableDiffusionGenerator:
    """Generator using Stable Diffusion.

    Args:
        prompts_path: Path to the prompts. It should point to a csv file, which
            will be loaded into a dataframe.
        stable_diffusion: The repo name of Stable Diffusion model.
        inference_cfg: Configurations for inference.
        from_case: Starting case index in the prompts dataframe (0-based).
            One case corresponds to one prompt.
        till_case: Ending case index in the prompts dataframe.
        device: Device name to perform inference.
    """

    def __init__(
        self,
        prompts_path: str,
        stable_diffusion: str = 'stabilityai/stable-diffusion-2-1-base',
        inference_cfg: Dict[str, Any] = dict(
            guidance_scale=7.5,
            height=512,
            width=512,
            num_inference_steps=100,
            num_images_per_prompt=1,
        ),
        from_case: int = 0,
        till_case: int = 1000000,
        device: Device = 'cuda:0',
    ) -> None:

        self.from_case = from_case
        self.till_case = till_case
        self.inference_cfg = deepcopy(inference_cfg)
        self.device = torch.device(device) if isinstance(device, str) else device

        self.pipeline = StableDiffusionPipeline.from_pretrained(stable_diffusion).to(
            self.device)

        self.prompts_df = pd.read_csv(prompts_path)

    def load_state_dict(self, ckpt_path: str) -> None:
        logger = setup_logger('uce')
        state_dict = torch.load(ckpt_path, map_location=self.device)
        self.pipeline.unet.load_state_dict(state_dict)
        logger.info(f'Loaded checkpoint: {ckpt_path}')

    @torch.no_grad()
    def generate(self, out_path: str) -> None:
        logger = setup_logger('uce')
        mkdir_or_exist(out_path)

        pbar = ProgressBar(len(self.prompts_df))

        for _, row in self.prompts_df.iterrows():
            prompt = str(row.prompt)
            seed = row.evaluation_seed
            case_number = row.case_number
            if not (self.from_case <= case_number <= self.till_case):
                continue

            manual_seed(seed)

            imgs: List[Image] = self.pipeline(prompt, **self.inference_cfg)[0]
            for img_ind, img in enumerate(imgs):
                img_path = osp.join(out_path, f'{case_number}_{img_ind}.png')
                img.save(img_path)

            pbar.update(1)

        pbar.file.write('\n')
        pbar.file.flush()
        logger.info(f'Generated images are saved to: {out_path}')
