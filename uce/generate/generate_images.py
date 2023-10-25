import os.path as osp
from typing import Tuple

import cv2
import pandas as pd
import torch
from diffusers import AutoencoderKL, LMSDiscreteScheduler, UNet2DConditionModel
from mmengine import ProgressBar, mkdir_or_exist
from transformers import CLIPTextModel, CLIPTokenizer

from ..utils import Device, batched_tensor_to_img_list, setup_logger


class ImageGenerator:

    def __init__(
        self,
        prompts_path: str,
        base='1.4',
        guidance_scale: float = 7.5,
        img_size: Tuple[int, int] = (512, 512),
        ddim_steps: int = 100,
        num_samples: int = 10,
        from_case: int = 0,
        till_case: int = 1000000,
        device: Device = 'cuda:0',
    ) -> None:

        self.guidance_scale = guidance_scale
        self.img_size = img_size
        self.ddim_steps = ddim_steps
        self.num_samples = num_samples
        self.from_case = from_case
        self.till_case = till_case
        self.device = device

        if base == '2.1':
            model_version = 'stabilityai/stable-diffusion-2-1-base'
        else:
            model_version = 'CompVis/stable-diffusion-v1-4'

        self.vae = AutoencoderKL.from_pretrained(
            model_version, subfolder='vae').to(device)

        self.tokenizer = CLIPTokenizer.from_pretrained(
            model_version, subfolder='tokenizer')
        self.text_encoder = CLIPTextModel.from_pretrained(
            model_version, subfolder='text_encoder').to(device)

        self.unet = UNet2DConditionModel.from_pretrained(
            model_version, subfolder='unet').to(device)
        self.scheduler = LMSDiscreteScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule='scaled_linear',
            num_train_timesteps=1000)

        self.prompts_df = pd.read_csv(prompts_path)

    def load_state_dict(self, ckpt_path: str) -> None:
        logger = setup_logger('uce')
        state_dict = torch.load(ckpt_path, map_location=self.device)
        self.unet.load_state_dict(state_dict)
        logger.info(f'Loaded checkpoint: {ckpt_path}')

    @torch.no_grad()
    def generate(self, out_path: str) -> None:
        logger = setup_logger('uce')
        mkdir_or_exist(out_path)

        pbar = ProgressBar(len(self.prompts_df))

        for _, row in self.prompts_df.iterrows():
            prompt = [str(row.prompt) for _ in range(self.num_samples)]
            seed = row.evaluation_seed
            case_number = row.case_number
            if not (self.from_case <= case_number <= self.till_case):
                continue

            generator = torch.manual_seed(seed)
            batch_size = len(prompt)
            img_height, img_width = self.img_size

            text_input = self.tokenizer(
                prompt,
                padding='max_length',
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors='pt')
            # text_embeddings: (batch_size, sequence_length, hidden_size)
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

            # TODO: check whether here should be shape[1],
            #  since the text_embeddings.shape[-1] is the hidden_size
            max_length = text_input.input_ids.shape[-1]
            uncond_input = self.tokenizer(
                ['' for _ in range(batch_size)],
                padding='max_length',
                max_length=max_length,
                return_tensors='pt')
            uncond_embeddings = self.text_encoder(
                uncond_input.input_ids.to(self.device))[0]

            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
            latents = torch.randn(
                (batch_size, self.unet.in_channels, img_height // 8, img_width // 8),
                generator=generator,
                device=self.device)
            self.scheduler.set_timesteps(self.ddim_steps)
            latents = latents * self.scheduler.init_noise_sigma

            for t in range(self.scheduler.timesteps):
                # expand teh latents if we are doing classifier-free guidance to avoid
                #  doing two forward passes.
                latent_model_input = torch.cat([latents] * 2)
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, timestep=t)

                noise_pred = self.unet(
                    latent_model_input, t, encoder_hidden_states=text_embeddings).sample
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (
                    noise_pred_text - noise_pred_uncond)

                # compute the previous sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample

            # scale and decode the image latents with VAE
            latents = 1 / 0.18215 * latents
            imgs = self.vae.decode(latents).sample
            imgs = batched_tensor_to_img_list(imgs)
            for img_ind, img in enumerate(imgs):
                img_path = osp.join(out_path, f'{case_number}_{img_ind}.png')
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imsave(img_path, img)

            pbar.update(1)

        pbar.file.write('\n')
        pbar.file.flush()
        logger.info(f'Generated images are saved to: {out_path}')
