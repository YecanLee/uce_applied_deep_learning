import os.path as osp
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import plotly.express as px
import timm
import torch
import torch.nn as nn
from alive_progress import alive_bar
from diffusers import DiffusionPipeline, StableDiffusionPipeline
from mmengine import mkdir_or_exist
from PIL import Image
from torch import Tensor
from torch.nn.parameter import is_lazy
from torch.utils.data import DataLoader
from transformers import CLIPTokenizerFast

from ..utils import Device, setup_logger
from .builder import IBA_COMPONENTS
from .info_bottleneck import InformationBottleneck


@IBA_COMPONENTS.register_module()
class IBARunner:

    def __init__(
            self,
            iba: Dict[str, Any],
            estimator_ckpt: Optional[str] = None,
            classifier: Dict[str, Any] = dict(
                model_name='maxvit_tiny_tf_512.in1k', pretrained=True),
            stable_diffusion: str = 'stabilityai/stable-diffusion-2-1-base',
            estimation_cfg: Dict[str, Any] = dict(num_samples=100),
            analysis_cfg: Dict[str, Any] = dict(
                min_num_samples=1000,
                info_loss_weight=1.0,
                lr=1.0,
                batch_size=1,
            ),
            inference_cfg: Dict[str, Any] = dict(
                guidance_scale=7.5,
                height=512,
                width=512,
                num_inference_steps=100,
                num_images_per_prompt=1,
            ),
            device: Device = 'cuda:0') -> None:
        self.device = device
        self.iba: InformationBottleneck = IBA_COMPONENTS.build(
            iba, default_args={'device': device})
        self.logger = setup_logger('uce')

        if estimator_ckpt is not None:
            state_dict = torch.load(estimator_ckpt, map_location=device)
            self.iba.estimator.load_state_dict(state_dict)
            self.iba.init_alpha()
            self.logger.info(f'Loaded IBA estimator checkpoint: {estimator_ckpt}')

        self.inference_cfg = deepcopy(inference_cfg)
        self.estimation_cfg = deepcopy(estimation_cfg)
        self.analysis_cfg = deepcopy(analysis_cfg)

        self.pipeline = StableDiffusionPipeline.from_pretrained(stable_diffusion).to(
            self.device)
        self.pipeline.set_progress_bar_config(disable=True)
        self.freeze_model(self.pipeline)
        self.fast_tokenizer = CLIPTokenizerFast.from_pretrained(
            'openai/clip-vit-base-patch32')

        self.classifier = timm.create_model(**classifier).to(self.device)
        self.freeze_model(self.classifier)

        self.optimizer = torch.optim.Adam(
            lr=self.analysis_cfg['lr'], params=[self.iba.alpha])
        self.loss_closure: Optional[Callable[[torch.Tensor], torch.Tensor]] = None

        self.ori_prompt_embeds: Optional[torch.Tensor] = None

    @staticmethod
    def freeze_model(model: Union[nn.Module, StableDiffusionPipeline]) -> None:
        if isinstance(model, nn.Module):
            model.eval()
            for p in model.parameters():
                p.requires_grad = False
        else:
            modules = [model.vae, model.text_encoder, model.unet]
            for module in modules:
                module.eval()
                for p in module.parameters():
                    p.requires_grad = False

    def reset_optimizer(self) -> None:
        self.optimizer.zero_grad()
        self.optimizer = torch.optim.Adam(
            lr=self.analysis_cfg['lr'], params=[self.iba.alpha])

    def build_loss_closure(self, class_label: int) -> None:

        def compute_cls_loss(latents: Tensor) -> torch.Tensor:
            factor = self.pipeline.vae.config.scaling_factor
            decoded_image = self.pipeline.vae.decode(
                latents / factor, return_dict=False)[0]
            loss = -torch.log_softmax(self.classifier(decoded_image), 1)[:, class_label]
            loss = loss.mean()
            return loss

        self.loss_closure = compute_cls_loss

    def estimate_callback(
            self,
            pipe: DiffusionPipeline,
            step_index: int,
            timestep: int,
            callback_kwargs: Dict) -> Dict:
        if step_index == 0:
            prompt_embeds = callback_kwargs['prompt_embeds']
            # in classifier guidance, prompt_embeds are concatenated by
            # unconditioned embeddings and prompt embeddings. We only need
            # the second half.
            self.iba.update_estimator(prompt_embeds.chunk(2)[1])

        return callback_kwargs

    def run_estimation(self, data_loader: DataLoader) -> None:
        self.logger.info('Running estimation.')
        num_samples = self.estimation_cfg['num_samples']
        device = self.iba.device

        est_inference_cfg = deepcopy(self.inference_cfg)
        if est_inference_cfg.get('num_images_per_prompt', 1) != 1:
            est_inference_cfg['num_images_per_prompt'] = 1
            self.logger.info(
                'During estimation, inference_cfg.num_images_per_prompt '
                'should be 1. Now it is manually set to 1.')

        with alive_bar(total=num_samples, enrich_print=False) as bar:

            for batch in data_loader:
                if self.iba.estimator.num_samples() >= num_samples:
                    break

                prompt_list = batch['prompt']
                seed_list = batch['seed']
                generator_list = [
                    torch.Generator(device=device).manual_seed(s) for s in seed_list
                ]

                _ = self.pipeline(
                    prompt_list,
                    generator=generator_list,
                    callback_on_step_end=self.estimate_callback,
                    callback_on_step_end_tensor_inputs=['prompt_embeds'],
                    **self.inference_cfg)

                bar(len(prompt_list))

        # After estimation, feature map dimensions are known
        # and we can initialize alpha
        self.iba.init_alpha()

    def reset_ori_prompt_embeds(self) -> None:
        if self.ori_prompt_embeds is not None:
            self.ori_prompt_embeds.detach_()
            self.ori_prompt_embeds = None

    @torch.enable_grad()
    def analyze_callback(
            self,
            pipe: DiffusionPipeline,
            step_index: int,
            timestep: int,
            callback_kwargs: Dict) -> Dict:
        num_timesteps = self.pipeline.num_timesteps
        if int(0.8 * num_timesteps) <= step_index:
            # The prompt_embeds will be distorted in the end of this function
            # therefore the prompt_embeds callback_kwargs at next step
            # are already distorted. However, we need to distort the original
            # prompt_embeds every time. So we save a copy first and retrieve it later.
            if self.ori_prompt_embeds is None:
                self.ori_prompt_embeds = callback_kwargs['prompt_embeds'].chunk(
                    2)[1].clone().detach()

            # latents obtained with distorted prompt_embeds
            latents = callback_kwargs['latents']
            info_loss_weight = self.analysis_cfg['info_loss_weight']

            with self.iba.restrict_flow():
                # in the first iteration, iba.capacity is not materialized yet.
                if not is_lazy(self.iba.capacity):
                    cls_loss = self.loss_closure(latents)
                    info_loss = self.iba.get_avg_capacity().mean()
                    loss = cls_loss + info_loss_weight * info_loss
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.iba.total_loss_history.append(loss.item())
                    self.iba.cls_loss_history.append(cls_loss.item())
                    self.iba.info_loss_history.append(info_loss.item())
                    log_str = (
                        f'IBA: step {step_index}, timestep {timestep}, '
                        f'total_loss: {self.iba.total_loss_history[-1]:.5f}, '
                        f'cls loss: {self.iba.cls_loss_history[-1]:.5f}, '
                        f'info loss: {self.iba.info_loss_history[-1]:.5f}')
                    self.logger.info(log_str)

                # distort the original prompt_embeds, the distortion is out-of-place.
                prompt_embeds = self.iba.forward(self.ori_prompt_embeds)
                negative_prompt_embeds = callback_kwargs['prompt_embeds'].chunk(2)[0]
                prompt_embeds = torch.cat(
                    [negative_prompt_embeds, prompt_embeds], dim=0)
                callback_kwargs['prompt_embeds'] = prompt_embeds

        return callback_kwargs

    @torch.no_grad()
    def inference_callback(
            self,
            pipe: DiffusionPipeline,
            step_index: int,
            timestep: int,
            callback_kwargs: Dict) -> Dict:
        if step_index == 0:
            lamb = self.iba.get_lamb().mean(-1)
            self.logger.info(f'DEBUG lamb.min: {lamb.min()}, lamb.max: {lamb.max()}')

        with self.iba.inv_restrict_flow():
            prompt_embeds = self.iba.forward(self.ori_prompt_embeds)
            negative_prompt_embeds = callback_kwargs['prompt_embeds'].chunk(2)[0]
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            callback_kwargs['prompt_embeds'] = prompt_embeds

        return callback_kwargs

    def run_analysis(self, data_loader: DataLoader, work_dir: str) -> None:
        assert data_loader.batch_size == 1
        analysis_image_dir = osp.join(work_dir, 'analysis_images')
        inference_image_dir = osp.join(work_dir, 'inference_images')
        mkdir_or_exist(analysis_image_dir)
        mkdir_or_exist(inference_image_dir)
        smap_list: List[np.ndarray] = []
        token_list: List[np.ndarray] = []

        with alive_bar(total=len(data_loader), enrich_print=False) as bar:

            for batch in data_loader:
                self.reset_ori_prompt_embeds()
                # first reset alpha and then reset optimizer
                self.iba.reset_alpha()
                self.iba.reset_loss_histories()
                self.reset_optimizer()

                case_id = batch['case_id'][0]
                prompt = batch['prompt'][0]
                seed = batch['seed'][0]
                class_label = batch['class_label'][0]
                generator = torch.Generator(self.device).manual_seed(seed)
                self.build_loss_closure(class_label)

                analysis_images: List[Image] = self.pipeline(
                    prompt,
                    generator=generator,
                    callback_on_step_end=self.analyze_callback,
                    callback_on_step_end_tensor_inputs=['prompt_embeds', 'latents'],
                    **self.inference_cfg)[0]
                for image_id, image in enumerate(analysis_images):
                    image_path = osp.join(
                        analysis_image_dir, f'{case_id}_{image_id}.png')
                    image.save(image_path)

                # prepare for visualization
                smap = self.iba.get_saliency()
                smap_list.append(smap)
                tokens = self.fast_tokenizer(
                    prompt,
                    padding='max_length',
                    max_length=self.pipeline.tokenizer.model_max_length,
                    return_tensors='np').tokens()
                token_list.append(tokens)
                assert len(smap) == len(tokens)

                self.logger.info('Analysis finished. Running inference.')
                generator = torch.Generator(self.device).manual_seed(seed)
                inference_images: List[Image] = self.pipeline(
                    prompt,
                    generator=generator,
                    callback_on_step_end=self.inference_callback,
                    callback_on_step_end_tensor_inputs=['prompt_embeds'],
                    **self.inference_cfg)[0]
                for image_id, image in enumerate(inference_images):
                    image_path = osp.join(
                        inference_image_dir, f'{case_id}_{image_id}.png')
                    image.save(image_path)

                bar(len(batch['prompt']))

        self.logger.info('Visualizing saliency maps.')
        self.visualize(
            smap_list, token_list, save_path=osp.join(work_dir, 'smaps.html'))
        self.logger.info(f'Images and saliency maps are saved to: {work_dir}')

    @staticmethod
    def visualize(
            smap_list: List[np.ndarray], token_list: List[np.ndarray],
            save_path: str) -> None:
        all_smap = np.stack(smap_list, axis=0)
        all_smap = all_smap / (all_smap.max(axis=-1, keepdims=True) + 1e-8)
        token_inds = np.arange(len(token_list[0]))
        fig = px.imshow(
            all_smap,
            x=token_inds,
            y=np.arange(len(token_list)),
            color_continuous_scale='Viridis',
            aspect='auto')
        fig.update_traces(
            text=token_list,
            texttemplate='%{text}',
            hovertemplate='%{x}, %{text}: %{customdata:.3f}',
            customdata=all_smap)
        fig.update_xaxes(side='top')
        fig.write_html(save_path)
