from contextlib import contextmanager
from math import log
from typing import Dict, Generator, List, Optional, Sequence, Union

import torch
import torch.nn as nn
from numpy import ndarray
from torch import Tensor
from torch.nn.parameter import is_lazy

from ..utils import Device, setup_logger
from .builder import IBA_COMPONENTS
from .estimator import WelfordEstimator
from .utils import IBAInterrupt


@IBA_COMPONENTS.register_module()
class InformationBottleneck(nn.Module):

    def __init__(
            self,
            init_alpha_val: float = 5.0,
            threshold: float = 0.01,
            min_noise_std: float = 0.01,
            estimate_cfg: Dict = dict(num_samples=1000),
            analyze_cfg: Dict = dict(
                info_loss_weight=10.0,
                num_opt_steps=20,
                lr=1.0,
                # batch_size=1,
                log_every_steps=1,
                num_drop_tokens=1,
            ),
            device: Device = 'cuda:0') -> None:
        super().__init__()
        self._restrict_flow = False
        self._inv_restrict_flow = False
        self._estimate = False
        self._interrupt_execution = False

        self.init_alpha_val = init_alpha_val
        self.device = device
        # capacity is initialized on the first forward pass,
        # capacity has shape (batch_size, seq_length, hidden_dim)
        self.register_buffer('capacity', nn.UninitializedBuffer(device=device))
        # alpha is initialized on the first forward pass,
        # alpha has shape (seq_length, hidden_dim)
        self.alpha = nn.UninitializedParameter(device=self.device, dtype=torch.float32)

        self.threshold = threshold
        self.min_noise_std = min_noise_std
        self.estimator = WelfordEstimator(device=self.device)

        self.estimate_cfg = estimate_cfg
        self.analyze_cfg = analyze_cfg

        self.total_loss_history: List[float] = []
        self.cls_loss_history: List[float] = []
        self.info_loss_history: List[float] = []

        self.logger = setup_logger('uce')

    def reset_estimator(self) -> None:
        self.estimator = WelfordEstimator(device=self.alpha.device)

    @contextmanager
    def interrupt_execution(self) -> Generator:
        self._interrupt_execution = True
        try:
            yield
        except IBAInterrupt:
            pass
        finally:
            self._interrupt_execution = False

    @contextmanager
    def enable_estimation(self) -> Generator:
        self._estimate = True
        try:
            yield
        finally:
            self._estimate = False

    @contextmanager
    def restrict_flow(self) -> Generator:
        """Context manager for restricting the information flow.

        Under this context, the noise should be added to the signal using interpolation.
        Then, the interpolated signal will be forwarded for computing the task loss.
        """
        self._restrict_flow = True
        try:
            yield
        finally:
            self._restrict_flow = False

    @contextmanager
    def inv_restrict_flow(self) -> Generator:
        self._inv_restrict_flow = True
        try:
            yield
        finally:
            self._inv_restrict_flow = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert not (self._restrict_flow and self._inv_restrict_flow)
        if self._restrict_flow:
            return self.do_restrict_flow(x, inverse=False)
        if self._inv_restrict_flow:
            return self.do_restrict_flow(x, inverse=True)
        if self._estimate:
            self.estimator(x)
        if self._interrupt_execution:
            raise IBAInterrupt()
        return x

    def init_alpha(self) -> None:
        num_samples = self.estimator.num_samples()
        if self.estimator.num_samples() <= 0:
            raise ValueError(
                f'Estimator has estimated for {num_samples} samples. Please estimate '
                'the feature distribution before using the bottleneck.')
        if not is_lazy(self.alpha):
            raise ValueError(
                'alpha (nn.UninitializedParameter) is already materialized, which '
                'means that init_alpha is already called. Please call reset_alpha to '
                'reset the parameter')
        shape = self.estimator.mean().shape
        with torch.no_grad():
            self.alpha.materialize(shape)
            self.alpha.fill_(self.init_alpha_val)
            self.logger.info(
                f'alpha is initialized with shape {self.alpha.shape}, '
                f'and filled with initial value: {self.init_alpha_val}')

    def update_estimator(self, x: torch.FloatTensor) -> None:
        with (torch.no_grad(), self.interrupt_execution(), self.enable_estimation()):
            # x is the prompt_embeds in StableDiffusion
            _ = self.forward(x)

    def do_restrict_flow(self, x: torch.Tensor, inverse: bool) -> torch.Tensor:
        if is_lazy(self.alpha):
            raise ValueError('alpha is not initialized.')

        noise_mean = self.estimator.mean()
        noise_std = self.estimator.std()
        noise_std = torch.maximum(
            noise_std, torch.full_like(noise_std, self.min_noise_std))
        active_neurons = self.estimator.active_neurons(self.threshold).float()

        # self.alpha shape is broadcasted
        lamb = self.get_lamb(x.shape)

        if is_lazy(self.capacity):
            with torch.no_grad():
                self.capacity.materialize(x.shape)
                self.logger.info(
                    f'capacity is initialized with shape {self.capacity.shape}')

        self.capacity = self.kl_div(x, lamb, noise_mean, noise_std) * active_neurons
        if not inverse:
            # broadcast will be performed on active_neurons in terms of shapes
            z = self.apply_noise_to_input(
                x=x, lamb=lamb, noise_mean=noise_mean, noise_std=noise_std)
        else:
            # use the average capacity as mask, because the values of lamb after
            # optimization is typically very small. E.g. ranging from 0.0013 to 0.0017,
            # therefore 1 - lamb is close to 1, resulting in a negligible distortion
            # to the features related to the target concept (tokens), which needs to
            # be avoided during image generation.
            # smap shape: (seq_length,)
            smap = self.get_saliency(out_format='tensor')
            # TODO: a question: the introduced noise might change the
            #  generated image, even when other non-target tokens are kept.
            #  This noise might suppress the information from non-target tokens.

            # TODO: think about how to sparsify the smap.
            smap = smap / (smap.max() + 1e-8)
            is_related_token = smap > smap.quantile(0.3)
            smap[is_related_token] = 1
            smap[~is_related_token] = 0

            # self.logger.info(f'DEBUG: smap: {smap}')

            # shape after un-squeeze: (seq_length, hidden_size)
            inv_smap = (1 - smap).unsqueeze(-1)
            z = self.apply_noise_to_input(
                x=x, lamb=inv_smap, noise_mean=noise_mean, noise_std=noise_std)

        z = z * active_neurons

        return z

    def reset_loss_histories(self) -> None:
        self.total_loss_history.clear()
        self.cls_loss_history.clear()
        self.info_loss_history.clear()

    def get_total_loss_history(self) -> List[float]:
        return self.total_loss_history

    def get_cls_loss_history(self) -> List[float]:
        return self.cls_loss_history

    def get_info_loss_history(self) -> List[float]:
        return self.info_loss_history

    @torch.no_grad()
    def reset_alpha(self) -> None:
        if is_lazy(self.alpha):
            raise ValueError('alpha is not materialized. Please call init_alpha first.')
        if self.alpha.grad is not None:
            self.alpha.grad = None
        self.alpha.fill_(self.init_alpha_val)

    def get_avg_capacity(self, detach: bool = False) -> torch.Tensor:
        # average over the batch dimension
        capacity = self.capacity.mean(dim=0)
        if detach:
            capacity = capacity.detach()
        # returned tensor has shape (img_token_length, hidden_dim)
        return capacity

    def get_saliency(self,
                     num_drop_tokens: int = 0,
                     out_format: str = 'ndarray') -> Union[ndarray, Tensor]:
        # sum over the hidden_dim dimension
        smap = torch.nansum(self.get_avg_capacity(detach=True), dim=-1)
        # convert into bits
        smap /= log(2)
        smap = smap[num_drop_tokens:]

        if out_format == 'ndarray':
            smap = smap.cpu().numpy()
        elif out_format == 'tensor':
            pass
        else:
            raise ValueError(
                f"out_format should be in ('ndarray', 'tensor'), "
                f'but got {out_format}')

        return smap

    def get_lamb(self, expand_shape: Optional[Sequence[int]] = None) -> torch.Tensor:
        # Smoothen and expand alpha on batch dimension
        lamb = torch.sigmoid(self.alpha)
        # repeat the mask for each sample.
        if expand_shape is not None:
            lamb = lamb.expand(expand_shape)
        return lamb

    @staticmethod
    def apply_noise_to_input(
            x: torch.Tensor,
            lamb: torch.Tensor,
            noise_mean: torch.Tensor,
            noise_std: torch.Tensor) -> torch.Tensor:
        # eps of each sample will be uncorrelated
        eps = torch.zeros_like(x).normal_()
        eps = noise_std * eps + noise_mean
        z = lamb * x + (1 - lamb) * eps

        return z

    @staticmethod
    def kl_div(
            r: torch.Tensor, lamb: torch.Tensor, mean_r: torch.Tensor,
            std_r: torch.Tensor) -> torch.Tensor:
        """Return the feature-wise KL-divergence of p(z|x) and q(z). See also.

        <original IBA's implementation https://github.com/BioroboticsLab/IBA/>_
        """
        r_norm = (r - mean_r) / std_r
        var_z = (1 - lamb)**2
        log_var_z = torch.log(var_z)
        mu_z = r_norm * lamb
        capacity = -0.5 * (1 + log_var_z - mu_z**2 - var_z)
        return capacity
