from typing import Any, Mapping, Optional, Sequence, Union

import torch
import torch.nn as nn
from torch.nn.parameter import UninitializedBuffer, is_lazy

from ..utils import Device


class WelfordEstimator(nn.Module):

    def __init__(self, device: Optional[Device] = None):
        super().__init__()
        self.register_buffer(
            '_num_samples',
            torch.tensor([0], dtype=torch.long, requires_grad=False, device=device))
        self.register_buffer(
            '_mean',
            UninitializedBuffer(
                dtype=torch.float32, requires_grad=False, device=device))
        self.register_buffer(
            '_std',
            UninitializedBuffer(
                dtype=torch.float32, requires_grad=False, device=device))
        self.register_buffer(
            '_neuron_nonzero',
            UninitializedBuffer(dtype=torch.long, requires_grad=False, device=device))

    def num_samples(self) -> int:
        return self._num_samples.item()

    def mean(self) -> torch.Tensor:
        return self._mean

    def std(self) -> torch.Tensor:
        if self.num_samples() < 2:
            raise ValueError(
                f'Number of tracked samples: {self.num_samples()} is smaller than 2.')
        return torch.sqrt(self._std / (self._num_samples.float() - 1 + 1e-8))

    def active_neurons(self, threshold: float) -> torch.BoolTensor:
        return (self._neuron_nonzero.float() / self._num_samples.float()) > threshold

    @torch.no_grad()
    def _init_buffers(self, shape: Union[torch.Size, Sequence[int]]) -> None:
        """Initialize the buffers.

        Shape should not contain batch dimension.
        """
        if is_lazy(self._mean):
            self._mean.materialize(shape)
            self._mean.fill_(0)
        if is_lazy(self._std):
            self._std.materialize(shape)
            self._std.fill_(0)
        if is_lazy(self._neuron_nonzero):
            self._neuron_nonzero.materialize(shape)
            self._neuron_nonzero.fill_(0)

    @property
    def buffers_initialized(self) -> bool:
        return not (
            is_lazy(self._mean) or is_lazy(self._std) or is_lazy(self._neuron_nonzero))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Update estimates without altering x.

        Args:
            x: Feature map with shape
                ``(batch_size, seq_length, hidden_dim_size)``.

        Returns:
            The same tensor as the input. The input is not modified.
        """
        if not self.buffers_initialized:
            if x.ndim != 3:
                raise ValueError(
                    f'Input tensor should have 3 dimensions, but got shape {x.shape}')
            self._init_buffers(x.shape[1:])

        for xi in x:
            self._neuron_nonzero += (xi != 0.).long()
            old_mean = self._mean.clone()
            self._mean += (xi - self._mean) / (self._num_samples.float() + 1)
            self._std += (xi - self._mean) * (xi - old_mean)
            self._num_samples += 1
        return x

    def load_state_dict(
            self,
            state_dict: Mapping[str, Any],
            strict: bool = True,
            assign: bool = False) -> None:
        assert ('_mean' in state_dict) and \
               ('_std' in state_dict) and \
               ('_neuron_nonzero' in state_dict)

        shape = state_dict['_mean'].shape
        self._init_buffers(shape)
        super().load_state_dict(state_dict, strict=strict, assign=assign)
