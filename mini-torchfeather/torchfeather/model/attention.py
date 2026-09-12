import torch
from torch import Tensor, nn

from .model_args import DeepSeekV3ModelArgs


class Attention(nn.Module):
    def __init__(self, model_args: DeepSeekV3ModelArgs):
        super().__init__()

    def init_weights(self, init_std: float | None = None, buffer_device: torch.device | None = None):
        pass

    def forward(self, x: Tensor, freqs_cis: Tensor):
        pass
