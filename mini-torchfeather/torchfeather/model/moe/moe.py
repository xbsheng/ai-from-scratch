from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

if TYPE_CHECKING:
    # 仅类型注解用；运行时导入会与 model_args → .moe 形成循环
    from ..model import DeepSeekV3ModelArgs


class MoE(nn.Module):
    def __init__(self, model_args: DeepSeekV3ModelArgs):
        super().__init__()

    def init_weights(self, init_std: float | None = None, buffer_device: torch.device | None = None):
        pass

    def forward(self, x: Tensor):
        pass
