import torch
from torch import Tensor, nn


class RMSNorm(nn.Module):
    """
    RMSNorm（Root Mean Square Normalization，均方根归一化）

    是对传统 LayerNorm 的一种简化变体。它去除了均值标准化操作，仅基于输入特征的均方根（RMS）进行缩放，从而在保持归一化效果的同时降低计算复杂度。
    实践表明，均值项在深层 Transformer 中对稳定性贡献有限，因此去除均值不会影响模型性能，反而带来更稳定、更高效的归一化形式。
    目前，RMSNorm 已成为主流 LLM 中最常用的归一化方式。
    """

    def __init__(self, emb_dim: int, eps=1e-6, qwen3_compatible=True):
        super().__init__()

        self.eps = eps  # 避免分母为0
        self.qwen3_compatible = qwen3_compatible

        self.weight = nn.Parameter(torch.ones(emb_dim))

    def forward(self, x: Tensor):
        x_dtype = x.dtype

        # 先把输入 x 暂时转换成 float32，优点：
        # - 提高数值稳定性
        # - 减少低精度（bf16 / fp16）下的舍入误差
        #
        # 在大模型里，LayerNorm / RMSNorm 是非常频繁的操作。
        # 如果直接在 bf16 中做，会容易出现数值不稳定，尤其在大规模训练或生成时
        if self.qwen3_compatible:
            x = x.to(torch.float32)

        variance = x.pow(2).mean(dim=-1, keepdim=True)

        # torch.rsqrt: 先求平方根的倒数
        # 为什么用 rsqrt 而不是 sqrt ？
        # 因为它更高效，直接拿到倒数平方根，避免多一步除法计算。
        # 在深度学习里，这种写法非常常见
        x_norm = x * torch.rsqrt(variance + self.eps) * self.weight

        return x_norm.to(x_dtype)
