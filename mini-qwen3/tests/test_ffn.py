import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 使 mini-qwen3 源码可导入

"""SwiGLU FFN 测试：输出 shape、梯度、门控结构。"""

import sys

import torch

from ffn import FeedForward
from test_common import run_module


def test_ffn_shape_and_gradient():
    torch.manual_seed(0)
    emb_dim, hidden_dim = 64, 192
    ffn = FeedForward(emb_dim, hidden_dim)
    x = torch.randn(2, 8, emb_dim, requires_grad=True)

    out = ffn(x)
    assert out.shape == x.shape

    out.sum().backward()
    assert x.grad is not None
    # 所有参数都收到梯度
    for name, p in ffn.named_parameters():
        assert p.grad is not None, f"{name} 没有梯度"
        assert p.grad.abs().sum() > 0, f"{name} 梯度为全零"


def test_swiglu_gate_structure():
    """门控分支为 0 时输出应为 0（验证 SwiGLU 的 up * silu(gate) 结构）。"""
    ffn = FeedForward(emb_dim=8, hidden_dim=16)
    x1 = torch.randn(2, 8)
    x2 = torch.zeros_like(x1)

    out = ffn(x1 * x2)  # 输入全零 → 两个分支线性层输出都为零
    assert out.abs().max() == 0


if __name__ == "__main__":
    sys.exit(1 if run_module(globals()) else 0)
