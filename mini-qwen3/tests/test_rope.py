import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 使 mini-qwen3 源码可导入

"""RoPE 测试：与 Qwen 官方一致、旋转性质、dtype。"""

import sys

import torch

from rope import apply_rope, build_rope_table
from test_common import run_module


def test_rope_position0_identity():
    """位置 0 时 sin=0、cos=1，旋转应为恒等。"""
    x = torch.randn(2, 3, 5, 8)
    sin, cos = build_rope_table(8, 8)
    out = apply_rope(x, sin, cos)
    torch.testing.assert_close(out[:, :, 0], x[:, :, 0])  # 仅位置 0 恒等，其余位置旋转


def test_rope_matches_qwen_official():
    """与 Qwen 官方 rotate_half 实现完全一致。"""

    def qwen_rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    pos = torch.arange(5)
    freqs = 1e6 ** (torch.arange(0, 8, 2).float() / -8)  # Qwen3 rope_base=1e6
    angles = pos.unsqueeze(-1) * freqs
    cos_q = torch.cat([angles.cos(), angles.cos()], -1)
    sin_q = torch.cat([angles.sin(), angles.sin()], -1)
    x = torch.randn(2, 4, 5, 8)
    torch.testing.assert_close(apply_rope(x, sin_q, cos_q), x * cos_q + qwen_rotate_half(x) * sin_q)


def test_rope_preserves_norm():
    """旋转不改变向量范数。"""
    x = torch.randn(2, 4, 7, 8)
    sin, cos = build_rope_table(8, 8)
    out = apply_rope(x, sin, cos)
    torch.testing.assert_close(out.pow(2).sum(-1), x.pow(2).sum(-1), atol=1e-5, rtol=0)


def test_rope_rotation_composition():
    """旋转可合成：先转 p 再转 q == 直接转 p+q（验证表与旋转公式自洽）。"""
    x = torch.randn(2, 3, 1, 8)
    sin, cos = build_rope_table(8, 32)
    p, q = 5, 9
    r1 = apply_rope(x, sin[p : p + 1], cos[p : p + 1])
    r2 = apply_rope(r1, sin[q : q + 1], cos[q : q + 1])
    r_direct = apply_rope(x, sin[p + q : p + q + 1], cos[p + q : p + q + 1])
    torch.testing.assert_close(r2, r_direct, atol=1e-5, rtol=0)


def test_rope_bf16_dtype():
    """bf16 输入 + fp32 sin/cos 表 → 输出保持 bf16（回归 dtype 提升 bug）。"""
    x = torch.randn(2, 3, 5, 8, dtype=torch.bfloat16)
    sin, cos = build_rope_table(8, 8)
    out = apply_rope(x, sin, cos)
    assert out.dtype == torch.bfloat16


if __name__ == "__main__":
    sys.exit(1 if run_module(globals()) else 0)
