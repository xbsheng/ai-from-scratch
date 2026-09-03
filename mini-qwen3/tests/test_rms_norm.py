import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 使 mini-qwen3 源码可导入

"""RMSNorm 测试：归一化性质 + fp32 计算（qwen3_compatible）。"""

import sys

import torch

from rms_norm import RMSNorm
from test_common import run_module


def test_rms_norm_shape_and_normalization():
    torch.manual_seed(0)
    x = torch.randn(4, 16) * 3 + 1  # 非零均值、任意尺度
    m = RMSNorm(16)
    out = m(x)

    assert out.shape == x.shape
    # RMSNorm 输出每行均方根≈1
    rms = out.pow(2).mean(-1).sqrt()
    torch.testing.assert_close(rms, torch.ones_like(rms), atol=1e-4, rtol=1e-4)


def test_rms_norm_fp32_computation():
    """qwen3_compatible=True + bf16 输入时，应先在 fp32 计算再转回 bf16。

    曾出现过 x.to(torch.float32) 未重新赋值导致全程 bf16 计算的 bug，此测试用于拦截。
    """
    torch.manual_seed(0)
    x = torch.randn(4, 16, dtype=torch.bfloat16) * 3 + 1
    m = RMSNorm(16, qwen3_compatible=True)
    out = m(x)

    # 手动 fp32 参考实现
    xf = x.float()
    variance = xf.pow(2).mean(-1, keepdim=True)
    ref = (xf * torch.rsqrt(variance + m.eps) * m.weight.float()).to(x.dtype)

    # bf16 本身有量化误差，容差取 bf16 精度量级（fp32 计算的误差远小于此）
    torch.testing.assert_close(out.float(), ref.float(), atol=0.002, rtol=0.002)


if __name__ == "__main__":
    sys.exit(1 if run_module(globals()) else 0)
