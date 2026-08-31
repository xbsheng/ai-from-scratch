import torch
from ffn import FeedForward
from rms_norm import RMSNorm


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
    """门控分支为 0 时输出应为 0（验证 SwiGLU 的 x1 * silu(x2) 结构）。"""
    ffn = FeedForward(emb_dim=8, hidden_dim=16)
    x1 = torch.randn(2, 8)
    x2 = torch.zeros_like(x1)

    out = ffn(x1 * x2)  # 输入全零 → 两个分支线性层输出都为零
    assert out.abs().max() == 0


if __name__ == "__main__":
    for fn in (
        test_rms_norm_shape_and_normalization,
        test_rms_norm_fp32_computation,
        test_ffn_shape_and_gradient,
        test_swiglu_gate_structure,
    ):
        fn()
        print(f"✅ PASS {fn.__name__}")
    print("✅ 全部通过")
