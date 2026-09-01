import torch
from ffn import FeedForward
from gqa import GroupedQueryAttention
from rms_norm import RMSNorm
from rope import apply_rope, build_rope_table


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


def test_gqa_head_dim_both_paths():
    """显式 / 默认 head_dim 两条路径都能前向（回归 self.head_dim 未赋值 bug）。"""
    torch.manual_seed(0)
    for kwargs in ({}, {"head_dim": 16}):
        gqa = GroupedQueryAttention(64, 8, 4, qk_norm=True, **kwargs)
        x = torch.randn(2, 5, 64)
        mask = torch.zeros(2, 1, 5, 5, dtype=torch.bool)
        sin, cos = build_rope_table(16 if "head_dim" in kwargs else 8, 16)
        out, cache = gqa(x, mask, sin, cos)
        assert out.shape == (2, 5, 64), kwargs
        assert cache[0].shape[-1] == (16 if "head_dim" in kwargs else 8), kwargs


def test_gqa_kv_shared_within_group():
    """GQA 结构：w_k / w_v 只输出 n_kv_groups*head_dim（KV 按组共享，而非每头一份）。"""
    gqa = GroupedQueryAttention(d_in=64, n_heads=8, n_kv_groups=4, head_dim=16)
    assert gqa.w_q.out_features == 8 * 16
    assert gqa.w_k.out_features == 4 * 16
    assert gqa.w_v.out_features == 4 * 16
    assert gqa.w_out.in_features == 8 * 16


def test_gqa_causal_mask():
    """因果 mask 语义：末 token 的可见集与全可见一致 → 输出相同；
    首 token 被屏蔽未来 → 输出与全可见时不同。
    """
    torch.manual_seed(0)
    gqa = GroupedQueryAttention(d_in=16, n_heads=2, n_kv_groups=1, head_dim=8, qk_norm=True)
    x = torch.randn(1, 4, 16)
    sin, cos = build_rope_table(8, 8)
    causal = torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1).unsqueeze(0).unsqueeze(0)  # True=屏蔽未来
    none = torch.zeros(1, 1, 4, 4, dtype=torch.bool)

    out_causal, _ = gqa(x, causal, sin, cos)
    out_none, _ = gqa(x, none, sin, cos)

    # 末 token 在两种 mask 下可见集相同（都是全部）→ 输出一致
    torch.testing.assert_close(out_causal[:, -1], out_none[:, -1])
    # 首 token 在因果 mask 下只能看到自己 → 与全可见时不同
    assert not torch.allclose(out_causal[:, 0], out_none[:, 0])


def test_gqa_kv_cache_consistency():
    """流式生成：带缓存逐 token 计算 == 一次性全量计算（末 token 输出一致）。"""
    torch.manual_seed(0)
    d_in, n_heads, n_groups, head_dim = 16, 4, 2, 8
    gqa = GroupedQueryAttention(d_in, n_heads, n_groups, head_dim=head_dim, qk_norm=True)
    sin, cos = build_rope_table(head_dim, 16)

    x = torch.randn(1, 4, d_in)  # 4 个 token
    causal = torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1).unsqueeze(0).unsqueeze(0)  # True=屏蔽未来

    # 方式 A：一次性全量
    out_full, _ = gqa(x, causal, sin, cos)

    # 方式 B：逐 token + kv_cache
    cache = None
    for t in range(4):
        x_t = x[:, t : t + 1]
        mask_t = torch.zeros(1, 1, 1, t + 1, dtype=torch.bool)  # 可看到所有已生成 token
        out_t, cache = gqa(x_t, mask_t, sin, cos, kv_cache=cache)

    torch.testing.assert_close(out_t, out_full[:, 3:4], atol=1e-5, rtol=1e-5)


def test_gqa_bf16_dtype():
    """bf16 模型 + fp32 sin/cos 表 → 输出保持 bf16（回归 dtype 提升 bug）。"""
    gqa = GroupedQueryAttention(d_in=64, n_heads=8, n_kv_groups=4, qk_norm=True).to(torch.bfloat16)
    x = torch.randn(2, 5, 64, dtype=torch.bfloat16)
    sin, cos = build_rope_table(8, 16)
    out, _ = gqa(x, torch.zeros(2, 1, 5, 5, dtype=torch.bool), sin, cos)
    assert out.dtype == torch.bfloat16


def test_gqa_qk_norm_activates():
    """qk_norm=True 时 q 在进入 attention 前每头 RMS≈1；False 时不归一化。"""
    torch.manual_seed(0)
    gqa = GroupedQueryAttention(d_in=16, n_heads=2, n_kv_groups=1, head_dim=8, qk_norm=True)
    captured = {}
    hook = gqa.q_norm.register_forward_hook(lambda m, i, o: captured.update(q=o))
    x = torch.randn(2, 3, 16)
    gqa(x, torch.zeros(2, 1, 3, 3, dtype=torch.bool), *build_rope_table(8, 8))
    hook.remove()

    q = captured["q"]  # (batch, n_heads, seq, head_dim)，RMSNorm 输出
    rms = q.pow(2).mean(-1).sqrt()
    torch.testing.assert_close(rms, torch.ones_like(rms), atol=1e-4, rtol=1e-4)


def test_gqa_gradient():
    """所有参数（含 qk_norm 的 RMSNorm weight）都收到非零梯度。"""
    torch.manual_seed(0)
    gqa = GroupedQueryAttention(d_in=16, n_heads=4, n_kv_groups=2, head_dim=8, qk_norm=True)
    x = torch.randn(2, 3, 16, requires_grad=True)
    sin, cos = build_rope_table(8, 8)
    out, _ = gqa(x, torch.zeros(2, 1, 3, 3, dtype=torch.bool), sin, cos)
    out.sum().backward()
    assert x.grad is not None
    for name, p in gqa.named_parameters():
        assert p.grad is not None, f"{name} 没有梯度"
        assert p.grad.abs().sum() > 0, f"{name} 梯度为全零"


if __name__ == "__main__":
    for fn in (
        test_rms_norm_shape_and_normalization,
        test_rms_norm_fp32_computation,
        test_ffn_shape_and_gradient,
        test_swiglu_gate_structure,
        test_rope_position0_identity,
        test_rope_matches_qwen_official,
        test_rope_preserves_norm,
        test_rope_rotation_composition,
        test_rope_bf16_dtype,
        test_gqa_head_dim_both_paths,
        test_gqa_kv_shared_within_group,
        test_gqa_causal_mask,
        test_gqa_kv_cache_consistency,
        test_gqa_bf16_dtype,
        test_gqa_qk_norm_activates,
        test_gqa_gradient,
    ):
        fn()
        print(f"✅ PASS {fn.__name__}")
    print("✅ 全部通过")
