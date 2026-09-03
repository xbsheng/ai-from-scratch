import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 使 mini-qwen3 源码可导入

"""GQA 注意力测试：结构、mask 语义、kv_cache、dtype、qk_norm、梯度。"""

import sys

import torch

from gqa import GroupedQueryAttention
from rope import build_rope_table
from test_common import run_module


def _causal(seq_len: int):
    """(1,1,seq,seq) 因果 mask，True=屏蔽未来。"""
    return torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1).unsqueeze(0).unsqueeze(0)


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
    causal = _causal(4)
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

    # 方式 A：一次性全量（因果 mask）
    out_full, _ = gqa(x, _causal(4), sin, cos)

    # 方式 B：逐 token + kv_cache（每步可见所有已生成 token）
    cache = None
    for t in range(4):
        x_t = x[:, t : t + 1]
        mask_t = torch.zeros(1, 1, 1, t + 1, dtype=torch.bool)
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
    sys.exit(1 if run_module(globals()) else 0)
