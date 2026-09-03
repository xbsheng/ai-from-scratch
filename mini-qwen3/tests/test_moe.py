import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 使 mini-qwen3 源码可导入

"""MoE 测试：前向正确性（vs 逐 token 参考）、梯度、router 归一化、边界、有限差分。"""

import sys

import torch
from torch.nn import functional as F

from moe import MoEBlock, TopKRouter
from test_common import run_module

H, E, K, I = 8, 4, 2, 16  # hidden / experts / top_k / moe_intermediate


def make_moe(top_k=K, norm_topk_prob=True) -> MoEBlock:
    torch.manual_seed(0)
    return MoEBlock(top_k=top_k, num_experts=E, hidden_size=H, moe_intermediate_size=I, norm_topk_prob=norm_topk_prob)


def reference(x: torch.Tensor, moe: MoEBlock) -> torch.Tensor:
    """逐 token 朴素参考实现（不依赖 index_add 的向量化路径）。"""
    B, T, Hd = x.shape
    xf = x.reshape(-1, Hd)
    logits = xf @ moe.router.weight.T
    probs = logits.softmax(-1, dtype=torch.float)
    top_p, top_i = probs.topk(moe.router.top_k, dim=-1)
    if moe.router.norm_topk_prob:
        top_p = top_p / top_p.sum(-1, keepdim=True)
    top_p = top_p.to(x.dtype)
    out = torch.zeros_like(xf)
    for t in range(xf.shape[0]):
        for k in range(top_i.shape[1]):
            e = top_i[t, k].item()
            g, u = (xf[t] @ moe.experts.gate_up_proj[e].T).chunk(2, dim=-1)
            out[t] += (F.silu(g) * u) @ moe.experts.down_proj[e].T * top_p[t, k]
    return out.reshape(B, T, Hd)


def test_moe_output_shape_and_nonzero():
    """输出 shape 正确且非零（回归 index_add 无下划线导致输出全零的 bug）。"""
    moe = make_moe()
    x = torch.randn(2, 5, H)
    out = moe(x)
    assert out.shape == x.shape
    assert out.abs().max().item() > 0  # 随机初始化下必须非零


def test_moe_matches_tokenwise_reference():
    """前向与逐 token 朴素参考实现一致（验证 fused gate_up / 加权 / index_add 数学）。"""
    moe = make_moe()
    x = torch.randn(3, 7, H)
    torch.testing.assert_close(moe(x), reference(x, moe), atol=1e-5, rtol=1e-5)


def test_moe_gradients_flow():
    """router / 专家 / 输入 x 梯度全部非零（回归 zeros 初始化的梯度死锁）。"""
    moe = make_moe()
    x = torch.randn(2, 5, H, requires_grad=True)
    moe(x).sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
    for name, p in moe.named_parameters():
        assert p.grad is not None, f"{name} 没有梯度"
        assert p.grad.abs().sum() > 0, f"{name} 梯度为全零"


def test_moe_router_scores_normalized():
    """norm_topk_prob=True：每 token 的 top-k 权重归一化且和 = 1，值 = softmax topk / sum。"""
    torch.manual_seed(0)
    router = TopKRouter(top_k=K, num_experts=E, hidden_size=H)
    x = torch.randn(10, H)
    logits, scores, indices = router(x)

    assert logits.shape == (10, E)
    assert indices.shape == (10, K)
    torch.testing.assert_close(scores.sum(-1), torch.ones(10), atol=1e-5, rtol=1e-5)
    # scores == 归一化后的 softmax topk
    ref = logits.softmax(-1, dtype=torch.float).gather(1, indices).to(logits.dtype)
    torch.testing.assert_close(scores, ref / ref.sum(-1, keepdim=True), atol=1e-5, rtol=1e-5)


def test_moe_router_scores_no_norm():
    """norm_topk_prob=False：scores = 原始 softmax topk 值（不做归一化）。"""
    torch.manual_seed(0)
    router = TopKRouter(top_k=K, num_experts=E, hidden_size=H, norm_topk_prob=False)
    x = torch.randn(10, H)
    logits, scores, indices = router(x)
    ref = logits.softmax(-1, dtype=torch.float).gather(1, indices).to(logits.dtype)
    torch.testing.assert_close(scores, ref, atol=1e-5, rtol=1e-5)


def test_moe_topk_edges():
    """top_k 边界：=1 与 =num_experts 都能前向，且与参考实现一致。"""
    for top_k in (1, E):
        moe = make_moe(top_k=top_k)
        x = torch.randn(2, 4, H)
        torch.testing.assert_close(moe(x), reference(x, moe), atol=1e-5, rtol=1e-5)


def test_moe_gradient_finite_difference():
    """router 权重解析梯度 == 有限差分数值梯度（反向传播正确性抽查）。"""
    moe = make_moe()
    x = torch.randn(1, 4, H)

    def loss_fn():
        return moe(x).pow(2).mean()

    loss_fn().backward()
    idx = (0, 0)
    ga = moe.router.weight.grad[idx].item()

    eps = 1e-3
    with torch.no_grad():
        w0 = moe.router.weight[idx].item()
        moe.router.weight[idx] = w0 + eps
        l_plus = loss_fn().item()
        moe.router.weight[idx] = w0 - eps
        l_minus = loss_fn().item()
        moe.router.weight[idx] = w0
    gn = (l_plus - l_minus) / (2 * eps)

    assert abs(ga - gn) < 1e-5, f"解析={ga} 数值={gn}"


if __name__ == "__main__":
    sys.exit(1 if run_module(globals()) else 0)
