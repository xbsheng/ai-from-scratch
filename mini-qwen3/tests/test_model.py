import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 使 mini-qwen3 源码可导入

"""Qwen3 模型测试：prefill/生成、kv_cache 一致性、结构与 config 一致、梯度。"""

import sys

import torch

from model import Qwen3
from test_common import SMALL_CFG, run_module


def test_qwen3_prefill_and_generate():
    """prefill 与生成（单 token + cache）输出 shape 与 offset 推进正确。"""
    torch.manual_seed(0)
    model = Qwen3(SMALL_CFG)
    in_idx = torch.randint(0, 1000, (2, 5))

    cache: dict = {}
    logits = model(in_idx, cache)
    assert logits.shape == (2, 5, 1000), logits.shape
    assert model.offset == 5

    logits2 = model(in_idx[:, :1], cache)
    assert logits2.shape == (2, 1, 1000), logits2.shape
    assert model.offset == 6


def test_qwen3_kv_cache_consistency():
    """逐 token 生成 == 一次性全量（同时隐式验证因果 mask 语义：
    逐 token 路径 mask 全可见，全量路径是因果 mask，末 token 输出一致
    说明两种 mask 下可见集等价）。"""
    torch.manual_seed(0)
    model = Qwen3(SMALL_CFG)
    in_idx = torch.randint(0, 1000, (2, 5))

    model.reset_kv_cache()
    logits_full = model(in_idx, cache={})

    model.reset_kv_cache()
    cache: dict = {}
    for t in range(5):
        logit_t = model(in_idx[:, t : t + 1], cache)

    torch.testing.assert_close(logit_t, logits_full[:, 4:5], atol=1e-5, rtol=1e-5)


def test_qwen3_structure_matches_config():
    """模型结构与 config 一致：层数、embedding/lm_head 形状、权重不共享（官方 tie_word_embeddings=false）。"""
    model = Qwen3(SMALL_CFG)
    assert len(model.tf_blocks) == SMALL_CFG["n_layers"]
    assert tuple(model.embedding.weight.shape) == (1000, 32)
    assert tuple(model.out.weight.shape) == (1000, 32)  # bias=False，无 out.bias
    assert not hasattr(model.out, "bias") or model.out.bias is None
    assert model.embedding.weight is not model.out.weight  # 不共享


def test_qwen3_gradient():
    """整个模型（含 qk_norm 参数）所有参数收到非零梯度。"""
    torch.manual_seed(0)
    model = Qwen3(SMALL_CFG)
    in_idx = torch.randint(0, 1000, (2, 5))

    logits = model(in_idx, cache={})
    logits.sum().backward()

    missing = [n for n, p in model.named_parameters() if p.grad is None or p.grad.abs().sum() == 0]
    assert not missing, f"无梯度参数: {missing}"


if __name__ == "__main__":
    sys.exit(1 if run_module(globals()) else 0)
