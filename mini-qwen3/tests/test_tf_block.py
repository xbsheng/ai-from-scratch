import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 使 mini-qwen3 源码可导入

"""TransformerBlock 测试：pre-norm + 残差结构、kv_cache 透传。"""

import sys

import torch

from rope import build_rope_table
from test_common import SMALL_CFG, causal_mask, run_module
from tf_block import TransformerBlock


def test_tf_block_pre_norm_residual():
    """pre-norm + 残差结构：手动复算（norm→attn→+residual→norm→ffn→+residual）应与 forward 一致。"""
    torch.manual_seed(0)
    block = TransformerBlock(SMALL_CFG)
    x = torch.randn(2, 5, 32)
    mask = causal_mask(5)
    sin, cos = build_rope_table(16, 16)

    out, _ = block(x, mask, sin, cos)

    h = block.norm_1(x)
    h_attn, _ = block.attn(h, mask, sin, cos)
    h1 = h_attn + x
    h2 = block.norm_2(h1)
    out_ref = block.ffn(h2) + h1

    torch.testing.assert_close(out, out_ref)


def test_tf_block_kv_cache_passthrough():
    """逐 token + kv_cache 的末 token 输出 == 一次性全量。"""
    torch.manual_seed(0)
    block = TransformerBlock(SMALL_CFG)
    x = torch.randn(2, 5, 32)
    sin, cos = build_rope_table(16, 16)

    out_full, _ = block(x, causal_mask(5), sin, cos)

    cache = None
    for t in range(5):
        x_t = x[:, t : t + 1]
        out_t, cache = block(x_t, causal_mask(1, t + 1), sin, cos, kv_cache=cache)

    torch.testing.assert_close(out_t, out_full[:, 4:5], atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    sys.exit(1 if run_module(globals()) else 0)
