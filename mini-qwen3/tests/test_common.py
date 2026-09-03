import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 使 mini-qwen3 源码可导入

"""共享测试基础设施：小模型配置、因果 mask 工具、模块运行器。"""

import torch

# 2 层小模型，测试秒级跑完
SMALL_CFG = {
    "vocab_size": 1000,
    "context_length": 64,
    "emb_dim": 32,
    "n_heads": 4,
    "n_layers": 2,
    "hidden_dim": 64,
    "head_dim": 16,
    "qk_norm": True,
    "n_kv_groups": 2,
    "rope_base": 1e6,
    "dtype": torch.float32,
}


def causal_mask(seq_len: int, total: int | None = None) -> torch.Tensor:
    """因果 mask：q 行在全局位置 total-seq_len..total-1，屏蔽 j > 自身全局位置。
    seq_len=total 时等价于 triu(diagonal=1)；seq_len=1（生成）时全 False。
    """
    total = total or seq_len
    offset = total - seq_len
    row = torch.arange(seq_len).unsqueeze(1) + offset  # q 的全局位置
    col = torch.arange(total).unsqueeze(0)
    return (col > row).unsqueeze(0).unsqueeze(0)


def run_module(g: dict) -> int:
    """运行模块命名空间里所有 test_ 开头的函数，返回失败数。"""
    passed, failed = 0, []
    for name in sorted(g):
        if name.startswith("test_") and callable(g[name]):
            try:
                g[name]()
                print(f"✅ PASS {name}")
                passed += 1
            except Exception as e:
                print(f"❌ FAIL {name}: {type(e).__name__}: {e}")
                failed.append(name)
    print(f"—— {passed} 通过, {len(failed)} 失败 ——")
    return len(failed)
