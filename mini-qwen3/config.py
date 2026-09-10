from typing import TypedDict

import torch


class QwenConfig(TypedDict):
    """Qwen3 模型配置。

    字段名与官方 config.json 保持一致
    """

    vocab_size: int
    max_position_embeddings: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    rope_theta: float
    qk_norm: bool
    torch_dtype: torch.dtype


# 0.6 billion parameters（数值与官方 Qwen/Qwen3-0.6B config.json 一致）
QWEN_CONFIG_0_6_B: QwenConfig = {
    "vocab_size": 151_936,  # 词表大小
    "max_position_embeddings": 40_960,  # 训练时使用的上下文长度
    "hidden_size": 1024,  # 嵌入维度（隐藏维）
    "num_hidden_layers": 28,  # 层数
    "num_attention_heads": 16,  # 注意力头数
    "num_key_value_heads": 8,  # GQA 的 Key-Value 组数
    "head_dim": 128,  # 每头维度
    "intermediate_size": 3072,  # FFN 中间层维度
    "rope_theta": 1.0e6,  # RoPE 的 theta base
    "qk_norm": True,  # 是否对 Query 和 Key 归一化（官方 config 无此字段，仓库开关）
    "torch_dtype": torch.bfloat16,  # 权重精度，降低内存占用
}
