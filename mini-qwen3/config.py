import torch

# 0.6 billion parameters
QWEN_CONFIG_06_B = {
    "vocab_size": 151_936,  # 词表大小
    "context_length": 40_960,  # 训练时使用的上下文长度
    "emb_dim": 1024,  # 嵌入维度
    "n_heads": 16,  # 注意力头数
    "n_layers": 28,  # 层数
    "hidden_dim": 3072,  # 中间层维度
    "head_dim": 128,  # GQA头维度
    "qk_norm": True,  # 是否需要对Query和Key进行归一化
    "n_kv_groups": 8,  # Key-Value groups for GQA
    "rope_base": 1.0e6,  # The base in RoPE's "theta"
    "dtype": torch.bfloat16,  # Lower-precision dtype to reduce memory
}
