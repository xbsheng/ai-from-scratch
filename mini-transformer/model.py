import math
from typing import override

import torch
from torch import Tensor, nn


class InputEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.embedding: nn.Embedding = nn.Embedding(vocab_size, d_model)

    @override
    def forward(self, x: Tensor) -> Tensor:
        # 论文 3.4 节:嵌入乘以 sqrt(d_model) 以缩放
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, seq_len: int, d_model: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(seq_len, d_model, dtype=torch.float)

        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)  # (seq_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(1e4) / d_model)
        )  # (d_model / 2)

        # 偶数位 sin、奇数位 cos
        pe[:, 0::2] = torch.sin(position * div_term)  # sin(position * (10000 ** (2i / d_model))
        pe[:, 1::2] = torch.cos(position * div_term)  # cos(position * (10000 ** (2i / d_model))

        pe = pe.unsqueeze(0)  # (1, seq_len, d_model)
        self.pe: Tensor
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor):
        (_, seq_len, _) = x.shape
        x = x + (self.pe[:, :seq_len, :]).requires_grad_(False)  # (batch, seq_len, d_model)
        return self.dropout(x)


class LayerNormal(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()

        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.ones(1))

    def forward(self, x: Tensor):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)

        return self.alpha * (x - mean) / (std + self.eps) + self.bias
