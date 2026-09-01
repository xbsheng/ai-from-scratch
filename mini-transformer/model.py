import math

import torch
from torch import Tensor, nn


class InputEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.embedding: nn.Embedding = nn.Embedding(vocab_size, d_model)

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
        _, seq_len, _ = x.shape
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


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor):
        x = self.linear_1(x).relu()
        x = self.dropout(x)
        return self.linear_2(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads

        assert d_model % n_heads == 0, "d_model % n_heads != 0"
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query: Tensor, key: Tensor, value: Tensor, mask: Tensor | None, dropout: nn.Dropout | None):
        # query/key/value shape: (batch_size, n_heads, seq_len, d_k)
        d_k = query.shape[-1]

        # (batch_size, n_heads, seq_len d_k) @ (batch_size, n_heads, d_k, seq_len)
        # -> (batch_size, n_heads, seq_len, seq_len)
        attention_score = (query @ key.mT) / math.sqrt(d_k)

        if mask is not None:
            attention_score = attention_score.masked_fill(mask == 0, -1e9)

        # 缩放点积注意力(scaled dot-product attention)
        attn_weights = torch.softmax(attention_score / (d_k**0.5), dim=-1)

        if dropout is not None:
            attn_weights = dropout(attn_weights)

        # attn_weights @ value -> (batch_size, n_heads, seq_len, d_k)
        context_vec = attn_weights @ value
        return context_vec, attn_weights

    def forward(self, x: Tensor, mask: Tensor | None):
        batch_size, seq_len, _ = x.shape
        reshape_shape = (batch_size, seq_len, self.n_heads, self.d_k)

        # x:           (batch_size, seq_len, d_model)
        # w_q(x)    -> (batch_size, seq_len, d_model)
        # reshape   -> (batch_size, seq_len, n_heads, d_k)
        # transpose -> (batch_size, n_heads, seq_len, d_k)
        query = self.w_q(x).reshape(reshape_shape).transpose(1, 2)
        key = self.w_k(x).reshape(reshape_shape).transpose(1, 2)
        value = self.w_v(x).reshape(reshape_shape).transpose(1, 2)

        context_vec, attention_score = self.attention(query, key, value, mask, self.dropout)

        # context_vec:       (batch_size, n_heads, seq_len, d_k)
        # transpose(1, 2) -> (batch_size, seq_len, n_heads, d_k)
        # flatten(-2)     -> (batch_size, seq_len, d_model)
        context_vec = context_vec.transpose(1, 2).flatten(-2)

        return self.w_o(context_vec)


class ResidualConnection(nn.Module):
    def __init__(self, dropout: float):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.normal = LayerNormal()

    def forward(self, x: Tensor, sub_layer: nn.Module):
        output = sub_layer(self.normal(x))
        output = self.dropout(output)
        return x + output


class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, d_ff: int):
        super().__init__()

        self.attn = MultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)
        self.ffn = FeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.res = ResidualConnection(dropout)

    def forward(self, x: Tensor, mask: Tensor | None):
        x = self.res(x, lambda x: self.attn(x, mask))
        return self.res(x, self.ffn)


class Encoder(nn.Module):
    def __init__(self, num_layers: int, d_model: int, n_heads: int, dropout: float, d_ff: int):
        super().__init__()
        self.normal = LayerNormal()
        self.layers = nn.ModuleList(
            [
                EncoderBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    dropout=dropout,
                    d_ff=d_ff,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: Tensor, mask: Tensor | None = None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.normal(x)
