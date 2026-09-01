import torch
from config import QwenConfig
from rms_norm import RMSNorm
from rope import build_rope_table
from tf_block import TransformerBlock
from torch import Tensor, nn


class Qwen3(nn.Module):
    def __init__(self, config: QwenConfig):
        super().__init__()

        self.embedding = nn.Embedding(config["vocab_size"], config["emb_dim"])
        self.tf_blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config["n_layers"])])
        self.norm = RMSNorm(config["emb_dim"])
        self.out = nn.Linear(config["emb_dim"], config["vocab_size"], bias=False)

        sin, cos = build_rope_table(
            head_dim=config["head_dim"],
            context_len=config["context_length"],
            theta_base=config["rope_base"],
        )

        self.register_buffer("sin", sin, persistent=False)
        self.register_buffer("cos", cos, persistent=False)

        self.offset = 0

    def reset_kv_cache(self):
        self.offset = 0

    def forward(self, in_idx: Tensor, cache: dict):
        # in_idx.shape (batch_size, seq_len)
        x = self.embedding(in_idx)  # (batch_size, seq_len, emb_dim)

        _batch_size, seq_len, _emb_dim = x.shape

        seq_len_total = self.offset + seq_len
        mask = torch.ones(seq_len_total, seq_len_total, device=x.device, dtype=torch.bool)
        mask = torch.triu(mask, diagonal=1)
        # self.offset == 0: prefill阶段
        if self.offset > 0:
            mask = mask[-seq_len:, :]

        for i, block in enumerate(self.tf_blocks):
            kv_cache = cache.get(i)

            x, next_cache = block(x, mask=mask, sin=self.sin, cos=self.cos, kv_cache=kv_cache)
            cache[i] = next_cache

        x = self.norm(x)
        logits: Tensor = self.out(x)  # (batch_size, seq_len, vocab_size)

        self.offset = seq_len_total

        return logits
