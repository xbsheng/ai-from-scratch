from config import QwenConfig
from ffn import FeedForward
from gqa import GroupedQueryAttention
from rms_norm import RMSNorm
from torch import Tensor, nn


class TransformerBlock(nn.Module):
    def __init__(self, config: QwenConfig):
        super().__init__()

        self.norm_1 = RMSNorm(emb_dim=config["hidden_size"])
        self.norm_2 = RMSNorm(emb_dim=config["hidden_size"])

        self.attn = GroupedQueryAttention(
            d_in=config["hidden_size"],
            n_heads=config["num_attention_heads"],
            n_kv_groups=config["num_key_value_heads"],
            head_dim=config["head_dim"],
            qk_norm=config["qk_norm"],
        )

        self.ffn = FeedForward(emb_dim=config["hidden_size"], hidden_dim=config["intermediate_size"])

    def forward(self, x: Tensor, mask: Tensor, sin: Tensor, cos: Tensor, kv_cache: tuple[Tensor, Tensor] | None = None):
        residual = x
        # pre-norm（norm first）
        x = self.norm_1(x)
        x, next_cache = self.attn(x, mask, sin, cos, kv_cache)
        x = x + residual

        residual = x
        x = self.norm_2(x)
        x = self.ffn(x)
        x = x + residual

        return x, next_cache
