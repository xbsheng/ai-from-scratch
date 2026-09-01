import torch
from rms_norm import RMSNorm
from rope import apply_rope
from torch import Tensor, nn


class GroupedQueryAttention(nn.Module):
    def __init__(
        self, d_in: int, n_heads: int, n_kv_groups: int, head_dim: int | None = None, qk_norm=False, bias=False
    ):
        super().__init__()

        self.n_heads = n_heads
        self.n_kv_groups = n_kv_groups
        self.bias = bias

        assert n_heads % n_kv_groups == 0, "n_heads % n_kv_groups != 0"
        self.group_size = n_heads // n_kv_groups

        if head_dim is None:
            assert d_in % n_heads == 0, "d_in % n_heads != 0"
            head_dim = d_in // n_heads
        self.head_dim = head_dim

        d_out = n_heads * head_dim

        self.w_q = nn.Linear(d_in, d_out, bias=bias)
        self.w_k = nn.Linear(d_in, n_kv_groups * head_dim, bias=bias)
        self.w_v = nn.Linear(d_in, n_kv_groups * head_dim, bias=bias)

        self.w_out = nn.Linear(d_out, d_in, bias=bias)

        self.q_norm = RMSNorm(head_dim) if qk_norm else None
        self.k_norm = RMSNorm(head_dim) if qk_norm else None

    def forward(self, x: Tensor, mask: Tensor, sin: Tensor, cos: Tensor, kv_cache: tuple[Tensor, Tensor] | None = None):
        batch_size, seq_len, emb_dim = x.shape

        offset = 0
        if kv_cache:
            k_cache, v_cache = kv_cache  # k、v cache shape: (batch_size, n_kv_groups, seq_len, head_dim)
            offset = k_cache.shape[-2]

        q: Tensor = self.w_q(x)  # (batch_size, seq_len, n_heads * head_dim)
        k: Tensor = self.w_k(x)  # (batch_size, seq_len, n_kv_groups * head_dim)
        v: Tensor = self.w_v(x)  # (batch_size, seq_len, n_kv_groups * head_dim)

        # q shape: (batch_size, n_heads, seq_len, head_dim)
        q = q.reshape(batch_size, seq_len, self.n_heads, -1).transpose(1, 2)

        # k、v shape: (batch_size, n_kv_groups, seq_len, head_dim)
        k = k.reshape(batch_size, seq_len, self.n_kv_groups, -1).transpose(1, 2)
        v = v.reshape(batch_size, seq_len, self.n_kv_groups, -1).transpose(1, 2)

        if self.q_norm:
            q = self.q_norm(q)
        if self.k_norm:
            k = self.k_norm(k)

        # 旋转位置编码
        q = apply_rope(q, sin=sin, cos=cos, offset=offset)
        k = apply_rope(k, sin=sin, cos=cos, offset=offset)

        if kv_cache:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=-2)
            v = torch.cat([v_cache, v], dim=-2)
        next_cache = (k, v)

        # (batch_size, n_kv_groups, seq_len, head_dim)
        # -> (batch_size, n_kv_groups * group_size, seq_len, head_dim)
        # n_kv_groups * group_size == n_heads
        k = k.repeat_interleave(self.group_size, dim=1)
        v = v.repeat_interleave(self.group_size, dim=1)

        attn_score = q @ k.mT  # (batch_size, n_heads, seq_len, seq_len)
        attn_score = attn_score.masked_fill(mask, -torch.inf)
        attn_weight = torch.softmax(attn_score / self.head_dim**0.5, dim=-1)

        context = attn_weight @ v  # (batch_size, n_heads, seq_len, head_dim)
        context = context.transpose(1, 2).reshape(batch_size, seq_len, -1)

        return self.w_out(context), next_cache
