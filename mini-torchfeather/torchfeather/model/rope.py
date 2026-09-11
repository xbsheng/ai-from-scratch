import torch
from torch import Tensor

from .model_args import DeepSeekV3ModelArgs


def precompute_freqs_cis(args: DeepSeekV3ModelArgs) -> Tensor:
    dim = args.qk_rope_head_dim
    max_seq_len = args.max_seq_len
    rope_theta = args.rope_theta  # 10000.0

    freqs = 1.0 / (rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))  # shape: (dim/2, )
    pos = torch.arange(max_seq_len)
    freqs = torch.outer(pos, freqs)  # shape: (max_seq_len, dim/2)

    # e^(i*freq*pos)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # 计算复数 shape: (max_seq_len, dim/2)

    return freqs_cis


def apply_rotary_emb(x: Tensor, freqs_cis: Tensor) -> Tensor:
    dtype = x.dtype

    # x shape: (B, S, H, D) -> (B, S, H, D/2, 2) -> (B, S, H, D/2)
    #
    # view_as_complex: 把最后一维长度为 2 的实数数组，解释成复数张量
    # .float() : 复数张量只能是 complex64 / complex128，所以代码里先 .float()，计算完再转回半精度
    # complex64: 实部float32 + 虚部float32 = 64 bit, 故名 complex64
    x = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))  # (B, S, H, D/2)

    freqs_cis = freqs_cis.reshape(1, x.size(1), 1, x.size(-1))  # shape (1, S, 1, D/2)

    # RoPE 的旋转操作
    # view_as_real->(B, S, H, D/2, 2)
    # flatten -> (B, S, H, D)
    y = torch.view_as_real(x * freqs_cis).flatten(3)

    return y.to(dtype)
