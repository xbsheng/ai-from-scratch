import math

import torch
from torch import Tensor

from .model_args import DeepSeekV3ModelArgs


# ============================ YaRN utils ============================
def find_correction_dim(num_rotations: float, dim: int, base: float, max_seq_len: int) -> float:
    """
    已知圈数，反解维度下标

    freq_i = 1 / base^(2i/dim)
    λ_i    = 2π / freq_i = 2π · base^(2i/dim)

    全长 seq_len 的序列在第 i 维上转的圈数
    rotations_i = seq_len / λ_i = seq_len / (2π · base^(2i/dim))
    -> i = dim·ln(seq_len/(num_rotations·2π)) / (2·ln(base))
    """
    return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))


def find_correction_range(low_rot: float, high_rot: float, dim: int, base: float, max_seq_len: int) -> tuple[int, int]:
    """
    圈数区间 → 维度区间
    """
    low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
    high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
    return max(low, 0), min(high, dim - 1)


def linear_ramp_factor(min: float, max: float, dim: int) -> Tensor:
    """
    区间内线性斜坡

    下标 ≤ min          → 0
    下标 ≥ max          → 1
    min < 下标 < max    → 线性插值 (idx - min)/(max - min)
    """
    if min == max:
        max += 0.001  # 防除零
    linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
    ramp_func = torch.clamp(linear_func, 0, 1)  # 截断到 [0,1]
    return ramp_func


# ============================ YaRN utils end ============================


def precompute_freqs_cis(args: DeepSeekV3ModelArgs) -> Tensor:
    dim = args.qk_rope_head_dim
    max_seq_len = args.max_seq_len
    base = args.rope_theta  # 10000.0

    beta_fast = args.beta_fast
    beta_slow = args.beta_slow

    factor = args.rope_factor

    # 基础频率（标准 RoPE）
    # 对 i = 0, 1, ..., dim/2-1，得到几何递减的频率序列 1/base^(2i/dim)
    # i 小 → 频率高（波长短），i 大 → 频率低（波长长）
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))  # shape: (dim/2, )

    if max_seq_len > args.original_seq_len:
        # YaRN "按部分 NTK 缩放"
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, args.original_seq_len)

        smooth = 1 - linear_ramp_factor(low, high, dim // 2)

        freqs = freqs / factor * (1 - smooth) + freqs * smooth

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
