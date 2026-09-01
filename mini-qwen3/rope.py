import torch
from torch import Tensor


def apply_rope(x: Tensor, sin: Tensor, cos: Tensor, offset=0):
    """
    RoPE旋转位置编码

    Args:
        x (Tensor): shape (batch_size, num_heads, seq_len, head_dim)
        sin (Tensor): shape (context_len, head_dim)
        cos (Tensor): shape (context_len, head_dim)
        offset (int, optional): 偏移量. Defaults to 0.

    Returns:
        x_rotated (Tensor): 旋转后的张量
    """
    _batch_size, _num_heads, seq_len, head_dim = x.shape

    assert head_dim % 2 == 0, "head_dim 必须是偶数"

    a = x[..., : head_dim // 2]
    b = x[..., head_dim // 2 :]

    # sin.shape: (context_len, head_dim)
    sin = sin[offset : offset + seq_len, :]
    cos = cos[offset : offset + seq_len, :]

    # 旋转：[a, b]
    #       -> [a * cos - b * sin, a * sin + b * cos]
    #       = sin * [-b, a] + cos * [a, b]
    return sin * torch.cat([-b, a], dim=-1) + cos * x


def build_rope_table(head_dim: int, context_len: int, theta_base=1e4, dtype=torch.float32):
    """
    计算 sin, cos 数值表

    分组配对策略：对于8维的向量
    -> (0, 4) (1, 5) (2, 6) (3, 7)

    Args:
        head_dim (int): 向量维度
        context_len (int): 最大上下文长度
        theta_base (_type_, optional): _description_. Defaults to 1e4.

    Returns:
        (sin, cos): _description_
    """
    assert head_dim % 2 == 0, "head_dim 必须是偶数"

    # torch.arange(0, 10, 2) -> [0, 2, 4, 6]
    freqs = torch.arange(0, head_dim, 2, dtype=dtype)  # 2i shape: (head_dim // 2, )
    freqs = theta_base ** (freqs / -head_dim)  # 10000^(−2i/d)

    positions = torch.arange(context_len, dtype=dtype)  # shape: (context_len,)

    angle = positions.unsqueeze(-1) @ freqs.unsqueeze(0)  # shape: (context_len, head_dim // 2)
    angle = torch.cat([angle, angle], dim=-1)

    return angle.sin(), angle.cos()


if __name__ == "__main__":
    sin, cos = build_rope_table(8, 16)
    print(sin.shape)  # [16, 8]

    x = torch.rand((32, 4, 12, 8))
    x_rotated = apply_rope(x, sin, cos)
    print(x.shape, x_rotated.shape)
