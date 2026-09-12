import math

import torch
from torch import Tensor, nn

from .model_args import DeepSeekV3ModelArgs


class Attention(nn.Module):
    def __init__(self, model_args: DeepSeekV3ModelArgs):
        super().__init__()

        self.dim = model_args.dim  # 2048
        self.n_heads = model_args.n_heads  # 16
        self.q_lora_rank = model_args.q_lora_rank  # 0
        self.kv_lora_rank = model_args.kv_lora_rank  # 512
        self.qk_nope_head_dim = model_args.qk_nope_head_dim  # 128
        self.qk_rope_head_dim = model_args.qk_rope_head_dim  # 64
        self.qk_head_dim = model_args.qk_nope_head_dim + model_args.qk_rope_head_dim  # 128 + 64 = 192
        self.v_head_dim = model_args.v_head_dim  # 128

        if self.q_lora_rank == 0:
            self.wq = nn.Linear(self.dim, self.n_heads * self.qk_head_dim, bias=False)
        else:
            self.wq_a = nn.Linear(self.dim, self.q_lora_rank, bias=False)
            self.q_norm = nn.RMSNorm(self.q_lora_rank, eps=model_args.norm_eps)
            self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.qk_head_dim, bias=False)

        self.wkv_a = nn.Linear(self.dim, self.qk_rope_head_dim + self.kv_lora_rank, bias=False)
        self.kv_norm = nn.RMSNorm(self.kv_lora_rank, eps=model_args.norm_eps)
        self.wkv_b = nn.Linear(self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim), bias=False)

        self.wo = nn.Linear(self.n_heads * self.v_head_dim, self.dim, bias=False)

        self.softmax_scale = self.qk_head_dim**-0.5

        if model_args.max_seq_len > model_args.original_seq_len:
            m_scale = 0.1 * model_args.m_scale * math.log(model_args.rope_factor) + 1.0
            self.softmax_scale = self.softmax_scale * m_scale**2

        self.inner_attn = ScaledDotProductAttentionWrapper()

    def init_weights(self, init_std: float | None = None, buffer_device: torch.device | None = None):
        pass

    def forward(self, x: Tensor, freqs_cis: Tensor):
        pass


class ScaledDotProductAttentionWrapper(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor):
        pass
