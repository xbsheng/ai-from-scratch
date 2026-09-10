from dataclasses import dataclass, field

from loguru import logger
from torch import nn
from torchfeather.model.moe import MoEArgs


@dataclass
class DeepSeekV3ModelArgs:
    max_seq_len: int = 4096 * 4
    vocab_size: int = 102400
    dim: int = 2048
    inter_dim: int = 10944
    moe_inter_dim: int = 1408
    n_layers: int = 27
    n_dense_layers: int = 1
    n_heads: int = 16
    norm_eps: float = 1e-5  # eps used for RMSNorm

    # MoE
    moe_args: MoEArgs = field(default_factory=MoEArgs)

    # Multi-Head Latent Attention (MLA)
    q_lora_rank: int = 0
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128

    # yarn
    original_seq_len: int = 4096
    rope_theta: float = 10000.0
    rope_factor: float = 40
    beta_fast: int = 32
    beta_slow: int = 1
    m_scale: float = 1.0

    def get_n_params_and_flops(self, model: nn.Module, seq_len: int) -> tuple[int, int]:
        n_dense_params = 0
        n_embedding_params = 0

        n_moe_router_params = 0
        n_experts_params = 0
        n_shared_experts_params = 0

        for name, param in model.named_parameters():
            n = param.numel()

            if "embedding" in name:
                n_embedding_params += n
                n_dense_params += n

            # moe sparse params
            elif "moe.router" in name:
                n_moe_router_params += n
            elif "moe.experts" in name:
                n_experts_params += n
            elif "moe.shared_experts" in name:
                n_shared_experts_params += n
            # moe sparse params end

            else:
                n_dense_params += n

        n_sparse_params = n_moe_router_params + n_experts_params + n_shared_experts_params
        n_total_params = n_dense_params + n_sparse_params
        n_active_params = (
            n_dense_params
            + n_moe_router_params
            + n_shared_experts_params
            + self.moe_args.top_k * (n_experts_params // self.moe_args.num_experts)  # active experts
        )

        logger.info(
            f"Total Params: {n_total_params} | "
            f"Active Params: {n_active_params} | "
            f"Dense Params: {n_dense_params} | "
            f"Sparse Params: {n_sparse_params}"
        )

        # Q·K^T and AV
        head_dims = self.qk_nope_head_dim + self.qk_rope_head_dim + self.v_head_dim

        # why 6 * ?
        # forward = 2 * N params per token (1 multiply + 1 add per MAC = 2 FLOPs) -> 2N
        # backward is 2x forward (grad wrt input + grad wrt weight) -> 4N
        # Training = forward + backward = 2N + 4N -> 6N
        n_flops_pre_token = 6 * (
            # embedding is a table lookup, does not perform MAC, and does not count as FLOPs.
            (n_active_params - n_embedding_params)
            # attention QK^T and AV computation, which is not captured by parameter count (activation-dependent)
            + self.n_layers * self.n_heads * head_dims * seq_len
        )

        return n_total_params, n_flops_pre_token
