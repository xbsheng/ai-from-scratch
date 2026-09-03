import torch
from torch import Tensor, nn
from torch.nn import functional as F


class MoEBlock(nn.Module):
    def __init__(self, top_k: int, num_experts: int, hidden_size: int, moe_intermediate_size: int, norm_topk_prob=True):
        super().__init__()

        self.router = TopKRouter(
            top_k=top_k,
            num_experts=num_experts,
            hidden_size=hidden_size,
            norm_topk_prob=norm_topk_prob,
        )

        self.experts = Experts(
            num_experts=num_experts,
            moe_intermediate_size=moe_intermediate_size,
            hidden_size=hidden_size,
        )

    def forward(self, hidden_states: Tensor):
        batch_size, seq_len, hidden_size = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_size)  # (batch_size * seq_len, hidden_size)

        _, router_score, router_indices = self.router(hidden_states)
        final_hidden_states = self.experts(hidden_states, router_score, router_indices)

        return final_hidden_states.reshape(batch_size, seq_len, hidden_size)


class TopKRouter(nn.Module):
    def __init__(self, top_k: int, num_experts: int, hidden_size: int, norm_topk_prob=True):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.norm_topk_prob = norm_topk_prob

        self.weight = nn.Parameter(torch.zeros(num_experts, hidden_size))

    def forward(self, hidden_states: Tensor):
        hidden_states = hidden_states.reshape(-1, self.hidden_size)  # (seq_len, hidden_size)
        router_logits = F.linear(hidden_states, self.weight)  # (seq_len, num_experts)

        # why dtype=torch.float ?
        # 路由决策需要高精度，直接影响专家选择的均衡性和准确性
        router_probs = router_logits.softmax(-1, dtype=torch.float)

        top_k_probs, top_k_indices = router_probs.topk(self.top_k, dim=-1)  # (seq_len, k)

        if self.norm_topk_prob:
            top_k_probs /= top_k_probs.sum(dim=-1, keepdim=True)

        # 前面设置的 torch.float，精度还原
        top_k_probs = top_k_probs.to(router_logits.dtype)

        return router_logits, top_k_probs, top_k_indices


class Experts(nn.Module):
    def __init__(self, num_experts: int, moe_intermediate_size: int, hidden_size: int):
        super().__init__()
        self.num_experts = num_experts

        # up / gate 共用权重
        self.gate_up_proj = nn.Parameter(torch.empty(num_experts, 2 * moe_intermediate_size, hidden_size))

        self.down_proj = nn.Parameter(torch.empty(num_experts, hidden_size, moe_intermediate_size))

        # init
        nn.init.normal_(self.gate_up_proj, mean=0.0, std=0.02)
        nn.init.normal_(self.down_proj, mean=0.0, std=0.02)

    def forward(self, hidden_states: Tensor, router_score: Tensor, router_indices: Tensor):
        # 此处x经过reshape seq_len: batch_size * seq_len
        # x shape: (seq_len, hidden_size)
        # router_score / router_score shape: (seq_len, k)

        final_hidden_states = torch.zeros_like(hidden_states)

        with torch.no_grad():
            expert_mask = F.one_hot(router_indices, num_classes=self.num_experts)  # (seq_len, k, num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)  # (num_experts, k, seq_len)

            # expert_mask:       (num_experts, k, seq_len)
            # expert_mask.sum -> (num_experts,)
            # greater(0) ->      (num_experts,) 选出k、seq_len维度被选中的expert [True, False, ..., True]
            # nonzero ->         (num_experts_hit, 1)
            expert_hit = expert_mask.sum((-1, -2)).greater(0).nonzero()

        for (expert_idx,) in expert_hit:
            # top_k_idx / token_idx shape: (seq_len_hit, )
            top_k_idx, token_idx = expert_mask[expert_idx].nonzero(as_tuple=True)

            # gate / up shape: (seq_len_hit, moe_intermediate_size)
            gate, up = F.linear(hidden_states[token_idx], self.gate_up_proj[expert_idx]).chunk(2, dim=-1)

            current_hidden_states = F.linear(
                F.silu(gate) * up,
                self.down_proj[expert_idx],
            )  # (seq_len_hit, hidden_size)

            current_hidden_states = current_hidden_states * router_score[token_idx, top_k_idx, None]

            final_hidden_states.index_add_(0, token_idx, current_hidden_states)

        return final_hidden_states
