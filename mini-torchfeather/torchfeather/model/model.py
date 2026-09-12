import torch
from torch import Tensor, nn

from .attention import Attention
from .model_args import DeepSeekV3ModelArgs
from .moe import MoE
from .rope import precompute_freqs_cis


class DeepSeekV3Model(nn.Module):
    def __init__(self, model_args: DeepSeekV3ModelArgs):
        super().__init__()

        self.model_args = model_args

        self.embeddings = nn.Embedding(model_args.vocab_size, model_args.dim)
        self.register_buffer("freqs_cis", precompute_freqs_cis(model_args), persistent=False)

        # why use ModuleDict not ModuleList ?
        # 分布式代码里层频繁被包裹替换，string key 的稳定引用比位置更安全
        self.layers: nn.ModuleDict = nn.ModuleDict()
        for layer_id in range(model_args.n_layers):
            self.layers[str(layer_id)] = TransformerBlock(model_args, layer_id)

        self.norm = nn.RMSNorm(model_args.dim)

        self.output = nn.Linear(model_args.dim, model_args.vocab_size, bias=False)

    def init_weights(self, init_std: float | None = None, buffer_device: torch.device | None = None):
        assert isinstance(self.freqs_cis, Tensor)
        buffer_device = buffer_device or self.freqs_cis.device

        with torch.device(buffer_device):
            self.freqs_cis = precompute_freqs_cis(self.model_args)

        nn.init.normal_(self.embeddings.weight)

        for layer in self.layers.values():
            assert isinstance(layer, TransformerBlock)
            layer.init_weights(init_std, buffer_device)

        self.norm.reset_parameters()

        # std = 1/√dim 让 logit 量级不随维度膨胀
        final_output_std = self.model_args.dim**-0.5
        cutoff_factor = 3
        # 截断正态，砍掉尾部 正态采样但拒绝落在 [a, b] 之外的样本
        nn.init.trunc_normal_(
            self.output.weight,
            std=final_output_std,
            a=-cutoff_factor,
            b=cutoff_factor,
        )

    def forward(self, x: Tensor):
        x = self.embeddings(x)

        for layer in self.layers.values():
            x = layer(x, self.freqs_cis)

        x = self.norm(x)

        return self.output(x)


class TransformerBlock(nn.Module):
    def __init__(self, model_args: DeepSeekV3ModelArgs, layer_id: int):
        super().__init__()

        self.model_args = model_args
        self.layer_id = layer_id

        self.weight_init_std = 0.02 / (2 * (layer_id + 1)) ** 0.5

        self.attn_norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)
        self.ffn_norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)

        self.attn = Attention(model_args)

        # 前 n_dense_layers 层是稠密 ffn , 后面的层是 MoE
        self.moe_enabled = layer_id >= model_args.n_dense_layers

        if self.moe_enabled:
            self.moe = MoE(model_args)
        else:
            self.ffn = FeedForward(model_args.dim, model_args.inter_dim)

    def init_weights(self, init_std: float | None = None, buffer_device: torch.device | None = None):
        for norm in (self.attn_norm, self.ffn_norm):
            norm.reset_parameters()

        self.attn.init_weights(self.weight_init_std, buffer_device)

        if self.moe_enabled:
            self.moe.init_weights(self.weight_init_std, buffer_device)
        else:
            self.ffn.init_weights(self.weight_init_std, buffer_device)

    def forward(self, x: Tensor, freqs_cis: Tensor):
        x = x + self.attn(self.attn_norm(x), freqs_cis)

        if self.moe_enabled:
            x = x + self.moe(self.ffn_norm(x))
        else:
            x = x + self.ffn(self.ffn_norm(x))

        return x


class FeedForward(nn.Module):
    def __init__(self, dim: int, inter_dim: int):
        super().__init__()

    def init_weights(self, init_std: float | None = None, buffer_device: torch.device | None = None):
        pass

    def forward(self, x: Tensor):
        pass
