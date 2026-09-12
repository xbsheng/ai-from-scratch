import torch
from torch import Tensor, nn

from .model_args import DeepSeekV3ModelArgs
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
            self.layers[str(layer_id)] = TransformerBlock(model_args)

        self.norm = nn.RMSNorm(model_args.dim)

        self.output = nn.Linear(model_args.dim, model_args.vocab_size, bias=False)

    def init_weights(self, init_std: float | None = None, buffer_device: torch.device | None = None):
        assert isinstance(self.freqs_cis, Tensor)
        buffer_device = buffer_device or self.freqs_cis.device

        with torch.device(buffer_device):
            self.freqs_cis = precompute_freqs_cis(self.model_args)

        if self.embeddings:
            nn.init.normal_(self.embeddings.weight)

        for layer in self.layers.values():
            if layer:
                assert isinstance(layer, TransformerBlock)
                layer.init_weights(init_std, buffer_device)

        if self.norm:
            self.norm.reset_parameters()

        # std = 1/√dim 让 logit 量级不随维度膨胀
        final_output_std = self.model_args.dim**-0.5
        cutoff_factor = 3
        if self.output:
            # 截断正态，砍掉尾部 正态采样但拒绝落在 [a, b] 之外的样本
            nn.init.trunc_normal_(
                self.output.weight,
                std=final_output_std,
                a=-cutoff_factor,
                b=cutoff_factor,
            )

    def forward(self, x: Tensor):
        pass


class TransformerBlock(nn.Module):
    def __init__(self, model_args: DeepSeekV3ModelArgs):
        super().__init__()

        self.model_args = model_args

    def init_weights(self, init_std: float | None = None, buffer_device: torch.device | None = None):
        pass

    def forward(self, x: Tensor):
        pass
