from torch import Tensor, nn


class FeedForward(nn.Module):
    """
    SwiGLU FFN

    GLU（Gated Linear Unit）及其一系列变体是当前大型语言模型（LLM）中最常用的 FFN 激活结构
    与传统 ReLU、GELU 等单路激活不同，GLU 类结构采用 “主分支 × 门控分支” 的双分支设计，
    通过引入门控机制，使模型能够对信息流进行更加细致的筛选与调控

    SwiGLU：使用SiLU作为门控函数
    """

    def __init__(self, emb_dim: int, hidden_dim: int, bias=False):
        super().__init__()

        # Qwen3 全系 bias=False
        self.up_proj = nn.Linear(emb_dim, hidden_dim, bias=bias)
        self.gate_proj = nn.Linear(emb_dim, hidden_dim, bias=bias)

        self.down_proj = nn.Linear(hidden_dim, emb_dim, bias=bias)

    def forward(self, x: Tensor):
        x_1 = self.up_proj(x)
        x_2 = self.gate_proj(x)

        # SwiGLU：使用SiLU作为门控函数
        # SwiGLU在性能与训练稳定性方面表现最优，因此成为当前主流 LLM（Qwen、DeepSeek等）的默认门控结构
        x = x_1 * nn.functional.silu(x_2)

        return self.down_proj(x)
