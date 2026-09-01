"""加载  Qwen3-0.6B 权重"""

import torch
from config import QWEN_CONFIG_0_6_B
from huggingface_hub import hf_hub_download
from model import Qwen3
from safetensors.torch import load_file


# 官方参数名 -> 本模型参数名（形状已对齐，纯改名）
def build_name_map():
    m = {
        "model.embed_tokens.weight": "embedding.weight",
        "model.norm.weight": "norm.weight",
        "lm_head.weight": "out.weight",
    }
    for i in range(QWEN_CONFIG_0_6_B["n_layers"]):
        m |= {
            f"model.layers.{i}.input_layernorm.weight": f"tf_blocks.{i}.norm_1.weight",
            f"model.layers.{i}.post_attention_layernorm.weight": f"tf_blocks.{i}.norm_2.weight",
            f"model.layers.{i}.self_attn.q_proj.weight": f"tf_blocks.{i}.attn.w_q.weight",
            f"model.layers.{i}.self_attn.k_proj.weight": f"tf_blocks.{i}.attn.w_k.weight",
            f"model.layers.{i}.self_attn.v_proj.weight": f"tf_blocks.{i}.attn.w_v.weight",
            f"model.layers.{i}.self_attn.o_proj.weight": f"tf_blocks.{i}.attn.w_out.weight",
            f"model.layers.{i}.self_attn.q_norm.weight": f"tf_blocks.{i}.attn.q_norm.weight",
            f"model.layers.{i}.self_attn.k_norm.weight": f"tf_blocks.{i}.attn.k_norm.weight",
            f"model.layers.{i}.mlp.gate_proj.weight": f"tf_blocks.{i}.ffn.gate_proj.weight",
            f"model.layers.{i}.mlp.up_proj.weight": f"tf_blocks.{i}.ffn.up_proj.weight",
            f"model.layers.{i}.mlp.down_proj.weight": f"tf_blocks.{i}.ffn.down_proj.weight",
        }
    return m


def load_official_weights(model: Qwen3) -> None:
    name_map = build_name_map()

    path = hf_hub_download("Qwen/Qwen3-0.6B", "model.safetensors")
    official = load_file(path)  # {官方参数名: 张量}

    # 加载前双重校验，防止静默错位
    for official_name, mine in name_map.items():
        assert official_name in official, f"官方权重缺少 {official_name}"
        o_shape = tuple(official[official_name].shape)
        m_shape = tuple(model.get_parameter(mine).shape)
        assert o_shape == m_shape, f"shape 不匹配: {official_name} {o_shape} vs {mine} {m_shape}"

    missing = [n for n, _ in model.named_parameters() if n not in name_map.values()]
    assert not missing, f"模型参数未覆盖: {missing}"

    with torch.no_grad():
        for official_name, mine in name_map.items():
            model.get_parameter(mine).copy_(official[official_name])
    print(f"权重加载完成: {len(name_map)} 个参数, 来源 {path}")


if __name__ == "__main__":
    model = Qwen3(QWEN_CONFIG_0_6_B).to(QWEN_CONFIG_0_6_B["dtype"]).eval()
    load_official_weights(model)

    # 快速 sanity：跑一次前向确认无 NaN
    sample = torch.randint(0, QWEN_CONFIG_0_6_B["vocab_size"], (1, 8))
    logits = model(sample, cache={})
    assert torch.isfinite(logits).all(), "前向出现 NaN/Inf"
