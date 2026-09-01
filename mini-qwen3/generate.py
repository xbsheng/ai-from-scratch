import torch
from config import QWEN_CONFIG_0_6_B
from load_qwen3 import load_official_weights
from model import Qwen3
from torch import Tensor
from transformers import AutoTokenizer, PreTrainedTokenizerBase


def generate_text(in_idx: Tensor, model: Qwen3, tokenizer: PreTrainedTokenizerBase, max_len=128):
    final_idx = in_idx.clone()  # (batch_size, seq_len)
    kv_cache = {}
    model.reset_kv_cache()

    with torch.no_grad():
        for _ in range(max_len):
            logits = model.forward(in_idx=in_idx, cache=kv_cache)  # (batch_size, seq_len, vocab_size)
            logits = logits[:, -1, :]
            probs = logits.softmax(dim=-1)
            in_idx = probs.multinomial(1)

            final_idx = torch.cat([final_idx, in_idx], dim=-1)

            print(tokenizer.decode(in_idx[0].tolist()), end="", flush=True)

            if in_idx[0][0] == tokenizer.eos_token_id:
                break

    return final_idx


if __name__ == "__main__":
    model = Qwen3(QWEN_CONFIG_0_6_B).to(QWEN_CONFIG_0_6_B["dtype"])
    model = model.eval()  # 开启评估模式
    load_official_weights(model)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

    messages = [
        {"role": "user", "content": "你好~"},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    input_ids = inputs["input_ids"]

    """
    <think>
    好的，用户发来消息，“你好~”，我需要友好回应。首先，确认称呼是“你好”，然后回应应保持亲切自然，例如“你好呀！有什么我可以帮你的吗？”。保持口语化，不需要长篇大论。另外，要确保回复符合角色设定，保持友好和帮助的态度。同时，检查是否有需要进一步互动的地方，比如询问用户是否有问题或需要帮助。最后，保持回复简洁，避免冗长。
    </think>

    你好呀！有什么我可以帮你的吗？<|im_end|>
    """

    generate_text(input_ids, model, tokenizer, max_len=256)
