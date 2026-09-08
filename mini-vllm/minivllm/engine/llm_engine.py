from dataclasses import fields

from minivllm.config import Config


class LLMEngine:
    def __init__(self, model: str, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        self.config = Config(model, **config_kwargs)


if __name__ == "__main__":
    LLMEngine("Qwen/Qwen3-0.6B", max_num_batched_tokens=1000)
