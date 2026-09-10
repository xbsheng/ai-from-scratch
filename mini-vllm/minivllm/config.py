from dataclasses import dataclass

from transformers import AutoConfig, PreTrainedConfig


@dataclass(slots=True)
class Config:
    model_name: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    max_num_cuda_graph_seqs: int = 512
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: PreTrainedConfig | None = None
    eos: int = -1
    kv_cache_block_size: int = 256
    num_kv_cache_blocks: int = -1

    def __post_init__(self):
        assert self.kv_cache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model_name)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
