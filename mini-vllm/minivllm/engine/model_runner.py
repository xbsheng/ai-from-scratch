from multiprocessing.synchronize import Event

import torch
import torch.distributed as disd
from minivllm.config import Config
from minivllm.engine.sequence import Sequence
from minivllm.sampling_params import SamplingParams
from torch import nn


class ModelRunner:
    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        self.rank = rank
        self.event = event

        assert config.hf_config
        self.hf_config = config.hf_config
        self.block_size = config.kv_cache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size

        self.model = nn.Module()

        disd.init_process_group(
            # 用 NVIDIA NCCL 库做 GPU 间通信（all-reduce、broadcast 等），CPU 通信一般用 "gloo"
            backend="nccl",
            # rendezvous 地址。rank=0 的进程监听 localhost:8888，
            # 其他进程连上来交换彼此的地址/hostname，组成一个进程组
            init_method="tcp:localhost:8888",
            world_size=self.world_size,  # 总进程数（比如 8 张卡就是 8）
            rank=rank,  # 当前进程的编号，0 ~ world_size-1
        )

        torch.cuda.set_device(rank)

        # warmup_model → allocate_kv_cache → capture_cuda_graph
        # 先测峰值再定 KV cache 大小，最后才抓 CUDA 图，抓图时 KV cache 地址已固定
        self.warmup_model()
        self.allocate_kv_cache()
        if not config.enforce_eager:
            self.capture_cuda_graph()

    def call(self, method_name: str):
        pass

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        return []

    def warmup_model(self):
        """
        1. 获取模型推理的显存峰值（主要目的） :

        如果不 warmup 直接算，等真正推理时 batch 大了才发现模型 + 中间激活吃掉的显存超过预算，KV cache 一撞上就 OOM。
        所以先用最大 batch 跑一遍，让 PyTorch 记录峰值，再从剩余配额里分给 KV cache——这是 vLLM 同款的 profile-and-allocate 思路

        2. 顺带完成惰性初始化：

        CUDA kernel 的 JIT 编译（比如 GQA 相关的 attention 核）、cuBLAS handle 创建、
        各种 lazy 分配都在这次假 forward 中触发，避免第一次真实请求变慢
        """
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_host_memory_stats()  # 清零显存峰值统计

        seq_len = min(self.config.max_num_batched_tokens, self.config.max_model_len)
        num_seqs = min(self.config.max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * seq_len, SamplingParams()) for _ in range(num_seqs)]  # 最大的seqs
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len

        self.run(seqs, is_prefill=True)

        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        """
        根据显存空间数据，分配kv cache
        """
        hf_config = self.hf_config

        free, total = torch.cuda.mem_get_info()
        used = total - free  # 已被占用的全部显存

        allocated_bytes_all = torch.cuda.memory_stats()["allocated_bytes.all"]
        model_peak = allocated_bytes_all["peak"] - allocated_bytes_all["current"]  # model显存使用峰值
        kv_cache_bytes = total * self.config.gpu_memory_utilization - used - model_peak  # 可用于kv cache的显存值

        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)

        # 每个block所需占用空间大小
        block_bytes = (
            2  # k, v
            * hf_config.num_hidden_layers
            * self.block_size
            * num_kv_heads
            * head_dim
            * hf_config.dtype.itemsize  # type: ignore
        )

        num_kv_cache_blocks = kv_cache_bytes // block_bytes
        assert num_kv_cache_blocks > 0
        self.config.num_kv_cache_blocks = num_kv_cache_blocks

        kv_cache = torch.empty(
            2,
            hf_config.num_hidden_layers,
            num_kv_cache_blocks,
            self.block_size,
            num_kv_heads,
            head_dim,
        )

        k_cache, v_cache = kv_cache

        for idx, module in enumerate(self.model.modules()):
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = k_cache[idx]
                module.v_cache = v_cache[idx]

    def capture_cuda_graph(self):
        pass
