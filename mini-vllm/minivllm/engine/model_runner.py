import pickle
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.synchronize import Event

import torch
import torch.distributed as dist
from minivllm.config import Config
from minivllm.engine.sequence import Sequence
from minivllm.sampling_params import SamplingParams
from minivllm.utils.context import get_context, reset_context, set_context
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

        dist.init_process_group(
            # 用 NVIDIA NCCL 库做 GPU 间通信（all-reduce、broadcast 等），CPU 通信一般用 "gloo"
            backend="nccl",
            # rendezvous 地址。rank=0 的进程监听 localhost:8888，
            # 其他进程连上来交换彼此的地址/hostname，组成一个进程组
            init_method="tcp:localhost:8888",
            world_size=self.world_size,  # 总进程数（比如 8 张卡就是 8）
            rank=rank,  # 当前进程的编号，0 ~ world_size-1
        )

        torch.cuda.set_device(rank)

        # ==========================================================

        default_dtype = torch.get_default_dtype()
        # 切换默认值是为了让 nn.Parameter 一出生就在正确的 device/dtype 上
        torch.set_default_dtype(config.hf_config.dtype)  # type: ignore
        torch.set_default_device("cuda")

        # self.model =
        # load_model()

        # warmup_model → allocate_kv_cache → capture_cuda_graph
        # 先测峰值再定 KV cache 大小，最后才抓 CUDA 图，抓图时 KV cache 地址已固定
        self.warmup_model()
        self.allocate_kv_cache()
        if not config.enforce_eager:
            self.capture_cuda_graph()

        torch.set_default_dtype(default_dtype)
        torch.set_default_device("cpu")

        # ==========================================================

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="minivllm", create=True, size=2**20)
                dist.barrier()  # 等所有 rank attach 完再往下走
            else:
                # rank0 建完立刻 barrier，子进程先 barrier 再 attach，避免 attach 到还没创建出来的段
                dist.barrier()
                self.shm = SharedMemory(name="minivllm")
                self.loop()  # 子进程从此阻塞在死循环里

    def call(self, method_name: str, *args):
        print(method_name, args)

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()  # type: ignore

        assert self.shm.buf
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4 : n + 4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0

        data = pickle.dumps([method_name, *args])
        n = len(data)

        assert self.shm.buf
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4 : n + 4] = data
        for event in self.event:  # type: ignore
            event.set()

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        return []

    def warmup_model(self):
        """
        1. 获取模型推理的显存峰值（主要目的） :

        如果不 warmup 直接算，等真正推理时 batch 大了才发现模型 + 中间激活吃掉的显存超过预算，KV cache 一撞上就 OOM。
        所以先用最大 batch 跑一遍，让 PyTorch 记录峰值，再从剩余配额里分给 KV cache——这是 vLLM 同款的 profile-and-allocate 思路

        2. 触发所有 kernel 的首次初始化：

        triton 的 store_kvcache_kernel 编译、flash-attn 的 varlen 路径选择、cuBLAS workspace 分配、Sampler 的 torch.compile 编译

        3. 提前暴露 shape 相关的错误
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

    @torch.inference_mode()
    def capture_cuda_graph(self):
        """
        正常 PyTorch 推理时，每个 op 都要单独发射 kernel，且每次 launch 有 CPU 开销（Python 层调度 + CUDA launch）
        decode 阶段每个序列只算 1 个 token，batch 小、kernel 极碎，CPU launch 开销占比甚至超过 GPU 计算
        CUDA Graph 把整段 kernel 录制成一张静态图，之后 graph.replay() 一次调用就把所有 kernel 依原样重放，几乎消除 CPU 开销
        代价是：图是静态的——形状、内存地址、kernel 序列都固定
        """
        max_batch_size = min(self.config.max_num_seqs, self.config.max_num_cuda_graph_seqs)
        max_num_blocks = (self.config.max_model_len + self.block_size - 1) // self.block_size

        # 录图前就分配好这些固定地址的输入/输出缓冲区
        # 因为图内 kernel 引用的是固定指针，replay 时不能换张量，只能往同一块显存里写新数据
        input_ids = torch.zeros(max_batch_size, dtype=torch.int64)
        positions = torch.zeros(max_batch_size, dtype=torch.int64)

        slot_mapping = torch.zeros(max_batch_size, dtype=torch.int32)
        context_lens = torch.zeros(max_batch_size, dtype=torch.int32)
        block_tables = torch.zeros(max_batch_size, max_num_blocks, dtype=torch.int32)

        outputs = torch.zeros(max_batch_size, self.hf_config.hidden_size)
        # =============================================================================

        self.graph_vars = {
            "input_ids": input_ids,
            "positions": positions,
            "slot_mapping": slot_mapping,
            "context_lens": context_lens,
            "block_tables": block_tables,
            "outputs": outputs,
        }

        # 小 batch 用细粒度（1/2/4/8），大 batch 每 16 一档
        # 运行时取 最小的 ≥ 实际 batch_size 的 bucket，多出来的行靠 padding 空转：
        # 最坏情况浪费 15 行算力，换来 bucket 数量可控（512 时最多 ~35 张图）
        self.graph_batch_size = [1, 2, 4, 8] + list(range(16, max_batch_size + 1, 16))

        self.graph_pool = None
        self.graphs = {}

        # reversed：先录最大的图，它的显存池最大， self.graph_pool 在这一轮建好
        # 后面小图都用同一个 pool，激活显存互不叠加 —— 这是显存能省下来的关键
        for batch_size in reversed(self.graph_batch_size):
            graph = torch.cuda.CUDAGraph()

            # 把 decode 模式的上下文（block_tables 等）设为录图用的哑张量，录完清掉
            set_context(
                is_prefill=True,
                slot_mapping=slot_mapping[:batch_size],
                context_lens=context_lens[:batch_size],
                block_tables=block_tables[:batch_size],
            )

            # 每张图捕获前先在同一组 buffer 上做一次 warmup，保证 pool 里有足够的块可复用
            outputs[:batch_size] = self.model(input_ids[:batch_size], positions[:batch_size])  # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:batch_size] = self.model(input_ids[:batch_size], positions[:batch_size])  # capture

            # graph_pool 共享内存池：
            # 所有图共用一个显存池，多张图的中间激活内存可以重叠复用，而不是每张图各占一份
            # 只在第一次录完拿 pool 交给后续的 torch.cuda.graph(graph, self.graph_pool)
            if self.graph_pool is None:
                self.graph_pool = graph.pool()

            self.graphs[batch_size] = graph

            # 保证 capture 干净结束
            torch.cuda.synchronize()

            reset_context()
