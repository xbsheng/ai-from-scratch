# mini-vllm

从零实现 vLLM 的推理引擎核心：PagedAttention 显存管理、连续批处理调度、CUDA Graph 加速。

参考 vLLM 的 profile-and-allocate 思路与分块 KV cache 设计，用尽可能少的代码复现其架构骨架。

## 目标

- [x] Config / SamplingParams（数据类，含 `__post_init__` 推导）
- [x] Sequence：请求状态机（WAITING / RUNNING / FINISHED）、block 视图（`get_block_token_ids` 等）
- [x] BlockManager：PagedAttention 显存管理（256 token/block，可 append / preempt / deallocate）
- [x] Prefix Caching：block 级 hash（xxhash，逐块前缀哈希），命中即复用已有 KV
- [x] Scheduler：连续批处理（continuous batching）——prefill 优先、按 `max_num_batched_tokens` / `max_num_seqs` 配额调度、显存不足时抢占（preempt）换页式回滚
- [x] LLMEngine：`add_request` → `step` 循环 + `generate` 流式输出
- [x] ModelRunner：warmup 假 forward 测显存峰值 → 按 vLLM 同款公式预留 KV cache（`total × util − used − model_peak`）
- [x] CUDA Graph：decode 阶段按 batch bucket（1/2/4/8 + 每 16 一档）录图，共享 `graph_pool` 复用激活显存
- [x] 全局 Context（`utils/context.py`）：向 attention 传递 `slot_mapping` / `block_tables` / cu_seqlens，prefill 与 decode 共用一套模型 forward
- [x] 多卡 IPC：rank > 0 经共享内存 + Event 接收指令进入事件循环（Tensor 并行的调度通路）
- [ ] 模型 forward（Qwen 加载 +权重）
- [ ] attention kernel（flash-attn varlen + store_kvcache triton kernel）
- [ ] `run()` 真实执行 + Sampler（含 torch.compile）
- [ ] TP all-reduce 通信

## 运行

```bash
uv sync --all-packages          # 仓库根目录执行一次，装齐所有轮子的依赖
```

尚未有可运行入口（`ModelRunner.run()` 还是空壳），当前可通过 `LLMEngine` 单步调试调度器与 block 管理逻辑。

## 结构

```
minivllm/
├── config.py            # 引擎配置（显存利用率、block 大小、TP 数等）
├── sampling_params.py   # 采样参数
├── utils/
│   └── context.py       # 全局 attention 上下文（set/get/reset）
└── engine/
    ├── sequence.py      # 请求状态机 + block 视图
    ├── block_manager.py # PagedAttention 分块管理 + 前缀缓存
    ├── scheduler.py     # 连续批处理调度（prefill/decode、抢占）
    ├── model_runner.py  # warmup → KV cache 分配 → CUDA Graph 捕获
    └── llm_engine.py    # 引擎入口：请求队列 + step 循环
```

## 参考

- vLLM 论文（PagedAttention）: <https://arxiv.org/abs/2309.06180>
- vLLM 源码: <https://github.com/vllm-project/vllm>
- PagedAttention 博客: <https://blog.vllm.ai/2023/06/20/vllm.html>
- FlashAttention: <https://arxiv.org/abs/2305.04560>
