# mini-qwen3

从零实现 Qwen3 的架构（目标：对齐官方 Qwen3-0.6B 的模型结构）。

## 目标

- [x] 配置文件：对齐官方 Qwen3-0.6B
- [x] RMSNorm（fp32 计算 + 可学习 weight）
- [x] SwiGLU FFN（gate / up / down 三投影，bias=False）
- [ ] RoPE 旋转位置编码（rope_base=1e6）
- [ ] GQA 注意力（n_kv_groups=8、head_dim=128、qk_norm）
- [ ] Decoder block 组装（残差连接、pre-norm）
- [ ] 推理 demo（加载官方权重或随机权重跑通生成）

## 运行

```bash
uv sync --all-packages   # 仓库根目录执行一次，装齐所有轮子的依赖
uv run python test.py    # 在本目录内运行单元测试
```

## 结构

```
config.py     # Qwen3-0.6B 配置
rms_norm.py   # RMSNorm
ffn.py        # SwiGLU FFN
model.py      # 注意力 / Decoder / 整体模型（待实现）
test.py       # 单元测试（uv run python test.py）
```

## 参考

- Qwen3 技术报告: <https://arxiv.org/abs/2505.09388>
- Qwen3-0.6B 模型卡（官方 config.json 来源）: <https://huggingface.co/Qwen/Qwen3-0.6B>
- RoPE 旋转位置编码: <https://arxiv.org/abs/2104.09864>
- GQA 分组查询注意力: <https://arxiv.org/abs/2305.13245>
- SwiGLU: <https://arxiv.org/abs/2002.05202>
- RMSNorm: <https://arxiv.org/abs/1910.07467>
