# mini-torchfeather

从零实现 [torchfeather](https://github.com/hkproj/torchfeather)——一个基于 PyTorch 官方 [torchtitan](https://github.com/pytorch/torchtitan) 裁剪而来的轻量分布式训练框架（FSDP / 多机多卡训练 LLM）。

## 目标

- [ ] 分布式启动（SLURM / torchrun）
- [ ] FSDP 数据并行（分片模型 / 优化器状态）
- [ ] 并行策略（DP / TP / PP 组合）
- [ ] checkpoint 保存与恢复
- [ ] 训练 MoE 模型（对齐上游的 DeepSeek-MoE 16B 示例）

## 运行

（待补充）

## 参考

- torchfeather: <https://github.com/hkproj/torchfeather>
- torchtitan: <https://github.com/pytorch/torchtitan>
- PyTorch FSDP: <https://arxiv.org/abs/2304.11277>
