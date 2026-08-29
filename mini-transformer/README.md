# mini-transformer

从零实现一个 Transformer

## 目标

- [ ] 缩放点积注意力（single-head → multi-head）
- [ ] 位置编码（sinusoidal / RoPE）
- [ ] Encoder / Decoder block（LayerNorm、FFN、残差连接）
- [ ] 在小数据集（如 tiny shakespeare）上训练一个能跑的小模型

## 运行

```bash
uv sync --all-packages   # 仓库根目录执行一次，装齐所有轮子的依赖
uv run python train.py   # 在本目录内运行（脚本待实现）
```

## 参考

- Attention Is All You Need: <https://arxiv.org/abs/1706.03762>
- The Annotated Transformer: <https://nlp.seas.harvard.edu/annotated-transformer/>
- Karpathy / nanoGPT: <https://github.com/karpathy/nanoGPT>
- Karpathy / Let's build GPT (video): <https://www.youtube.com/watch?v=kCc8FmEb1nY>
