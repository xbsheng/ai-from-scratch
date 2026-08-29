# mini-tokenizer

从零实现一个 BPE 分词器

## 目标

- [ ] 训练：在语料上合并出词表（BPE merge 规则）
- [ ] 编码：text → ids
- [ ] 解码：ids → text
- [ ] 特殊 token 处理
- [ ] 与 `tiktoken` 结果对拍验证

## 运行

```bash
uv sync --all-packages      # 仓库根目录执行一次（dev 组的 tiktoken 默认会装，用于对拍）
uv run python tokenizer.py  # 在本目录内运行（脚本待实现）
```

## 参考

- Karpathy / minbpe: <https://github.com/karpathy/minbpe>
- Hugging Face tokenizer 总结: <https://huggingface.co/docs/transformers/tokenizer_summary>
- Sennrich et al., Neural Machine Translation of Rare Words with Subword Units: <https://arxiv.org/abs/1508.07909>
