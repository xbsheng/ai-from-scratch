# mini-rag

从零实现一个最小的 RAG 流水线

## 目标

- [ ] 文档加载与切分（chunking：固定长度 / 按语义边界）
- [ ] 向量化（embedding，可调用现成 API 或本地小模型）
- [ ] 相似度检索（先暴力搜索，向量库部分见 mini-vector-db）
- [ ] 检索结果拼 Prompt → 生成回答
- [ ] 一个端到端示例：对一批本地文档提问

## 运行

uv sync --all-packages   # 仓库根目录执行一次，装齐所有轮子的依赖
uv run python rag.py     # 在本目录内运行（脚本待实现）
```

> 本轮子当前未声明依赖，动手实现时先加：`uv add --package mini-rag numpy openai python-dotenv`

## 参考

- Lewis et al., RAG 原始论文: <https://arxiv.org/abs/2005.11401>
- Chunking 策略综述: <https://www.pinecone.io/learn/chunking-strategies/>
