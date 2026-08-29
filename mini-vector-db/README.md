# mini-vector-db

从零实现一个最小的向量检索引擎

## 目标

- [ ] 暴力检索：余弦/内积相似度 + top-k
- [ ] 索引加速：实现一种（IVF 或 HNSW），对比召回率与速度
- [ ] 持久化：保存/加载索引
- [ ] 简单过滤查询（如按 metadata 过滤）
- [ ] Benchmark：与暴力搜索对比加速比

## 运行

```bash
uv sync --all-packages   # 仓库根目录执行一次，装齐所有轮子的依赖
uv run python benchmark.py # 对比索引检索与暴力搜索（脚本待实现）
```

## 参考

- Malkov & Yashunin, HNSW: <https://arxiv.org/abs/1603.09320>
- FAISS 入门文档: <https://faiss.ai/>
