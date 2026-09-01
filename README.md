# ai-from-scratch

从零实现 AI 相关的工具与框架，理解底层原理。

每个 `mini-*` 目录是一个独立的"轮子"

## 目录

| 轮子                                   | 内容                                                     | 状态      |
| -------------------------------------- | -------------------------------------------------------- | --------- |
| [mini-transformer](./mini-transformer) | 从零实现 Transformer（注意力、位置编码、训练一个小模型） | 🚧 规划中 |
| [mini-qwen3](./mini-qwen3)             | 从零实现 Qwen3 架构（RMSNorm / SwiGLU / RoPE / GQA）     | ✅ 完成   |
| [mini-tokenizer](./mini-tokenizer)     | 从零实现 BPE 分词器                                      | 🚧 规划中 |
| [mini-agent](./mini-agent)             | 从零实现 Agent 循环（工具调用 / ReAct）                  | 🚧 规划中 |
| [mini-rag](./mini-rag)                 | 从零实现 RAG 流水线（切分、检索、生成）                  | 🚧 规划中 |
| [mini-vector-db](./mini-vector-db)     | 从零实现向量检索（暴力搜索 → 索引加速）                  | 🚧 规划中 |

## 约定

- 每个 `mini-*` 自带 README：写明目标清单、运行方式、参考资料
- 依赖尽量少：核心实现只用 Python 标准库 + numpy；涉及训练的可用 PyTorch
- Python 依赖用 uv workspace 管理：
  - 每个 `mini-*` 在自己的 `pyproject.toml` 里声明依赖，全仓库共享一个 `.venv` 与 `uv.lock`
  - 常用命令：`uv sync --all-packages` 装齐所有轮子的依赖（注意：根目录裸 `uv sync` 只同步根项目本身，而根项目没有依赖）、在轮子目录内 `uv run python xxx.py` 运行、`uv add --package mini-<name> <pkg>` 加依赖
- 命名惯例：`mini-<name>`，新增轮子时复制一个现有目录做起点
- 完成一个轮子后更新上面的状态列（🚧 规划中 → 🏗️ 进行中 → ✅ 完成）
