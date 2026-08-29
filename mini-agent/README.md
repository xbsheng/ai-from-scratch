# mini-agent

从零实现一个最小的 Agent 循环

## 目标

- [ ] 基本循环：LLM 调用 → 解析动作 → 执行工具 → 回填结果 → 再调用
- [ ] 工具注册与 schema 定义（function calling）
- [ ] ReAct 风格的思考-行动-观察循环
- [ ] 多轮上下文管理与终止条件
- [ ] 一个端到端示例（如：让 Agent 用计算器和搜索工具回答问题）

## 运行

```bash
# 先在本目录创建 .env 配置 API key（已被 gitignore）：
#   OPENAI_API_KEY=sk-xxx
# 兼容任意 OpenAI 兼容端点（如 DeepSeek），需要时另配 base_url
uv run python agent.py   # 在本目录内运行（脚本待实现）
```

> 依赖暂未声明，动手实现时先加：`uv add --package mini-agent openai python-dotenv`

## 参考

- ReAct: Synergizing Reasoning and Acting in Language Models: <https://arxiv.org/abs/2210.03629>
- OpenAI Function Calling 文档: <https://platform.openai.com/docs/guides/function-calling>
- Anthropic Tool Use 文档: <https://docs.anthropic.com/en/docs/build-with-claude/tool-use>
