# 劳动法知识库 Agent

面向 Agent 应用开发学习与实习作品集的问答项目。基于 DeepSeek、LangGraph、FastAPI 和 Streamlit，将中文向量检索、BM25、证据引用校验、会话持久化和用户记忆串成端到端应用。

本仓库从学习目录中独立整理，保留 `projects.enterprise_knowledge_agent` 包路径，避免修改已验证的业务代码。详细架构、接口与评估说明见 [项目说明](projects/enterprise_knowledge_agent/README.md)；项目答辩参考见 [技术问答](projects/enterprise_knowledge_agent/INTERVIEW.md)。

## 能做什么

- 检索《劳动法》《劳动合同法》共 205 条条文，使用本地中文 Embedding 与 BM25 融合排序，并执行检索门控。
- LangGraph 编排模型调用、白名单工具执行与最终结构化答案校验。
- 校验引用的来源 ID 和原文片段；证据不足时允许拒答。
- SQLite Checkpoint 保存短期会话，SQLite Store 保存跨会话用户记忆，独立聊天表支持历史列表。
- 长期记忆经过提取、规则筛选和确认；后端验证一次性审核编号，防止篡改和重放。
- 提供普通 HTTP 与 SSE 阶段事件接口、Bearer 令牌身份校验、请求 ID 日志和自动化测试。

```text
Streamlit → HTTP 客户端 → FastAPI → AgentService → LangGraph
                                                    ├─ DeepSeek
                                                    ├─ 混合检索 → 法条索引
                                                    └─ 答案与引用校验
                             SQLite：会话检查点 / 长期记忆 / 聊天记录
```

## 本地运行（Windows CMD / Python 3.13）

```cmd
git clone https://github.com/zyh141665/labor-law-knowledge-agent.git
cd labor-law-knowledge-agent
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install -r requirements-dev.txt
copy projects\enterprise_knowledge_agent\.env.example projects\enterprise_knowledge_agent\.env
notepad projects\enterprise_knowledge_agent\.env
```

在 `.env` 中填写自己的 `DEEPSEEK_API_KEY`、可用的模型名，以及访问令牌配置。`AGENT_API_TOKENS` 为 `随机令牌:用户ID`，`AGENT_API_TOKEN` 填同一随机令牌；示例值不能作为部署密钥。调用真实模型会产生费用。

程序使用 Responses API、工具调用与结构化输出。更换模型或兼容服务时，应确认这些能力均受支持，不能只看“OpenAI 兼容”。

首次生成索引（会下载本地 Embedding 模型）：

```cmd
python -m projects.enterprise_knowledge_agent.scripts.build_index
```

终端 1 启动后端，保持窗口打开：

```cmd
.venv\Scripts\python.exe -m projects.enterprise_knowledge_agent.scripts.run_backend
```

终端 2 进入同一个仓库根目录，启动前端：

```cmd
.venv\Scripts\python.exe -m projects.enterprise_knowledge_agent.scripts.run_frontend --port 8502
```

前端：`http://127.0.0.1:8502`；接口文档：`http://127.0.0.1:8000/docs`。

## 测试

```cmd
.venv\Scripts\python.exe -m pytest -q projects\enterprise_knowledge_agent\tests
```

测试使用替身模型和临时数据，不要求填写真实 API Key，也不调用付费模型。检索与生成评估的运行方法见详细项目说明；真实生成评估需要 API 调用。

## 边界与安全

- 这是学习与演示项目，不是法律意见。法源完整性、时效性、适用地区与具体案情需要另外核验。
- 引用能在原文中找到，不代表回答的所有推论都正确；测试通过率也不等于真实业务准确率。
- SSE 当前用于显示执行阶段，不应把阶段事件等同于逐 Token 生成。
- 默认绑定本机地址；前端使用进程级令牌，不是面向公网的多用户登录系统。
- 审核记录暂存在单进程内存中，重启后需重新提取；不宣称支持多进程共享审核凭证。
- `.env`、数据库、真实聊天与记忆、索引、模型缓存和运行日志均不提交。索引需在本地生成。
- 删除会话会删除该会话的检查点与聊天记录，但不删除用户的长期记忆；删除前应等待当前问答完成。

本仓库不包含原学习目录中的练习、私人学习记录或其他项目。
