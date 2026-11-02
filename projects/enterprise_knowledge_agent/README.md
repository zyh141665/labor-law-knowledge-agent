# 劳动法知识库 Agent

> **2026-11-02账号登录升级**：默认使用注册/密码登录，不再让浏览器共享`.env`令牌；新增第四个数据库`data/accounts.sqlite`。旧用户数据未删除，也不会自动迁入新账号。下文旧令牌配置、认证探测和“三个SQLite”描述属于兼容模式或升级前说明，当前登录步骤与边界以[登录说明](docs/LOGIN.md)为准。账号模式应重启前后端后先注册再登录。

一个用于 Agent 应用开发学习与实习作品集的端到端项目。系统通过 DeepSeek、LangGraph、本地中文 Embedding 与 BM25 混合检索、《中华人民共和国劳动法》《中华人民共和国劳动合同法》条文知识库，完成证据约束的问答、会话记忆、HTTP 服务、日志追踪与离线评估。

知识库当前收录两部法律：《中华人民共和国劳动法》（2018 年修正，107 条，来源：人力资源和社会保障部官网）和《中华人民共和国劳动合同法》（2012 年修正，98 条，来源：中国人大网），合计 205 条条文。检索与问答框架本身与具体领域解耦，在 `knowledge/` 目录下新增 `*.txt`（一行一条，以“第X条”开头）并在 `app/retrieval.py` 的 `LAW_TITLES` 里登记法律简称，即可扩展法源。

## 项目能力

- 使用 LangGraph 编排“模型 → 工具 → 模型 → 校验”循环
- 使用 DeepSeek Responses API 完成工具选择和结构化回答
- 使用 `BAAI/bge-small-zh-v1.5` 生成真实文本向量
- 混合检索：向量召回（语义）+ BM25 关键词召回（字面），两路分数融合排序
- 双通道门控：向量相似度达标或关键词覆盖率达标的片段才进入证据
- 使用 Pydantic 校验工具参数、回答结构和引用内容
- 只允许模型调用工具白名单中的本地函数
- 检索结果不足时拒答，避免模型凭空补充事实
- 使用 SQLite Checkpoint 保存短期会话状态
- 使用 SQLite Store 保存跨会话的用户长期记忆
- 长期记忆写入经过模型提取、本地规则和明确确认；后端校验一次性审核编号，拒绝篡改、跨用户复用和重放
- 每轮问答投影到独立的聊天日志表，支持历史会话列表与刷新后恢复
- 前端与后端通过 HTTP 分离：Streamlit 只持有一个 `AgentApiClient`，不加载模型、不连数据库
- 问答走 SSE 事件流，逐步显示「检索 → 取到 N 条条文 → 生成 → 校验」四个阶段
- 后端错误带稳定 `code`（余额不足 / Key 失效 / 引用校验失败），前端按码出提示
- 同时提供命令行、普通 HTTP、SSE 事件流和 Streamlit 界面
- 使用 request ID、UTF-8 轮转日志追踪请求
- 提供单元测试、检索评估和端到端生成评估

## 核心流程

```text
用户问题
  ↓
FastAPI / AgentService
  ↓
恢复 thread_id 对应的短期状态，并读取 user_id 对应的长期记忆
  ↓
LangGraph 要求模型选择知识库工具
  ↓
Pydantic 校验工具参数，执行本地混合检索（向量 + BM25）
  ↓
双通道检索门控：向量相似度或关键词覆盖率任达标
  ↓
模型基于证据生成结构化回答和引用
  ↓
本地校验来源 ID 与逐字引用
  ↓
返回回答、拒答状态、引用和模型调用轮数
```

## 目录结构

```text
enterprise_knowledge_agent/
├─ app/                 核心业务、LangGraph、RAG、记忆和 HTTP 接口
│  ├─ api.py            FastAPI 路由（后端唯一入口）
│  ├─ api_client.py     HTTP 客户端（前端唯一入口）
│  ├─ service.py        AgentService：准备 → 跑图 → 收尾
│  └─ ...
├─ knowledge/           法条原文（一行一条）：labor_law.txt、labor_contract_law.txt
├─ scripts/             建索引、记忆演示、评估入口和启动脚本
│  ├─ run_backend.py    单独启动后端（终端 1）
│  ├─ run_frontend.py   单独启动前端（终端 2）
│  └─ run_demo.py       可选：一条命令同时起两边
├─ evals/               人工标注的检索与生成评估集
├─ tests/               不调用真实模型的自动化测试
├─ data/                运行时索引和 SQLite 数据库（不提交 Git）
├─ logs/                运行日志（不提交 Git）
├─ reports/             评估报告（不提交 Git）
├─ streamlit_app.py     Streamlit 前端（只通过 HTTP 调后端）
├─ start_backend.bat    双击启动后端（等价于 run_backend.py）
├─ start_frontend.bat   双击启动前端（等价于 run_frontend.py）
├─ .env.example         环境变量模板
├─ requirements.txt     运行依赖
└─ requirements-dev.txt 开发和测试依赖
```

### 前后端怎么分工

```text
Streamlit（界面）  ──HTTP/JSON──▶  FastAPI（AgentService）
  api_client.py                     api.py
  不加载模型                         模型 + 检索 + 三个 SQLite
  不连数据库                         唯一写数据的人
```

前端不加载 Embedding 模型，昂贵的模型资源由后端复用。启动时间以本机实测为准。
前后端不再各持一套数据库连接，但后端并发请求仍可能竞争 SQLite，不能据此宣称没有并发问题。

## 环境要求

- Windows 11
- Python 3.13
- DeepSeek API Key
- 首次下载本地 Embedding 模型时需要网络连接

以下命令均在仓库根目录 `labor-law-knowledge-agent` 中执行（原学习目录中的命令也可以继续使用）。

## 1. 创建环境并安装依赖

如果还没有虚拟环境：

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install -r projects\enterprise_knowledge_agent\requirements-dev.txt
```

已经存在 `.venv` 时，只需激活并安装依赖：

```cmd
.venv\Scripts\activate.bat
python -m pip install -r projects\enterprise_knowledge_agent\requirements-dev.txt
```

只运行应用、不运行测试时，可以安装较小的运行依赖：

```cmd
python -m pip install -r projects\enterprise_knowledge_agent\requirements.txt
```

## 2. 配置 DeepSeek 与访问令牌

复制项目模板：

```cmd
copy projects\enterprise_knowledge_agent\.env.example projects\enterprise_knowledge_agent\.env
```

打开新建的 `.env`，填写自己的密钥和令牌：

```dotenv
DEEPSEEK_API_KEY=你的真实密钥
DEEPSEEK_MODEL=deepseek-flash

# 后端：令牌表，格式 令牌:用户ID，多个用逗号分隔
AGENT_API_TOKENS=replace-with-your-generated-token:student-001
# 前端：从上面挑一个令牌原样填进来
AGENT_API_TOKEN=replace-with-your-generated-token
```

程序优先读取项目目录下的 `.env`，如果不存在，则兼容读取学习仓库根目录下的 `.env`。真实 `.env` 已被 Git 忽略。

**令牌就是身份**：后端从 `Authorization: Bearer <token>` 认出你是谁，请求里带别人的
`user_id` 会直接返回 403，不会静默返回空列表。没配 `AGENT_API_TOKENS` 时后端**拒绝启动**——
认证静默失效比启动报错危险得多。

生成一个像样的令牌：

```cmd
python -c "import secrets;print(secrets.token_urlsafe(24))"
```

## 3. 建立向量索引

首次运行或修改 `knowledge/` 下任意条文文件后执行：

```cmd
python -m projects.enterprise_knowledge_agent.scripts.build_index
```

首次执行会下载 `BAAI/bge-small-zh-v1.5`。索引保存到 `data/index/`，以后正常启动不会重复建立。

## 4. 使用命令行 Agent

```cmd
python -m projects.enterprise_knowledge_agent.app.main
```

可选参数决定用哪份记忆、恢复哪段会话：

```cmd
python -m projects.enterprise_knowledge_agent.app.main --user alice --thread job-hunting
```

默认身份是 `cli-user` / `cli-session`，与前端演示用户（`student-001`）隔离，避免串数据。

CLI 和 HTTP 入口共用同一套服务层，所以命令行聊的内容同样会写进 `chat_history.sqlite`，
检查点 thread_id 也是同一种格式——换个入口不会「失忆」。

支持的控制命令：

```text
/remember <内容>  提取候选长期记忆，通过本地门控并经用户确认后保存
/memories         查看当前用户的长期记忆
quit              保存当前会话并退出
```

长期记忆写入前会自动判重：同一件事（内容相同，或 key 属于同一个前缀家族）只保留一条，
合并时保留更短的 key、内容取更完整的那条。想检查历史数据里有没有重复：

```cmd
python -m projects.enterprise_knowledge_agent.scripts.dedup_memories
```

该命令默认只报告，不会改动数据；确认无误后加 `--apply` 才真正合并。

## 4.5 启动前端与后端（两个终端）

前端只通过 HTTP 访问后端，**后端必须先启动**。开两个 cmd 窗口，各跑一条命令。

### 终端 1：后端

```cmd
cd /d D:\量化agent\labor-law-knowledge-agent
.venv\Scripts\activate.bat
python -m projects.enterprise_knowledge_agent.scripts.run_backend
```

看到 `Uvicorn running on http://127.0.0.1:8000` 就是就绪了。首次启动要加载 Embedding 模型，
约 10~20 秒，这段时间前端会连不上，属正常。

这个窗口会一直停着输出访问日志（每来一个请求打一行 `INFO: 127.0.0.1 - "GET /threads ..."`），
**不要关**。停后端就在这个窗口按 `Ctrl+C`。

### 终端 2：前端

```cmd
cd /d D:\量化agent\labor-law-knowledge-agent
.venv\Scripts\activate.bat
python -m projects.enterprise_knowledge_agent.scripts.run_frontend
```

启动前会先探一次后端 `/info`：

- 连得上 → 打印 `后端已就绪：deepseek-flash · 205 条法条 · v0.4.0`，然后打开界面
- 连不上 → 打印提示，**界面照常打开**，顶部显示未连接和「重试连接」按钮

后端还在加载模型时有这个差异很重要：不必等它加载完才能开前端，加载好了点一下重试即可。

### 也可以直接双击

项目根目录下有两个批处理，双击各开一个 cmd 窗口，等价于上面两条命令：

```text
start_backend.bat     双击 → 后端窗口
start_frontend.bat    双击 → 前端窗口
```

### 换端口

两个脚本用参数对齐，不用改代码：

```cmd
:: 终端 1
python -m projects.enterprise_knowledge_agent.scripts.run_backend --port 8010

:: 终端 2：地址要跟着改，否则前端找的还是 8000
python -m projects.enterprise_knowledge_agent.scripts.run_frontend --api http://127.0.0.1:8010
```

### 开发时想改代码自动重启

```cmd
python -m projects.enterprise_knowledge_agent.scripts.run_backend --reload
```

只监听 `app/` 目录。不这么限定的话，`data/` 下 SQLite 每次写入都会触发重启，
刚问完一句话后端就自己关了。

### 只想开一个窗口

`run_demo.py` 保留着，它会先起后端、轮询 `/health`、再起前端，Ctrl+C 一起停：

```cmd
python -m projects.enterprise_knowledge_agent.scripts.run_demo
```

代价是两边的日志混在同一个窗口里，后端报错和前端报错分不开。日常开发还是推荐两个终端。

浏览器打开 `http://localhost:8501`。页面提供：

- 聊天式问答，回答过程中逐条显示「检索 → 取到 N 条条文 → 生成 → 校验」阶段
- 展示答案、拒答状态、逐字引用来源
- 侧边栏的历史会话列表，点击即可切回任意一段过往对话，可单条删除
- 顶部的「已连接 · 模型 · 法条数」取自后端 `/info`，不在前端硬编码
- 查看长期记忆，并可通过「提取候选记忆 → 本地门控 → 人工确认」三步写入

两个设计细节：

**身份由后端根据令牌确定，不由 URL 中的 user 参数决定。** 前端请求 `/me` 获取身份，
会话编号写在 `?thread=`，刷新后可恢复该账号的会话。后端会拒绝令牌身份与请求 `user_id` 不符的访问。
当前 Streamlit 进程从 `.env` 读取一把 `AGENT_API_TOKEN`，所有访问这个实例的浏览器仍然共用
该令牌绑定的账号；这不是完整的多用户登录系统，不应将当前界面公开给不可信访客。

**前端不再持有数据库连接。** 检查点、聊天日志、长期记忆三个 SQLite 都只由后端写入，
避免两个进程并发写同一批文件的锁竞争。

### 历史会话是怎么存的

检查点数据库里其实保存着完整对话，但不适合直接拿来渲染：`conversation_input`
里混着工具调用、推理片段和原始 JSON，引用信息也只在最后一轮的 `final_answer`
里保留。因此单独建了一张投影表（`data/chat_history.sqlite`）：

| 字段 | 含义 |
| --- | --- |
| `user_id` / `thread_id` | 与前端的用户、会话两级隔离保持一致 |
| `role` | `user` 或 `assistant` |
| `content` | 问题原文或已通过校验的答案 |
| `refused` | 该轮是否属于证据不足的拒答 |
| `citations` | 引用来源与逐字引文，JSON 存储 |
| `created_at` | 本地时间；从旧检查点导入的记录没有可靠时间，留空 |

写入发生在 `AgentService.answer()` 里，也就是**只有通过引用校验的答案才会入库**，
被拦下的半成品不会污染历史记录。写日志失败（磁盘满、库被锁）只降级为「本次不可回看」，
不会让已经算出来的答案丢失。

旧会话可以通过一次性脚本搬进来：

```cmd
python -m projects.enterprise_knowledge_agent.scripts.import_chat_history
```

脚本会按检查点的写入顺序还原会话新旧。注意当前版本的 LangGraph 不写
`metadata["ts"]`，所以导入的记录**没有真实时间**，界面上只显示消息条数 ——
这一点在脚本里被显式处理成空字符串，而不是填一个“导入时刻”冒充历史时间。

对应的只读接口：

- `GET /threads?user_id=...`：该用户最近的会话列表
- `GET /threads/{thread_id}/messages?user_id=...`：某个会话的完整消息

## 5. HTTP 接口清单

后端就是 4.5 里的「终端 1」，只需要那一条命令；这里列它对外提供的接口。

**除 `/health` 和 `/docs` 外，所有接口都需要访问令牌**：

```cmd
curl -H "Authorization: Bearer YOUR_GENERATED_TOKEN" http://127.0.0.1:8000/me
```

启动后可访问：

| 接口 | 用途 | 需要令牌 |
|---|---|---|
| `GET /health` | 健康检查，不碰模型和数据库 | 否 |
| `GET /me` | 我是谁（由令牌决定），前端据此显示身份 | 是 |
| `GET /info` | 后端自述：模型名、法条数、实际生效的检索参数、接口清单 | 是 |
| `POST /chat` | 普通问答，等整轮跑完返回 | 是 |
| `POST /chat/stream` | SSE，逐条推送阶段进度，最后推结果 | 是 |
| `GET /threads?user_id=` | 该用户的历史会话列表 | 是 |
| `GET /threads/{id}/messages?user_id=` | 某个会话的全部消息 | 是 |
| `DELETE /threads/{id}?user_id=` | 删除该会话的聊天记录及 Checkpoint，保留用户长期记忆 | 是 |
| `GET /memories?user_id=` | 该用户的长期记忆 | 是 |
| `POST /memories` | 校验一次性审核编号和明确确认后，写入原候选记忆 | 是 |
| `POST /memories/extract` | 从一句话提取候选记忆并跑本地门控 | 是 |
| `GET /docs` | Swagger UI | 否 |

`/chat` 请求示例：

```json
{
  "user_id": "demo-user-001",
  "thread_id": "demo-thread-001",
  "question": "法定节假日加班工资怎么算？"
}
```

其中 `user_id` 用于隔离长期记忆，`thread_id` 用于隔离短期对话。相同用户可以拥有多个不同会话。

### 长期记忆审核协议

先调用 `POST /memories/extract`，提交 `user_id` 和 `text`。仅通过本地规则的候选会返回
非空 `review_id`。用户核对候选后，向 `POST /memories` 提交：

```json
{
  "user_id": "student-001",
  "memory_key": "answer_style",
  "category": "preference",
  "content": "回答先给结论",
  "review_id": "原样复制提取接口返回的编号",
  "confirmed": true
}
```

key、类别和内容必须与后端已审核候选完全相同。缺字段返回 422；无效、过期、篡改、
已消费的编号或未确认请求返回 400 `memory_review_rejected`，不会写入。
编号有效期十分钟、成功写入后失效，审核记录只在当前后端进程内保存；服务重启后需重新提取。
最终写入时再次检查原始输入与候选的敏感信息规则。

`confirmed=true` 表达客户端明确提交的确认，不能证明屏幕前确实有真人点击；
后端保证的是已审核内容不能被替换，不是完整的人机认证。低层 `save_memory` 是测试与迁移使用的
可信内部函数，不是 HTTP 写入入口。当前审核缓存适用于单进程，多 worker 部署前需替换为共享存储。

### 会话删除的范围

删除接口先调用 Checkpointer 的 `delete_thread`，再删除聊天历史；目标键由
`[user_id, thread_id]` 编码，只影响这一用户的这一会话，不清理长期记忆。
两份数据库不共享事务，失败时应重试同一删除请求。当前未协调删除与正在执行的同一会话请求，
删除前应等该会话回答完成，避免进行中的请求重新写入状态。

### SSE 事件长什么样

`/chat/stream` 会依次推送这些事件，客户端可以据此显示进度：

```text
status  {"stage": "accepted",   "message": "Agent 已开始处理问题。"}
status  {"stage": "thinking",   "message": "正在理解问题并读取长期记忆"}
status  {"stage": "retrieving", "message": "正在检索劳动法条文"}
status  {"stage": "retrieved",  "message": "已取得 10 条候选条文", "match_count": 10}
status  {"stage": "generating", "message": "正在组织答案并标注引用"}
status  {"stage": "validating", "message": "正在校验引用是否逐字来自原文"}
result  {"answer": "...", "refused": false, "citations": [...], "round_number": 2}
```

失败时用 `error` 事件代替 `result`，带 `code` 和 `message`。

**为什么不是逐 Token 流。** 模型必须输出完整的 JSON（answer + citations），
而引用真实性校验要等 JSON 收全才能做。在「先校验再展示」这条规则下，
逐 Token 输出意味着用户可能先看到一个后来被判定为编造条文的答案。
这里选择保留校验，把可见的进度做成阶段事件。

### 错误码

失败时返回 HTTP 状态码 + 结构化 `detail`，前端按 `code` 决定提示文案：

| 状态码 | code | 含义 |
|---|---|---|
| 401 | `unauthorized` | 访问令牌缺失或错误（后端这扇门没开） |
| 403 | `user_mismatch` | 令牌身份与请求里的 `user_id` 不一致，越权 |
| 400 | `memory_review_rejected` | 记忆未确认、审核编号无效或候选被修改，需重新提取 |
| 503 | `auth_not_configured` | 后端没加载到令牌表，属于启动配置问题 |
| 402 | `upstream_balance` | DeepSeek 余额不足，Key 仍然有效 |
| 401 | `upstream_auth` | DeepSeek API Key 无效或被吊销 |
| 502 | `upstream_error` | 上游模型其他故障 |
| 500 | `citation_failed` | 回答未通过引用真实性校验，已拦截 |
| 500 | `agent_error` | Agent 执行失败 |

注意 `unauthorized` 和 `upstream_auth` 都是 401，但原因完全不同：前者是**我们后端**拒绝了
你的令牌，后者是 **DeepSeek** 拒绝了我们的 API Key。查错方向不一样，别混。

## 6. 运行自动化测试

单元测试使用替身对象，不调用真实 DeepSeek，也不会消耗 API Token：

```cmd
python -m pytest -q projects\enterprise_knowledge_agent\tests
```

## 7. 运行质量评估

检索评估不调用 DeepSeek，检查正确文档是否排在第一位以及无关问题能否被门控：

```cmd
python -m projects.enterprise_knowledge_agent.scripts.evaluate_retrieval
```

生成评估调用真实 DeepSeek，检查拒答状态、引用来源和关键事实：

```cmd
python -m projects.enterprise_knowledge_agent.scripts.evaluate_generation
```

两个评估都使用 80% 作为当前学习项目的质量门槛。低于门槛时进程返回非零退出码，报告写入 `reports/`。

### 引用校验规则（以及它管不到的地方）

模型输出离开工作流前要过 `validate_answer`，规则只有三条：

1. **引用真实性是硬约束**：来源必须出现在本次检索结果里，引文必须是该来源原文的逐字子串——防的是模型编造条文；
2. **作答必须至少给出一条引用**，否则等于没有证据就下结论；
3. **拒答不要求引用为空**。

第 3 条与早期“拒答时必须无引用”的规则不同。拒答时附上“查过但不足以作答”的真实条文是合理的透明度表达，不应直接判成故障。目前提示词、本地校验与生成评估已统一，前端会将拒答场景的引用标题显示为“检索到但不足以支撑回答的条文”。

**这条规则管不到的，必须说清楚**：引用真实不等于推理正确。模型可能引用某条真实原文，却推出原文没有规定的责任或条件。当前缺少“结论是否被证据支持”的完整语义校验，不能把引用通过率当成回答正确率。旧笔记中的个别重复运行次数不构成稳定错误率估计，本次修复未重新调用真实模型测量这一比例。

### 检索泛化验证（零成本、可反复运行）

固定检索评估集有 7 条，额外诊断脚本包含 18 条域内自然问法和 13 条域外问题。
这些诊断问题与调参过程存在重合，不应宣称它们是从未参与调参的独立测试集：

```cmd
python -m projects.enterprise_knowledge_agent.scripts.validate_retrieval_quality
```

下表是旧版单法源的历史记录，不是当前 205 条双法源版本的复测结果。
改动代码、知识库或模型后，应重新运行脚本并保存样本范围、配置和输出，再引用指标：

| 指标 | 结果 | 说明 |
| --- | --- | --- |
| 域内 Top-1 命中 | 13/17（76%） | 排序精度 |
| 域内证据召回（正确条文进入前 4 条） | 17/17（100%） | 决定模型“有没有机会看到正确答案”的关键指标 |
| 域内误拒答 | 0/17 | 没有把该答的问题挡掉 |
| 域外正确拒答 | 10/13（77%） | 漏检 3 条：竞业限制（关键词通道）、社保缴费基数与工伤医疗费（向量通道） |

**两个指标必须分开看。** Top-1 衡量第一名是否可接受；这里的 Top-K 命中衡量至少一条
可接受来源是否进入前 K 条，不等于全部所需证据均已召回，更不等于最终回答正确。

### 混合检索的来历（为什么不是一开始就加）

旧版笔记记录了以下检索对比；当前缺少完整可核验的同批运行记录，不能把它们当作本次修复的实测收益：

| 配置 | 域内 Top-1 | 域内证据召回 | 域外正确拒答 |
| --- | --- | --- | --- |
| 纯向量 0.5 + Top-K=2 | 61% | 67% | 42% |
| 纯向量 0.6 + Top-K=4 | 65% | 94% | 85% |
| **向量 0.6 或关键词 0.15 + Top-K=4** | **76%** | **100%** | 77% |

每次调整都是先在真实数据上测出失败案例才改的：

1. **阈值 0.5 → 0.6**：换成 107 条真实法条后，域外问题的向量相似度普遍落在 0.50~0.67，12 条里 7 条能通过门控（误放行 58%）。提高阈值后降到 8%，且域内问题没有被误拒。
2. **Top-K 2 → 4**：线上提问「法定节假日加班工资怎么算」时，模型回答“检索结果不足以确定倍数”。排查发现正确答案第四十四条得分 0.6216，**虽然超过阈值，但排在第五十一条（0.6519）和第四十条（0.6267）之后被 `top_k=2` 挤掉了**——模型没有幻觉，是证据没喂到。
3. **加入 BM25 关键词通道**：Top-K 放宽到 4 之后，仍有「法定节假日加班给几倍工资」检不到，因为它的向量分只有 0.584，连门控都过不了。而第四十四条原文里就有“法定休假日”四个字——这是向量不擅长、关键词一抓就准的场景。加入 BM25 后该问题首次被正确召回，模型也答出了 300%。

**这次调整有真实代价，不藏起来**：关键词通道会把「竞业限制补偿金标准」放行到第九十一条（经济补偿），域外正确拒答从 85% 降到 77%。但生成层还有第二道门——实测该问题放行后，模型看到第九十一条仍会拒答，因为条文内容答不了这个问题。**检索层放宽 + 生成层兜底**，是有意的设计取舍：宁可在检索层多给一条证据，也不要因为漏检让能答的问题答不出来。

### 关于中文 BM25 的工程细节

中文没有空格，BM25 必须先分词。这里没有用 jieba，而是**字符 bigram**（`app/bm25.py`）：

- 法条里全是“法定休假日”“用人单位”这类专业术语，通用词典经常切错，切错就直接导致匹配失败；
- 字符 bigram 对术语已经够敏感：“法定休假日”和“法定节假日”整体不同，但共享“法定”“假日”两个 token，正好覆盖同义换写；
- 零额外分词依赖，关键词索引由已加载的 chunks 在内存重建；知识源变化后仍须重新建立磁盘索引，否则加载的 chunks 和向量都会保持旧版本。

另外两个实现细节：

- **BM25 分数没有固定上界**，直接与余弦相似度相加可能由关键词分主导。当前用 `b/(b+5)` 压缩到 0~1；这是单调映射，但不是概率校准，不保证两路分数完全可比。
- **门控用“关键词覆盖率”而不是按每次查询最大值归一化的 BM25 分数**。若采用后一种归一化，非零最高分总会是 1，难以做绝对拒答阈值。当前 BM25 排序实际使用前述 `b/(b+5)`，并没有按最大值归一化。覆盖率衡量查询词的 idf 权重有多大比例被该文档覆盖，也可能放行没有答案但共享关键词的问题。

阈值和权重都收敛在 `app/config.py`（`KNOWLEDGE_MIN_SCORE` / `KNOWLEDGE_KEYWORD_COVERAGE` / `KNOWLEDGE_TOP_K` / 融合权重），不在各入口重复硬编码。

### 标注错误也值得记录

调参过程中出现过三次**我自己标错答案**的情况，这类错误会反过来把优化带向错误方向：

| 问题 | 我原来的标注 | 原文实际内容 |
| --- | --- | --- |
| 劳动合同必须书面订立吗 | 第十六条 | **第十九条**才是“应当以书面形式订立”，第十六条只是定义劳动合同 |
| 用人单位拖欠工资怎么办 | 第五十条 | **第九十一条**才是“拖欠后的处置”（第九十一条：责令支付 + 赔偿金） |
| 工伤医疗费由谁承担 | 第七十三条 | 第七十三条只说“依法享受社会保险待遇”，**完全没有讲医疗费谁付**——这是《工伤保险条例》的内容，属于域外问题 |

所以脚本现在允许一条问题配置多个可接受条文（例如“拖欠工资怎么办”同时接受第五十条与第九十一条），第三条则被移出域内集合。**先确认标注是对的，再谈指标提升。**

## 运行数据与日志

- `data/checkpoints_v2.sqlite`：按 `user_id + thread_id` 保存短期会话状态
- `data/memory.sqlite`：按 `user_id` 保存跨会话长期记忆
- `data/chat_history.sqlite`：界面用的对话日志，支持历史会话列表与刷新恢复
- `logs/agent_api.log`：FastAPI 请求和 Agent 指标日志
- `X-Request-ID` 响应头：用于把一次 HTTP 请求与日志对应起来

日志不会记录问题正文、模型回答或 API Key，避免把敏感内容直接写入文件。
注意 `chat_history.sqlite` 保存的是完整问答正文 —— 它是功能数据而不是日志，
删掉 `data/` 下的文件会同时清空这些历史会话。

## 常见问题

### 前端提示「无法连接后端」

前端启动时会先请求 `GET /info`，拿不到就只显示一条提示和「重试连接」按钮。常见原因：

1. 只起了前端没起后端。在另一个终端跑 `python -m projects.enterprise_knowledge_agent.scripts.run_backend`。
2. 后端还在加载模型（约 10~20 秒）。等终端出现 `Uvicorn running` 后点重试。
3. 端口不是 8000。启动前端时用 `--api http://127.0.0.1:端口` 指过去，或设 `AGENT_API_BASE_URL` 环境变量。

### 前端能连上，但点历史会话返回 404

**这是系统代理导致的，不是代码问题。** 判断特征：第一次请求成功，第二次开始全部 404；
用 `curl` 单独请求同一个地址却是 200。

原因是 httpx 默认会读 `HTTP_PROXY` / `HTTPS_PROXY` 环境变量。一旦走代理，
它按代理协议把请求行写成绝对 URI：

```text
GET http://127.0.0.1:8000/threads HTTP/1.1    ← 代理格式，uvicorn 不认
GET /threads HTTP/1.1                          ← 正确格式
```

uvicorn 只匹配路径，所以返回 `{"detail":"Not Found"}`。更麻烦的是它往往
第一次成功、之后复用连接才失败，看起来像随机 bug。

`AgentApiClient` 里已经用 `trust_env=False` 关掉了这个行为。如果你在其他地方
自己 new 了 `httpx.Client`，记得也加这个参数。检查当前是否有代理：

```cmd
set | findstr /i proxy
```

### 改了 `app/` 下的代码，前端行为没变

前后端分离之后，前端只通过 HTTP 说话，改 `app/` 里的代码**必须重启 uvicorn** 才生效。
如果后端是用普通命令起的（没有 `--reload`），改完 `app/config.py` 的阈值后
界面会一直用旧值，很容易误判成"改了没用"。

开发时给后端加 `--reload`，改完保存即自动重启：

```cmd
python -m projects.enterprise_knowledge_agent.scripts.run_backend --reload
```

它只监听 `app/` 目录，所以聊天记录写库不会误触发重启。

### 返回 422

说明请求已经到达 FastAPI，但 JSON 不符合 `ChatRequest`。请求字段必须是 `user_id`、`thread_id` 和 `question`，不是 `message`。

### 提示尚未建立向量索引

先运行：

```cmd
python -m projects.enterprise_knowledge_agent.scripts.build_index
```

### 启动时提示缺少 API Key

确认 `projects/enterprise_knowledge_agent/.env` 或学习仓库根目录 `.env` 中存在 `DEEPSEEK_API_KEY`。

### 提示 402 Insufficient Balance

这不是 Key 失效，而是账户余额不足。可以先用余额接口确认（Key 有效时能正常返回余额，Key 失效才会返回 401）：

```cmd
curl -H "Authorization: Bearer %DEEPSEEK_API_KEY%" https://api.deepseek.com/user/balance
```

充值后无需改任何代码。如果暂时不想充值，可以把 `.env` 里的 `DEEPSEEK_BASE_URL` 指向任意 OpenAI 兼容服务（硅基流动、阿里百炼、本地 Ollama 等），并同步修改 `DEEPSEEK_MODEL`——项目只依赖 OpenAI 兼容协议，不需要改代码。注意：换模型后检索门控阈值和生成评估都应按新模型重新验证一次。

### 相同问题得到略有不同的文字

模型生成具有一定随机性。系统检查输出结构和逐字引用来源，但不保证回答中的所有结论均正确或完整。

## 当前边界

- 知识库为两部法律（《劳动法》107 条 + 《劳动合同法》98 条，共 205 条），索引保存在本地文件中，尚未接入专用向量数据库
- 混合检索的两路分数用等权相加，没有做学习式重排（rerank）；排序质量需重新评估，旧文档中的 78% / 94% / 69% 不作为本次复测指标
- **语义角色反转会检索失败**：问“什么情况下用人单位可以解除劳动合同”，前四条命中的全是“劳动者可以解除”（第三十八条、第三十七条），真正对的第三十九条、第四十条排在第七、第八名之外。向量模型看到“用人单位、解除、劳动合同”就往一起凑，分不清谁解除谁。要修得靠 query 改写或 cross-encoder 重排，当前未做
- 关键词通道可能放行域外问题，生成层仍需判断证据是否足够；增加资料后指标可能涨跌，需要区分样本变更、排序变化与阈值失配，不是必然规律
- 引用校验只检查“引文真实、来源存在”，**不能保证“结论被条文蕴含”**，没有当前可复核的稳定错误率
- 关键词覆盖率阈值 0.15 对语料规模敏感：语料越小，查询里“知识库中不存在的词”占比越高，覆盖率整体会被压低，换知识库后需要重新校准
- 阈值、Top-K 和融合权重是在 18 条域内 / 13 条域外问题上校准的经验值，更换知识库或 Embedding 模型后需要重新验证
- 建索引时 `batch_size` 必须显式限制为 16。默认一次性喂入全部条文时，同一批要按最长那条 padding，205 条会直接把 ONNX 撑到 `bad allocation`
- 生成评估的关键事实判断目前使用字符串匹配，可能受空格和措辞变化影响
- SSE 是工作流事件流，不是逐 Token 流
- 后端已提供令牌绑定用户的认证与越权检查；尚未提供每位网页访客独立登录、限流和云端部署
- 记忆审核编号为单进程内存态，后端重启需重新提取候选；敏感词门控也不等于完整的隐私识别
- Docker 容器化因本机资源限制暂缓，不影响本地运行与项目主体能力

## 安全说明

- 不提交 `.env`、SQLite 数据库、日志、模型缓存、向量索引和评估报告
- 不在代码、README、截图或聊天消息中公开真实 API Key
- 正常 CLI 和 HTTP 写入必须经过后端审核编号、明确确认及再次规则检查；模型不能直接操作数据库
