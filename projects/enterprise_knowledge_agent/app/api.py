"""企业知识库 Agent 的 FastAPI HTTP 服务入口。"""

import json  # 把流式事件数据编码成前端可以解析的 JSON 文本。
from collections.abc import AsyncIterator  # 标注生命周期和事件流生成器的返回类型。
from contextlib import asynccontextmanager  # 在服务启动和关闭时管理共享资源。
from time import perf_counter  # 统计每个 HTTP 请求从进入到返回的耗时。
from typing import Annotated, Any, Literal  # 把服务类型与 FastAPI 依赖注入声明组合起来。
from uuid import uuid4  # 为每个请求生成难以重复的追踪 ID。

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request  # 创建路由、依赖和 HTTP 错误。
from fastapi.responses import StreamingResponse  # 持续发送 SSE 事件而不是一次性 JSON。
from starlette.concurrency import iterate_in_threadpool  # 逐步拉取同步生成器而不阻塞事件循环。
from openai import OpenAI, OpenAIError  # 创建 DeepSeek 客户端并识别上游 API 错误。
from pydantic import BaseModel, Field, StrictBool, ValidationError  # 严格区分明确确认和字符串等伪布尔值。

from .auth import (  # 令牌认证：从凭证推导身份，不信任客户端自报。
    Principal,
    authenticate,
    load_token_map,
)
from .chat_history import (  # 把每轮问答投影成可回看的历史会话。
    ChatHistory,
    ChatMessage,
    ThreadSummary,
    open_chat_history,
)
from .checkpoint import open_checkpointer  # 在应用生命周期中打开短期检查点数据库。
from .config import (  # 读取配置与统一阈值，并对外报告实际生效的检索参数。
    KNOWLEDGE_KEYWORD_COVERAGE,
    KNOWLEDGE_MIN_SCORE,
    KNOWLEDGE_TOP_K,
    load_settings,
)
from .memory import open_memory_store  # 长期记忆存储的打开与关闭。
from .memory_policy import MemoryCandidate  # 候选记忆的数据结构，用于响应模型。
from .memory_service import MemoryService  # 记忆的唯一业务入口，CLI 与接口共用。
from .memory_review import MemoryReviewError, MemoryReviews  # 跨请求保留一次性候选审核记录。
from .logging_config import (  # 配置日志并在请求调用链中传播 request_id。
    get_logger,
    request_id_context,
    setup_logging,
)
from .retrieval import (  # 加载本地向量索引和 Embedding 模型。
    LAW_TITLES,
    VectorKnowledgeBase,
)
from .service import AgentService, AgentServiceProtocol, AgentTurnResult  # 复用 Agent 业务层。
from .tools import ToolRegistry  # 把知识库检索器注册成模型可调用工具。


logger = get_logger("api")  # 日志名称会显示为 enterprise_agent.api。


class HealthResponse(BaseModel):
    """健康检查接口对外承诺的固定响应格式。"""

    status: str  # 表示当前进程是否可以正常响应 HTTP 请求。
    service: str  # 标识返回结果来自哪个服务。
    version: str  # 方便部署和排查时确认当前运行版本。


class ChatRequest(BaseModel):
    """客户端调用 Agent 时必须提供的三个隔离字段。"""

    user_id: str = Field(  # user_id 用于读取跨会话长期记忆。
        min_length=1,
        max_length=64,
        pattern=r"^[a-zA-Z0-9_-]+$",
    )
    thread_id: str = Field(  # thread_id 用于恢复该用户的一段短期会话。
        min_length=1,
        max_length=64,
        pattern=r"^[a-zA-Z0-9_-]+$",
    )
    question: str = Field(  # 限制空问题和异常大的请求体。
        min_length=1,
        max_length=2000,
    )


class ServiceInfo(BaseModel):
    """``GET /info`` 的响应：前端据此显示连接状态，不用硬编码任何配置。"""

    service: str  # 服务标识，确认连对了后端。
    version: str  # 后端版本，排查问题时有用。
    model: str  # 当前使用的模型名称。
    knowledge_chunks: int  # 知识库里有多少条法条。
    knowledge_source: str  # 知识库文件路径的相对形式。
    retrieval: dict[str, Any]  # 实际生效的检索参数，方便核对阈值。
    capabilities: list[str]  # 后端支持的接口列表。


class WhoAmI(BaseModel):
    """``GET /me`` 的响应：我是谁，由令牌决定，不用客户端猜。"""

    user_id: str  # 该令牌绑定的用户身份。
    token_label: str  # 令牌末四位掩码，便于确认“现在用的是哪把钥匙”。


class MemoryItem(BaseModel):
    """一条对外展示的长期记忆。"""

    memory_key: str  # 稳定键，例如 learning_goal。
    category: str  # profile / preference / goal。
    content: str  # 记忆内容。
    updated_at: str  # 最近一次写入时间（UTC）。


class MemorySaveRequest(BaseModel):
    """人工确认后写入长期记忆的请求体。"""

    user_id: str = Field(min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_-]+$")
    memory_key: str = Field(min_length=2, max_length=64, pattern=r"^[a-z][a-z0-9_]*$")
    category: Literal["profile", "preference", "goal"]
    content: str = Field(min_length=1, max_length=500)
    review_id: str = Field(min_length=1, max_length=128)  # 必须来自 /memories/extract。
    confirmed: StrictBool  # 不能把 "false" 等字符串误当成用户确认。


class MemoryExtractRequest(BaseModel):
    """请后端从一句话里提取候选记忆。

    ``user_id`` 是必需字段：判重要拿这句话和该用户已有的记忆比对，
    不带用户身份就只能提出候选，说不出「这条跟哪条重复」。
    """

    user_id: str = Field(min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_-]+$")
    text: str = Field(min_length=1, max_length=500)


class MemorySaveResponse(MemoryItem):
    """写入结果：除了记忆本身，还要告诉前端到底发生了什么。

    把 ``action`` 带回来，界面才能区分“新增了一条”“和旧记忆合并了”
    和“本来就有，没重复写”，而不是一律显示「已保存」。
    """

    action: Literal["insert", "update", "merge", "skip"]
    duplicate_of: str | None = None  # 判定为重复的已有 key。
    removed_key: str | None = None  # 合并时被清理的冗余 key。
    reason: str  # 一句话说明为什么是这个动作。


class MemoryExtractResponse(BaseModel):
    """候选记忆 + 本地门控结论；前端据此决定是否显示「确认保存」。"""

    candidate: MemoryCandidate  # 模型给出的候选项。
    allowed: bool  # 本地规则是否放行。
    reason: str  # 放行或拒绝的原因，展示给用户看。
    duplicate_of: str | None = None  # 与已有记忆重复时填该 key，界面可提示“将合并”。
    review_id: str | None = None  # 被拒绝的候选不会拿到可写入凭证。


class DeleteThreadResponse(BaseModel):
    """删除会话的结果。"""

    thread_id: str
    deleted: int  # 实际删除的消息条数。


def api_error(status_code: int, code: str, message: str) -> HTTPException:
    """构造带稳定错误码的 HTTPException。

    前端不再靠匹配异常字符串来决定提示文案，而是读 ``detail.code``。
    这样新增接口时不用把错误映射再写一遍。
    """
    return HTTPException(
        status_code=status_code,
        detail={"code": code, "message": message},
    )


# ---------------------------------------------------------------------------
# 认证：所有会读写用户数据的接口都挂这两个依赖
# ---------------------------------------------------------------------------


def require_principal(
    request: Request,
    authorization: Annotated[str | None, Header()] = None,
) -> Principal:
    """校验 ``Authorization: Bearer <token>``，返回该令牌对应的身份。

    令牌表在启动时加载到 ``app.state``，所以这里不读环境变量，
    也就能被测试用依赖覆盖的方式替换掉。
    """
    token_map = getattr(request.app.state, "auth_tokens", None)
    if not token_map:  # 正常启动不会走到这里，属于防御性检查。
        raise api_error(
            503,
            "auth_not_configured",
            "后端没有加载访问令牌，认证功能不可用。",
        )

    principal = authenticate(token_map, authorization)
    if principal is None:
        # 401 而不是 403：客户端还没证明自己是谁，应该去拿凭证重试。
        # 带上 WWW-Authenticate 才是合规的 Bearer 认证响应。
        raise HTTPException(
            status_code=401,
            detail={
                "code": "unauthorized",
                "message": "缺少或错误访问令牌，请在请求头带上 Authorization: Bearer <token>。",
            },
            headers={"WWW-Authenticate": "Bearer"},
        )
    return principal


def ensure_owner(principal: Principal, user_id: str) -> None:
    """确认请求里的 user_id 就是令牌本人，否则拒绝。

    为什么不干脆忽略客户端传的 user_id：那会让“你试图读别人的数据”
    这件事悄无声息地发生。显式 403 能让越权尝试立刻暴露出来，
    排查时一眼就知道是身份对不上，而不是“数据怎么是空的”。
    """
    if principal.user_id != user_id:
        raise api_error(
            403,
            "user_mismatch",
            f"令牌对应的用户是 {principal.user_id}，与请求中的 user_id 不一致。",
        )


def translate_upstream_error(error: Exception) -> HTTPException:
    """把业务层抛出的异常翻译成带 code 的 HTTP 错误。"""
    if isinstance(error, OpenAIError):
        text = str(error)
        # 402 / Insufficient Balance 最常见：账户没钱了，但 API Key 仍然有效。
        if "402" in text or "Insufficient Balance" in text:
            return api_error(
                402,
                "upstream_balance",
                "DeepSeek 账户余额不足，请充值后重试；API Key 本身仍然有效。",
            )
        if "401" in text or "Unauthorized" in text:
            return api_error(
                401,
                "upstream_auth",
                "DeepSeek API Key 无效或已被吊销。",
            )
        return api_error(502, "upstream_error", "上游模型服务调用失败。")

    message = str(error)
    # 引用校验失败是“模型这次的输出不可信”，重试通常能恢复，要和真故障分开。
    if "引用真实性校验" in message or "拒答状态" in message:
        return api_error(
            500,
            "citation_failed",
            "本次回答未通过引用真实性校验，已拦截，没有返回给用户。重试一次通常即可。",
        )
    if isinstance(error, ValueError | RuntimeError):
        return api_error(500, "agent_error", "Agent 执行失败。")
    return api_error(500, "agent_error", "Agent 执行失败。")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """服务启动时加载昂贵资源，关闭时安全释放 SQLite 连接。"""
    setup_logging()  # 在接收第一个 HTTP 请求前建立 UTF-8 轮转日志。
    logger.info("service_starting")  # 记录服务生命周期开始，但不记录密钥配置。
    # 认证配置先于模型加载检查：配错令牌时立刻失败，
    # 不用白等二十秒把 Embedding 模型装进来才报错。
    app.state.auth_tokens = load_token_map()  # 没有配置时抛错，后端拒绝启动。
    logger.info("auth_tokens_loaded count=%s", len(app.state.auth_tokens))
    settings = load_settings()  # API Key 或模型配置缺失时直接阻止错误服务启动。
    knowledge_base = VectorKnowledgeBase.from_disk()  # Embedding 模型和索引只加载一次。
    client = OpenAI(  # 所有 HTTP 请求复用同一个 DeepSeek 客户端。
        api_key=settings.api_key,
        base_url=settings.base_url,
    )
    registry = ToolRegistry(knowledge_base, min_score=KNOWLEDGE_MIN_SCORE)  # 所有请求复用同一检索器。
    app.state.memory_reviews = MemoryReviews()  # 提取与确认是两次请求，必须共用审核表。

    with (
        open_memory_store() as memory_store,
        open_checkpointer() as checkpointer,
        open_chat_history() as chat_history,
    ):
        app.state.chat_history = chat_history  # 供历史会话接口读取同一份日志。
        app.state.memory_store = memory_store  # 供长期记忆接口读写同一份 Store。
        app.state.client = client  # 供候选记忆提取复用同一个模型客户端。
        app.state.model = settings.model  # 供 /info 报告当前模型。
        app.state.knowledge_base = knowledge_base  # 供 /info 报告知识库规模。
        app.state.started_at = perf_counter()  # 供 /info 报告运行时长。
        app.state.agent_service = AgentService(  # 将共享业务服务挂到 FastAPI 应用状态。
            client,
            settings.model,
            registry,
            checkpointer,
            memory_store,
            history=chat_history,
        )
        yield  # yield 期间 Uvicorn 才会真正接收和处理 HTTP 请求。
    logger.info("service_stopped")  # 所有数据库连接关闭后记录正常停止。


app = FastAPI(  # 创建 Uvicorn 将要加载的 ASGI 应用对象。
    title="Labor Law Knowledge Agent API",  # 显示在自动生成的 Swagger 文档中。
    description="基于 DeepSeek、LangGraph 和 RAG 的劳动法知识库 Agent。",
    version="0.4.0",  # 与 /health 和 /info 对外报告的版本保持一致。
    lifespan=lifespan,  # 注册共享资源的启动和关闭流程。
)


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    """为每个请求生成追踪 ID，并记录方法、路径、状态码和耗时。"""
    request_id = uuid4().hex  # 服务端生成 ID，不信任客户端传入内容进入日志。
    context_token = request_id_context.set(request_id)  # 当前异步调用链共享该 ID。
    started_at = perf_counter()  # 请求刚进入 FastAPI 时开始计时。
    status_code = 500  # 若发生未处理异常，日志默认按服务端错误记录。
    try:
        response = await call_next(request)  # 继续执行路由、依赖和业务层。
        status_code = response.status_code  # 取得最终 HTTP 状态码。
        response.headers["X-Request-ID"] = request_id  # 客户端可用此 ID 查询日志。
        return response
    finally:
        elapsed_seconds = perf_counter() - started_at  # 无论成功失败都记录耗时。
        logger.info(  # 不记录 URL 查询参数和请求体，避免意外写入敏感数据。
            "http_request method=%s path=%s status=%s elapsed_seconds=%.3f",
            request.method,
            request.url.path,
            status_code,
            elapsed_seconds,
        )
        request_id_context.reset(context_token)  # 请求结束后清除上下文，避免污染下一请求。


def get_agent_service(request: Request) -> AgentServiceProtocol:
    """从应用状态取得共享服务；测试可以覆盖这个依赖而不调用真实模型。"""
    service = getattr(request.app.state, "agent_service", None)
    if service is None:  # 服务尚未完成启动时返回明确的暂不可用状态。
        raise HTTPException(status_code=503, detail="Agent 服务尚未准备好。")
    return service


def get_chat_history(request: Request) -> ChatHistory:
    """从应用状态取得聊天日志；与业务服务共用同一个数据库连接。"""
    history = getattr(request.app.state, "chat_history", None)
    if history is None:
        raise HTTPException(status_code=503, detail="历史会话存储尚未准备好。")
    return history


def get_memory_store(request: Request):
    """从应用状态取得长期记忆 Store。"""
    store = getattr(request.app.state, "memory_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="长期记忆存储尚未准备好。")
    return store


def get_memory_reviews(request: Request) -> MemoryReviews:
    """提取与保存共用审核记录；未完成启动时不允许退化为直接写入。"""
    reviews = getattr(request.app.state, "memory_reviews", None)
    if reviews is None:
        raise api_error(503, "memory_review_unavailable", "候选审核服务尚未准备好。")
    return reviews


def get_memory_service(
    store: Annotated[Any, Depends(get_memory_store)],
    reviews: Annotated[MemoryReviews, Depends(get_memory_reviews)],
    request: Request,
) -> MemoryService:
    """把 Store 和模型客户端组装成记忆服务。

    刻意依赖 ``get_memory_store`` 而不是直接读 ``app.state``：测试覆盖
    这个依赖就能换成内存数据库，不用启动真实后端。MemoryService 本身
    很轻（只保存三个引用），每个请求新建一次不会带来额外开销。

    模型客户端用 ``getattr`` 取而不是写进依赖：读写记忆不需要模型，
    只有「提取候选」需要。取不到时传 None，让该接口自己报错，
    不至于把整个 /memories 一起拖成 503。
    """
    return MemoryService(
        getattr(request.app.state, "client", None),
        getattr(request.app.state, "model", ""),
        store,
        reviews=reviews,
    )


UserIdQuery = Annotated[  # 历史会话按用户隔离，参数校验规则与 /chat 保持一致。
    str,
    Query(min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_-]+$"),
]


def encode_sse_event(event: str, data: dict[str, object]) -> str:
    """把事件名称和字典编码成符合 Server-Sent Events 格式的文本。"""
    json_data = json.dumps(  # SSE 的 data 行仍使用 JSON 表达结构化内容。
        data,
        ensure_ascii=False,  # 中文直接输出，便于调试和前端展示。
    )
    return f"event: {event}\ndata: {json_data}\n\n"  # 空行表示一个 SSE 事件结束。


@app.get(  # 注册一个只读的 HTTP GET 路由。
    "/health",  # 客户端访问 http://127.0.0.1:8000/health。
    response_model=HealthResponse,  # FastAPI 会校验并生成响应文档。
    tags=["system"],  # 在 Swagger 页面中归入 system 分组。
)
def health_check() -> HealthResponse:
    """返回进程级健康状态；这个接口不会调用模型或数据库。"""
    return HealthResponse(  # 显式创建 Pydantic 对象，避免响应字段随意变化。
        status="ok",
        service="labor-law-knowledge-agent",
        version="0.4.0",
    )


@app.get(  # 前端启动时先问这里，不用把模型名和阈值硬编码在界面里。
    "/info",
    response_model=ServiceInfo,
    tags=["system"],
)
def service_info(
    request: Request,
    principal: Annotated[Principal, Depends(require_principal)],
) -> ServiceInfo:
    """返回当前后端实际生效的配置，供前端显示连接状态。

    需要令牌：它暴露了模型名称和知识库规模，没有理由给匿名访问者看。
    """
    knowledge_base = getattr(request.app.state, "knowledge_base", None)
    chunk_count = len(getattr(knowledge_base, "chunks", []) or [])
    return ServiceInfo(
        service="labor-law-knowledge-agent",
        version="0.4.0",
        model=getattr(request.app.state, "model", "unknown"),
        knowledge_chunks=chunk_count,
        # 从 LAW_TITLES 生成，加新法律后这里自动跟着变，不用手工维护。
        knowledge_source=", ".join(f"knowledge/{name}" for name in LAW_TITLES),
        retrieval={
            "min_score": KNOWLEDGE_MIN_SCORE,
            "top_k": KNOWLEDGE_TOP_K,
            "keyword_coverage": KNOWLEDGE_KEYWORD_COVERAGE,
        },
        capabilities=[
            "/health",
            "/me",
            "/info",
            "/chat",
            "/chat/stream",
            "/threads",
            "/memories",
        ],
    )


@app.get(  # 前端靠它回答“我现在是谁”，不再自己编一个随机身份。
    "/me",
    response_model=WhoAmI,
    tags=["auth"],
)
def who_am_i(
    principal: Annotated[Principal, Depends(require_principal)],
) -> WhoAmI:
    """返回令牌对应的用户身份。"""
    return WhoAmI(user_id=principal.user_id, token_label=principal.token_label)


@app.get(  # 历史会话只读接口，供前端渲染侧边栏列表。
    "/threads",
    response_model=list[ThreadSummary],
    tags=["history"],
)
def list_threads(
    # 认证声明在业务依赖之前：未认证的请求应停在 401，
    # 而不是先探到「历史存储未就绪」的 503，白暴露一层内部状态。
    principal: Annotated[Principal, Depends(require_principal)],
    user_id: UserIdQuery,
    history: Annotated[ChatHistory, Depends(get_chat_history)],
) -> list[ThreadSummary]:
    """列出该用户最近的会话，按最近一次对话时间倒序。"""
    ensure_owner(principal, user_id)
    return history.list_threads(user_id)


@app.get(  # 读取单个会话的全部消息，用于刷新页面后恢复聊天记录。
    "/threads/{thread_id}/messages",
    response_model=list[ChatMessage],
    tags=["history"],
)
def list_thread_messages(
    principal: Annotated[Principal, Depends(require_principal)],
    thread_id: str,
    user_id: UserIdQuery,
    history: Annotated[ChatHistory, Depends(get_chat_history)],
) -> list[ChatMessage]:
    """按时间顺序返回一个会话的消息；会话不存在时返回空列表。"""
    ensure_owner(principal, user_id)
    return history.list_messages(user_id, thread_id)


@app.delete(  # 同时清理会话检查点与聊天记录，用户长期记忆不受影响。
    "/threads/{thread_id}",
    response_model=DeleteThreadResponse,
    tags=["history"],
)
def delete_thread(
    principal: Annotated[Principal, Depends(require_principal)],
    thread_id: str,
    user_id: UserIdQuery,
    service: Annotated[AgentServiceProtocol, Depends(get_agent_service)],
) -> DeleteThreadResponse:
    """删除指定用户的会话状态与消息，返回删除的界面消息条数。"""
    ensure_owner(principal, user_id)
    deleted = service.delete_thread(user_id, thread_id)
    return DeleteThreadResponse(thread_id=thread_id, deleted=deleted)


@app.get(  # 长期记忆按 user_id 隔离，与会话无关。
    "/memories",
    response_model=list[MemoryItem],
    tags=["memory"],
)
def get_memories(
    principal: Annotated[Principal, Depends(require_principal)],
    user_id: UserIdQuery,
    service: Annotated[MemoryService, Depends(get_memory_service)],
) -> list[MemoryItem]:
    """列出该用户的全部长期记忆，按 key 排序。"""
    ensure_owner(principal, user_id)
    memories = service.list(user_id)
    return [
        MemoryItem(
            memory_key=key,
            category=record.category,
            content=record.content,
            updated_at=record.updated_at,
        )
        for key, record in sorted(memories.items())
    ]


@app.post(  # 写入前必须经过界面上的「提取 → 门控 → 人工确认」三步。
    "/memories",
    response_model=MemorySaveResponse,
    tags=["memory"],
)
def create_memory(
    payload: MemorySaveRequest,
    principal: Annotated[Principal, Depends(require_principal)],
    service: Annotated[MemoryService, Depends(get_memory_service)],
) -> MemorySaveResponse:
    """保存一条已经人工确认的长期记忆；同一件事只保留一条。"""
    ensure_owner(principal, payload.user_id)
    try:
        outcome = service.save(
            payload.user_id,
            payload.memory_key,
            payload.category,
            payload.content,
            review_id=payload.review_id,
            confirmed=payload.confirmed,
        )
    except MemoryReviewError as error:
        raise api_error(400, "memory_review_rejected", str(error)) from error
    saved = service.list(payload.user_id).get(outcome.memory_key)
    return MemorySaveResponse(
        memory_key=outcome.memory_key,
        # 合并时最终内容可能来自旧记忆，所以以库里实际存的为准，不回声请求体。
        category=saved.category if saved else payload.category,
        content=saved.content if saved else payload.content,
        updated_at=saved.updated_at if saved else "",
        action=outcome.action,
        duplicate_of=outcome.duplicate_of,
        removed_key=outcome.removed_key,
        reason=outcome.reason,
    )


@app.post(  # 提取候选记忆需要调用模型，所以放在后端而不是前端。
    "/memories/extract",
    response_model=MemoryExtractResponse,
    tags=["memory"],
)
def extract_memory(
    payload: MemoryExtractRequest,
    principal: Annotated[Principal, Depends(require_principal)],
    service: Annotated[MemoryService, Depends(get_memory_service)],
) -> MemoryExtractResponse:
    """让模型提出候选记忆，再由本地规则决定是否允许保存。"""
    ensure_owner(principal, payload.user_id)
    try:
        proposal = service.propose(payload.user_id, payload.text)
    except OpenAIError as error:
        raise translate_upstream_error(error) from error
    except RuntimeError as error:  # 没有模型客户端时说明服务尚未装配完整。
        raise api_error(503, "model_unavailable", str(error)) from error
    except ValidationError as error:
        raise api_error(502, "extract_failed", "模型返回的记忆候选格式不合法。") from error
    except MemoryReviewError as error:
        raise api_error(400, "memory_review_rejected", str(error)) from error
    return MemoryExtractResponse(
        candidate=proposal.candidate,
        allowed=proposal.allowed,
        reason=proposal.reason,
        duplicate_of=proposal.duplicate_of,
        review_id=proposal.review_id,
    )


@app.post(  # POST 适合携带问题并产生一次新的 Agent 执行结果。
    "/chat",
    response_model=AgentTurnResult,  # 固定答案、拒答、引用和轮数四个响应字段。
    tags=["agent"],  # 在 Swagger 页面中归入 agent 分组。
)
def chat(
    chat_request: ChatRequest,  # FastAPI 自动把请求 JSON 解析并校验成 Pydantic 对象。
    # 认证声明在业务依赖之前：没通过认证的请求应该在 401 停下，
    # 而不是先探到「服务未就绪」再返回 503，白暴露一层内部状态。
    principal: Annotated[Principal, Depends(require_principal)],
    service: Annotated[AgentServiceProtocol, Depends(get_agent_service)],  # 注入共享业务层。
) -> AgentTurnResult:
    """运行一次带短期状态、长期记忆、RAG 和引用校验的 Agent 回合。"""
    ensure_owner(principal, chat_request.user_id)
    try:
        return service.answer(  # HTTP 层只转交参数，不包含 Agent 编排细节。
            chat_request.user_id,
            chat_request.thread_id,
            chat_request.question,
        )
    except (OpenAIError, RuntimeError, ValueError, ValidationError) as error:
        raise translate_upstream_error(error) from error


@app.post(  # 使用独立路径，保留原有一次性 JSON 接口供简单客户端使用。
    "/chat/stream",
    tags=["agent"],
    response_class=StreamingResponse,
)
async def chat_stream(
    chat_request: ChatRequest,  # 请求结构与普通 /chat 完全相同。
    principal: Annotated[Principal, Depends(require_principal)],
    service: Annotated[AgentServiceProtocol, Depends(get_agent_service)],
) -> StreamingResponse:
    """通过 SSE 推送阶段进度，最后推送已通过校验的完整结果。

    事件顺序固定为：status(accepted) → status(thinking) → status(retrieving)
    → status(retrieved) → status(generating) → status(validating) → result。
    任何一步失败都替换成 error 事件；流已经开始后无法再改 HTTP 状态码。
    """
    ensure_owner(principal, chat_request.user_id)

    async def generate_events() -> AsyncIterator[str]:
        """按发生顺序生成客户端可以逐个接收的 SSE 文本块。"""
        yield encode_sse_event(  # 不等待模型，先让前端知道请求已经进入处理阶段。
            "status",
            {"stage": "accepted", "message": "Agent 已开始处理问题。"},
        )
        try:
            # 业务层的 answer_stream 是同步生成器，放进线程池逐步拉取，
            # 避免一次模型调用把整个事件循环卡住。
            async for event in iterate_in_threadpool(
                service.answer_stream(
                    chat_request.user_id,
                    chat_request.thread_id,
                    chat_request.question,
                )
            ):
                if event["type"] == "status":
                    yield encode_sse_event(
                        "status",
                        {key: value for key, value in event.items() if key != "type"},
                    )
                elif event["type"] == "result":
                    yield encode_sse_event("result", event["data"])
        except (OpenAIError, RuntimeError, ValueError, ValidationError) as error:
            http_error = translate_upstream_error(error)
            detail = (
                http_error.detail
                if isinstance(http_error.detail, dict)
                else {"code": "agent_error", "message": str(http_error.detail)}
            )
            yield encode_sse_event("error", detail)

    return StreamingResponse(  # FastAPI 不会等待生成器全部结束才开始响应。
        generate_events(),
        media_type="text/event-stream",  # 告诉客户端按 SSE 协议解释响应体。
        headers={
            "Cache-Control": "no-cache",  # 中间缓存不得保存旧的流式事件。
            "X-Accel-Buffering": "no",  # 告诉兼容代理不要攒齐内容后再一次性转发。
        },
    )
