"""劳动法知识库 Agent 的 Streamlit 前端。

本文件只做界面：所有后端能力都通过 ``AgentApiClient`` 走 HTTP 调用。
它不 import 任何 ``app.*`` 业务模块，也不加载 Embedding 模型、不打开数据库，
因此启动只要一两秒；后端没起时前端只是连不上，不会崩在 import 阶段。

启动方式（需要后端已经跑起来）：

    streamlit run projects/enterprise_knowledge_agent/streamlit_app.py

配套的后端地址由环境变量 ``AGENT_API_BASE_URL`` 指定，默认 127.0.0.1:8000。
"""

import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4

# 把项目根目录加入模块搜索路径，确保 ``from app.api_client import ...`` 可以解析。
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from app.api_client import AgentApiClient, AgentApiError


# ---------------------------------------------------------------------------
# 后端客户端：整个 Streamlit 进程共用一份，保留 httpx 连接池。
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="正在连接后端…")
def get_client() -> AgentApiClient:
    return AgentApiClient()


client = get_client()


# ---------------------------------------------------------------------------
# 身份：由后端根据访问令牌推导，前端不再自己编随机 ID。
#
# 以前这里是 uuid4() 生成一个随机用户名写进 URL，谁都能改 ?user=xxx
# 冒充别人。现在令牌就是身份：后端从 Authorization 头认出你是谁，
# 请求里带别的 user_id 会被直接 403。
# ---------------------------------------------------------------------------


def load_identity() -> dict:
    """向后端问一次“我是谁”，结果缓存在本次浏览器会话里。

    只查一次而不是每次调用都问，是因为 Streamlit 每次交互都会重跑脚本，
    不缓存就会每按一次按钮多打一次 HTTP 请求。
    """
    if "identity" not in st.session_state:
        st.session_state["identity"] = client.me()
    return st.session_state["identity"]


def current_user_id() -> str:
    return load_identity()["user_id"]


def current_thread_id() -> str:
    if "thread_id" not in st.session_state:
        existing = st.query_params.get("thread")
        thread_id = existing or f"thread-{uuid4().hex[:8]}"
        st.session_state["thread_id"] = thread_id
        st.query_params["thread"] = thread_id
    return st.session_state["thread_id"]


def switch_thread(target: str) -> None:
    """切到另一个会话；清掉已加载标记，让主区域重新向后端拉消息。"""
    st.session_state["thread_id"] = target
    st.session_state["messages"] = []
    st.session_state["loaded_thread"] = None
    st.query_params["thread"] = target


def start_new_thread() -> None:
    switch_thread(f"thread-{uuid4().hex[:8]}")


def remove_thread(uid: str, tid: str) -> None:
    """删除一个会话的聊天记录。

    清掉 loaded_thread 是为了让主区域在回调后的自动重跑里重新向后端拉一次，
    否则删掉的会话还会留在屏幕上。
    """
    try:
        client.delete_thread(uid, tid)
    except AgentApiError as error:
        st.error(describe_error(error))  # 删除失败要明确提示，不能伪装成已删除。
        return
    if st.session_state.get("thread_id") == tid:
        start_new_thread()
    else:
        st.session_state["loaded_thread"] = None


def describe_error(error: AgentApiError) -> str:
    """把后端的错误码翻译成用户能照着做的提示。"""
    if error.is_connection_error:
        return (
            f"无法连接后端（{client.base_url}）。请先在另一个终端运行：\n\n"
            "`python -m projects.enterprise_knowledge_agent.scripts.run_backend`\n\n"
            "后端首次启动要加载模型，看到 `Uvicorn running` 后再点下面的重试。"
        )
    if error.is_unauthorized:
        return (
            "后端拒绝了本次请求（HTTP 401）：访问令牌缺失或不匹配。\n\n"
            "检查两处是否一致——后端的 `AGENT_API_TOKENS`，"
            "和前端进程的 `AGENT_API_TOKEN`（前者是 `token:用户` 的列表，"
            "后者取其中一个 token 原样填入）。改完两个进程都要重启。"
        )
    if error.is_user_mismatch:
        return (
            "身份对不上（HTTP 403）：令牌属于另一个用户，"
            "但请求里带的是别的 user_id。刷新页面重新获取身份再试。"
        )
    if error.is_balance_error:
        return (
            "DeepSeek 账户余额不足（HTTP 402），请前往 platform.deepseek.com 充值后重试；"
            "API Key 本身仍然有效，不需要重新配置。"
        )
    if error.is_auth_error:
        return f"DeepSeek API Key 无效或已失效（HTTP 401）：{error.message}"
    if error.is_citation_error:
        return (
            "本次回答未通过引用真实性校验（模型可能引用了检索结果里不存在的条文），"
            "已被拦截，没有展示给你。重试一次通常即可。"
        )
    return f"Agent 执行失败：{error.message}"


def format_timestamp(raw: str) -> str:
    """把 ISO 时间显示成“今天 18:02”这类更短的相对格式。"""
    if not raw:
        return ""
    try:
        moment = datetime.fromisoformat(raw).astimezone()
    except ValueError:
        return raw
    if moment.date() == datetime.now().astimezone().date():
        return f"今天 {moment:%H:%M}"
    return f"{moment:%m-%d %H:%M}"


def describe_save_action(action: str, memory_key: str) -> str:
    """把后端的写入动作翻译成人话，避免“跳过”也显示成“已保存”。"""
    if action == "merge":
        return f"已合并到已有记忆 `{memory_key}`，没有产生重复条目"
    if action == "skip":
        return f"`{memory_key}` 已经记过同样的内容，本次未重复写入"
    if action == "update":
        return f"已更新记忆 `{memory_key}`"
    return "已保存"


def render_citations(citations: list[dict], refused: bool) -> None:
    """把引用折叠展示；拒答时说明这些条文「查过但不足以支撑结论」。"""
    if not citations:
        return
    title = (
        f"检索到但不足以支撑回答的条文（{len(citations)} 条）"
        if refused
        else f"引用来源（{len(citations)} 条）"
    )
    with st.expander(title):
        for citation in citations:
            st.markdown(f"**{citation['source_id']}**")
            st.markdown(f"> {citation['quote']}")


def render_answer(answer: dict) -> None:
    """渲染一次已通过校验的 Agent 结果。"""
    if answer.get("refused"):
        st.warning("证据不足，Agent 拒绝回答")
    st.markdown(answer["answer"])
    render_citations(answer.get("citations") or [], bool(answer.get("refused")))


# ---------------------------------------------------------------------------
# 连接状态：先问后端要 /info，连不上就只显示提示，不渲染其余界面。
# ---------------------------------------------------------------------------

try:
    service_info = client.info()
    # 身份也放在同一个 try 里：令牌不对时两个请求都会 401，
    # 统一在这里拦下，避免下半页代码拿着空身份继续请求。
    identity = load_identity()
except AgentApiError as error:
    st.title("劳动法知识库 RAG Agent")
    st.error(describe_error(error))
    if st.button("重试连接", type="primary"):
        st.cache_resource.clear()
        st.session_state.pop("identity", None)
        st.rerun()
    st.stop()

user_id = identity["user_id"]
thread_id = current_thread_id()

if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "loaded_thread" not in st.session_state:
    st.session_state["loaded_thread"] = None


# ---------------------------------------------------------------------------
# 侧边栏：连接状态、会话切换、历史会话、长期记忆。
# ---------------------------------------------------------------------------

with st.sidebar:
    st.caption(
        f"已连接 · {service_info['model']} · "
        f"{service_info['knowledge_chunks']} 条法条 · v{service_info['version']}"
    )
    st.caption(f"你的身份：`{user_id}` · 令牌 {identity['token_label']}")

    st.button("新建会话", use_container_width=True, on_click=start_new_thread)

    st.divider()
    st.subheader("历史会话")
    try:
        thread_summaries = client.list_threads(user_id)
    except AgentApiError as error:
        st.caption(describe_error(error))
        thread_summaries = []

    if not thread_summaries:
        st.caption("还没有历史会话，问一个问题就会出现在这里")
    for summary in thread_summaries:
        is_current = summary["thread_id"] == thread_id
        open_column, delete_column = st.columns([0.84, 0.16], gap="small")
        open_column.button(
            f"{'▶ ' if is_current else ''}{summary['title']}",
            key=f"open-{summary['thread_id']}",
            use_container_width=True,
            on_click=switch_thread,
            args=(summary["thread_id"],),
            help=f"会话 ID：{summary['thread_id']}",
        )
        delete_column.button(
            "🗑",
            key=f"delete-{summary['thread_id']}",
            on_click=remove_thread,
            args=(user_id, summary["thread_id"]),
            help="删除这条会话",
        )
        caption = f"{summary['message_count']} 条消息"
        if summary["updated_at"]:
            caption = f"{caption} · {format_timestamp(summary['updated_at'])}"
        st.caption(caption)

    st.divider()
    st.subheader("长期记忆")
    try:
        memories = client.list_memories(user_id)
    except AgentApiError as error:
        memories = []
        st.caption(describe_error(error))
    if memories:
        for item in memories:
            st.markdown(f"**{item['category']}** · `{item['memory_key']}`")
            st.markdown(f"- {item['content']}")
    else:
        st.caption("暂无长期记忆")

    st.divider()
    st.subheader("记住一句话")
    memory_text = st.text_area("输入要长期记住的内容", key="memory_input")
    if st.button("提取候选记忆", use_container_width=True):
        if not memory_text.strip():
            st.warning("请先输入内容")
        else:
            try:
                st.session_state["memory_candidate"] = client.extract_memory(
                    user_id, memory_text
                )
            except AgentApiError as error:
                st.error(describe_error(error))

    candidate_payload = st.session_state.get("memory_candidate")
    if candidate_payload:
        candidate = candidate_payload["candidate"]
        st.markdown(f"**候选记忆**：{candidate['content']}")
        st.caption(
            f"类别：{candidate['category']} · key：{candidate['memory_key']} "
            f"· 门控：{candidate_payload['reason']}"
        )
        if candidate_payload["allowed"]:
            duplicate_of = candidate_payload.get("duplicate_of")
            if duplicate_of:
                st.info(f"这条与已有记忆 `{duplicate_of}` 是同一件事，保存时会合并。")
            if st.button("确认保存到长期记忆", use_container_width=True):
                try:
                    saved = client.save_memory(
                        user_id,
                        candidate["memory_key"],
                        candidate["category"],
                        candidate["content"],
                        review_id=candidate_payload["review_id"],  # 原样带回后端审核编号。
                        confirmed=True,  # 只在用户点击确认按钮后提交。
                    )
                    # 写入结果可能是新增、覆盖、合并或跳过，提示语要如实反映。
                    action = saved.get("action", "insert")
                    st.success(describe_save_action(action, saved.get("memory_key", "")))
                    del st.session_state["memory_candidate"]
                    st.rerun()
                except AgentApiError as error:
                    st.error(describe_error(error))


# ---------------------------------------------------------------------------
# 主区域：标题、历史消息、输入框。
# ---------------------------------------------------------------------------

st.title("劳动法知识库 RAG Agent")
st.caption(
    "DeepSeek + LangGraph + 向量/BM25 混合检索 + 引用逐字校验。"
    f"检索参数 min_score={service_info['retrieval']['min_score']}、"
    f"top_k={service_info['retrieval']['top_k']}。"
)

# 切换会话后向后端拉一次消息；切回来时不必重复请求。
if st.session_state["loaded_thread"] != thread_id:
    try:
        st.session_state["messages"] = client.list_messages(user_id, thread_id)
    except AgentApiError as error:
        st.session_state["messages"] = []
        st.caption(describe_error(error))
    st.session_state["loaded_thread"] = thread_id

for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_citations(
                message.get("citations") or [],
                bool(message.get("refused")),
            )

if prompt := st.chat_input("输入劳动法相关问题，例如：法定节假日加班工资怎么算？"):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # 阶段提示占位符：后端每推一个 status 事件就换一行文字，
        # 最终结果到达时清掉它，再渲染答案。
        status_placeholder = st.empty()
        answer_payload: dict | None = None
        try:
            for event_name, event in client.chat_stream(user_id, thread_id, prompt):
                if event_name == "status":
                    status_placeholder.caption(f"⏳ {event.get('message', '处理中')}…")
                elif event_name == "result":
                    status_placeholder.empty()
                    answer_payload = event
                elif event_name == "error":
                    status_placeholder.empty()
                    raise AgentApiError(
                        str(event.get("code", "agent_error")),
                        str(event.get("message", "Agent 执行失败。")),
                    )
        except AgentApiError as error:
            status_placeholder.empty()
            st.error(describe_error(error))

        if answer_payload is not None:
            render_answer(answer_payload)
            st.session_state["messages"].append(
                {
                    "role": "assistant",
                    "content": answer_payload["answer"],
                    "refused": answer_payload.get("refused", False),
                    "citations": answer_payload.get("citations") or [],
                    "created_at": "",
                }
            )
            # 侧边栏在本轮之前已经渲染过，重跑一次才能让新会话出现在历史列表里。
            st.rerun()
