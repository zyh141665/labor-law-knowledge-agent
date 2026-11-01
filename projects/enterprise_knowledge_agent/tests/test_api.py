"""从 HTTP 客户端视角测试健康检查和 Agent 对话接口。"""

import sqlite3  # 历史会话与长期记忆都用内存数据库测试，不触碰真实文件。

from collections.abc import Iterator
from typing import Any

from fastapi.testclient import TestClient  # 不启动真实端口即可模拟 HTTP 请求。
from langgraph.store.sqlite import SqliteStore  # 长期记忆接口用真实 Store 验证读写。
from langgraph.checkpoint.sqlite import SqliteSaver

from projects.enterprise_knowledge_agent.app.api import (
    app,
    get_agent_service,
    get_chat_history,
    get_memory_store,
)
from projects.enterprise_knowledge_agent.app.chat_history import ChatHistory
from projects.enterprise_knowledge_agent.app.schemas import Citation
from projects.enterprise_knowledge_agent.app.service import AgentService, AgentTurnResult
from projects.enterprise_knowledge_agent.app.memory_policy import MemoryCandidate


class FakeAgentService:
    """返回固定结果并记录参数，确保 HTTP 测试不会调用 DeepSeek。"""

    def __init__(self) -> None:
        self.last_request: tuple[str, str, str] | None = None

    def answer(
        self,
        user_id: str,
        thread_id: str,
        question: str,
    ) -> AgentTurnResult:
        self.last_request = (user_id, thread_id, question)
        return AgentTurnResult(
            answer="退款申请必须在购买后 7 天内提交。",
            refused=False,
            citations=[
                Citation(
                    source_id="company_policy.txt#chunk-1",
                    quote="退款申请必须在购买后 7 天内提交。",
                )
            ],
            round_number=2,
        )

    def answer_stream(
        self,
        user_id: str,
        thread_id: str,
        question: str,
    ) -> Iterator[dict[str, Any]]:
        """模拟业务层的阶段事件序列，最后给出完整结果。"""
        yield {"type": "status", "stage": "thinking", "message": "正在理解问题"}
        yield {"type": "status", "stage": "retrieving", "message": "正在检索条文"}
        yield {"type": "status", "stage": "validating", "message": "正在校验引用"}
        yield {"type": "result", "data": self.answer(user_id, thread_id, question).model_dump()}


client = TestClient(app)  # 不进入 lifespan，测试通过依赖覆盖注入假服务。


def test_health_endpoint() -> None:
    """健康检查应返回 200 状态码和稳定的结构化内容。"""
    response = client.get("/health")

    assert response.status_code == 200
    assert len(response.headers["X-Request-ID"]) == 32  # 客户端可以用该 ID 关联服务端日志。
    assert response.json() == {
        "status": "ok",
        "service": "labor-law-knowledge-agent",
        "version": "0.4.0",
    }


def test_info_endpoint_reports_effective_config() -> None:
    """前端启动时靠 /info 显示连接状态，不硬编码模型名和阈值。"""
    response = client.get("/info")

    assert response.status_code == 200
    body = response.json()
    assert body["service"] == "labor-law-knowledge-agent"
    assert body["retrieval"]["top_k"] == 4  # 与 config.py 的常量保持一致。
    assert "/chat/stream" in body["capabilities"]


def test_chat_endpoint_validates_and_forwards_request() -> None:
    """合法 JSON 应转交服务层，并返回结构化 Agent 结果。"""
    fake_service = FakeAgentService()
    app.dependency_overrides[get_agent_service] = lambda: fake_service
    try:
        response = client.post(
            "/chat",
            json={
                "user_id": "student-001",
                "thread_id": "refund-chat",
                "question": "退款期限是多少？",
            },
        )
    finally:
        app.dependency_overrides.clear()  # 防止依赖覆盖污染其他测试。

    assert response.status_code == 200
    assert fake_service.last_request == (
        "student-001",
        "refund-chat",
        "退款期限是多少？",
    )
    assert response.json()["citations"][0]["source_id"].endswith("chunk-1")


def test_chat_endpoint_rejects_empty_question() -> None:
    """空问题应在进入服务层之前由 FastAPI/Pydantic 返回 422。"""
    fake_service = FakeAgentService()  # 先满足依赖，测试才能只观察请求体校验。
    app.dependency_overrides[get_agent_service] = lambda: fake_service
    try:
        response = client.post(
            "/chat",
            json={
                "user_id": "student-001",
                "thread_id": "refund-chat",
                "question": "",
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 422
    assert fake_service.last_request is None  # 非法请求不能进入 Agent 业务层。


def test_chat_stream_sends_status_before_result() -> None:
    """SSE 接口应先报告开始，再发送最终结构化结果。"""
    fake_service = FakeAgentService()
    app.dependency_overrides[get_agent_service] = lambda: fake_service
    try:
        with client.stream(
            "POST",
            "/chat/stream",
            json={
                "user_id": "student-001",
                "thread_id": "refund-chat",
                "question": "退款期限是多少？",
            },
        ) as response:
            response_text = "".join(response.iter_text())
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response_text.index("event: status") < response_text.index("event: result")
    assert "company_policy.txt#chunk-1" in response_text


def build_history() -> ChatHistory:
    """构造一个只存在内存里的聊天日志，供接口测试注入。

    同步路由会被 FastAPI 放进线程池执行，因此连接必须允许跨线程使用，
    这一点与生产环境的 ``open_chat_history`` 保持一致。
    """
    connection = sqlite3.connect(":memory:", check_same_thread=False)
    history = ChatHistory(connection)
    history.setup()
    return history


def test_threads_endpoint_lists_recent_conversations() -> None:
    """历史会话接口应返回标题、消息条数和会话标识。"""
    history = build_history()
    history.record_turn(
        "student-001",
        "refund-chat",
        "退款期限是多少？",
        answer="购买后 7 天内提交。",
        refused=False,
        citations=[],
    )
    app.dependency_overrides[get_chat_history] = lambda: history
    try:
        response = client.get("/threads", params={"user_id": "student-001"})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body[0]["thread_id"] == "refund-chat"
    assert body[0]["title"] == "退款期限是多少？"
    assert body[0]["message_count"] == 2


def test_thread_messages_endpoint_returns_ordered_messages() -> None:
    """刷新页面后要能按原顺序恢复一问一答。"""
    history = build_history()
    history.record_turn(
        "student-001",
        "refund-chat",
        "退款期限是多少？",
        answer="购买后 7 天内提交。",
        refused=False,
        citations=[{"source_id": "劳动法#第四十四条", "quote": "百分之三百"}],
    )
    app.dependency_overrides[get_chat_history] = lambda: history
    try:
        response = client.get(
            "/threads/refund-chat/messages",
            params={"user_id": "student-001"},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    messages = response.json()
    assert [message["role"] for message in messages] == ["user", "assistant"]
    assert messages[1]["citations"][0]["source_id"] == "劳动法#第四十四条"


def test_threads_endpoint_rejects_invalid_user_id() -> None:
    """用户 ID 含空格等非法字符时应由参数校验拦下，而不是拼进查询。"""
    history = build_history()
    app.dependency_overrides[get_chat_history] = lambda: history
    try:
        response = client.get("/threads", params={"user_id": "student 001"})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 422


def test_delete_thread_removes_all_messages() -> None:
    """删除会话走业务服务，不再仅清理界面历史。"""
    history = build_history()
    history.record_turn(
        "student-001",
        "refund-chat",
        "退款期限是多少？",
        answer="购买后 7 天内提交。",
        refused=False,
        citations=[],
    )
    with SqliteSaver.from_conn_string(":memory:") as saver:
        service = AgentService(None, "fake-model", None, saver, None, history)
        app.dependency_overrides[get_agent_service] = lambda: service
        try:
            response = client.delete("/threads/refund-chat", params={"user_id": "student-001"})
        finally:
            app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {"thread_id": "refund-chat", "deleted": 2}
    assert history.list_messages("student-001", "refund-chat") == []


def build_memory_store() -> SqliteStore:
    """构造内存版长期记忆 Store，测试不会写进 data/memory.sqlite。"""
    connection = sqlite3.connect(":memory:", check_same_thread=False)
    store = SqliteStore(connection)
    store.setup()
    # setup() 里的 CREATE TABLE 会留下一个未提交事务，SqliteStore 之后每次
    # put/search 都会自己 BEGIN，残留事务会让它报 “within a transaction”。
    connection.commit()
    return store


def approved_memory_payload(reviews, key: str, category: str, content: str) -> dict:
    """在测试审核表中登记候选，模拟提取接口已通过规则的结果。"""
    candidate = MemoryCandidate(
        should_remember=True, memory_key=key, category=category, content=content,
        sensitive=False, reason="用户明确表达了偏好或目标。",
    )
    return {
        "user_id": "student-001", "memory_key": key, "category": category,
        "content": content, "review_id": reviews.issue("student-001", candidate, content),
        "confirmed": True,
    }


def test_memory_endpoints_roundtrip(review_records) -> None:
    """前端把「读记忆 / 存记忆」改成 HTTP 调用后，这两条路必须仍然通。"""
    store = build_memory_store()
    app.dependency_overrides[get_memory_store] = lambda: store
    try:
        created = client.post(
            "/memories",
            json=approved_memory_payload(
                review_records, "learning_goal", "goal", "学习 Agent 应用开发并寻找相关实习"
            ),
        )
        listed = client.get("/memories", params={"user_id": "student-001"})
        other_user = client.get("/memories", params={"user_id": "student-002"})
    finally:
        app.dependency_overrides.clear()

    assert created.status_code == 200
    assert created.json()["memory_key"] == "learning_goal"
    assert listed.status_code == 200
    assert len(listed.json()) == 1
    assert listed.json()[0]["content"].startswith("学习 Agent")
    # 认证之后「换个 user_id 就看不到别人」这件事更严格了：
    # 不是返回空列表，而是直接 403 —— 越权尝试应该立刻暴露，而不是静默变空。
    assert other_user.status_code == 403
    assert other_user.json()["detail"]["code"] == "user_mismatch"


def test_memory_endpoint_rejects_bad_category() -> None:
    """类别必须在白名单内，否则拒绝写入。"""
    store = build_memory_store()
    app.dependency_overrides[get_memory_store] = lambda: store
    try:
        response = client.post(
            "/memories",
            json={
                "user_id": "student-001",
                "memory_key": "some_key",
                "category": "随便写的类别",
                "content": "内容",
                "review_id": "invalid-review",
                "confirmed": True,
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 422


def test_chat_stream_emits_error_event_when_validation_fails() -> None:
    """引用校验失败时，流已经开始，只能发 error 事件而不是改状态码。"""

    class FailingService(FakeAgentService):
        def answer_stream(
            self,
            user_id: str,
            thread_id: str,
            question: str,
        ) -> Iterator[dict[str, Any]]:
            raise ValueError("模型回答未通过拒答状态或引用真实性校验。")
            yield  # pragma: no cover - 保持生成器语义。

    app.dependency_overrides[get_agent_service] = lambda: FailingService()
    try:
        with client.stream(
            "POST",
            "/chat/stream",
            json={
                "user_id": "student-001",
                "thread_id": "refund-chat",
                "question": "退款期限是多少？",
            },
        ) as response:
            response_text = "".join(response.iter_text())
    finally:
        app.dependency_overrides.clear()

    assert "event: error" in response_text
    assert "citation_failed" in response_text  # 前端据此显示「被安全机制拦下」。


def test_memory_endpoint_reports_merge_instead_of_duplicating(review_records) -> None:
    """同一件事用不同 key 存两次，第二次必须合并，库里只能剩一条。"""
    store = build_memory_store()
    app.dependency_overrides[get_memory_store] = lambda: store
    try:
        first = client.post(
            "/memories",
            json=approved_memory_payload(
                review_records, "answer_style", "preference", "回答尽量简洁，并优先解释结论"
            ),
        )
        second = client.post(
            "/memories",
            json=approved_memory_payload(
                review_records, "answer_style_conclusion_first", "preference",
                "用户偏好回答时先给结论，再解释原因。",
            ),
        )
        listed = client.get("/memories", params={"user_id": "student-001"})
    finally:
        app.dependency_overrides.clear()

    assert first.json()["action"] == "insert"
    assert second.json()["action"] == "merge"
    assert second.json()["memory_key"] == "answer_style"  # 保留更短的 key。
    assert second.json()["duplicate_of"] == "answer_style"
    assert len(listed.json()) == 1
    assert listed.json()[0]["content"].startswith("用户偏好回答时先给结论")


def test_memory_extract_requires_user_id() -> None:
    """判重要和该用户已有记忆比对，缺了身份就该直接拒绝。"""
    store = build_memory_store()
    app.dependency_overrides[get_memory_store] = lambda: store
    try:
        response = client.post("/memories/extract", json={"text": "我更喜欢简洁的回答"})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 422
