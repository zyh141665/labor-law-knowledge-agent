"""测试前端 HTTP 客户端：错误码翻译、SSE 解析、请求参数。"""

import json

import httpx
import pytest

from projects.enterprise_knowledge_agent.app.api_client import (
    AgentApiClient,
    AgentApiError,
    _iter_sse_events,
)


def build_client(handler) -> AgentApiClient:
    """用 MockTransport 替换真实网络，测试不会碰 127.0.0.1:8000。"""
    client = AgentApiClient(base_url="http://backend.test")
    client._client.close()
    client._client = httpx.Client(
        base_url="http://backend.test",
        transport=httpx.MockTransport(handler),
    )
    return client


def test_chat_returns_parsed_json() -> None:
    """普通问答应原样返回后端结构。"""
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/chat"
        body = json.loads(request.content)
        assert body == {
            "user_id": "u1",
            "thread_id": "t1",
            "question": "加班工资怎么算？",
        }
        return httpx.Response(
            200,
            json={
                "answer": "支付不低于百分之三百的工资",
                "refused": False,
                "citations": [{"source_id": "劳动法#第四十四条", "quote": "百分之三百"}],
                "round_number": 2,
            },
        )

    result = build_client(handler).chat("u1", "t1", "加班工资怎么算？")

    assert result["answer"].startswith("支付不低于")
    assert result["citations"][0]["source_id"] == "劳动法#第四十四条"


def test_balance_error_is_recognisable() -> None:
    """402 必须能被前端识别成「余额不足」，而不是笼统的 502。"""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            402,
            json={
                "detail": {
                    "code": "upstream_balance",
                    "message": "DeepSeek 账户余额不足",
                }
            },
        )

    with pytest.raises(AgentApiError) as excinfo:
        build_client(handler).chat("u1", "t1", "问题")

    error = excinfo.value
    assert error.is_balance_error
    assert not error.is_connection_error
    assert error.status_code == 402


def test_auth_error_is_recognisable() -> None:
    """401 要单独区分，提示用户换 Key，而不是让他去充值。"""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            json={"detail": {"code": "upstream_auth", "message": "Key 无效"}},
        )

    with pytest.raises(AgentApiError) as excinfo:
        build_client(handler).chat("u1", "t1", "问题")

    assert excinfo.value.is_auth_error
    assert not excinfo.value.is_balance_error


def test_citation_error_is_recognisable() -> None:
    """引用校验失败要能和普通故障分开，提示「重试一次」。"""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            500,
            json={
                "detail": {
                    "code": "citation_failed",
                    "message": "未通过引用真实性校验",
                }
            },
        )

    with pytest.raises(AgentApiError) as excinfo:
        build_client(handler).chat("u1", "t1", "问题")

    assert excinfo.value.is_citation_error


def test_network_failure_becomes_connection_error() -> None:
    """后端没启动时抛连接错误，前端据此提示「先起后端」。"""
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    with pytest.raises(AgentApiError) as excinfo:
        build_client(handler).chat("u1", "t1", "问题")

    assert excinfo.value.is_connection_error


def test_sse_events_are_parsed_in_order() -> None:
    """流式接口要按顺序还原 status 和 result 事件。"""
    body = (
        'event: status\ndata: {"stage":"thinking","message":"正在理解问题"}\n\n'
        'event: status\ndata: {"stage":"retrieving","message":"正在检索条文"}\n\n'
        'event: result\ndata: {"answer":"百分之三百","refused":false,'
        '"citations":[],"round_number":2}\n\n'
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/chat/stream"
        return httpx.Response(
            200,
            content=body.encode("utf-8"),
            headers={"content-type": "text/event-stream"},
        )

    events = list(build_client(handler).chat_stream("u1", "t1", "加班工资怎么算？"))

    assert [name for name, _ in events] == ["status", "status", "result"]
    assert events[0][1]["stage"] == "thinking"
    assert events[-1][1]["answer"] == "百分之三百"


def test_sse_parser_handles_multiline_data() -> None:
    """data 行可能被拆成多行（服务端格式化 JSON 时），解析器要拼回去再解码。"""
    raw = [
        "event: result",
        "data: {",
        'data:   "answer": "百分之三百",',
        'data:   "refused": false',
        "data: }",
        "",
    ]

    events = list(_iter_sse_events(raw))

    assert len(events) == 1
    assert events[0][1]["answer"] == "百分之三百"


def test_sse_parser_falls_back_on_invalid_json() -> None:
    """服务端发来非 JSON 时不能让前端崩掉，退化成 raw 字段即可。"""
    raw = ["event: status", "data: 这不是 JSON", ""]

    events = list(_iter_sse_events(raw))

    assert events[0][1] == {"raw": "这不是 JSON"}


def test_delete_thread_returns_deleted_count() -> None:
    """删除会话要返回实际条数，供界面提示。"""
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "DELETE"
        assert request.url.path == "/threads/thread-abc"
        assert request.url.params["user_id"] == "u1"
        return httpx.Response(200, json={"thread_id": "thread-abc", "deleted": 2})

    assert build_client(handler).delete_thread("u1", "thread-abc") == 2


def test_save_memory_sends_category() -> None:
    """保存记忆必须带上类别，后端按白名单校验。"""
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["category"] == "goal"
        assert body["memory_key"] == "learning_goal"
        assert body["review_id"] == "test-review"
        assert body["confirmed"] is True
        return httpx.Response(
            200,
            json={
                "memory_key": "learning_goal",
                "category": "goal",
                "content": body["content"],
                "updated_at": "2026-10-31T00:00:00+00:00",
            },
        )

    saved = build_client(handler).save_memory(
        "u1", "learning_goal", "goal", "学习 Agent 应用开发",
        review_id="test-review", confirmed=True,
    )

    assert saved["category"] == "goal"


def test_info_is_used_for_connection_check() -> None:
    """前端启动时靠 /info 决定显示「已连接」还是提示先起后端。"""
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/info"
        return httpx.Response(
            200,
            json={
                "service": "labor-law-knowledge-agent",
                "version": "0.4.0",
                "model": "deepseek-flash",
                "knowledge_chunks": 107,
                "knowledge_source": "knowledge/labor_law.txt",
                "retrieval": {"min_score": 0.6, "top_k": 4, "keyword_coverage": 0.15},
                "capabilities": ["/chat", "/chat/stream"],
            },
        )

    info = build_client(handler).info()

    assert info["knowledge_chunks"] == 107
    assert info["retrieval"]["top_k"] == 4
