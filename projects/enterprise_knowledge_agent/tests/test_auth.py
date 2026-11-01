"""测试令牌认证：配置解析、Bear 头解析、以及 HTTP 层的真实拦截。

这些用例刻意**去掉** ``conftest.py`` 里的依赖覆盖，走真实的校验路径，
否则测的只是自己写的假函数。
"""

import sqlite3

import pytest
from fastapi.testclient import TestClient

from projects.enterprise_knowledge_agent.app.api import (
    app,
    get_chat_history,
    require_principal,
)
from projects.enterprise_knowledge_agent.app.auth import (
    AuthConfigError,
    authenticate,
    parse_bearer,
    parse_token_map,
)
from projects.enterprise_knowledge_agent.app.chat_history import ChatHistory


ALICE_TOKEN = "alice-secret-token-001"
BOB_TOKEN = "bob-secret-token-002"
TOKEN_MAP = {ALICE_TOKEN: "alice", BOB_TOKEN: "bob"}


def _build_history() -> ChatHistory:
    """内存版聊天日志，避免认证测试碰到真实的 chat_history.sqlite。"""
    connection = sqlite3.connect(":memory:", check_same_thread=False)
    return ChatHistory(connection)


# ---------------------------------------------------------------------------
# 纯函数：解析与校验
# ---------------------------------------------------------------------------


def test_parse_token_map_accepts_commas_spaces_and_newlines() -> None:
    """配置写在 .env 里，逗号和换行都该能用——多令牌挤一行很难读。"""
    raw = "token-aaaa1111:alice, token-bbbb2222:bob\n token-cccc3333:carol"

    mapping = parse_token_map(raw)

    assert mapping == {
        "token-aaaa1111": "alice",
        "token-bbbb2222": "bob",
        "token-cccc3333": "carol",
    }


def test_parse_token_map_rejects_missing_colon() -> None:
    """写错格式必须报错，而不是悄悄跳过——静默跳过会让人“怎么也登不上”。"""
    with pytest.raises(AuthConfigError, match="格式"):
        parse_token_map("token-aaaa1111")


def test_parse_token_map_rejects_short_token() -> None:
    """太短的令牌既好猜，也容易和别的环境变量撞上。"""
    with pytest.raises(AuthConfigError, match="长度"):
        parse_token_map("short:alice")


def test_parse_token_map_rejects_illegal_user_id() -> None:
    """用户 ID 的字符集必须和 API 层的校验规则一致，否则会互相打架。"""
    with pytest.raises(AuthConfigError, match="用户 ID"):
        parse_token_map("token-aaaa1111:张三")


def test_parse_token_map_rejects_empty_config() -> None:
    """空配置等于没有认证，必须当成错误处理。"""
    with pytest.raises(AuthConfigError, match="没有任何有效令牌"):
        parse_token_map("  ,  \n ")


def test_parse_bearer_is_case_insensitive_and_strict() -> None:
    """scheme 大小写不敏感，但格式不对就得返回 None。"""
    assert parse_bearer("Bearer abc") == "abc"
    assert parse_bearer("bearer abc") == "abc"
    assert parse_bearer("Basic abc") is None
    assert parse_bearer("Bearer   ") is None
    assert parse_bearer(None) is None


def test_authenticate_maps_token_to_user() -> None:
    """令牌决定身份，这是“绑定用户”模式的核心。"""
    principal = authenticate(TOKEN_MAP, f"Bearer {BOB_TOKEN}")

    assert principal is not None
    assert principal.user_id == "bob"
    # 标签只留末四位，日志里能区分是哪把钥匙，又不泄露令牌本身。
    assert principal.token_label == "…-002"
    assert BOB_TOKEN not in principal.token_label


def test_authenticate_rejects_unknown_token() -> None:
    """不认识的令牌一律不通过。"""
    assert authenticate(TOKEN_MAP, "Bearer not-a-real-token") is None
    assert authenticate(TOKEN_MAP, "Token alice-secret-token-001") is None


# ---------------------------------------------------------------------------
# HTTP 层：真的把请求打到接口上
# ---------------------------------------------------------------------------


@pytest.fixture(name="client")
def fixture_client():
    """不带认证依赖覆盖的裸客户端，配合真实的令牌表使用。

    只覆盖历史存储：``get_chat_history`` 在服务未启动时抛 503，
    如果不替换，越权用例会被 503 挡住，看不到真正想验证的 403。
    """
    removed = app.dependency_overrides.pop(require_principal, None)
    app.dependency_overrides[get_chat_history] = _build_history
    app.state.auth_tokens = dict(TOKEN_MAP)
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.pop(require_principal, None)
        app.dependency_overrides.pop(get_chat_history, None)
        if removed is not None:  # 把 conftest 的覆盖还回去，别影响后面的用例。
            app.dependency_overrides[require_principal] = removed
        del app.state.auth_tokens


def test_health_stays_public(client: TestClient) -> None:
    """健康检查必须开放：run_demo 靠它轮询后端是否就绪。"""
    assert client.get("/health").status_code == 200


def test_missing_token_is_rejected_with_401(client: TestClient) -> None:
    """没带令牌就 401，并且带上 WWW-Authenticate，符合 Bearer 规范。"""
    response = client.get("/me")

    assert response.status_code == 401
    assert response.json()["detail"]["code"] == "unauthorized"
    assert response.headers["www-authenticate"] == "Bearer"


def test_wrong_token_is_rejected_with_401(client: TestClient) -> None:
    """令牌不对同样 401，且不透露“错在哪一位”。"""
    response = client.get("/me", headers={"Authorization": "Bearer wrong-token-xxx"})

    assert response.status_code == 401


def test_correct_token_returns_identity(client: TestClient) -> None:
    """/me 应该告诉我们令牌是谁，前端据此显示身份、发后续请求。"""
    response = client.get("/me", headers={"Authorization": f"Bearer {ALICE_TOKEN}"})

    assert response.status_code == 200
    assert response.json()["user_id"] == "alice"


def test_token_cannot_read_another_users_data(client: TestClient) -> None:
    """令牌属于 alice，却想查 bob 的会话：必须 403 而不是返回别人的数据。"""
    response = client.get(
        "/threads",
        params={"user_id": "bob"},
        headers={"Authorization": f"Bearer {ALICE_TOKEN}"},
    )

    assert response.status_code == 403
    assert response.json()["detail"]["code"] == "user_mismatch"


def test_chat_requires_token_before_touching_business_layer(
    client: TestClient,
) -> None:
    """未认证的请求应该停在 401，而不是先探到“服务未就绪”的 503。

    服务层没有启动（这里没跑 lifespan），如果认证排在它后面，
    就会返回 503，等于告诉陌生人“这个后端装了 Agent 但还没起好”。
    """
    response = client.post(
        "/chat",
        json={"user_id": "alice", "thread_id": "t1", "question": "加班费怎么算"},
    )

    assert response.status_code == 401


def test_protected_endpoint_without_configured_tokens(
    client: TestClient,
) -> None:
    """令牌表为空时给出 503 而不是放行，防止配置缺失变成后门。"""
    app.state.auth_tokens = {}

    response = client.get("/me", headers={"Authorization": f"Bearer {ALICE_TOKEN}"})

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "auth_not_configured"
