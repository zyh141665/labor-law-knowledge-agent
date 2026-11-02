"""账号密码、令牌注销/过期、限流与真实HTTP隔离回归；不调用模型。"""

import sqlite3

import pytest
from fastapi.testclient import TestClient

from projects.enterprise_knowledge_agent.app.accounts import (
    AccountError,
    AccountStore,
    SESSION_SECONDS,
)
from projects.enterprise_knowledge_agent.app.accounts_api import get_account_store
from projects.enterprise_knowledge_agent.app.api import (
    app,
    get_chat_history,
    require_principal,
)
from projects.enterprise_knowledge_agent.app.chat_history import ChatHistory


PASSWORD = "demo-only-password-2026"


def test_password_salts_and_session_hashes_are_not_plaintext(tmp_path):
    store = AccountStore(tmp_path / "accounts.sqlite")
    alice = store.register("Alice", PASSWORD)
    bob = store.register("bob", PASSWORD)
    login = store.login("ALICE", PASSWORD)
    assert store.authenticate(login["access_token"]).user_id == alice["user_id"]
    assert alice["user_id"] != bob["user_id"]
    with sqlite3.connect(store.path) as db:
        rows = db.execute("SELECT salt,password_hash FROM accounts").fetchall()
        assert rows[0] != rows[1]
        assert all(PASSWORD not in str(row) for row in rows)
        assert (
            db.execute("SELECT token_hash FROM account_sessions").fetchone()[0]
            != login["access_token"]
        )
    assert (
        AccountStore(store.path).authenticate(login["access_token"]) is not None
    )  # 重建连接仍有效。
    store.logout(login["access_token"])
    assert store.authenticate(login["access_token"]) is None


def test_expiry_and_wrong_password_or_unknown_user(tmp_path):
    now = [1000]
    store = AccountStore(tmp_path / "accounts.sqlite", clock=lambda: now[0])
    store.register("alice", PASSWORD)
    token = store.login("alice", PASSWORD)["access_token"]
    for username in ["alice", "nobody"]:
        with pytest.raises(AccountError, match="用户名或密码错误"):
            store.login(username, "wrong-but-long-password")
    now[0] += SESSION_SECONDS
    assert store.authenticate(token) is None


def test_duplicate_user_and_rate_limit_persist_across_connections(tmp_path):
    now = [1000]
    path = tmp_path / "accounts.sqlite"
    store = AccountStore(path, clock=lambda: now[0])
    store.register("alice", PASSWORD)
    with pytest.raises(AccountError) as duplicate:
        store.register("ALICE", PASSWORD)
    assert duplicate.value.status == 409
    for _ in range(10):
        with pytest.raises(AccountError) as error:
            store.login("alice", "incorrect-password")
        assert error.value.status == 401
    reopened = AccountStore(path, clock=lambda: now[0])
    with pytest.raises(AccountError) as error:
        reopened.login("alice", PASSWORD)
    assert error.value.status == 429
    now[0] += 301
    assert reopened.login("alice", PASSWORD)["user_id"]


@pytest.fixture
def account_api(tmp_path, monkeypatch):
    store = AccountStore(tmp_path / "accounts.sqlite")
    saved_override = app.dependency_overrides.pop(require_principal, None)
    previous = {
        key: getattr(app.state, key, None)
        for key in ["account_store", "auth_tokens", "auth_mode"]
    }
    connection = sqlite3.connect(":memory:", check_same_thread=False)
    history = ChatHistory(connection)
    history.setup()
    app.state.account_store, app.state.auth_tokens, app.state.auth_mode = (
        store,
        {"old-static-secret": "alice"},
        "accounts",
    )
    app.dependency_overrides[get_account_store] = lambda: store
    app.dependency_overrides[get_chat_history] = lambda: history
    monkeypatch.setenv("AGENT_ALLOW_REGISTRATION", "true")
    try:
        yield TestClient(app), store, history
    finally:
        connection.close()
        app.dependency_overrides.pop(get_account_store, None)
        app.dependency_overrides.pop(get_chat_history, None)
        if saved_override is not None:
            app.dependency_overrides[require_principal] = saved_override
        for key, value in previous.items():
            if value is None:
                if hasattr(app.state, key):
                    delattr(app.state, key)
            else:
                setattr(app.state, key, value)


def test_http_register_login_me_isolation_and_logout(account_api):
    client, _, history = account_api
    alice = client.post(
        "/auth/register", json={"username": "alice", "password": PASSWORD}
    ).json()
    bob = client.post(
        "/auth/register", json={"username": "bob", "password": PASSWORD}
    ).json()
    response = client.post(
        "/auth/login", json={"username": "alice", "password": PASSWORD}
    )
    assert (
        response.status_code == 200 and response.headers["cache-control"] == "no-store"
    )
    token = response.json()["access_token"]
    headers = {"Authorization": "Bearer " + token}
    history.record_turn(
        bob["user_id"],
        "bobs-thread",
        "private",
        answer="private",
        refused=True,
        citations=[],
    )
    assert client.get("/me", headers=headers).json()["user_id"] == alice["user_id"]
    assert (
        client.get(
            "/threads", params={"user_id": alice["user_id"]}, headers=headers
        ).json()
        == []
    )
    assert (
        client.get(
            "/threads", params={"user_id": bob["user_id"]}, headers=headers
        ).status_code
        == 403
    )
    assert (
        client.get(
            "/me", headers={"Authorization": "Bearer old-static-secret"}
        ).status_code
        == 401
    )
    assert client.post("/auth/logout", headers=headers).status_code == 200
    assert client.get("/me", headers=headers).status_code == 401
    assert len(history.list_messages(bob["user_id"], "bobs-thread")) == 2


def test_auth_payload_never_echoes_password_or_allows_user_id_assignment(account_api):
    client, _, _ = account_api
    secret = "do-not-echo-this-password"
    for payload in [
        {"username": "x", "password": secret},
        {"username": "alice", "password": secret, "user_id": "victim"},
        {"username": "alice", "password": "short"},
    ]:
        response = client.post("/auth/register", json=payload)
        assert response.status_code == 422
        assert secret not in response.text and "short" not in response.text
        assert "input" not in response.text


def test_auth_registration_switch_invalid_credentials_and_missing_login(
    account_api, monkeypatch
):
    client, _, _ = account_api
    assert client.get("/auth/config").json()["registration_enabled"] is True
    assert client.get("/me").status_code == 401
    assert client.post("/auth/logout").status_code == 401
    for username in ["alice", "nobody"]:
        assert (
            client.post(
                "/auth/login", json={"username": username, "password": PASSWORD}
            ).status_code
            == 401
        )
    monkeypatch.setenv("AGENT_ALLOW_REGISTRATION", "false")
    assert client.get("/auth/config").json()["registration_enabled"] is False
    assert (
        client.post(
            "/auth/register", json={"username": "alice", "password": PASSWORD}
        ).status_code
        == 403
    )
