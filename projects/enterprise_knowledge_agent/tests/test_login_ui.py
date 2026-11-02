"""Streamlit真实组件测试：不启动浏览器、不连模型，验证登录页与会话清理。"""

import importlib
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest


PROJECT = Path(__file__).resolve().parent.parent
PASSWORD = "ui-demo-password-only"


@pytest.fixture
def fake_ui_backend(monkeypatch):
    monkeypatch.syspath_prepend(str(PROJECT))
    ui = importlib.import_module("app.login_ui")
    error_type = importlib.import_module("app.api_client").AgentApiError
    created = []

    class FakeClient:
        base_url = "http://local.test"

        def __init__(self, base_url=None, token=None):
            self.token, self.closed, self.expired = token, False, False
            created.append(self)

        def close(self):
            self.closed = True

        def auth_config(self):
            return {"mode": "accounts", "registration_enabled": True}

        def login(self, username, password):
            if password != PASSWORD:
                raise error_type("invalid_credentials", "用户名或密码错误。", 401)
            return {
                "access_token": "ui-session-token",
                "username": username,
                "user_id": "acct_test",
            }

        def register(self, username, password):
            return {"user_id": "acct_test", "username": username}

        def me(self):
            return {"user_id": "acct_test", "token_label": "…test"}

        def info(self):
            if self.expired:
                raise error_type("unauthorized", "已失效", 401)
            assert self.token == "ui-session-token"
            return {
                "model": "test",
                "knowledge_chunks": 205,
                "version": "test",
                "retrieval": {"min_score": 0.6, "top_k": 4},
            }

        def list_threads(self, uid):
            return []

        def list_messages(self, uid, tid):
            return []

        def list_memories(self, uid):
            return []

        def logout(self):
            return {"logged_out": True}

    monkeypatch.setattr(ui, "AgentApiClient", FakeClient)
    return created


def click(at, label):
    next(button for button in at.button if button.label == label).click()
    return at.run(timeout=15)


def log_in(at):
    at.text_input(key="login_username").set_value("alice")
    at.text_input(key="login_password").set_value(PASSWORD)
    return click(at, "登录")


def test_login_page_hides_chat_and_register_passwords_must_match(fake_ui_backend):
    at = AppTest.from_file(str(PROJECT / "streamlit_app.py")).run(timeout=15)
    assert not at.exception and len(at.chat_input) == 0
    assert [tab.label for tab in at.tabs] == ["账号登录", "注册账号"]
    assert fake_ui_backend[0].token == ""  # 绝不使用.env中的共享身份。
    at.text_input(key="register_username").set_value("alice")
    at.text_input(key="register_password").set_value(PASSWORD)
    at.text_input(key="register_confirmation").set_value("different-password")
    at.checkbox[0].check()
    click(at, "创建账号")
    assert not at.exception
    assert any("不一致" in item.value for item in at.error)


def test_login_logout_and_independent_browser_clients(fake_ui_backend):
    first = AppTest.from_file(str(PROJECT / "streamlit_app.py")).run(timeout=15)
    log_in(first)
    assert not first.exception and len(first.chat_input) == 1
    assert first.session_state["identity"]["username"] == "alice"
    second = AppTest.from_file(str(PROJECT / "streamlit_app.py")).run(timeout=15)
    assert not second.exception and len(second.chat_input) == 0
    assert first.session_state["http_client"] is not second.session_state["http_client"]
    first.session_state["memory_candidate"] = {"private": "sentinel"}
    click(first, "退出登录")
    assert not first.exception and len(first.chat_input) == 0
    assert "access_token" not in first.session_state
    assert "memory_candidate" not in first.session_state


def test_expired_login_returns_to_login_without_showing_chat(fake_ui_backend):
    at = AppTest.from_file(str(PROJECT / "streamlit_app.py")).run(timeout=15)
    log_in(at)
    at.session_state["http_client"].expired = True
    at.run(timeout=15)
    assert not at.exception and len(at.chat_input) == 0
    click(at, "返回登录")
    assert not at.exception and "access_token" not in at.session_state
