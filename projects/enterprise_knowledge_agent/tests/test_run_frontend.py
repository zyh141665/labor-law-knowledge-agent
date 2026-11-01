"""用 HTTP 替身检查启动探测，不启动前端也不请求真实服务。"""

import httpx

from projects.enterprise_knowledge_agent.scripts import run_frontend
from projects.enterprise_knowledge_agent.app import api_client


def install_mock_client(monkeypatch, handler):
    """保留真实客户端的令牌加载逻辑，只替换网络传输。"""
    monkeypatch.setenv("AGENT_API_TOKEN", "test-probe-token")
    monkeypatch.setenv("HTTP_PROXY", "http://invalid-proxy:9999")
    real_client = httpx.Client

    def mock_client(**kwargs):
        assert kwargs["trust_env"] is False  # 本地后端不能被系统代理截走。
        return real_client(**kwargs, transport=httpx.MockTransport(handler))

    monkeypatch.setattr(api_client.httpx, "Client", mock_client)


def test_probe_sends_bearer_and_returns_info(monkeypatch):
    def handler(request):
        assert request.url.path == "/info"
        assert request.headers["Authorization"] == "Bearer test-probe-token"
        return httpx.Response(200, json={"model": "fake", "knowledge_chunks": 205})

    install_mock_client(monkeypatch, handler)

    assert run_frontend.probe_backend("http://backend.test")["knowledge_chunks"] == 205


def test_probe_reports_auth_failure_instead_of_only_connection_failure(
    monkeypatch, capsys
):
    def handler(request):
        return httpx.Response(
            401,
            json={
                "detail": {
                    "code": "unauthorized",
                    "message": "令牌无效",
                }
            },
        )

    install_mock_client(monkeypatch, handler)

    assert run_frontend.probe_backend("http://backend.test") is None
    assert "unauthorized" in capsys.readouterr().out
