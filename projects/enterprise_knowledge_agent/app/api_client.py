"""调用后端 FastAPI 服务的 HTTP 客户端。

Streamlit 前端只通过这个模块与后端通信，不再 import 任何业务模块：
Embedding 模型、三个 SQLite 连接和 Agent 编排都留在后端进程里。
这样前端重启只要 1 秒，也避免两个进程同时写同一批数据库文件。

后端地址优先读环境变量 ``AGENT_API_BASE_URL``，默认 http://127.0.0.1:8000。
不要用 0.0.0.0 作为客户端地址：它在某些系统上不是可路由的目标。
"""

import json
import os
from collections.abc import Iterable, Iterator
from typing import Any

import httpx

from .config import load_env  # 与后端共用同一套 .env 读取规则。


DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_TIMEOUT = 180.0  # Agent 一轮可能包含检索和两次模型调用，留足时间。


class AgentApiError(Exception):
    """后端返回错误、或根本连不上。

    ``code`` 来自后端响应体中的 error code，前端据此决定提示文案，
    不再靠匹配错误字符串。连不上时 code 固定为 ``connection_failed``。
    """

    def __init__(self, code: str, message: str, status_code: int = 0) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code

    @property
    def is_connection_error(self) -> bool:
        """后端没启动，或地址写错。"""
        return self.code == "connection_failed"

    @property
    def is_balance_error(self) -> bool:
        """DeepSeek 账户余额不足；API Key 本身仍然有效。"""
        return self.code == "upstream_balance"

    @property
    def is_auth_error(self) -> bool:
        """API Key 无效或已被吊销。"""
        return self.code == "upstream_auth"

    @property
    def is_citation_error(self) -> bool:
        """模型输出没通过引用真实性校验，被安全机制拦下。"""
        return self.code == "citation_failed"

    @property
    def is_unauthorized(self) -> bool:
        """后端拒绝了我们的令牌（HTTP 401）。

        注意和 ``is_auth_error`` 区分：那个是 DeepSeek 的 Key 失效，
        这个是后端自己这扇门没开。两者提示语完全不同，混在一起会查错方向。
        """
        return self.code == "unauthorized"

    @property
    def is_user_mismatch(self) -> bool:
        """令牌身份和请求里的 user_id 不一致（HTTP 403）。"""
        return self.code == "user_mismatch"


def _parse_error(response: httpx.Response) -> AgentApiError:
    """把 FastAPI 的错误响应还原成带 code 的异常。"""
    status_code = response.status_code
    try:
        payload = response.json()
    except ValueError:  # 响应不是 JSON，多半是反向代理吐的 HTML 错误页。
        return AgentApiError("unexpected_response", response.text[:200], status_code)

    detail = payload.get("detail", payload) if isinstance(payload, dict) else payload
    if isinstance(detail, dict):
        return AgentApiError(
            str(detail.get("code", "unknown_error")),
            str(detail.get("message", "后端返回了未说明的错误。")),
            status_code,
        )
    return AgentApiError("unknown_error", str(detail), status_code)


def _iter_sse_events(lines: Iterable[str]) -> Iterator[tuple[str, dict[str, Any]]]:
    """把 SSE 文本行还原成 (事件名, 数据) 序列。

    服务端格式为 ``event: xxx\\ndata: {...}\\n\\n``，空行表示一个事件结束。
    这里不依赖具体事件名，前端只关心它收到的字典。
    """
    event_name = "message"
    data_lines: list[str] = []
    for raw_line in lines:
        line = raw_line.rstrip("\r")
        if not line:  # 空行：一个事件收尾。
            if data_lines:
                try:
                    payload = json.loads("\n".join(data_lines))
                except json.JSONDecodeError:
                    payload = {"raw": "\n".join(data_lines)}
                yield event_name, payload
            event_name = "message"
            data_lines = []
            continue
        if line.startswith("event:"):
            event_name = line[6:].strip()
        elif line.startswith("data:"):
            data_lines.append(line[5:].lstrip())

    if data_lines:  # 服务端没发结尾空行时，补一次收尾。
        try:
            payload = json.loads("\n".join(data_lines))
        except json.JSONDecodeError:
            payload = {"raw": "\n".join(data_lines)}
        yield event_name, payload


class AgentApiClient:
    """后端 HTTP 接口的同步封装。

    与 agent-service-toolkit 的 AgentClient 一样，这里不依赖任何 Agent 框架的
    SDK：协议就是后端自己定义的 REST + SSE，客户端和服务端各写一半、配对使用。
    区别是这里复用同一个 httpx.Client，保留连接池，而不是每次调用都新建。
    """

    def __init__(
        self,
        base_url: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        token: str | None = None,
    ) -> None:
        load_env()  # Streamlit 不会自动读 .env，这里补上，保证和后端读到同一份配置。
        resolved = base_url or os.getenv("AGENT_API_BASE_URL") or DEFAULT_BASE_URL
        self.base_url = resolved.rstrip("/")
        self.token = token if token is not None else os.getenv("AGENT_API_TOKEN")
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=timeout,
            headers=headers,
            # 不读 HTTP_PROXY / HTTPS_PROXY 等环境变量。
            # 这台机器上配了系统代理，一旦 httpx 走代理，它会按代理协议把请求行
            # 写成绝对 URI（GET http://127.0.0.1:8000/threads HTTP/1.1），
            # uvicorn 只认 GET /threads，于是返回 404 —— 而且第一次请求往往成功，
            # 之后复用连接才失败，极难排查。本地后端不需要代理。
            trust_env=False,
        )

    def close(self) -> None:
        """释放连接池；Streamlit 进程退出时由 atexit 调用。"""
        self._client.close()

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        """发起请求并把所有失败统一翻译成 AgentApiError。"""
        try:
            response = self._client.request(method, path, **kwargs)
        except httpx.TimeoutException as error:
            raise AgentApiError("timeout", f"后端响应超时：{error}") from error
        except httpx.HTTPError as error:
            raise AgentApiError(
                "connection_failed",
                f"无法连接后端 {self.base_url}，请确认它已经启动：{error}",
            ) from error
        if response.status_code >= 400:
            raise _parse_error(response)
        if not response.content:
            return None
        return response.json()

    # ------------------------------------------------------------------
    # 探活与服务信息
    # ------------------------------------------------------------------

    def health(self) -> bool:
        """后端是否可用；用于前端启动时的连接状态检查。"""
        try:
            self._request("GET", "/health")
        except AgentApiError:
            return False
        return True

    def info(self) -> dict[str, Any]:
        """取得后端自述的运行信息（模型、知识库规模、接口列表）。"""
        return self._request("GET", "/info")

    def me(self) -> dict[str, Any]:
        """取得当前令牌对应的身份。

        前端不再自己编一个随机用户 ID：身份由后端从令牌推导，
        客户端说什么都不算数。否则认证只挡住了“能不能进”，
        挡不住“进来以后看谁的数据”。
        """
        return self._request("GET", "/me")

    def auth_config(self) -> dict[str, Any]:
        return self._request("GET", "/auth/config")  # 登录前公开的模式与注册开关。

    def register(self, username: str, password: str) -> dict[str, Any]:
        return self._request("POST", "/auth/register", json={"username": username, "password": password})

    def login(self, username: str, password: str) -> dict[str, Any]:
        return self._request("POST", "/auth/login", json={"username": username, "password": password})

    def logout(self) -> dict[str, Any]:
        return self._request("POST", "/auth/logout")  # 后端删除当前令牌摘要。

    # ------------------------------------------------------------------
    # 对话
    # ------------------------------------------------------------------

    def chat(self, user_id: str, thread_id: str, question: str) -> dict[str, Any]:
        """普通问答：等 Agent 整轮跑完再返回。"""
        return self._request(
            "POST",
            "/chat",
            json={"user_id": user_id, "thread_id": thread_id, "question": question},
        )

    def chat_stream(
        self,
        user_id: str,
        thread_id: str,
        question: str,
    ) -> Iterator[tuple[str, dict[str, Any]]]:
        """流式问答：逐条产出 SSE 事件。

        事件约定：
        - ``status``：{"stage": "thinking"|"retrieving"|"generating"|"validating", ...}
        - ``result``：{"answer": ..., "refused": ..., "citations": [...], "round_number": ...}
        - ``error``：{"code": ..., "message": ...}
        """
        payload = {
            "user_id": user_id,
            "thread_id": thread_id,
            "question": question,
        }
        try:
            with self._client.stream("POST", "/chat/stream", json=payload) as response:
                if response.status_code >= 400:
                    response.read()  # 先把错误响应体读完才能解析。
                    raise _parse_error(response)
                yield from _iter_sse_events(response.iter_lines())
        except httpx.TimeoutException as error:
            raise AgentApiError("timeout", f"后端响应超时：{error}") from error
        except httpx.HTTPError as error:
            raise AgentApiError(
                "connection_failed",
                f"流式连接中断，请确认后端仍在运行：{error}",
            ) from error

    # ------------------------------------------------------------------
    # 历史会话
    # ------------------------------------------------------------------

    def list_threads(self, user_id: str) -> list[dict[str, Any]]:
        """该用户的历史会话列表，按最近一次对话倒序。"""
        return self._request("GET", "/threads", params={"user_id": user_id})

    def list_messages(self, user_id: str, thread_id: str) -> list[dict[str, Any]]:
        """某个会话的全部消息，用于刷新页面后恢复。"""
        return self._request(
            "GET",
            f"/threads/{thread_id}/messages",
            params={"user_id": user_id},
        )

    def delete_thread(self, user_id: str, thread_id: str) -> int:
        """删除一个会话的聊天记录，返回删除条数。"""
        payload = self._request(
            "DELETE",
            f"/threads/{thread_id}",
            params={"user_id": user_id},
        )
        return int((payload or {}).get("deleted", 0))

    # ------------------------------------------------------------------
    # 长期记忆
    # ------------------------------------------------------------------

    def list_memories(self, user_id: str) -> list[dict[str, Any]]:
        """该用户的全部长期记忆。"""
        return self._request("GET", "/memories", params={"user_id": user_id})

    def save_memory(
        self,
        user_id: str,
        memory_key: str,
        category: str,
        content: str,
        *,
        review_id: str,
        confirmed: bool,
    ) -> dict[str, Any]:
        """写入一条已经人工确认过的长期记忆。"""
        return self._request(
            "POST",
            "/memories",
            json={
                "user_id": user_id,
                "memory_key": memory_key,
                "category": category,
                "content": content,
                "review_id": review_id,
                "confirmed": confirmed,
            },
        )

    def extract_memory(self, user_id: str, text: str) -> dict[str, Any]:
        """让后端调用模型提取候选记忆，并返回门控与判重结果。

        必须带 user_id：后端要拿这句话和该用户已有的记忆比对，
        才能说出「这条会跟哪条合并」。
        """
        return self._request(
            "POST",
            "/memories/extract",
            json={"user_id": user_id, "text": text},
        )
