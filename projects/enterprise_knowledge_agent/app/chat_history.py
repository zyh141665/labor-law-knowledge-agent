"""把每一轮问答写入独立的 SQLite 日志，供前端回看历史会话。

检查点数据库里其实也保存着完整对话，但那是 LangGraph 的内部状态结构：
``conversation_input`` 里混着工具调用、推理片段和原始 JSON，引用信息也只在
最后一轮的 ``final_answer`` 里保留。直接读它来渲染界面等于把内部实现当接口用。
因此这里单独建一张只读投影表，字段就是界面真正需要的东西。
"""

import json  # 引用列表按 JSON 存成一列，避免再拆一张关联表。
import sqlite3  # 项目统一使用标准库内置的 SQLite，不引入额外依赖。
from collections.abc import Iterator, Sequence  # 标注上下文管理器和参数类型。
from contextlib import contextmanager  # 用 with 保证连接一定被关闭。
from datetime import datetime  # 记录每条消息的本地时间。
from pathlib import Path  # 以跨平台方式拼出数据库文件路径。
from typing import Literal  # 限制角色只能是用户或助手。

from pydantic import BaseModel, Field  # 读出数据库后用同一套模型校验并传递。


PROJECT_ROOT = Path(__file__).resolve().parent.parent  # 得到 enterprise_knowledge_agent 根目录。
CHAT_HISTORY_PATH = PROJECT_ROOT / "data" / "chat_history.sqlite"  # 对话日志独立成一个文件。

_SQLITE_ERRORS = (sqlite3.Error,)  # 写日志失败时只吞掉数据库错误，不掩盖业务异常。

_SCHEMA = (  # 建表语句写成元组，首次运行逐条执行即可重复执行。
    """
    CREATE TABLE IF NOT EXISTS chat_messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        thread_id TEXT NOT NULL,
        role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
        content TEXT NOT NULL,
        refused INTEGER NOT NULL DEFAULT 0,
        citations TEXT NOT NULL DEFAULT '[]',
        created_at TEXT NOT NULL
    )
    """,
    # 按会话取消息是最频繁的查询，user_id + thread_id + id 一起建索引可以顺带完成排序。
    """
    CREATE INDEX IF NOT EXISTS idx_chat_messages_thread
        ON chat_messages (user_id, thread_id, id)
    """,
    # 侧边栏要按用户列出最近的会话，单独给 user_id 建一条索引。
    """
    CREATE INDEX IF NOT EXISTS idx_chat_messages_user
        ON chat_messages (user_id, id)
    """,
)


class ChatMessage(BaseModel):
    """一条可以渲染到界面上的对话消息。"""

    role: Literal["user", "assistant"]  # 区分问题气泡和回答气泡。
    content: str  # 已经通过校验的答案正文或用户原始问题。
    refused: bool = False  # 助手消息是否属于证据不足的拒答。
    citations: list[dict[str, str]] = Field(default_factory=list)  # 来源编号与逐字引文。
    created_at: str = ""  # 本地时间带时区偏移；从旧检查点导入的数据没有真实时间，留空。


class ThreadSummary(BaseModel):
    """侧边栏里的一行历史会话。"""

    thread_id: str  # 点击后要切回去的会话标识。
    title: str  # 用该会话的第一个问题充当标题。
    message_count: int  # 消息条数，帮助判断会话长短。
    updated_at: str  # 最后一次对话时间；旧数据没有时间时为空字符串。


def _now() -> str:
    """返回带时区偏移的本地时间，避免跨时区显示时被误读成 UTC。"""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _row_to_message(row: tuple[object, ...]) -> ChatMessage:
    """把数据库的一行还原成带类型的信息对象。"""
    role, content, refused, citations, created_at = row
    try:  # 历史数据可能来自旧版本，引用列解析失败不应该让整个会话打不开。
        parsed = json.loads(citations) if citations else []
    except (TypeError, json.JSONDecodeError):
        parsed = []
    return ChatMessage(
        role=role,
        content=str(content),
        refused=bool(refused),
        citations=parsed if isinstance(parsed, list) else [],
        created_at=str(created_at),
    )


class ChatHistory:
    """面向界面的对话日志读写层。"""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection  # 复用外部传入的连接，便于测试换成内存库。

    def setup(self) -> None:
        """创建首次运行所需的表和索引。"""
        for statement in _SCHEMA:
            self._connection.execute(statement)
        self._connection.commit()

    def _insert(
        self,
        user_id: str,
        thread_id: str,
        role: str,
        content: str,
        refused: bool,
        citations: Sequence[dict[str, str]],
        created_at: str,
    ) -> None:
        """写入一条消息；调用方负责事务边界。"""
        self._connection.execute(
            "INSERT INTO chat_messages "
            "(user_id, thread_id, role, content, refused, citations, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                user_id,
                thread_id,
                role,
                content,
                int(refused),
                json.dumps(list(citations), ensure_ascii=False),
                created_at,
            ),
        )

    def record_turn(
        self,
        user_id: str,
        thread_id: str,
        question: str,
        *,
        answer: str,
        refused: bool,
        citations: Sequence[dict[str, str]],
        asked_at: str | None = None,
        answered_at: str | None = None,
    ) -> None:
        """把一问一答写在同一事务里，避免界面上出现只有问题没有回答的半截会话。"""
        with self._connection:  # sqlite3 的 with 会在异常时自动回滚。
            self._insert(
                user_id,
                thread_id,
                "user",
                question,
                False,
                [],
                _now() if asked_at is None else asked_at,
            )
            self._insert(
                user_id,
                thread_id,
                "assistant",
                answer,
                refused,
                citations,
                _now() if answered_at is None else answered_at,
            )

    def record_message(
        self,
        user_id: str,
        thread_id: str,
        role: Literal["user", "assistant"],
        content: str,
        *,
        refused: bool = False,
        citations: Sequence[dict[str, str]] | None = None,
        created_at: str | None = None,
    ) -> None:
        """单独写入一条消息，供导入历史数据和测试使用。"""
        with self._connection:
            self._insert(
                user_id,
                thread_id,
                role,
                content,
                refused,
                citations or [],
                _now() if created_at is None else created_at,
            )

    def list_messages(
        self,
        user_id: str,
        thread_id: str,
        *,
        limit: int = 500,
    ) -> list[ChatMessage]:
        """按写入顺序取出一个会话的全部消息。"""
        rows = self._connection.execute(
            "SELECT role, content, refused, citations, created_at "
            "FROM chat_messages WHERE user_id = ? AND thread_id = ? "
            "ORDER BY id LIMIT ?",
            (user_id, thread_id, limit),
        ).fetchall()
        return [_row_to_message(row) for row in rows]

    def list_threads(self, user_id: str, *, limit: int = 20) -> list[ThreadSummary]:
        """列出该用户最近的会话，标题取第一个问题。"""
        rows = self._connection.execute(
            """
            SELECT m.thread_id,
                   m.content,
                   m.created_at,
                   (SELECT COUNT(*) FROM chat_messages AS c
                     WHERE c.user_id = m.user_id AND c.thread_id = m.thread_id) AS message_count,
                   (SELECT MAX(created_at) FROM chat_messages AS c
                     WHERE c.user_id = m.user_id AND c.thread_id = m.thread_id) AS updated_at,
                   (SELECT MAX(id) FROM chat_messages AS c
                     WHERE c.user_id = m.user_id AND c.thread_id = m.thread_id) AS last_id
              FROM chat_messages AS m
             WHERE m.user_id = ?
               AND m.role = 'user'
               AND m.id = (SELECT MIN(id) FROM chat_messages AS c
                            WHERE c.user_id = m.user_id
                              AND c.thread_id = m.thread_id
                              AND c.role = 'user')
             ORDER BY last_id DESC
             LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()

        summaries = []
        for thread_id, first_question, created_at, message_count, updated_at, _ in rows:
            summaries.append(
                ThreadSummary(
                    thread_id=str(thread_id),
                    title=_shorten(str(first_question)),
                    message_count=int(message_count),
                    updated_at=str(updated_at or created_at or ""),
                )
            )
        return summaries

    def count_threads(self, user_id: str) -> int:
        """统计该用户有多少个会话，用于界面提示。"""
        row = self._connection.execute(
            "SELECT COUNT(DISTINCT thread_id) FROM chat_messages WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        return int(row[0]) if row else 0

    def delete_thread(self, user_id: str, thread_id: str) -> int:
        """删除一个会话的全部消息，返回实际删除条数。"""
        with self._connection:
            cursor = self._connection.execute(
                "DELETE FROM chat_messages WHERE user_id = ? AND thread_id = ?",
                (user_id, thread_id),
            )
        return cursor.rowcount or 0


def _shorten(text: str, limit: int = 22) -> str:
    """把问题压成侧边栏里一行放得下的标题。"""
    cleaned = " ".join(text.split())  # 去掉换行和多余空格，避免标题撑破布局。
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[:limit]}…"


@contextmanager
def open_chat_history(path: Path = CHAT_HISTORY_PATH) -> Iterator[ChatHistory]:
    """打开对话日志数据库，并在 with 结束时不提交残留事务地关闭连接。"""
    path.parent.mkdir(parents=True, exist_ok=True)  # 首次运行时补上 data 目录。
    connection = sqlite3.connect(str(path), check_same_thread=False)
    try:
        history = ChatHistory(connection)
        history.setup()
        yield history
    finally:
        connection.close()


def safe_record_turn(
    history: ChatHistory | None,
    user_id: str,
    thread_id: str,
    question: str,
    *,
    answer: str,
    refused: bool,
    citations: Sequence[dict[str, str]],
) -> bool:
    """写日志失败只影响“能不能回看”，不应该让已经算出来的答案丢失。"""
    if history is None:
        return False
    try:
        history.record_turn(
            user_id,
            thread_id,
            question,
            answer=answer,
            refused=refused,
            citations=citations,
        )
    except _SQLITE_ERRORS:  # 磁盘满、库被锁等情况降级为无日志，由调用方决定是否提示。
        return False
    return True
