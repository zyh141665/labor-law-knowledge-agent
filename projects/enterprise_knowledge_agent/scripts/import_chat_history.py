"""把检查点数据库里已有的对话导入聊天日志表。

检查点保存的是 LangGraph 的完整内部状态，字段结构会随版本变化；聊天日志是
界面用的稳定投影。这个脚本只做一次性搬运，让加功能之前的会话也能被回看。

用法（在 agent-learning 目录下）：

    python -m projects.enterprise_knowledge_agent.scripts.import_chat_history
    python -m projects.enterprise_knowledge_agent.scripts.import_chat_history --force
"""

import sqlite3  # 直接读检查点库，只为拿到 thread_id 列表。
import sys  # 解析命令行参数。
from datetime import datetime  # 把检查点里的 UTC 时间转换成展示用的本地时间。

from langgraph.checkpoint.sqlite import SqliteSaver  # 复用官方实现反序列化检查点。

from projects.enterprise_knowledge_agent.app.chat_history import (
    CHAT_HISTORY_PATH,
    ChatHistory,
    open_chat_history,
)
from projects.enterprise_knowledge_agent.app.checkpoint import CHECKPOINT_PATH
from projects.enterprise_knowledge_agent.app.service import parse_checkpoint_thread_id


def local_time(raw: str | None) -> str:
    """把检查点里的 ISO 时间转成本地时区的展示字符串。

    当前版本的 LangGraph 不往 metadata 写 ``ts``（实测为 None），
    此时返回空字符串，让界面知道这条记录没有可靠时间，而不是编一个导入时刻。
    """
    if not raw:
        return ""
    try:
        return datetime.fromisoformat(raw).astimezone().isoformat(timespec="seconds")
    except (TypeError, ValueError):
        return ""


def last_user_question(items: list[dict]) -> str | None:
    """取对话输入中离当前最近的用户提问。"""
    for item in reversed(items):
        if item.get("role") == "user" and isinstance(item.get("content"), str):
            return item["content"]
    return None


def iter_turns(saver: SqliteSaver, thread_key: str) -> list[tuple[str, dict, str]]:
    """还原一个会话里的每一轮问答，按时间正序返回。

    同一轮会产生多个检查点（模型、工具、校验各一步），它们的 ``final_answer``
    可能相同，因此按“问题 + 答案正文”去重，只保留第一次出现的记录。
    """
    turns: list[tuple[str, dict, str]] = []
    seen: set[tuple[str, str]] = set()
    # saver.list 返回由新到旧，先收集再反转，得到对话发生顺序。
    for item in saver.list({"configurable": {"thread_id": thread_key}}):
        values = item.checkpoint.get("channel_values", {})
        final_answer = values.get("final_answer")
        if not final_answer:  # 工具调用轮次还没有生成最终答案。
            continue
        question = last_user_question(values.get("conversation_input", []))
        if question is None:
            continue
        answer_text = str(final_answer.get("answer", ""))
        fingerprint = (question, answer_text)
        if fingerprint in seen:  # 同一轮的后续检查点不重复导入。
            continue
        seen.add(fingerprint)
        turns.append((question, final_answer, local_time(item.metadata.get("ts"))))
    turns.reverse()
    return turns


def import_thread(
    history: ChatHistory,
    user_id: str,
    thread_id: str,
    turns: list[tuple[str, dict, str]],
) -> int:
    """把一个会话的问答写入聊天日志，返回导入的轮数。"""
    for question, final_answer, timestamp in turns:
        history.record_turn(
            user_id,
            thread_id,
            question,
            answer=str(final_answer.get("answer", "")),
            refused=bool(final_answer.get("refused", False)),
            citations=final_answer.get("citations") or [],
            asked_at=timestamp,  # 空字符串表示“来源没有可靠时间”，不要替换成当前时间。
            answered_at=timestamp,
        )
    return len(turns)


def main() -> None:
    """遍历检查点库中的全部会话，逐条导入聊天日志。"""
    force = "--force" in sys.argv[1:]

    connection = sqlite3.connect(str(CHECKPOINT_PATH), check_same_thread=False)
    try:
        # 检查点没有保存真实时间，但 rowid 是写入顺序；按它升序导入，
        # 聊天日志的自增 id 就与真实先后顺序一致，侧边栏的“最近会话”才不会乱。
        thread_keys = [
            row[0]
            for row in connection.execute(
                "SELECT thread_id FROM checkpoints GROUP BY thread_id ORDER BY MAX(rowid)"
            ).fetchall()
        ]
        saver = SqliteSaver(connection)
        imported: list[tuple[str, str, int]] = []
        skipped: list[tuple[str, str, str]] = []

        with open_chat_history() as history:
            for thread_key in thread_keys:
                user_id, thread_id = parse_checkpoint_thread_id(thread_key)
                existing = history.list_messages(user_id, thread_id, limit=1)
                if existing and not force:
                    skipped.append((user_id, thread_id, "已有聊天记录"))
                    continue
                turns = iter_turns(saver, thread_key)
                if not turns:
                    skipped.append((user_id, thread_id, "没有可还原的完整问答"))
                    continue
                count = import_thread(history, user_id, thread_id, turns)
                imported.append((user_id, thread_id, count))
    finally:
        connection.close()

    for user_id, thread_id, count in imported:
        print(f"导入 {user_id} / {thread_id}：{count} 轮")
    for user_id, thread_id, reason in skipped:
        print(f"跳过 {user_id} / {thread_id}：{reason}")

    print(f"\n总计导入 {len(imported)} 个会话、{sum(c for _, _, c in imported)} 轮问答")
    print(f"聊天日志：{CHAT_HISTORY_PATH}")


if __name__ == "__main__":
    main()
