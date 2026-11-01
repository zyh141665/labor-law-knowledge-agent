"""测试用户和会话组合后的检查点标识不会发生简单串线。"""

from projects.enterprise_knowledge_agent.app.service import build_checkpoint_thread_id

import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.store.sqlite import SqliteStore

from projects.enterprise_knowledge_agent.app.chat_history import ChatHistory
from projects.enterprise_knowledge_agent.app.memory import get_memory, save_memory
from projects.enterprise_knowledge_agent.app.service import AgentService
from projects.enterprise_knowledge_agent.app.session import build_turn_state


def test_checkpoint_thread_id_contains_both_boundaries() -> None:
    """不同的 user/thread 组合必须生成不同的检查点标识。"""
    first = build_checkpoint_thread_id("user-a", "thread-b")
    second = build_checkpoint_thread_id("user", "a-thread-b")

    assert first != second


def test_same_user_and_thread_produce_stable_id() -> None:
    """相同组合必须稳定，程序重启后才能恢复同一会话。"""
    first = build_checkpoint_thread_id("student-001", "refund-chat")
    second = build_checkpoint_thread_id("student-001", "refund-chat")

    assert first == second


def test_delete_thread_clears_checkpoints_but_preserves_other_threads_and_memory():
    """真实 SQLite 保存器验证删除后的同一会话不会恢复旧消息。"""
    with (
        SqliteSaver.from_conn_string(":memory:") as saver,
        SqliteStore.from_conn_string(":memory:") as store,
        sqlite3.connect(":memory:", check_same_thread=False) as connection,
    ):
        store.setup()
        history = ChatHistory(connection)
        history.setup()
        builder = StateGraph(dict)
        builder.add_node("remember", lambda state: state)
        builder.add_edge(START, "remember")
        builder.add_edge("remember", END)
        graph = builder.compile(checkpointer=saver)
        configs = {}
        for user, thread in [("alice", "same"), ("alice", "other"), ("bob", "same")]:
            config = {"configurable": {"thread_id": build_checkpoint_thread_id(user, thread)}}
            configs[user, thread] = config
            graph.invoke({"conversation_input": [{"role": "user", "content": "旧问题"}]}, config)
            history.record_turn(user, thread, "旧问题", answer="旧答案", refused=False, citations=[])
        save_memory(store, "alice", "answer_style", "preference", "先给结论")
        service = AgentService(None, "fake", None, saver, store, history)

        assert service.delete_thread("alice", "same") == 2
        assert list(saver.list(configs["alice", "same"])) == []
        assert graph.get_state(configs["alice", "same"]).values == {}
        new_state = build_turn_state(graph, configs["alice", "same"], "新问题")
        assert new_state["conversation_input"] == [{"role": "user", "content": "新问题"}]
        assert history.list_messages("alice", "same") == []
        for identity in [("alice", "other"), ("bob", "same")]:
            assert graph.get_state(configs[identity]).values
            assert len(history.list_messages(*identity)) == 2
        assert get_memory(store, "alice", "answer_style").content == "先给结论"
        assert service.delete_thread("alice", "same") == 0  # 重复删除保持幂等。
