"""测试聊天日志的写入、回读、会话列表和隔离规则。"""

import sqlite3  # 用内存数据库跑测试，不碰项目里的真实日志文件。
from types import SimpleNamespace  # 构造服务层需要的最小替身对象。

import pytest  # 提供夹具和异常断言工具。

from projects.enterprise_knowledge_agent.app import service as service_module
from projects.enterprise_knowledge_agent.app.chat_history import (
    ChatHistory,
    safe_record_turn,
)
from projects.enterprise_knowledge_agent.app.service import (
    AgentService,
    parse_checkpoint_thread_id,
)


@pytest.fixture()
def history() -> ChatHistory:
    """每个测试拿到独立的空日志库，互不影响。"""
    connection = sqlite3.connect(":memory:")
    log = ChatHistory(connection)
    log.setup()
    yield log
    connection.close()


def test_record_turn_persists_question_and_answer(history: ChatHistory) -> None:
    """一轮问答应写成两条消息，并保留拒答标记与逐字引文。"""
    history.record_turn(
        "u1",
        "t1",
        "法定节假日加班给几倍工资？",
        answer="支付不低于工资的百分之三百的工资报酬。",
        refused=False,
        citations=[
            {"source_id": "劳动法#第四十四条", "quote": "支付不低于工资的百分之三百的工资报酬"}
        ],
    )

    messages = history.list_messages("u1", "t1")

    assert [message.role for message in messages] == ["user", "assistant"]
    assert messages[0].content == "法定节假日加班给几倍工资？"
    assert messages[1].refused is False
    assert messages[1].citations[0]["source_id"] == "劳动法#第四十四条"


def test_refused_answer_keeps_flag(history: ChatHistory) -> None:
    """拒答也要入库，否则回看时会误以为那一轮没有回答。"""
    history.record_turn(
        "u1",
        "t1",
        "男方有陪产假吗？",
        answer="无法确定，《劳动法》中没有相应规定。",
        refused=True,
        citations=[],
    )

    assistant = history.list_messages("u1", "t1")[1]

    assert assistant.refused is True
    assert assistant.citations == []


def test_threads_are_sorted_by_recency(history: ChatHistory) -> None:
    """后发生的会话应排在前面，标题取该会话的第一个问题。"""
    history.record_turn("u1", "old", "第一个问题", answer="甲", refused=False, citations=[])
    history.record_turn("u1", "new", "第二个问题", answer="乙", refused=False, citations=[])
    history.record_turn("u1", "new", "追问", answer="丙", refused=False, citations=[])

    summaries = history.list_threads("u1")

    assert [summary.thread_id for summary in summaries] == ["new", "old"]
    assert summaries[0].title == "第二个问题"  # 标题固定取首问，不随追问变化。
    assert summaries[0].message_count == 4


def test_conversations_are_isolated_by_user_and_thread(history: ChatHistory) -> None:
    """不同用户或不同会话的消息不能互相串。"""
    history.record_turn("u1", "t1", "问题A", answer="答A", refused=False, citations=[])
    history.record_turn("u2", "t1", "问题B", answer="答B", refused=False, citations=[])
    history.record_turn("u1", "t2", "问题C", answer="答C", refused=False, citations=[])

    assert [m.content for m in history.list_messages("u1", "t1")] == ["问题A", "答A"]
    assert [m.content for m in history.list_messages("u2", "t1")] == ["问题B", "答B"]
    assert [m.content for m in history.list_messages("u1", "t2")] == ["问题C", "答C"]


def test_long_title_is_truncated(history: ChatHistory) -> None:
    """超长问题要截断，避免侧边栏被一行字撑破。"""
    question = "我在公司连续加班一个月每天三小时而且周末也不休息这样到底算不算违法"
    history.record_turn("u1", "t1", question, answer="答", refused=False, citations=[])

    title = history.list_threads("u1")[0].title

    assert len(title) <= 23
    assert title.endswith("…")


def test_delete_thread_removes_only_that_thread(history: ChatHistory) -> None:
    """删除会话返回删除条数，且不影响其他会话。"""
    history.record_turn("u1", "t1", "问题A", answer="答A", refused=False, citations=[])
    history.record_turn("u1", "t2", "问题B", answer="答B", refused=False, citations=[])

    removed = history.delete_thread("u1", "t1")

    assert removed == 2
    assert history.list_messages("u1", "t1") == []
    assert len(history.list_messages("u1", "t2")) == 2


def test_setup_is_idempotent(history: ChatHistory) -> None:
    """重复初始化不应报错，也不应清掉已有数据。"""
    history.record_turn("u1", "t1", "问题", answer="答", refused=False, citations=[])

    history.setup()

    assert len(history.list_messages("u1", "t1")) == 2


def test_missing_timestamp_stays_empty(history: ChatHistory) -> None:
    """旧检查点没有可靠时间，导入时不能伪造一个当前时间。"""
    history.record_turn(
        "u1",
        "t1",
        "旧问题",
        answer="旧答案",
        refused=False,
        citations=[],
        asked_at="",
        answered_at="",
    )

    message = history.list_messages("u1", "t1")[0]

    assert message.created_at == ""


def test_safe_record_turn_tolerates_missing_history() -> None:
    """没有配置日志时返回 False，而不是让已经算出的答案抛异常。"""
    assert (
        safe_record_turn(
            None,
            "u1",
            "t1",
            "问题",
            answer="答",
            refused=False,
            citations=[],
        )
        is False
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('["demo-user-001","thread-abcd1234"]', ("demo-user-001", "thread-abcd1234")),
        ("demo-user-001", ("demo-user-001", "__legacy__")),  # 旧版本没有用户维度。
        ('{"user": "u1"}', ('{"user": "u1"}', "__legacy__")),  # 结构不符时兜底。
    ],
)
def test_parse_checkpoint_thread_id(raw: str, expected: tuple[str, str]) -> None:
    """检查点标识要能还原成用户和会话，并为旧数据保留兜底分支。"""
    assert parse_checkpoint_thread_id(raw) == expected


class FakeGraph:
    """替身图：直接返回一个已经通过校验的回答，不调用模型。"""

    def get_state(self, config: dict) -> SimpleNamespace:
        return SimpleNamespace(values={})  # 模拟“空会话”，不需要历史。

    def invoke(self, state: dict, config: dict) -> dict:
        return {
            "final_answer": {
                "answer": "支付不低于工资的百分之三百的工资报酬。",
                "refused": False,
                "citations": [
                    {"source_id": "劳动法#第四十四条", "quote": "百分之三百的工资报酬"}
                ],
            },
            "round_number": 2,
        }


def test_service_writes_turn_into_history(history: ChatHistory, monkeypatch) -> None:
    """服务层跑完一轮后，问题和答案都应出现在聊天日志里。"""
    monkeypatch.setattr(service_module, "build_agent_graph", lambda *a, **k: FakeGraph())
    service = AgentService(
        client=None,  # 替身图不会用到客户端。
        model="fake-model",
        registry=None,
        checkpointer=None,
        memory_store=SimpleNamespace(search=lambda namespace: []),  # 没有长期记忆。
        history=history,
    )

    service.answer("u1", "t1", "法定节假日加班给几倍工资？")

    messages = history.list_messages("u1", "t1")
    assert [message.role for message in messages] == ["user", "assistant"]
    assert messages[0].content == "法定节假日加班给几倍工资？"
    assert messages[1].citations[0]["source_id"] == "劳动法#第四十四条"
    assert messages[1].created_at != ""  # 新写入的轮次一定带真实时间。
