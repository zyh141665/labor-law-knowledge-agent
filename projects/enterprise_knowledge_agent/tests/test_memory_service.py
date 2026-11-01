"""测试记忆服务层：提取 → 门控 → 判重 → 写入的完整协作。

这些测试用假的模型客户端，不联网、不花钱，只验证服务层自己的逻辑：
提示词里有没有带上已有记忆、门控拒绝时会不会误报重复、
模型编造的 key 会不会被拦下。
"""

import json
import sqlite3

import pytest
from langgraph.store.sqlite import SqliteStore

from projects.enterprise_knowledge_agent.app.memory import (
    list_memories,
    save_memory,
)
from projects.enterprise_knowledge_agent.app.memory_service import MemoryService


class FakeResponse:
    """模拟 OpenAI Responses API 的返回值，只保留 output_text。"""

    def __init__(self, text: str) -> None:
        self.output_text = text


class FakeResponses:
    """记录最后一次发给模型的输入，便于断言提示词里带了已有记忆。"""

    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.last_input = ""

    def create(self, **kwargs) -> FakeResponse:
        self.last_input = str(kwargs.get("input", ""))
        return FakeResponse(json.dumps(self.payload, ensure_ascii=False))


class FakeClient:
    """最小的 OpenAI 替身，只需要 responses.create 一条路。"""

    def __init__(self, payload: dict) -> None:
        self.responses = FakeResponses(payload)


def build_service(payload: dict, store: SqliteStore) -> MemoryService:
    """构造一个接在内存数据库上的记忆服务。"""
    return MemoryService(FakeClient(payload), "fake-model", store)


@pytest.fixture(name="store")
def fixture_store():
    """提供独立的内存 Store，避免用例之间互相污染。"""
    connection = sqlite3.connect(":memory:", check_same_thread=False)
    store = SqliteStore(connection)
    store.setup()
    connection.commit()  # setup 留下的未提交事务会让后续 put 报错。
    return store


def candidate_payload(**overrides) -> dict:
    """构造一份结构完整的候选记忆，用例只覆盖关心的字段。"""
    payload = {
        "should_remember": True,
        "memory_key": "answer_style",
        "category": "preference",
        "content": "回答先给结论，再解释原因。",
        "sensitive": False,
        "duplicate_of": "",
        "reason": "用户明确表达了自己的偏好。",
    }
    payload.update(overrides)
    return payload


def test_propose_feeds_existing_memories_into_prompt(store: SqliteStore) -> None:
    """提示词必须带上已有记忆，模型才有能力判断“这句话是不是记过了”。"""
    save_memory(store, "user-A", "learning_goal", "goal", "学习 Agent 应用开发")
    service = build_service(candidate_payload(), store)

    service.propose("user-A", "回答先给结论")

    prompt = service.client.responses.last_input  # type: ignore[union-attr]
    assert "learning_goal" in prompt
    assert "学习 Agent 应用开发" in prompt
    assert "回答先给结论" in prompt


def test_propose_uses_deterministic_rules_before_model_claim(
    store: SqliteStore,
) -> None:
    """本地规则能判出重复时，以规则为准，不采信模型给的 key。"""
    save_memory(
        store,
        "user-A",
        "answer_style",
        "preference",
        "回答尽量简洁，并优先解释结论",
    )
    payload = candidate_payload(
        memory_key="answer_style_conclusion_first",
        content="用户偏好回答时先给结论，再解释原因。",
        duplicate_of="另一个不存在的 key",
    )
    service = build_service(payload, store)

    proposal = service.propose("user-A", "回答先给结论再解释")

    assert proposal.allowed is True
    assert proposal.duplicate_of == "answer_style"  # 规则命中的结果，不是模型说的那个。


def test_propose_accepts_model_claim_only_when_key_exists(
    store: SqliteStore,
) -> None:
    """模型指认一个真实存在的 key 时采信；指认不存在的 key 时忽略。"""
    save_memory(store, "user-A", "study_goal", "goal", "找一份 Agent 实习")

    trusted = build_service(
        candidate_payload(
            memory_key="career_goal",
            category="goal",
            content="希望从事 Agent 应用开发",
            duplicate_of="study_goal",
        ),
        store,
    )
    assert trusted.propose("user-A", "我想做 Agent").duplicate_of == "study_goal"

    invented = build_service(
        candidate_payload(
            memory_key="career_goal",
            category="goal",
            content="希望从事 Agent 应用开发",
            duplicate_of="ghost_key",
        ),
        store,
    )
    assert invented.propose("user-A", "我想做 Agent").duplicate_of is None


def test_propose_reports_gate_failure_without_duplicate_hint(
    store: SqliteStore,
) -> None:
    """门控拒绝时不该再提示“与某条重复”，那会让人以为保存只是被去重挡住。"""
    save_memory(store, "user-A", "answer_style", "preference", "回答要简洁")
    payload = candidate_payload(
        memory_key="answer_style_conclusion_first",
        content="回答要简洁",
        sensitive=True,
    )
    service = build_service(payload, store)

    proposal = service.propose("user-A", "我的密码是 123456")

    assert proposal.allowed is False
    assert proposal.duplicate_of is None
    assert "敏感" in proposal.reason


def test_save_merges_into_existing_key(store: SqliteStore) -> None:
    """服务层写入必须走判重，不能退化成无脑 upsert。"""
    save_memory(
        store,
        "user-A",
        "answer_style",
        "preference",
        "回答尽量简洁，并优先解释结论",
    )
    service = build_service(candidate_payload(
        memory_key="answer_style_conclusion_first",
        content="用户偏好回答时先给结论，再解释原因。",
    ), store)
    proposal = service.propose("user-A", "用户偏好回答时先给结论，再解释原因。")

    outcome = service.save(
        "user-A",
        "answer_style_conclusion_first",
        "preference",
        "用户偏好回答时先给结论，再解释原因。",
        review_id=proposal.review_id,
        confirmed=True,
    )

    assert outcome.action == "merge"
    assert outcome.memory_key == "answer_style"
    assert list(list_memories(store, "user-A")) == ["answer_style"]


def test_context_renders_current_memories(store: SqliteStore) -> None:
    """context() 是 CLI 和接口共用的展示入口，内容要和库一致。"""
    save_memory(store, "user-A", "learning_goal", "goal", "学习 Agent 应用开发")
    service = MemoryService(None, "fake-model", store)

    context = service.context("user-A")

    assert "learning_goal（goal）" in context
    assert "学习 Agent 应用开发" in context


def test_propose_requires_model_client(store: SqliteStore) -> None:
    """没有模型客户端时给出明确错误，而不是抛 AttributeError。"""
    service = MemoryService(None, "fake-model", store)

    with pytest.raises(RuntimeError, match="没有模型客户端"):
        service.propose("user-A", "随便说点什么")
