"""验证提取与保存之间的信任边界，模型、网络与业务数据库都使用替身。"""

import json
import sqlite3
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from langgraph.store.sqlite import SqliteStore

from projects.enterprise_knowledge_agent.app.api import app, get_memory_service
from projects.enterprise_knowledge_agent.app.memory_service import MemoryService
from projects.enterprise_knowledge_agent.app import memory_review


@pytest.fixture
def memory_api(review_records):
    """两次 HTTP 请求共享审核表，存储只存在于测试进程内。"""
    connection = sqlite3.connect(":memory:", check_same_thread=False)
    store = SqliteStore(connection)
    store.setup()
    connection.commit()
    payload = {
        "should_remember": True,
        "memory_key": "answer_style",
        "category": "preference",
        "content": "回答先给结论",
        "sensitive": False,
        "reason": "用户明确偏好",
    }
    fake_client = SimpleNamespace(
        responses=SimpleNamespace(
            create=lambda **kwargs: SimpleNamespace(output_text=json.dumps(payload)),
        )
    )

    def service_factory():
        return MemoryService(fake_client, "fake", store, reviews=review_records)

    app.dependency_overrides[get_memory_service] = service_factory
    try:
        yield TestClient(app), service_factory, payload
    finally:
        app.dependency_overrides.pop(get_memory_service, None)
        connection.close()


def extract_and_confirm_payload(client):
    response = client.post(
        "/memories/extract",
        json={
            "user_id": "student-001",
            "text": "我希望回答先给结论",
        },
    )
    assert response.status_code == 200
    proposal = response.json()
    assert proposal["allowed"] is True
    assert proposal["review_id"]
    candidate = proposal["candidate"]
    return {
        "user_id": "student-001",
        "memory_key": candidate["memory_key"],
        "category": candidate["category"],
        "content": candidate["content"],
        "review_id": proposal["review_id"],
        "confirmed": True,
    }


def test_extract_confirm_and_replay(memory_api):
    client, service_factory, _ = memory_api
    body = extract_and_confirm_payload(client)

    saved = client.post("/memories", json=body)
    replayed = client.post("/memories", json=body)

    assert saved.status_code == 200
    assert replayed.status_code == 400
    assert replayed.json()["detail"]["code"] == "memory_review_rejected"
    assert len(service_factory().list("student-001")) == 1


@pytest.mark.parametrize(
    "changed",
    [
        {"review_id": "invented-id"},
        {"confirmed": False},
        {"content": "我的密码是 secret"},
        {"memory_key": "other_key"},
        {"category": "goal"},
    ],
)
def test_unreviewed_or_modified_writes_are_rejected(memory_api, changed):
    client, service_factory, _ = memory_api
    body = extract_and_confirm_payload(client)
    body.update(changed)

    response = client.post("/memories", json=body)

    assert response.status_code == 400
    assert service_factory().list("student-001") == {}


def test_missing_review_is_not_a_legacy_bypass(memory_api):
    client, service_factory, _ = memory_api
    response = client.post(
        "/memories",
        json={
            "user_id": "student-001",
            "memory_key": "answer_style",
            "category": "preference",
            "content": "回答先给结论",
        },
    )
    assert response.status_code == 422
    assert service_factory().list("student-001") == {}


def test_confirmation_must_be_a_boolean_not_a_truthy_string(memory_api):
    client, service_factory, _ = memory_api
    body = extract_and_confirm_payload(client)
    body["confirmed"] = "true"

    assert client.post("/memories", json=body).status_code == 422
    assert service_factory().list("student-001") == {}


def test_database_failure_keeps_review_available_for_retry(memory_api, monkeypatch):
    """模拟写入前失败，凭证不能提前消耗导致正常请求无法重试。"""
    from projects.enterprise_knowledge_agent.app import memory_service

    client, service_factory, _ = memory_api
    body = extract_and_confirm_payload(client)
    real_save = memory_service.save_memory_deduped

    def fail_before_write(*args):
        raise RuntimeError("模拟数据库暂不可用")

    monkeypatch.setattr(memory_service, "save_memory_deduped", fail_before_write)
    with pytest.raises(RuntimeError, match="模拟数据库"):
        client.post("/memories", json=body)
    assert service_factory().list("student-001") == {}
    monkeypatch.setattr(memory_service, "save_memory_deduped", real_save)

    assert client.post("/memories", json=body).status_code == 200


def test_review_can_only_be_consumed_by_one_concurrent_confirmation():
    """两个线程抢同一个编号，只有一个能进入写入区间。"""
    from concurrent.futures import ThreadPoolExecutor
    from projects.enterprise_knowledge_agent.app.memory_policy import MemoryCandidate

    reviews = memory_review.MemoryReviews()
    candidate = MemoryCandidate(
        should_remember=True,
        memory_key="answer_style",
        category="preference",
        content="回答先给结论",
        sensitive=False,
        reason="用户偏好",
    )
    review_id = reviews.issue("u1", candidate, candidate.content)

    def confirm_once(_):
        try:
            with reviews.confirm(
                review_id, "u1", "answer_style", "preference", candidate.content, True
            ):
                return True
        except memory_review.MemoryReviewError:
            return False

    with ThreadPoolExecutor(max_workers=2) as executor:
        assert sorted(executor.map(confirm_once, range(2))) == [False, True]


def test_sensitive_original_cannot_get_a_review_id(memory_api):
    client, service_factory, _ = memory_api
    response = client.post(
        "/memories/extract",
        json={
            "user_id": "student-001",
            "text": "我的密码是 123456，也希望回答先给结论",
        },
    )
    assert response.json()["allowed"] is False
    assert response.json()["review_id"] is None
    assert service_factory().list("student-001") == {}


def test_expired_review_is_rejected(memory_api, monkeypatch):
    client, service_factory, _ = memory_api
    monkeypatch.setattr(memory_review, "monotonic", lambda: 100.0)
    body = extract_and_confirm_payload(client)
    monkeypatch.setattr(memory_review, "monotonic", lambda: 701.0)

    response = client.post("/memories", json=body)

    assert response.status_code == 400
    assert "过期" in response.json()["detail"]["message"]
    assert service_factory().list("student-001") == {}


def test_review_is_bound_to_user_even_for_internal_calls(memory_api):
    client, service_factory, _ = memory_api
    body = extract_and_confirm_payload(client)
    with pytest.raises(memory_review.MemoryReviewError):
        service_factory().save(
            "other-user",
            body["memory_key"],
            body["category"],
            body["content"],
            review_id=body["review_id"],
            confirmed=True,
        )
    assert service_factory().list("other-user") == {}


def test_write_rechecks_policy(memory_api, monkeypatch):
    """候选签发之后规则也可能更新，因此最终写入不能只信旧的 allowed。"""
    from projects.enterprise_knowledge_agent.app import memory_service

    client, service_factory, _ = memory_api
    body = extract_and_confirm_payload(client)
    monkeypatch.setattr(
        memory_service, "check_memory_candidate", lambda *args: (False, "规则已更新")
    )

    assert client.post("/memories", json=body).status_code == 400
    assert service_factory().list("student-001") == {}
