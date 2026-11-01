"""测试长期记忆本地门控，不调用真实模型 API。"""

from projects.enterprise_knowledge_agent.app.memory_policy import (  # 导入待测试组件。
    MemoryCandidate,
    check_memory_candidate,
)


def test_stable_preference_is_allowed() -> None:
    """明确、稳定且不敏感的用户偏好应进入人工确认阶段。"""
    candidate = MemoryCandidate(
        should_remember=True,
        memory_key="answer_style",
        category="preference",
        content="回答尽量简洁",
        sensitive=False,
        reason="这是跨会话仍有价值的稳定偏好。",
    )

    allowed, _ = check_memory_candidate(candidate, "以后回答尽量简洁。")

    assert allowed is True


def test_sensitive_content_is_rejected_even_if_model_allows_it() -> None:
    """即使模型误判为可保存，本地关键词仍应拦截密码。"""
    candidate = MemoryCandidate(
        should_remember=True,
        memory_key="account_password",
        category="profile",
        content="我的密码是 example123",
        sensitive=False,
        reason="错误的模型判断。",
    )

    allowed, reason = check_memory_candidate(candidate, "请记住我的密码是 example123。")

    assert allowed is False
    assert "敏感" in reason


def test_temporary_information_is_rejected() -> None:
    """模型已经判断为临时信息时，不应进入人工确认阶段。"""
    candidate = MemoryCandidate(
        should_remember=False,
        memory_key="",
        category="profile",
        content="",
        sensitive=False,
        reason="今天有点累属于临时状态。",
    )

    allowed, _ = check_memory_candidate(candidate, "我今天有点累。")

    assert allowed is False
