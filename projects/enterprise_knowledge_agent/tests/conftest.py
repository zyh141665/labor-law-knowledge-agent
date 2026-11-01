"""测试公共配置。

绝大多数测试关心的是业务行为，不是令牌校验，所以这里统一把
``require_principal`` 这个**依赖**替换成固定身份，省得每个用例都拼
Authorization 头。

注意这里替换的是依赖，不是校验函数：``auth.authenticate`` 里的令牌比较、
Bearer 解析、恒定时间对比，在 ``test_auth.py`` 里仍然被完整执行。
换句话说，测试没有把认证逻辑删掉，只是给它换了一条入口。
"""

import pytest

from projects.enterprise_knowledge_agent.app.api import (
    app,
    get_memory_reviews,
    require_principal,
)
from projects.enterprise_knowledge_agent.app.auth import Principal
from projects.enterprise_knowledge_agent.app.memory_review import MemoryReviews


TEST_USER_ID = "student-001"  # 与 test_api.py 各处使用的身份保持一致。


@pytest.fixture(autouse=True)
def authenticated_as_default_user(review_records):
    """默认以 student-001 身份通过认证；用例结束后清理覆盖。"""
    app.dependency_overrides[require_principal] = lambda: Principal(
        user_id=TEST_USER_ID,
        token_label="…test",
    )
    app.dependency_overrides[get_memory_reviews] = lambda: review_records
    yield
    app.dependency_overrides.pop(require_principal, None)
    app.dependency_overrides.pop(get_memory_reviews, None)


@pytest.fixture
def review_records():
    """每个测试一份审核表，避免凭证跨测试复用。"""
    return MemoryReviews()
