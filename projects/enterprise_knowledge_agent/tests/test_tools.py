"""测试知识库工具的参数传递、门控和白名单行为。"""

import pytest

from projects.enterprise_knowledge_agent.app.tools import ToolRegistry


class FakeKnowledgeBase:
    """测试替身：无需加载真实模型就能观察工具传入的参数。"""

    def __init__(self, matches: list[dict[str, object]]) -> None:
        self.matches = matches
        self.last_request: tuple[str, int, float] | None = None

    def search(
        self,
        query: str,
        top_k: int = 2,
        min_score: float = 0.5,
    ) -> list[dict[str, object]]:
        self.last_request = (query, top_k, min_score)
        return self.matches[:top_k]


def test_search_tool_passes_validated_arguments() -> None:
    knowledge_base = FakeKnowledgeBase(
        [
            {
                "source_id": "company_policy.txt#chunk-3",
                "text": "忘记密码时，可以点击重置密码。",
                "score": 0.75,
            }
        ]
    )
    registry = ToolRegistry(knowledge_base, min_score=0.6)

    result = registry.execute(
        {
            "name": "search_knowledge_base",
            "arguments": {"query": "我忘记密码了", "top_k": 1},
        }
    )

    assert result["matches"][0]["source_id"] == "company_policy.txt#chunk-3"
    assert result["gate"]["passed"] is True
    assert knowledge_base.last_request == ("我忘记密码了", 1, 0.6)


def test_empty_result_fails_retrieval_gate() -> None:
    registry = ToolRegistry(FakeKnowledgeBase([]))

    result = registry.execute(
        {
            "name": "search_knowledge_base",
            "arguments": {"query": "公司今年营业收入是多少"},
        }
    )

    assert result["matches"] == []
    assert result["gate"]["passed"] is False


def test_unknown_tool_is_rejected() -> None:
    registry = ToolRegistry(FakeKnowledgeBase([]))

    with pytest.raises(ValueError, match="未知工具"):
        registry.execute({"name": "delete_all_files", "arguments": {}})
