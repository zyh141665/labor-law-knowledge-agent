"""使用假检索器测试评估指标和质量门控，不加载真实模型。"""

from projects.enterprise_knowledge_agent.app.evaluation import (
    RetrievalCase,
    evaluate_retrieval,
)


class FakeSearcher:
    """根据问题返回预设结果，隔离评估公式与向量模型。"""

    def __init__(self, results: dict[str, list[dict[str, object]]]) -> None:
        self.results = results

    def search(
        self,
        query: str,
        top_k: int = 1,
        min_score: float = 0.5,
    ) -> list[dict[str, object]]:
        return self.results.get(query, [])[:top_k]


def test_evaluation_counts_hits_and_applies_gate() -> None:
    """一条命中、一条错误时准确率应为 50%，并低于 80% 门槛。"""
    cases = [
        RetrievalCase(case_id="hit", question="退款", expected_source_ids=["chunk-1"]),
        RetrievalCase(case_id="miss", question="发票", expected_source_ids=["chunk-2"]),
    ]
    searcher = FakeSearcher(
        {
            "退款": [{"source_id": "chunk-1", "score": 0.8}],
            "发票": [{"source_id": "chunk-3", "score": 0.7}],
        }
    )

    report = evaluate_retrieval(searcher, cases, minimum_accuracy=0.8)

    assert report.passed_count == 1
    assert report.top1_accuracy == 0.5
    assert report.gate_passed is False


def test_expected_no_result_counts_as_correct() -> None:
    """无关问题被门控拒绝也属于正确检索行为。"""
    cases = [
        RetrievalCase(case_id="reject", question="天气", expected_source_ids=None),
    ]

    report = evaluate_retrieval(FakeSearcher({}), cases)

    assert report.top1_accuracy == 1.0
    assert report.gate_passed is True
