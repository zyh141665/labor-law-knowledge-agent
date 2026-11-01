"""使用假 Agent 服务测试生成评估公式，不调用 DeepSeek。"""

from projects.enterprise_knowledge_agent.app.generation_evaluation import (
    GenerationCase,
    evaluate_generation,
)
from projects.enterprise_knowledge_agent.app.schemas import Citation
from projects.enterprise_knowledge_agent.app.service import AgentTurnResult


class FakeAnswerService:
    """按照问题返回预设结构化答案。"""

    def __init__(self, answers: dict[str, AgentTurnResult]) -> None:
        self.answers = answers

    def answer(
        self,
        user_id: str,
        thread_id: str,
        question: str,
    ) -> AgentTurnResult:
        return self.answers[question]


def test_generation_evaluation_reports_independent_failures() -> None:
    """来源正确但缺少关键事实时，整条样本仍应失败。"""
    case = GenerationCase(
        case_id="refund",
        question="退款期限？",
        expected_refused=False,
        expected_source_ids=["chunk-1"],
        required_facts=["7 天", "不予受理"],
    )
    service = FakeAnswerService(
        {
            "退款期限？": AgentTurnResult(
                answer="退款期限为 7 天。",
                refused=False,
                citations=[Citation(source_id="chunk-1", quote="退款期限为 7 天。")],
                round_number=2,
            )
        }
    )

    report = evaluate_generation(service, [case], run_id="test")

    result = report.cases[0]
    assert result.refusal_correct is True
    assert result.source_correct is True
    assert result.facts_correct is False
    assert result.passed is False
    assert report.gate_passed is False


def test_correct_refusal_passes_without_citations() -> None:
    """知识库外问题正确拒答且没有引用时应通过。"""
    case = GenerationCase(
        case_id="unsupported",
        question="今年营收？",
        expected_refused=True,
        expected_source_ids=None,
        required_facts=[],
    )
    service = FakeAnswerService(
        {
            "今年营收？": AgentTurnResult(
                answer="根据现有资料无法确定。",
                refused=True,
                citations=[],
                round_number=2,
            )
        }
    )

    report = evaluate_generation(service, [case], run_id="test")

    assert report.pass_rate == 1.0
    assert report.gate_passed is True


def test_refusal_may_include_verified_but_insufficient_evidence() -> None:
    """正式服务已校验真实引文，评估不能再把带引用的拒答判错。"""
    case = GenerationCase(
        case_id="unsupported", question="资料没写的具体标准？", expected_refused=True,
        expected_source_ids=None, required_facts=[],
    )
    service = FakeAnswerService({case.question: AgentTurnResult(
        answer="条文提到这件事，但不足以确定具体标准。", refused=True,
        citations=[Citation(source_id="real-source", quote="相关条文")], round_number=2,
    )})

    report = evaluate_generation(service, [case], run_id="test")

    assert report.cases[0].source_correct is True
    assert report.pass_rate == 1.0
