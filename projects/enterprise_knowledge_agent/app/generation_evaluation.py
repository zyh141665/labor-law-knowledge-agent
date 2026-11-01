"""评估 Agent 最终回答的拒答、来源和关键事实是否符合人工标注。"""

import json  # 从 JSON 文件读取人工标注的生成评估集。
from pathlib import Path  # 接收跨平台的评估集文件路径。
from typing import Protocol  # 让评估器可以使用真实服务或测试假服务。

from pydantic import BaseModel, Field  # 校验样本、明细和汇总报告结构。

from .service import AgentTurnResult  # 复用正式业务层的结构化回答格式。


class AnswerService(Protocol):
    """生成评估只依赖 answer()，无需了解 Agent 内部实现。"""

    def answer(
        self,
        user_id: str,
        thread_id: str,
        question: str,
    ) -> AgentTurnResult: ...


class GenerationCase(BaseModel):
    """一条人工标注的端到端回答评估样本。"""

    case_id: str = Field(min_length=1)  # 稳定 ID 用于定位具体失败样本。
    question: str = Field(min_length=1)  # 真正交给 Agent 的用户问题。
    expected_refused: bool  # 证据不足时应为 True。
    # 可接受来源集合：两部法律都答得出的问题（试用期上限、辞职通知期）
    # 不能只认其中一部，否则正确答案会被判成失败。
    expected_source_ids: list[str] | None
    required_facts: list[str]  # 最终答案必须包含的关键事实片段。


class GenerationCaseResult(BaseModel):
    """单条问题的真实回答以及三个独立判定结果。"""

    case_id: str
    question: str
    answer: str
    expected_refused: bool
    actual_refused: bool
    refusal_correct: bool
    source_correct: bool
    facts_correct: bool
    passed: bool


class GenerationEvaluationReport(BaseModel):
    """整批生成评估的明细、通过率和发布门控结论。"""

    passed_count: int
    total_count: int
    pass_rate: float
    minimum_pass_rate: float
    gate_passed: bool
    cases: list[GenerationCaseResult]


def load_generation_cases(path: Path) -> list[GenerationCase]:
    """读取 JSON 评估集，并用 Pydantic 逐条检查人工标签。"""
    raw_cases = json.loads(path.read_text(encoding="utf-8"))
    return [GenerationCase.model_validate(item) for item in raw_cases]


def evaluate_generation(
    service: AnswerService,
    cases: list[GenerationCase],
    run_id: str,
    minimum_pass_rate: float = 0.8,
) -> GenerationEvaluationReport:
    """逐条调用 Agent，并分别检查拒答、来源和关键事实。"""
    if not cases:  # 没有样本时不能产生有意义的通过率。
        raise ValueError("生成评估集不能为空。")

    results = []  # 收集每条样本的三个子判定和最终结论。
    for case in cases:
        result = service.answer(  # 每条样本使用独立 thread，避免对话历史相互污染。
            user_id="generation-eval-user",
            thread_id=f"{run_id}-{case.case_id}",
            question=case.question,
        )
        actual_source_ids = {  # 将引用来源转换成集合，方便判断期望来源是否出现。
            citation.source_id
            for citation in result.citations
        }
        refusal_correct = result.refused == case.expected_refused
        # 拒答可以附真实的不足证据；回答样本需命中集合里至少一条。
        if case.expected_source_ids is None:
            source_correct = result.refused  # 引文真实性由正式 Agent 的校验节点负责。
        else:
            source_correct = bool(actual_source_ids & set(case.expected_source_ids))
        facts_correct = all(  # 所有人工标注的关键事实都必须出现在回答正文中。
            fact in result.answer
            for fact in case.required_facts
        )
        passed = refusal_correct and source_correct and facts_correct
        results.append(
            GenerationCaseResult(
                case_id=case.case_id,
                question=case.question,
                answer=result.answer,
                expected_refused=case.expected_refused,
                actual_refused=result.refused,
                refusal_correct=refusal_correct,
                source_correct=source_correct,
                facts_correct=facts_correct,
                passed=passed,
            )
        )

    passed_count = sum(result.passed for result in results)  # True 在求和时等于 1。
    pass_rate = passed_count / len(results)  # 端到端通过数除以全部样本数。
    return GenerationEvaluationReport(
        passed_count=passed_count,
        total_count=len(results),
        pass_rate=pass_rate,
        minimum_pass_rate=minimum_pass_rate,
        gate_passed=pass_rate >= minimum_pass_rate,
        cases=results,
    )
