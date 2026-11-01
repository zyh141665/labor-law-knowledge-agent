"""加载固定评估集并计算知识库检索的 Top-1 准确率。"""

import json  # 读取评估数据集并生成结构化评估报告。
from pathlib import Path  # 接收跨平台的评估集文件路径。
from typing import Protocol  # 定义评估器依赖的最小检索接口。

from pydantic import BaseModel, Field  # 校验评估样本和报告结构。


class RetrievalSearcher(Protocol):
    """任何提供相同 search() 接口的真实或假检索器都可以接受评估。"""

    def search(
        self,
        query: str,
        top_k: int = 1,
        min_score: float = 0.5,
    ) -> list[dict[str, object]]: ...


class RetrievalCase(BaseModel):
    """一条由人工标注期望来源的检索评估样本。

    ``expected_source_ids`` 是**集合**而不是单值，两个原因：

    1. 一个问题常有多个同样正确的落点（“拖欠工资怎么办”可以引第五十条，
       也可以引第九十一条）；
    2. 加入《劳动合同法》后，同一个问题往往两部法律都答得出
       （试用期上限、辞职提前通知期），只认一部会把另一部判成错误。

    值为 None 表示期望检索门控拒绝所有文档。
    """

    case_id: str = Field(min_length=1)  # 稳定名称便于定位失败样本。
    question: str = Field(min_length=1)  # 输入检索器的自然语言问题。
    expected_source_ids: list[str] | None  # None 表示期望门控拒答。


class RetrievalCaseResult(BaseModel):
    """一条评估样本实际得到的结果。"""

    case_id: str
    question: str
    expected_source_ids: list[str] | None
    actual_source_id: str | None
    score: float | None
    passed: bool


class RetrievalEvaluationReport(BaseModel):
    """整批检索评估的明细和汇总指标。"""

    passed_count: int
    total_count: int
    top1_accuracy: float
    minimum_accuracy: float
    gate_passed: bool
    cases: list[RetrievalCaseResult]


def load_retrieval_cases(path: Path) -> list[RetrievalCase]:
    """从 JSON 读取评估集，并对每条人工标签执行结构校验。"""
    raw_cases = json.loads(path.read_text(encoding="utf-8"))  # 先解析为普通列表和字典。
    return [RetrievalCase.model_validate(item) for item in raw_cases]  # 再逐条校验。


def evaluate_retrieval(
    searcher: RetrievalSearcher,
    cases: list[RetrievalCase],
    min_score: float = 0.5,
    minimum_accuracy: float = 0.8,
) -> RetrievalEvaluationReport:
    """逐条执行 Top-1 检索，并判断整体准确率是否达到最低门槛。"""
    if not cases:  # 空评估集无法计算有意义的准确率。
        raise ValueError("评估集不能为空。")

    results = []  # 保存每条样本的期望结果、实际结果和是否通过。
    for case in cases:
        matches = searcher.search(  # 只取第一名，因为本节评估 Top-1 命中率。
            case.question,
            top_k=1,
            min_score=min_score,
        )
        top_match = matches[0] if matches else None  # 门控没有结果时用 None 表示拒答。
        actual_source_id = (  # 从第一名中提取来源编号。
            str(top_match["source_id"])
            if top_match is not None
            else None
        )
        score = (  # 保存融合分（向量+关键词），方便分析阈值附近的失败样本。
            float(top_match["score"])
            if top_match is not None
            else None
        )
        # 期望值为 None 时要求门控确实拒答；否则命中集合中任意一条即算通过。
        if case.expected_source_ids is None:
            passed = actual_source_id is None
        else:
            passed = actual_source_id in case.expected_source_ids
        results.append(
            RetrievalCaseResult(
                case_id=case.case_id,
                question=case.question,
                expected_source_ids=case.expected_source_ids,
                actual_source_id=actual_source_id,
                score=score,
                passed=passed,
            )
        )

    passed_count = sum(result.passed for result in results)  # True 在求和时等于 1。
    accuracy = passed_count / len(results)  # Top-1 正确样本数除以全部样本数。
    return RetrievalEvaluationReport(
        passed_count=passed_count,
        total_count=len(results),
        top1_accuracy=accuracy,
        minimum_accuracy=minimum_accuracy,
        gate_passed=accuracy >= minimum_accuracy,  # 未达到门槛时阻止错误版本发布。
        cases=results,
    )
