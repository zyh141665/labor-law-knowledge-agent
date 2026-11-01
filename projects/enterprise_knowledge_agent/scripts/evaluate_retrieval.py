"""运行真实向量检索评估，打印结果并保存 JSON 报告。"""

from pathlib import Path  # 构造评估数据和报告的项目内路径。

from projects.enterprise_knowledge_agent.app.config import KNOWLEDGE_MIN_SCORE  # 复用生产检索阈值。
from projects.enterprise_knowledge_agent.app.evaluation import (  # 复用评估核心逻辑。
    evaluate_retrieval,
    load_retrieval_cases,
)
from projects.enterprise_knowledge_agent.app.retrieval import VectorKnowledgeBase  # 加载真实索引。


PROJECT_ROOT = Path(__file__).resolve().parent.parent  # 得到企业知识库 Agent 根目录。
CASES_PATH = PROJECT_ROOT / "evals" / "retrieval_cases.json"  # 固定人工评估集。
REPORT_PATH = PROJECT_ROOT / "reports" / "retrieval_evaluation.json"  # 最新评估报告。


def main() -> None:
    """执行评估，并使用进程退出码表示是否达到质量门槛。"""
    cases = load_retrieval_cases(CASES_PATH)  # 加载并校验人工标注样本。
    knowledge_base = VectorKnowledgeBase.from_disk()  # 复用项目真实 Embedding 和索引。
    report = evaluate_retrieval(  # 当前要求 Top-1 准确率至少达到 80%。
        knowledge_base,
        cases,
        min_score=KNOWLEDGE_MIN_SCORE,
        minimum_accuracy=0.8,
    )

    for result in report.cases:  # 输出每条结果，失败时可以直接定位 case_id。
        status = "通过" if result.passed else "失败"
        score_text = f"{result.score:.4f}" if result.score is not None else "None"
        print(f"{result.case_id}：{status}")
        expected = result.expected_source_ids
        print(f"  期望来源：{'/'.join(expected) if expected else '（应拒答）'}")
        print(f"  实际来源：{result.actual_source_id}")
        print(f"  融合分：{score_text}")

    print(f"\nTop-1 准确率：{report.top1_accuracy:.0%}")
    print(f"质量门槛：{report.minimum_accuracy:.0%}")
    print(f"评估门控：{'通过' if report.gate_passed else '失败'}")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)  # 首次运行时建立报告目录。
    REPORT_PATH.write_text(  # 保存机器可读报告，便于 CI 或后续对比。
        report.model_dump_json(indent=2),
        encoding="utf-8",
    )
    print(f"评估报告：{REPORT_PATH}")

    if not report.gate_passed:  # 非零退出码让自动化流程明确知道质量不达标。
        raise SystemExit(1)


if __name__ == "__main__":  # 使用 python -m 运行本模块时才开始评估。
    main()
