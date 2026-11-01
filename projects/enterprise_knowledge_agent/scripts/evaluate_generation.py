"""使用真实 DeepSeek、RAG 和校验流程运行项目级生成评估。"""

from pathlib import Path  # 构造评估集和报告路径。
from uuid import uuid4  # 为每次评估生成全新的 thread 前缀。

from langgraph.checkpoint.memory import InMemorySaver  # 评估状态无需污染正式 SQLite。
from langgraph.store.memory import InMemoryStore  # 评估用户不需要正式长期记忆。
from openai import OpenAI, OpenAIError  # 创建 DeepSeek 客户端并捕获上游错误。
from pydantic import ValidationError  # 捕获结构化输出和本地校验错误。

from projects.enterprise_knowledge_agent.app.config import (  # 读取 DeepSeek 配置与统一检索阈值。
    KNOWLEDGE_MIN_SCORE,
    load_settings,
)
from projects.enterprise_knowledge_agent.app.generation_evaluation import (  # 复用评估逻辑。
    evaluate_generation,
    load_generation_cases,
)
from projects.enterprise_knowledge_agent.app.retrieval import VectorKnowledgeBase  # 加载真实索引。
from projects.enterprise_knowledge_agent.app.service import AgentService  # 运行正式业务流程。
from projects.enterprise_knowledge_agent.app.tools import ToolRegistry  # 注册真实知识库工具。


PROJECT_ROOT = Path(__file__).resolve().parent.parent  # 得到企业知识库 Agent 根目录。
CASES_PATH = PROJECT_ROOT / "evals" / "generation_cases.json"  # 人工标注的回答评估集。
REPORT_PATH = PROJECT_ROOT / "reports" / "generation_evaluation.json"  # 最新评估报告。


def main() -> None:
    """运行五条真实 Agent 问答，并根据 80% 门槛返回退出码。"""
    try:
        settings = load_settings()  # API Key 缺失时给出明确配置错误。
        knowledge_base = VectorKnowledgeBase.from_disk()  # 加载真实向量检索组件。
    except RuntimeError as error:
        print(f"评估启动失败：{error}")
        return

    client = OpenAI(api_key=settings.api_key, base_url=settings.base_url)
    registry = ToolRegistry(knowledge_base, min_score=KNOWLEDGE_MIN_SCORE)
    service = AgentService(  # 使用内存状态运行，评估结束后自动丢弃对话。
        client,
        settings.model,
        registry,
        InMemorySaver(),
        InMemoryStore(),
    )
    cases = load_generation_cases(CASES_PATH)

    try:
        report = evaluate_generation(  # 五条样本通常会产生约十次模型调用。
            service,
            cases,
            run_id=uuid4().hex[:8],
            minimum_pass_rate=0.8,
        )
    except (OpenAIError, RuntimeError, ValueError, ValidationError) as error:
        print(f"评估执行失败：{error}")
        raise SystemExit(1) from error

    for result in report.cases:  # 展示三个子指标，便于判断失败来自哪一层。
        print(f"{result.case_id}：{'通过' if result.passed else '失败'}")
        print(f"  回答：{result.answer}")
        print(f"  拒答判断：{result.refusal_correct}")
        print(f"  来源判断：{result.source_correct}")
        print(f"  关键事实判断：{result.facts_correct}")

    print(f"\n端到端通过率：{report.pass_rate:.0%}")
    print(f"质量门槛：{report.minimum_pass_rate:.0%}")
    print(f"评估门控：{'通过' if report.gate_passed else '失败'}")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(  # 保存完整答案和子指标，供后续版本对比。
        report.model_dump_json(indent=2),
        encoding="utf-8",
    )
    print(f"评估报告：{REPORT_PATH}")

    if not report.gate_passed:  # 低于门槛时用非零退出码阻止发布。
        raise SystemExit(1)


if __name__ == "__main__":  # 使用 python -m 运行时才发起真实模型调用。
    main()
