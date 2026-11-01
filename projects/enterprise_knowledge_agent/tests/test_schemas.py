"""测试结构化回答的引用真实性校验规则。"""

from projects.enterprise_knowledge_agent.app.schemas import (
    StructuredAnswer,
    validate_answer,
)


MATCHES = [
    {
        "source_id": "劳动法#第四十四条",
        "text": (
            "第四十四条　有下列情形之一的，用人单位应当按照下列标准支付高于劳动者正常工作时间工资的工资报酬："
            "（一）安排劳动者延长工作时间的，支付不低于工资的百分之一百五十的工资报酬；"
            "（二）休息日安排劳动者工作又不能安排补休的，支付不低于工资的百分之二百的工资报酬；"
            "（三）法定休假日安排劳动者工作的，支付不低于工资的百分之三百的工资报酬。"
        ),
        "score": 0.78,
    }
]

QUOTE = "（三）法定休假日安排劳动者工作的，支付不低于工资的百分之三百的工资报酬。"


def test_real_quote_passes_validation() -> None:
    answer = StructuredAnswer.model_validate(
        {
            "answer": "法定休假日加班应支付不低于工资的百分之三百。",
            "refused": False,
            "citations": [{"source_id": "劳动法#第四十四条", "quote": QUOTE}],
        }
    )

    assert validate_answer(answer, MATCHES) is True


def test_fake_quote_fails_validation() -> None:
    answer = StructuredAnswer.model_validate(
        {
            "answer": "法定休假日加班应支付五倍工资。",
            "refused": False,
            "citations": [
                {"source_id": "劳动法#第四十四条", "quote": "支付不低于工资的百分之五百。"}
            ],
        }
    )

    assert validate_answer(answer, MATCHES) is False


def test_unknown_source_id_fails_validation() -> None:
    answer = StructuredAnswer.model_validate(
        {
            "answer": "法定休假日加班应支付百分之三百。",
            "refused": False,
            "citations": [{"source_id": "劳动法#第一百零八条", "quote": QUOTE}],
        }
    )

    assert validate_answer(answer, MATCHES) is False


def test_refusal_without_citations_passes_validation() -> None:
    answer = StructuredAnswer(
        answer="根据现有资料无法确定。",
        refused=True,
        citations=[],
    )

    assert validate_answer(answer, []) is True


def test_refusal_with_real_citations_passes_validation() -> None:
    """拒答时附上“查过但不相关”的条文属于合理表达，不应被判为失败。

    早期要求“拒答 ⇒ citations=[]”，实测约 1/8 的正常拒答会因此被判成系统错误，
    用户看到的是红色报错框而不是拒答答案。
    """
    answer = StructuredAnswer.model_validate(
        {
            "answer": "第二十九条只规定不得解除劳动合同的情形，没有涉及医疗费由谁承担。",
            "refused": True,
            "citations": [{"source_id": "劳动法#第四十四条", "quote": QUOTE}],
        }
    )

    assert validate_answer(answer, MATCHES) is True


def test_refusal_with_fake_citation_still_fails() -> None:
    """拒答时放宽的是“能否带引用”，不是“引用可以不真实”。"""
    answer = StructuredAnswer.model_validate(
        {
            "answer": "无法确定。",
            "refused": True,
            "citations": [{"source_id": "劳动法#第四十四条", "quote": "编造的引文。"}],
        }
    )

    assert validate_answer(answer, MATCHES) is False


def test_non_refusal_without_citations_fails_validation() -> None:
    answer = StructuredAnswer(answer="法定休假日加班应支付百分之三百。", refused=False, citations=[])

    assert validate_answer(answer, MATCHES) is False
