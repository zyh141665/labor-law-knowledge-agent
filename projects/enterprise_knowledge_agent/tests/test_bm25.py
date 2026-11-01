"""测试 BM25 分词、打分和关键词覆盖率的行为。"""

import pytest

from projects.enterprise_knowledge_agent.app.bm25 import BM25Index, tokenize


def test_chinese_is_split_into_character_bigrams() -> None:
    tokens = tokenize("用人单位")

    assert tokens == ["用人", "人单", "单位"]


def test_question_words_are_removed_before_tokenizing() -> None:
    # “怎么”是疑问词，不是检索内容；如果保留，它会被算成法条中不存在的
    # 稀有词，把覆盖率的分母撑大，导致正确条文被误判成不相关。
    tokens = tokenize("加班工资怎么算")

    assert "怎么" not in tokens
    assert "加班" in tokens
    assert "工资" in tokens


def test_single_character_run_is_kept() -> None:
    assert tokenize("法") == ["法"]


def test_bm25_ranks_literal_match_above_unrelated_document() -> None:
    index = BM25Index(
        [
            "法定休假日安排劳动者工作的，支付不低于工资的百分之三百的工资报酬。",
            "劳动者在法定休假日和婚丧假期间，用人单位应当依法支付工资。",
            "国家发展社会保险事业，建立社会保险制度。",
        ]
    )

    scores = index.score("加班工资怎么算")

    assert scores[0] > scores[2]
    assert scores[1] > scores[2]


def test_coverage_is_low_for_unrelated_query() -> None:
    index = BM25Index(
        [
            "法定休假日安排劳动者工作的，支付不低于工资的百分之三百的工资报酬。",
            "国家发展社会保险事业，建立社会保险制度。",
        ]
    )

    related = index.coverage("法定休假日加班")
    unrelated = index.coverage("公司年会抽奖奖品发放")

    assert related[0] > unrelated[0]
    assert unrelated[0] < 0.1


def test_empty_query_scores_zero_instead_of_raising() -> None:
    index = BM25Index(["用人单位应当依法支付工资。"])

    assert index.score("  ") == [0.0]
    assert index.coverage("  ") == [0.0]


def test_empty_document_list_is_rejected() -> None:
    with pytest.raises(ValueError, match="至少需要一篇文档"):
        BM25Index([])
