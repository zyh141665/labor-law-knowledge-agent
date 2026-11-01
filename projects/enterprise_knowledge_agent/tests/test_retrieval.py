"""测试知识切分、余弦相似度和混合检索的基础行为。"""

from collections import Counter

import numpy as np
import pytest

from projects.enterprise_knowledge_agent.app.retrieval import (
    HybridKnowledgeBase,
    cosine_similarity,
    load_knowledge_chunks,
)


def test_knowledge_file_is_split_into_stable_chunks() -> None:
    """两部法律各自条数正确，且 source_id 唯一。"""
    chunks = load_knowledge_chunks()
    by_law = Counter(chunk["source_id"].split("#")[0] for chunk in chunks)

    assert by_law == {"劳动法": 107, "劳动合同法": 98}
    # 两部法律都有“第一条”，source_id 必须带法律名才不会互相覆盖。
    # 引用校验就是拿 source_id 去原文里找逐字片段的，一旦撞号，
    # 模型引“劳动合同法#第一条”会被拿去《劳动法》里核对，判成引用造假。
    assert len({chunk["source_id"] for chunk in chunks}) == len(chunks)
    assert all("#第" in chunk["source_id"] for chunk in chunks)


def test_chunks_from_both_laws_are_loaded() -> None:
    """抽查两法各自的关键条文都进了索引。"""
    chunks = {chunk["source_id"]: chunk["text"] for chunk in load_knowledge_chunks()}

    assert "取得劳动报酬的权利" in chunks["劳动法#第三条"]
    assert "二倍的工资" in chunks["劳动合同法#第八十二条"]


def test_cosine_similarity_of_same_direction_is_one() -> None:
    score = cosine_similarity(
        np.array([1.0, 2.0]),
        np.array([2.0, 4.0]),
    )

    assert score == pytest.approx(1.0)


class FakeEmbeddingModel:
    """测试替身：查询向量固定，避免单元测试加载真实 Embedding 模型。"""

    def query_embed(self, queries: list[str]):
        for _ in queries:
            yield np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


def test_keyword_channel_rescues_answer_when_vector_score_fails() -> None:
    """向量完全不看好时，关键词通道仍应把正确条文召回。

    这正是加 BM25 的目的：真实数据上“法定节假日加班给几倍工资”的正确答案
    第四十四条向量分只有 0.58，过不了 0.6 的门控，但它和问题共享“法定”
    “假日”“工资”这些字面 token。
    """
    chunks = load_knowledge_chunks()
    vectors = np.zeros((len(chunks), 4), dtype=np.float32)
    vectors[:, 1] = 1.0  # 除下面一条外，所有条文与查询向量正交，向量分都是 0。

    decoy = next(
        index
        for index, chunk in enumerate(chunks)
        if chunk["source_id"] == "劳动法#第五十一条"
    )
    vectors[decoy] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # 余弦相似度 1.0

    knowledge_base = HybridKnowledgeBase(chunks, vectors, FakeEmbeddingModel())
    matches = knowledge_base.search("法定节假日加班给几倍工资", top_k=4)

    sources = [match["source_id"] for match in matches]
    assert "劳动法#第四十四条" in sources

    rescued = next(match for match in matches if match["source_id"] == "劳动法#第四十四条")
    assert rescued["vector_score"] < 0.6  # 确认它是靠关键词通道进来的
    assert rescued["keyword_coverage"] >= 0.15


def test_search_returns_empty_when_both_channels_reject() -> None:
    """两条门控都不通过时应返回空列表，交给上层拒答。"""
    chunks = load_knowledge_chunks()
    vectors = np.zeros((len(chunks), 4), dtype=np.float32)
    vectors[:, 1] = 1.0  # 与查询向量正交，向量通道全挡。

    knowledge_base = HybridKnowledgeBase(chunks, vectors, FakeEmbeddingModel())
    matches = knowledge_base.search("公司年会抽奖的奖品怎么发放", top_k=4)

    assert matches == []
