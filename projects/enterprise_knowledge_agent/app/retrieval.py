"""建立和查询劳动法知识库的混合索引（向量召回 + BM25 关键词召回）。

为什么需要两路信号：

- 向量召回擅长“换个说法也能找到”（辞退 ↔ 解除劳动合同），但对术语不敏感，
  实测“法定节假日加班给几倍工资”的正确答案第四十四条只有 0.58 分，过不了
  0.6 的门控；
- BM25 只看字面，恰好补上这一类：第四十四条原文就有“法定休假日”，
  字符 bigram 共享“法定”“假日”，关键词通道一抓就准；
- 两路各有各的盲区，所以用融合分排序、用两条独立门控放行，而不是二选一。
"""

import json
import re
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding

from .bm25 import BM25Index
from .config import (
    KNOWLEDGE_BM25_SCALE,
    KNOWLEDGE_KEYWORD_COVERAGE,
    KNOWLEDGE_KEYWORD_WEIGHT,
    KNOWLEDGE_MIN_SCORE,
    KNOWLEDGE_TOP_K,
    KNOWLEDGE_VECTOR_WEIGHT,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
LEARNING_ROOT = PROJECT_ROOT.parent.parent
KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge"
# 文件名 → source_id 里显示的法律简称。加新法律时只需在这里补一行，
# 不写文件名映射会退化成文件主干名，读起来像 "labor_x#第一条"。
LAW_TITLES = {
    "labor_law.txt": "劳动法",
    "labor_contract_law.txt": "劳动合同法",
}
ARTICLE_PATTERN = re.compile(r"第[一二三四五六七八九十百零]+条")
INDEX_DIR = PROJECT_ROOT / "data" / "index"
VECTORS_PATH = INDEX_DIR / "document_vectors.npy"
CHUNKS_PATH = INDEX_DIR / "chunks.json"
MODEL_CACHE_DIR = LEARNING_ROOT / ".model_cache"
MODEL_NAME = "BAAI/bge-small-zh-v1.5"


def load_knowledge_chunks() -> list[dict[str, str]]:
    """按行读取 knowledge/ 下所有法律条文，每行一条。

    使用 match 只匹配行首的“第X条”，避免把条文正文中引用的其他条号
    （例如“本法第三十六条规定的工时制度”）误判成新的条文起点。

    source_id 必须带上法律简称：两部法律都有“第一条”，只用条号做标识
    会让引用校验误判——模型引“劳动合同法#第一条”时，校验会去《劳动法》
    的第一条里找原文，然后判成引用造假。
    """
    chunks: list[dict[str, str]] = []
    for path in sorted(KNOWLEDGE_DIR.glob("*.txt")):
        law_title = LAW_TITLES.get(path.name, path.stem)
        text = path.read_text(encoding="utf-8")
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            match = ARTICLE_PATTERN.match(line)
            if match is None:
                continue
            chunks.append(
                {
                    "source_id": f"{law_title}#{match.group()}",
                    "text": line,
                }
            )
    if not chunks:
        raise RuntimeError(f"知识库为空：{KNOWLEDGE_DIR} 下没有可解析的条文。")
    return chunks


def cosine_similarity(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    """计算两个向量的余弦相似度，并防止零向量造成除零错误。"""
    denominator = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    if denominator == 0:
        return 0.0
    return float(np.dot(vector_a, vector_b) / denominator)


def build_vector_index() -> tuple[int, tuple[int, ...]]:
    """把知识片段转换为向量，并将向量和片段信息保存到磁盘。

    关键词索引不需要落盘：它由条文原文在内存中重建（两百多条只需几毫秒），
    多存一份反而会和条文文件不同步。

    batch_size 必须显式限制。默认一次性喂入全部条文时，同一批内要按最长
    那条 padding，条文一多（尤其加入了篇幅更长的《劳动合同法》）张量会
    撑爆内存，ONNX 直接抛 bad allocation。16 条一批既稳又快。
    """
    chunks = load_knowledge_chunks()
    embedding_model = TextEmbedding(
        model_name=MODEL_NAME,
        cache_dir=str(MODEL_CACHE_DIR),
    )
    vectors = np.asarray(
        list(
            embedding_model.passage_embed(
                [chunk["text"] for chunk in chunks],
                batch_size=16,
            )
        ),
        dtype=np.float32,
    )

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.save(VECTORS_PATH, vectors)
    CHUNKS_PATH.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return len(chunks), vectors.shape


class HybridKnowledgeBase:
    """向量召回 + BM25 关键词召回的混合检索器。"""

    def __init__(
        self,
        chunks: list[dict[str, str]],
        vectors: np.ndarray,
        embedding_model: TextEmbedding,
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("索引损坏：知识片段数量与向量数量不一致。")
        self.chunks = chunks
        self.vectors = vectors
        self.embedding_model = embedding_model
        # 预先把文档向量归一化，查询时一次矩阵乘法就能算完所有余弦相似度。
        self.normalized_vectors = vectors / np.linalg.norm(
            vectors, axis=1, keepdims=True
        )
        self.keyword_index = BM25Index([chunk["text"] for chunk in chunks])

    @classmethod
    def from_disk(cls) -> "HybridKnowledgeBase":
        """从磁盘加载索引；索引不存在时给出明确的建立命令。"""
        if not VECTORS_PATH.exists() or not CHUNKS_PATH.exists():
            raise RuntimeError(
                "尚未建立向量索引，请先运行："
                "python -m projects.enterprise_knowledge_agent.scripts.build_index"
            )

        chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
        vectors = np.load(VECTORS_PATH)
        embedding_model = TextEmbedding(
            model_name=MODEL_NAME,
            cache_dir=str(MODEL_CACHE_DIR),
        )
        return cls(chunks, vectors, embedding_model)

    def search(
        self,
        query: str,
        top_k: int = KNOWLEDGE_TOP_K,
        min_score: float = KNOWLEDGE_MIN_SCORE,
    ) -> list[dict[str, object]]:
        """混合检索：两条门控任一通过即视为“知识库可能有答案”。

        - 向量门控：余弦相似度 >= min_score，负责“语义上像不像”；
        - 关键词门控：idf 加权覆盖率 >= KNOWLEDGE_KEYWORD_COVERAGE，
          负责“这些字有没有真的出现在法条里”。

        只保留通过门控的片段，再按两路融合分排序取前 top_k。
        """
        query_vector = next(self.embedding_model.query_embed([query]))
        normalized_query = query_vector / np.linalg.norm(query_vector)
        vector_scores = self.normalized_vectors @ normalized_query

        keyword_scores = np.asarray(self.keyword_index.score(query), dtype=np.float32)
        keyword_coverages = np.asarray(
            self.keyword_index.coverage(query), dtype=np.float32
        )
        # BM25 无固定上界，用 b/(b+K) 压到 0~1，才能和余弦相似度加权相加。
        keyword_normalized = keyword_scores / (keyword_scores + KNOWLEDGE_BM25_SCALE)
        fused_scores = (
            KNOWLEDGE_VECTOR_WEIGHT * vector_scores
            + KNOWLEDGE_KEYWORD_WEIGHT * keyword_normalized
        )

        passed = (vector_scores >= min_score) | (
            keyword_coverages >= KNOWLEDGE_KEYWORD_COVERAGE
        )

        matches = []
        for index in np.flatnonzero(passed):
            matches.append(
                {
                    **self.chunks[index],
                    "score": float(fused_scores[index]),
                    "vector_score": float(vector_scores[index]),
                    "keyword_score": float(keyword_scores[index]),
                    "keyword_coverage": float(keyword_coverages[index]),
                }
            )

        matches.sort(key=lambda item: item["score"], reverse=True)
        return matches[:top_k]


# 兼容旧引用：类名从 VectorKnowledgeBase 改为 HybridKnowledgeBase 后，
# 仍有脚本用旧名字导入，保留别名避免大面积改动。
VectorKnowledgeBase = HybridKnowledgeBase
