"""为企业知识库离线建立向量索引。"""

from projects.enterprise_knowledge_agent.app.retrieval import (
    CHUNKS_PATH,
    VECTORS_PATH,
    build_vector_index,
)


def main() -> None:
    """建立索引并打印便于人工检查的结果。"""
    chunk_count, vector_shape = build_vector_index()
    print(f"文档块数量：{chunk_count}")
    print(f"向量矩阵形状：{vector_shape}")
    print(f"向量文件：{VECTORS_PATH}")
    print(f"文档信息：{CHUNKS_PATH}")


if __name__ == "__main__":
    main()
