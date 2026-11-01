"""定义最终回答的数据结构以及引用真实性校验。"""

from pydantic import BaseModel, Field


class Citation(BaseModel):
    """一条可追溯到知识片段的逐字引用。"""

    source_id: str = Field(min_length=1)
    quote: str = Field(min_length=1)


class StructuredAnswer(BaseModel):
    """Agent 最终回答必须满足的固定结构。"""

    answer: str = Field(min_length=1)
    refused: bool
    citations: list[Citation] = Field(max_length=3)


def validate_answer(
    answer: StructuredAnswer,
    retrieved_matches: list[dict[str, object]],
) -> bool:
    """检查来源编号和引用原文是否真实存在于本次检索结果中。

    规则分两层：

    1. **引用真实性是硬约束**，与是否拒答无关：来源必须出现在本次检索结果里，
       引文必须是该来源原文的逐字子串。这一层防的是模型编造条文。
    2. **拒答时不要求引用为空**。模型拒答时附上“我查过这几条”的条文，是合理的
       透明度表达。早期版本要求“拒答 ⇒ citations=[]”，实测会让约 1/8 的正常
       拒答被判成系统错误（模型一边说无法确定、一边列出它查过的条文），
       用户看到的是红色报错框而不是拒答答案——这既冤枉了模型，也误导了用户。
    3. **作答必须给出至少一条引用**，否则等于没有证据就下结论。
    """
    allowed_sources = {
        str(match["source_id"]): str(match["text"])
        for match in retrieved_matches
    }
    if not all(
        citation.source_id in allowed_sources
        and citation.quote in allowed_sources[citation.source_id]
        for citation in answer.citations
    ):
        return False
    if answer.refused:
        return True
    return bool(answer.citations)
