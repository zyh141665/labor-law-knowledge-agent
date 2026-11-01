"""知识库工具的参数校验、白名单注册和安全执行入口。"""

from typing import Protocol

from pydantic import BaseModel, Field

from .config import KNOWLEDGE_MIN_SCORE, KNOWLEDGE_TOP_K


class KnowledgeBase(Protocol):
    """规定检索器必须提供的最小接口，便于测试时替换为假检索器。"""

    def search(
        self,
        query: str,
        top_k: int = 2,
        min_score: float = KNOWLEDGE_MIN_SCORE,
    ) -> list[dict[str, object]]: ...


class SearchKnowledgeArguments(BaseModel):
    """知识库工具允许模型提供的参数。"""

    query: str = Field(min_length=1)
    # 上限放宽到 5，让模型在追问细节时也可以自己多取几条；
    # 默认值跟随 KNOWLEDGE_TOP_K（4），实测能显著提升“正确条文进入证据”的比例。
    top_k: int = Field(default=KNOWLEDGE_TOP_K, ge=1, le=5)


class ToolCall(BaseModel):
    """模型返回的一次工具调用。"""

    name: str
    arguments: dict[str, object]


class ToolRegistry:
    """集中保存允许模型调用的工具，并注入真实检索器。"""

    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        min_score: float = KNOWLEDGE_MIN_SCORE,
    ) -> None:
        self.knowledge_base = knowledge_base
        self.min_score = min_score

    def search_knowledge_base(self, raw_arguments: dict) -> dict[str, object]:
        """校验模型参数，执行混合检索，并返回检索门控信息。"""
        arguments = SearchKnowledgeArguments.model_validate(raw_arguments)
        matches = self.knowledge_base.search(
            arguments.query,
            top_k=arguments.top_k,
            min_score=self.min_score,
        )
        return {
            "matches": matches,
            "gate": {
                "passed": bool(matches),
                "min_score": self.min_score,
            },
        }

    def execute(self, raw_tool_call: dict) -> dict[str, object]:
        """只执行显式注册在白名单中的工具。"""
        tool_call = ToolCall.model_validate(raw_tool_call)
        executors = {
            "search_knowledge_base": self.search_knowledge_base,
        }
        executor = executors.get(tool_call.name)
        if executor is None:
            raise ValueError(f"未知工具：{tool_call.name}")
        return executor(tool_call.arguments)

    @staticmethod
    def build_definitions() -> list[dict]:
        """返回给模型阅读的工具名称、用途和参数格式。"""
        return [
            {
                "type": "function",
                "name": "search_knowledge_base",
                "description": (
                    "检索劳动法条文知识库（语义 + 关键词混合检索）。"
                    "知识库收录《中华人民共和国劳动法》和《中华人民共和国劳动合同法》。"
                    "涉及这两部法律的问题必须调用；"
                    "回答只能使用返回的文本，并逐字引用原文；"
                    "source_id 必须照抄返回结果，不得改写法律名称或条号。"
                ),
                "parameters": SearchKnowledgeArguments.model_json_schema(),
            }
        ]
