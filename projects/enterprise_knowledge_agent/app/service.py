"""封装一次企业知识库 Agent 对话所需的业务流程。"""

import json  # 将 user_id 和 thread_id 组合成不会混淆的检查点标识。
from collections.abc import Iterator  # 标注流式问答生成器的返回类型。
from time import perf_counter  # 使用单调时钟统计一次 Agent 回合的耗时。
from typing import Any, Protocol  # 定义 HTTP 层依赖的最小服务接口，方便测试替换。

from langgraph.checkpoint.base import BaseCheckpointSaver  # 持久化短期会话状态。
from langgraph.store.base import BaseStore  # 读取跨会话的用户长期记忆。
from openai import OpenAI  # 调用 DeepSeek Responses API。
from pydantic import BaseModel  # 定义服务层返回的固定数据结构。

from .chat_history import ChatHistory, safe_record_turn  # 把每轮问答投影成可回看的日志。
from .graph import build_agent_graph  # 构建模型、工具和校验节点组成的图。
from .logging_config import get_logger  # 记录不含问题正文的业务运行指标。
from .memory import build_memory_context, list_memories  # 读取并格式化当前用户记忆。
from .schemas import Citation, StructuredAnswer  # 复用已验证的回答和引用结构。
from .session import build_turn_state  # 恢复短期历史并加入本轮问题。
from .tools import ToolRegistry  # 提供经过白名单约束的知识库工具。


logger = get_logger("service")  # 日志名称会显示为 enterprise_agent.service。


class AgentTurnResult(BaseModel):
    """业务层完成一次对话后返回给 HTTP 层的数据。"""

    answer: str  # 已经通过结构与引用校验的自然语言答案。
    refused: bool  # 证据不足时为 True。
    citations: list[Citation]  # 本轮答案使用的真实来源和逐字引文。
    round_number: int  # 本轮内部调用模型的次数。


class AgentServiceProtocol(Protocol):
    """FastAPI 只依赖这个接口，测试可以注入不调用模型的假服务。"""

    def answer(
        self,
        user_id: str,
        thread_id: str,
        question: str,
    ) -> AgentTurnResult: ...

    def answer_stream(
        self,
        user_id: str,
        thread_id: str,
        question: str,
    ) -> Iterator[dict[str, Any]]: ...

    def delete_thread(self, user_id: str, thread_id: str) -> int: ...


def build_checkpoint_thread_id(user_id: str, thread_id: str) -> str:
    """把用户和会话共同编码，防止不同用户使用相同 thread_id 时串会话。"""
    return json.dumps(  # JSON 数组能保留两个字段的边界，不使用容易冲突的字符串拼接。
        [user_id, thread_id],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def parse_checkpoint_thread_id(raw: str) -> tuple[str, str]:
    """把检查点标识还原成 user_id 和 thread_id。

    早期版本只用 thread_id 作键，这类历史数据没有用户维度，
    统一归到 ``__legacy__`` 会话下，避免导入时丢数据。
    """
    try:
        parts = json.loads(raw)
    except json.JSONDecodeError:  # 旧格式是裸字符串，不是 JSON 数组。
        return raw, "__legacy__"
    if not isinstance(parts, list) or len(parts) != 2:  # 结构不符时同样按旧数据兜底。
        return raw, "__legacy__"
    user_id, thread_id = parts
    return str(user_id), str(thread_id)


class AgentService:
    """组合模型、工具、短期检查点和长期记忆的一次对话服务。"""

    def __init__(
        self,
        client: OpenAI,
        model: str,
        registry: ToolRegistry,
        checkpointer: BaseCheckpointSaver,
        memory_store: BaseStore,
        history: ChatHistory | None = None,
    ) -> None:
        self.client = client  # 复用同一个 API 客户端，避免每个请求重复创建。
        self.model = model  # 保存当前 DeepSeek 模型名称。
        self.registry = registry  # 保存知识库工具注册表。
        self.checkpointer = checkpointer  # 保存按 thread 隔离的短期状态。
        self.memory_store = memory_store  # 保存按 user 隔离的长期记忆。
        self.history = history  # 可选的对话日志；不传时保持原有的纯内存行为。

    # ------------------------------------------------------------------
    # 一次回合的三个阶段：准备 → 跑图 → 收尾
    # 拆开是为了让普通调用和流式调用共用同一套逻辑，避免两处出现偏差。
    # ------------------------------------------------------------------

    def _prepare(
        self,
        user_id: str,
        thread_id: str,
        question: str,
    ) -> tuple[Any, dict[str, Any], dict[str, Any]]:
        """读取长期记忆、编译图、恢复短期历史并追加本轮问题。"""
        memories = list_memories(self.memory_store, user_id)  # 每次请求读取最新长期记忆。
        memory_context = build_memory_context(memories)  # 转为模型可阅读但不可充当条文证据的文本。
        graph = build_agent_graph(  # 为当前用户记忆编译一次运行图。
            self.client,
            self.model,
            self.registry,
            self.checkpointer,
            memory_context,
        )
        config = {  # LangGraph 的运行配置不属于业务 State。
            "configurable": {
                "thread_id": build_checkpoint_thread_id(user_id, thread_id),
            },
            "recursion_limit": 20,
        }
        initial_state = build_turn_state(  # 恢复该用户该会话的历史并追加当前问题。
            graph,
            config,
            question,
        )
        return graph, config, initial_state

    @staticmethod
    def _to_result(final_state: dict[str, Any]) -> AgentTurnResult:
        """把图的最终状态收敛成对外的固定结构。"""
        raw_answer = final_state["final_answer"]
        if raw_answer is None:  # 防御模型未产生最终回答的异常情况。
            raise RuntimeError("Agent 未生成最终回答。")
        answer = StructuredAnswer.model_validate(raw_answer)  # 再恢复成受 Pydantic 校验的对象。
        return AgentTurnResult(
            answer=answer.answer,
            refused=answer.refused,
            citations=answer.citations,
            round_number=final_state["round_number"],
        )

    def _finalize(
        self,
        user_id: str,
        thread_id: str,
        question: str,
        result: AgentTurnResult,
        started_at: float,
    ) -> AgentTurnResult:
        """写聊天日志与指标日志；普通调用和流式调用共用。"""
        elapsed_seconds = perf_counter() - started_at  # 计算完整 Agent 回合耗时。
        # 只有通过校验的答案才会走到这里，因此日志里不会混入被拦下的半成品。
        recorded = safe_record_turn(
            self.history,
            user_id,
            thread_id,
            question,
            answer=result.answer,
            refused=result.refused,
            citations=[citation.model_dump() for citation in result.citations],
        )
        if self.history is not None and not recorded:  # 降级为“本次不回看”，不影响答案返回。
            logger.warning(
                "chat_history_write_failed user_id=%s thread_id=%s",
                user_id,
                thread_id,
            )
        logger.info(  # 不记录 question 和 answer 正文，降低隐私与日志泄露风险。
            "agent_turn_completed user_id=%s thread_id=%s rounds=%s "
            "refused=%s citations=%s elapsed_seconds=%.3f",
            user_id,
            thread_id,
            result.round_number,
            result.refused,
            len(result.citations),
            elapsed_seconds,
        )
        return result

    # ------------------------------------------------------------------
    # 对外入口
    # ------------------------------------------------------------------

    def delete_thread(self, user_id: str, thread_id: str) -> int:
        """只清理该用户该会话的状态与展示记录，保留跨会话长期记忆。"""
        if self.history is None:  # 不允许只清一半却告诉界面已经完整删除。
            raise RuntimeError("聊天历史存储尚未准备好，无法删除会话。")
        checkpoint_id = build_checkpoint_thread_id(user_id, thread_id)
        self.checkpointer.delete_thread(checkpoint_id)  # 官方 API 会清理检查点与待处理写入。
        return self.history.delete_thread(user_id, thread_id)  # 状态清理成功后再移除展示记录。

    def answer(
        self,
        user_id: str,
        thread_id: str,
        question: str,
    ) -> AgentTurnResult:
        """恢复用户上下文、运行 Agent，并返回通过校验的最终结果。"""
        started_at = perf_counter()  # 从进入业务层开始统计总耗时。
        graph, config, initial_state = self._prepare(user_id, thread_id, question)
        final_state = graph.invoke(initial_state, config)  # 执行并自动保存新的检查点。
        result = self._to_result(final_state)
        return self._finalize(user_id, thread_id, question, result, started_at)

    def answer_stream(
        self,
        user_id: str,
        thread_id: str,
        question: str,
    ) -> Iterator[dict[str, Any]]:
        """边跑图边产出进度事件，最后再产出已通过校验的完整结果。

        为什么不是逐 token 流：模型必须输出完整的 JSON（answer + citations），
        引用真实性校验也要等 JSON 收全才能做。在“先校验再展示”这条规则下，
        逐 token 输出意味着用户可能先看到一个后来被判定为编造条文的答案。
        这里选择保留校验，把可见的进度做成阶段事件。

        产出的事件：
        - ``{"type": "status", "stage": ..., "message": ...}``
        - ``{"type": "result", "data": {...}}``
        """
        started_at = perf_counter()
        graph, config, initial_state = self._prepare(user_id, thread_id, question)
        yield {
            "type": "status",
            "stage": "thinking",
            "message": "正在理解问题并读取长期记忆",
        }

        for chunk in graph.stream(initial_state, config, stream_mode="updates"):
            for node_name, update in chunk.items():
                if node_name == "model":
                    # 有 tool_calls 说明这一轮要去查知识库；没有则是最终答案已写完。
                    # update 是节点「完成之后」的输出，所以这里不能写「正在生成」，
                    # 否则用户会看到「生成中」和「校验中」几乎同一秒跳过去。
                    if update.get("tool_calls"):
                        yield {
                            "type": "status",
                            "stage": "retrieving",
                            "message": "正在调用工具：" + "、".join(str(call["name"]) for call in update["tool_calls"]),
                        }
                    else:
                        yield {
                            "type": "status",
                            "stage": "validating",
                            "message": "正在校验引用是否逐字来自原文",
                        }
                elif node_name == "tools":
                    match_count = len(update.get("retrieved_matches") or [])
                    yield {
                        "type": "status",
                        "stage": "retrieved",
                        "message": f"工具执行完成，当前取得 {match_count} 条法条或官方资料证据",
                        "match_count": match_count,
                    }
                    # 工具跑完，下一步必然是再调一次模型，此时预告「生成中」最准确。
                    yield {
                        "type": "status",
                        "stage": "generating",
                        "message": "正在组织答案并标注引用",
                    }

        final_state = graph.get_state(config).values
        result = self._to_result(final_state)
        self._finalize(user_id, thread_id, question, result, started_at)
        yield {"type": "result", "data": result.model_dump(mode="json")}
