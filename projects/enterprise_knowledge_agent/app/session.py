"""根据 thread_id 恢复旧状态，并构造一次新的用户回合。"""

from typing import Any  # 不绑定具体 CompiledGraph 类型，方便测试时使用简单替身。

from .models import AgentState  # 复用模型、工具和校验节点共同使用的状态结构。


def build_turn_state(
    graph: Any,
    config: dict[str, object],
    question: str,
) -> AgentState:
    """读取当前 thread 的历史，并在末尾追加一条新的用户问题。"""
    snapshot = graph.get_state(config)  # checkpointer 根据 thread_id 读取最新状态快照。
    previous_input = list(  # 复制旧历史，避免直接修改检查点返回的原对象。
        snapshot.values.get("conversation_input", [])
    )

    return {
        "conversation_input": [  # 旧历史和本轮问题一起发送给模型。
            *previous_input,
            {"role": "user", "content": question},
        ],
        "tool_calls": [],  # 新回合开始时还没有产生工具请求。
        "retrieved_matches": [],  # 只允许本轮重新检索到的证据支持本轮答案。
        "final_answer": None,  # 新回合开始时还没有最终回答。
        "round_number": 0,  # 最大轮数限制应当按每个用户回合重新计算。
        "max_rounds": 5,  # 单个问题最多允许调用模型五轮，防止无限循环。
    }
