"""定义 LangGraph 节点之间传递的数据结构。"""

from typing_extensions import TypedDict

class AgentState(TypedDict):
    """模型节点和工具节点共享的运行状态。"""

    conversation_input: list[object]
    tool_calls: list[dict[str, object]]
    retrieved_matches: list[dict[str, object]]
    final_answer: dict[str, object] | None
    round_number: int
    max_rounds: int
