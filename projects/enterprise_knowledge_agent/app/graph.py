"""构建 DeepSeek、向量检索和本地校验协作的 LangGraph。"""

import json
from functools import partial
from typing import Literal

from langgraph.checkpoint.base import BaseCheckpointSaver  # 所有 LangGraph 检查点保存器的共同类型。
from langgraph.graph import END, START, StateGraph
from openai import OpenAI

from .models import AgentState
from .schemas import StructuredAnswer, validate_answer
from .tools import ToolRegistry


def serialize_output_item(output_item: object) -> dict[str, object]:
    """把 OpenAI SDK 输出对象转换成适合写入 SQLite 的普通字典。"""
    if isinstance(output_item, dict):
        return output_item
    model_dump = getattr(output_item, "model_dump", None)
    if callable(model_dump):
        return model_dump(exclude_none=True)
    return vars(output_item)


def extract_tool_calls(output_items: list[object]) -> list[dict[str, object]]:
    """从 Responses API 输出中提取模型提出的全部函数调用。"""
    tool_calls = []
    for output_item in output_items:
        if output_item.type == "function_call":
            tool_calls.append(
                {
                    "name": output_item.name,
                    "arguments": json.loads(output_item.arguments),
                    "call_id": output_item.call_id,
                }
            )
    return tool_calls


def call_model_node(
    state: AgentState,
    *,
    client: OpenAI,
    model: str,
    tools: list[dict],
    memory_context: str,
) -> dict:
    """让 DeepSeek 决定调用知识库工具或生成结构化最终回答。"""
    next_round = state["round_number"] + 1
    if next_round > state["max_rounds"]:
        raise RuntimeError(f"Agent 超过最大轮数 {state['max_rounds']}。")

    print(f"第 {next_round} 轮：调用模型……")
    response = client.responses.create(
        model=model,
        instructions=(
            "你是严谨的劳动法知识库助手。知识库收录《中华人民共和国劳动法》"
            "和《中华人民共和国劳动合同法》两部法律。"
            "涉及这两部法律条文的问题必须先调用知识库工具。"
            "只能陈述检索结果直接支持的事实，不得添加常识、建议或推测。"
            "能够回答时 refused=false，citations 至少一项，quote 必须从对应来源逐字复制，"
            "source_id 必须照抄检索结果里的写法（形如“劳动合同法#第八十二条”），"
            "不得改写法律名称、补全全称或改换条号。"
            "没有检索结果或证据不足时 refused=true，answer 明确说明无法确定；"
            "拒答可以 citations=[]，也可以引用查过但不足以回答的真实条文并说明局限，"
            "所有引用仍必须逐字来自本次检索结果；不得用相关条文推断未规定的结论。"
            "若两部法律均有规定，应分别引用，不要只挑一部。"
            "下面的用户长期记忆只用于调整交流方式，不属于法律条文证据，不得作为条文引用。"
            f"\n\n{memory_context}"
        ),
        input=state["conversation_input"],
        tools=tools,
        tool_choice=(
            "required" if state["round_number"] == 0 else "auto"
        ),
        text={
            "format": {
                "type": "json_schema",
                "name": "grounded_answer",
                "schema": StructuredAnswer.model_json_schema(),
            }
        },
        reasoning={"effort": "none"},
        max_output_tokens=500,
        timeout=60.0,
    )
    tool_calls = extract_tool_calls(response.output)
    serialized_output = [serialize_output_item(item) for item in response.output]
    final_answer = None
    if not tool_calls:
        final_answer = StructuredAnswer.model_validate_json(
            response.output_text
        ).model_dump()

    return {
        "conversation_input": [*state["conversation_input"], *serialized_output],
        "tool_calls": tool_calls,
        "final_answer": final_answer,
        "round_number": next_round,
    }


def route_after_model(state: AgentState) -> Literal["tools", "validate"]:
    """有工具请求就执行工具，否则进入本地答案校验节点。"""
    return "tools" if state["tool_calls"] else "validate"


def execute_tools_node(state: AgentState, *, registry: ToolRegistry) -> dict:
    """执行白名单工具，把结果交回模型并保存本轮允许引用的证据。"""
    conversation_input = list(state["conversation_input"])
    retrieved_matches = list(state["retrieved_matches"])

    for tool_call in state["tool_calls"]:
        result = registry.execute(tool_call)
        print(f"执行工具：{tool_call['name']}")
        retrieved_matches.extend(result["matches"])
        conversation_input.append(
            {
                "type": "function_call_output",
                "call_id": tool_call["call_id"],
                "output": json.dumps(result, ensure_ascii=False),
            }
        )
    return {
        "conversation_input": conversation_input,
        "tool_calls": [],
        "retrieved_matches": retrieved_matches,
    }


def validate_answer_node(state: AgentState) -> dict:
    """阻止无引用、伪造来源或伪造引文的候选答案离开工作流。"""
    raw_answer = state["final_answer"]
    if raw_answer is None:
        raise ValueError("模型没有生成最终回答。")
    answer = StructuredAnswer.model_validate(raw_answer)
    if not validate_answer(answer, state["retrieved_matches"]):
        raise ValueError("模型回答未通过拒答状态或引用真实性校验。")
    print("最终答案校验：通过")
    return {}


def build_agent_graph(
    client: OpenAI,
    model: str,
    registry: ToolRegistry,
    checkpointer: BaseCheckpointSaver | None = None,  # 未传入时保持原来的无持久化行为。
    memory_context: str = "暂无用户长期记忆。",
):
    """编译模型—工具—校验组成的企业知识库 Agent。"""
    tools = registry.build_definitions()
    model_node = partial(
        call_model_node,
        client=client,
        model=model,
        tools=tools,
        memory_context=memory_context,
    )
    tools_node = partial(execute_tools_node, registry=registry)

    builder = StateGraph(AgentState)
    builder.add_node("model", model_node)
    builder.add_node("tools", tools_node)
    builder.add_node("validate", validate_answer_node)
    builder.add_edge(START, "model")
    builder.add_conditional_edges(
        "model",
        route_after_model,
        {"tools": "tools", "validate": "validate"},
    )
    builder.add_edge("tools", "model")
    builder.add_edge("validate", END)
    return builder.compile(checkpointer=checkpointer)  # 编译图时安装检查点保存器。
