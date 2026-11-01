"""使用假模型响应测试完整 LangGraph 工具与校验循环。"""

import json
from types import SimpleNamespace

from projects.enterprise_knowledge_agent.app.graph import build_agent_graph
from projects.enterprise_knowledge_agent.app.models import AgentState
from projects.enterprise_knowledge_agent.app.tools import ToolRegistry


class FakeKnowledgeBase:
    """固定返回密码政策，避免测试依赖真实向量模型。"""

    def search(
        self,
        query: str,
        top_k: int = 2,
        min_score: float = 0.5,
    ) -> list[dict[str, object]]:
        return [
            {
                "source_id": "company_policy.txt#chunk-3",
                "text": "忘记密码时，可以在登录页面点击“重置密码”。",
                "score": 0.75,
            }
        ]


class FakeResponsesAPI:
    """按顺序返回预设响应，并保存模型请求。"""

    def __init__(self, responses: list[SimpleNamespace]) -> None:
        self.responses = responses
        self.requests: list[dict] = []

    def create(self, **request: object) -> SimpleNamespace:
        self.requests.append(request)
        return self.responses[len(self.requests) - 1]


class FakeClient:
    """提供与 OpenAI 客户端相同的 responses 属性。"""

    def __init__(self, responses: list[SimpleNamespace]) -> None:
        self.responses = FakeResponsesAPI(responses)


def test_graph_searches_then_validates_structured_answer() -> None:
    """模型调用检索后，只有引用通过校验的结构化回答才能结束。"""
    first_response = SimpleNamespace(
        output=[
            SimpleNamespace(
                type="function_call",
                name="search_knowledge_base",
                arguments='{"query":"忘记密码怎么办","top_k":1}',
                call_id="call_search_001",
            )
        ],
        output_text="",
    )
    second_response = SimpleNamespace(
        output=[SimpleNamespace(type="message")],
        output_text=json.dumps(
            {
                "answer": "可以在登录页面点击重置密码。",
                "refused": False,
                "citations": [
                    {
                        "source_id": "company_policy.txt#chunk-3",
                        "quote": "忘记密码时，可以在登录页面点击“重置密码”。",
                    }
                ],
            },
            ensure_ascii=False,
        ),
    )
    client = FakeClient([first_response, second_response])
    registry = ToolRegistry(FakeKnowledgeBase())
    graph = build_agent_graph(client, "fake-model", registry)
    initial_state: AgentState = {
        "conversation_input": [{"role": "user", "content": "忘记密码怎么办？"}],
        "tool_calls": [],
        "retrieved_matches": [],
        "final_answer": None,
        "round_number": 0,
        "max_rounds": 5,
    }

    final_state = graph.invoke(initial_state)

    assert final_state["round_number"] == 2
    assert final_state["final_answer"]["refused"] is False
    assert final_state["final_answer"]["citations"][0]["source_id"].endswith("chunk-3")
    assert len(client.responses.requests) == 2
    assert client.responses.requests[0]["tool_choice"] == "required"
    assert client.responses.requests[1]["tool_choice"] == "auto"
    instructions = client.responses.requests[0]["instructions"]
    assert "拒答可以 citations=[]" in instructions
    assert "所有引用仍必须逐字来自本次检索结果" in instructions
    assert "refused=true、citations=[]" not in instructions

    second_input = client.responses.requests[1]["input"]
    tool_output = json.loads(second_input[-1]["output"])
    assert tool_output["gate"]["passed"] is True
    assert tool_output["matches"][0]["source_id"].endswith("chunk-3")
