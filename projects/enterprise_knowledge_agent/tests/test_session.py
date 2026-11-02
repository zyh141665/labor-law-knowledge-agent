"""测试新用户回合会恢复历史，同时重置只属于本轮的字段。"""

from types import SimpleNamespace  # 用简单对象模拟 LangGraph 返回的 StateSnapshot。

from projects.enterprise_knowledge_agent.app.session import build_turn_state


class FakeGraph:
    """只实现 build_turn_state() 所需的 get_state() 接口。"""

    def get_state(self, config: dict[str, object]) -> SimpleNamespace:
        return SimpleNamespace(
            values={
                "conversation_input": [
                    {"role": "user", "content": "退款期限是多少？"},
                    {"role": "assistant", "content": "退款期限为 7 天。"},
                ],
                "retrieved_matches": [{"source_id": "old-source"}],
                "round_number": 2,
                "tool_results": [{"name": "calculate_overtime_pay", "result": {"total_overtime_pay": "999"}}],
            }
        )


def test_new_turn_keeps_history_but_resets_turn_fields() -> None:
    config = {"configurable": {"thread_id": "test-user"}}

    state = build_turn_state(FakeGraph(), config, "那第八天可以吗？")

    assert len(state["conversation_input"]) == 3
    assert state["conversation_input"][-1]["content"] == "那第八天可以吗？"
    assert state["retrieved_matches"] == []
    assert state["tool_calls"] == []
    assert state["tool_results"] == []  # 不能把上一轮算出的金额用于当前回合。
    assert state["final_answer"] is None
    assert state["round_number"] == 0
