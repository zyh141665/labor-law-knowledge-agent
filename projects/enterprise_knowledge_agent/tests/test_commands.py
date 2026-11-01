"""测试本地控制命令与普通用户问题的路由。"""

from projects.enterprise_knowledge_agent.app.commands import parse_control_command


def test_remember_command_extracts_payload() -> None:
    """记忆命令应保留用户希望长期保存的原话。"""
    command = parse_control_command("/remember 以后请简洁回答")

    assert command == ("remember", "以后请简洁回答")


def test_memories_command_needs_no_payload() -> None:
    """查看命令应被识别为不带参数的本地操作。"""
    assert parse_control_command("/memories") == ("memories", "")


def test_empty_remember_command_is_recognized() -> None:
    """缺少内容时应由入口给出具体提示，而不是报告未知命令。"""
    assert parse_control_command("/remember") == ("remember", "")


def test_normal_question_is_not_a_control_command() -> None:
    """普通企业政策问题应继续交给 LangGraph Agent。"""
    assert parse_control_command("退款期限是多少？") is None


def test_unknown_slash_command_is_intercepted() -> None:
    """拼错的命令不应浪费一次模型调用。"""
    assert parse_control_command("/unknown") == ("unknown", "/unknown")
