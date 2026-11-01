"""识别命令行中的控制命令，避免把它们误当成普通知识库问题。"""


def parse_control_command(text: str) -> tuple[str, str] | None:
    """返回“命令名称、参数”；普通用户问题返回 None。"""
    if text == "/memories":  # 查看命令不需要额外参数。
        return "memories", ""
    if text == "/remember":  # 缺少内容时仍识别命令，交给入口输出针对性提示。
        return "remember", ""
    if text.startswith("/remember "):  # 记忆命令后面的全部文字都是待提取原文。
        payload = text.removeprefix("/remember ").strip()
        return "remember", payload
    if text.startswith("/"):  # 拦截拼错的控制命令，避免浪费模型调用。
        return "unknown", text
    return None  # 不以斜杠开头的内容继续进入正常 Agent 工作流。
