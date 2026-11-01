"""从用户文本提取候选记忆，并在写入前执行本地安全门控。"""

import re  # 使用正则表达式限制 memory_key 的格式。

from openai import OpenAI  # 调用 DeepSeek Responses API 提取候选记忆。
from pydantic import BaseModel  # 定义并校验模型必须返回的固定结构。

from .memory import MemoryRecord  # 把已有记忆一并交给模型，用于识别重复。


class MemoryCandidate(BaseModel):
    """模型只能提出候选记忆，不能直接操作长期记忆数据库。"""

    should_remember: bool  # 这句话是否包含值得跨会话保留的稳定信息。
    memory_key: str  # 稳定英文键；不应保存时返回空字符串。
    category: str  # 候选类别，之后还会由本地白名单再次检查。
    content: str  # 归一化后的记忆内容；不应保存时返回空字符串。
    sensitive: bool  # 是否可能包含密码、令牌、证件号等敏感信息。
    duplicate_of: str = ""  # 与已有记忆重复时填那条记忆的 key，否则为空字符串。
    reason: str  # 说明为什么建议保存或拒绝保存，便于人工审核。


ALLOWED_CATEGORIES = {"profile", "preference", "goal"}  # 当前项目允许的三类长期记忆。
MEMORY_KEY_PATTERN = re.compile(r"^[a-z][a-z0-9_]{1,63}$")  # 限制为稳定英文 snake_case。
SENSITIVE_TERMS = {  # 即使模型漏判，程序仍会用关键词做第二层拦截。
    "密码",
    "口令",
    "验证码",
    "身份证",
    "银行卡",
    "api key",
    "token",
    "secret",
}


def format_existing_memories(existing: dict[str, MemoryRecord]) -> str:
    """把已有记忆排成清单，交给模型判断“这句话是不是已经记过了”。

    记忆总量很小（每个用户通常个位数），把它整份塞进提示词是可行的，
    比另建一套相似度计算更准：判断“回答要简洁”和“先给结论再解释”
    是不是同一件事，本来就属于语义判断。
    """
    if not existing:
        return "该用户目前没有任何长期记忆。"
    lines = ["该用户已有的长期记忆（key（类别）：内容）："]
    for key, record in sorted(existing.items()):
        lines.append(f"- {key}（{record.category}）：{record.content}")
    return "\n".join(lines)


def extract_memory_candidate(
    client: OpenAI,
    model: str,
    user_text: str,
    existing: dict[str, MemoryRecord] | None = None,
) -> MemoryCandidate:
    """让模型把用户原话转换成结构化候选，但不执行数据库写入。

    传入 ``existing`` 后，模型会同时看到已有记忆并给出 ``duplicate_of``。
    这只是**建议**：真正的合并判断还要经过 ``find_duplicate`` 的确定性规则，
    模型判错也不会让数据出错。
    """
    response = client.responses.create(  # 只调用模型进行信息提取。
        model=model,  # 使用项目 .env 中配置的 DeepSeek 模型。
        instructions=(
            "你是长期记忆提取器，只能提出候选，不能虚构信息。"
            "仅保存用户明确表达且跨会话仍有价值的稳定资料、偏好或目标。"
            "临时状态、普通问题、猜测、第三方信息不得保存。"
            "密码、验证码、密钥、令牌、身份证、银行卡等敏感信息必须 sensitive=true，"
            "并且 should_remember=false。"
            "memory_key 使用简短稳定的英文 snake_case；不应保存时 key 和 content 为空。"
            "如果这句话讲的就是已有记忆里的同一件事（哪怕换了说法），"
            "duplicate_of 必须填那条已有记忆的 key，并优先沿用它的 key 作为 memory_key，"
            "content 写成合并后更完整的一版；确实是新信息时 duplicate_of 留空字符串。"
            "不要把同一件事拆成多条记忆。"
        ),
        input=(
            f"{format_existing_memories(existing or {})}\n\n"
            f"用户这次说的话：\n{user_text}"
        ),
        text={  # 强制模型按照 MemoryCandidate 的 JSON Schema 输出。
            "format": {
                "type": "json_schema",
                "name": "memory_candidate",
                "schema": MemoryCandidate.model_json_schema(),
            }
        },
        reasoning={"effort": "none"},  # 简单抽取任务不需要额外推理模式。
        max_output_tokens=300,  # 限制输出长度，防止产生无关长文本。
        timeout=60.0,  # 网络异常时最多等待 60 秒。
    )
    return MemoryCandidate.model_validate_json(  # 本地再次校验模型返回的 JSON。
        response.output_text
    )


def check_memory_candidate(
    candidate: MemoryCandidate,
    original_text: str,
) -> tuple[bool, str]:
    """使用确定性的本地规则决定候选记忆能否进入人工确认阶段。"""
    if not candidate.should_remember:  # 模型认为没有长期价值时不写入。
        return False, "模型判断这句话不应形成长期记忆。"
    if candidate.sensitive:  # 模型标记为敏感时直接拒绝。
        return False, "候选内容可能包含敏感信息。"
    if candidate.category not in ALLOWED_CATEGORIES:  # 类别必须进入本地白名单。
        return False, "候选记忆类别不在允许范围内。"
    if not MEMORY_KEY_PATTERN.fullmatch(candidate.memory_key):  # key 必须稳定且可控。
        return False, "memory_key 格式不合法。"
    if not candidate.content.strip():  # 空内容没有保存价值。
        return False, "候选记忆内容为空。"

    combined_text = f"{original_text}\n{candidate.content}".lower()  # 同时检查原文和提取结果。
    if any(term in combined_text for term in SENSITIVE_TERMS):  # 本地关键词提供第二道防线。
        return False, "本地规则检测到潜在敏感信息。"
    return True, "候选记忆通过本地规则，可以交给用户确认。"  # 通过不等于自动写入。
