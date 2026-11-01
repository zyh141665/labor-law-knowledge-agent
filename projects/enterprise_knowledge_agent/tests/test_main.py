"""测试命令行入口的参数解析和提示语。

只测纯函数：这些函数不读文件、不调模型，因此可以快速且稳定地跑。
真正的交互（读输入、打印）留给手工冒烟测试。
"""

from projects.enterprise_knowledge_agent.app.main import (
    DEFAULT_THREAD_ID,
    DEFAULT_USER_ID,
    describe_outcome,
    parse_args,
)
from projects.enterprise_knowledge_agent.app.memory import MemorySaveOutcome


def test_default_identity_is_isolated_from_frontend_demo_user() -> None:
    """CLI 默认身份不能是前端的演示用户，否则两个入口会互相污染数据。"""
    args = parse_args([])

    assert args.user == DEFAULT_USER_ID
    assert args.thread == DEFAULT_THREAD_ID
    assert args.user != "student-001"  # 前端演示账号。


def test_identity_can_be_overridden() -> None:
    """同一个 CLI 要能切换用户和会话。"""
    args = parse_args(["--user", "alice", "--thread", "job-hunting"])

    assert args.user == "alice"
    assert args.thread == "job-hunting"


def test_describe_outcome_distinguishes_skip_from_saved() -> None:
    """“跳过”和“保存成功”必须说清楚，否则用户会以为库里多了一条。"""
    common = {"memory_key": "answer_style"}

    merged = describe_outcome(
        MemorySaveOutcome(action="merge", reason="", **common)
    )
    skipped = describe_outcome(
        MemorySaveOutcome(action="skip", reason="", **common)
    )
    inserted = describe_outcome(
        MemorySaveOutcome(action="insert", reason="", **common)
    )

    assert "合并" in merged
    assert "未重复写入" in skipped
    assert inserted == "长期记忆已保存。"