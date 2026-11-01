"""企业知识库 Agent 的命令行入口。

这个文件**只做输入输出**：读一行文字、打印结果、问一句「确认保存吗」。
真正的业务流程全部在 ``AgentService`` 和 ``MemoryService`` 里。

以前的版本不是这样：它自己 ``build_agent_graph`` + ``graph.invoke``，
还自己复制了一份记忆提取和门控逻辑。后果有三个：

1. CLI 聊的内容不会写进 ``chat_history.sqlite``，前端看不到；
2. 检查点 thread_id 用的是裸字符串，和 HTTP 入口的 ``["user","thread"]``
   格式对不上，同一个人换个入口就「失忆」；
3. 记忆流程存在三份实现（CLI、HTTP 接口、演示脚本），改一处漏两处。

现在只剩一份实现，CLI 和 HTTP 只是同一套服务的两种外壳。
"""

import argparse  # 用参数而不是硬编码来指定用户和会话身份。
from collections.abc import Sequence

from openai import OpenAI, OpenAIError  # 创建 DeepSeek 客户端并识别上游错误。
from pydantic import ValidationError  # 捕获结构化输出校验失败。

from .chat_history import open_chat_history  # 与 HTTP 入口共用同一份聊天日志。
from .checkpoint import CHECKPOINT_PATH, open_checkpointer  # 短期会话状态。
from .commands import parse_control_command  # 区分控制命令和知识库问题。
from .config import KNOWLEDGE_MIN_SCORE, load_settings  # 集中读取阈值与模型配置。
from .logging_config import setup_logging  # 与后端输出同格式的日志。
from .memory import MemorySaveOutcome, open_memory_store  # 长期记忆存储。
from .memory_service import MemoryProposal, MemoryService  # 记忆的统一业务入口。
from .memory_review import MemoryReviewError  # 审核过期等问题不应让 CLI 崩溃。
from .retrieval import VectorKnowledgeBase  # 加载磁盘上的向量索引。
from .service import AgentService, AgentTurnResult  # 对话的统一业务入口。
from .tools import ToolRegistry  # 把检索器注册成模型可调用工具。


DEFAULT_USER_ID = "cli-user"  # 默认身份与前端演示用户隔离，避免串数据。
DEFAULT_THREAD_ID = "cli-session"  # 固定默认会话，重复启动能接着上次聊。
COMMAND_HELP = "命令：/remember <内容>、/memories、quit"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析身份参数，让同一个 CLI 可以切换成不同用户或不同会话。"""
    parser = argparse.ArgumentParser(
        description="劳动法知识库 Agent 命令行入口",
    )
    parser.add_argument(
        "--user",
        default=DEFAULT_USER_ID,
        help=f"用户 ID，决定读取哪份长期记忆（默认 {DEFAULT_USER_ID}）",
    )
    parser.add_argument(
        "--thread",
        default=DEFAULT_THREAD_ID,
        help=f"会话 ID，决定恢复哪段短期历史（默认 {DEFAULT_THREAD_ID}）",
    )
    return parser.parse_args(argv)


def print_answer(result: AgentTurnResult) -> None:
    """展示已经通过结构校验和引用校验的回答。"""
    print("\nAgent 回答：")
    print(result.answer)
    print(f"是否拒答：{result.refused}")
    print("引用：")
    for citation in result.citations:
        print(f"- {citation.source_id}：{citation.quote}")
    print(f"模型调用轮数：{result.round_number}")


def describe_outcome(outcome: MemorySaveOutcome) -> str:
    """把写入动作翻译成人话，避免“跳过”也显示成“已保存”。"""
    if outcome.action == "merge":
        return f"已合并到已有记忆 {outcome.memory_key}，没有产生重复条目。"
    if outcome.action == "skip":
        return f"{outcome.memory_key} 已经记过同样的内容，本次未重复写入。"
    if outcome.action == "update":
        return f"已更新记忆 {outcome.memory_key}。"
    return "长期记忆已保存。"


def handle_remember(
    memory_service: MemoryService,
    user_id: str,
    payload: str,
) -> None:
    """走完「模型提候选 → 本地门控 → 人工确认 → 判重写入」四步。"""
    try:
        proposal: MemoryProposal = memory_service.propose(user_id, payload)
    except (OpenAIError, ValidationError, MemoryReviewError) as error:
        print(f"候选记忆提取失败：{error}")
        return

    print(f"候选记忆：{proposal.candidate.model_dump()}")
    print(f"本地门控：{proposal.reason}")
    if not proposal.allowed:
        print("未写入长期记忆。")
        return
    if proposal.duplicate_of:
        print(f"判重：与已有记忆 {proposal.duplicate_of} 是同一件事，保存时会合并。")

    approval = input("确认保存？(y/N)：").strip().lower()
    if approval != "y":
        print("用户未确认，未写入长期记忆。")
        return

    try:
        outcome = memory_service.save(
            user_id,
            proposal.candidate.memory_key,
            proposal.candidate.category,
            proposal.candidate.content,
            review_id=proposal.review_id,
            confirmed=True,
        )
    except MemoryReviewError as error:
        print(f"未写入长期记忆：{error}")
        return
    print(describe_outcome(outcome))


def main(argv: Sequence[str] | None = None) -> None:
    """启动交互式命令行，所有业务调用都经过服务层。"""
    args = parse_args(argv)
    user_id: str = args.user
    thread_id: str = args.thread

    try:
        settings = load_settings()  # 从 .env 读取 DeepSeek 配置。
        knowledge_base = VectorKnowledgeBase.from_disk()  # 加载向量索引与 Embedding 模型。
    except RuntimeError as error:
        print(f"启动失败：{error}")
        return

    setup_logging()  # 与后端保持一致，让日志落到同一个目录。
    client = OpenAI(
        api_key=settings.api_key,
        base_url=settings.base_url,
    )
    registry = ToolRegistry(knowledge_base, min_score=KNOWLEDGE_MIN_SCORE)

    with (
        open_memory_store() as memory_store,
        open_checkpointer() as checkpointer,
        open_chat_history() as chat_history,
    ):
        agent_service = AgentService(  # 对话走统一入口，和 HTTP 是同一个实现。
            client,
            settings.model,
            registry,
            checkpointer,
            memory_store,
            history=chat_history,
        )
        memory_service = MemoryService(client, settings.model, memory_store)

        print(f"用户 ID：{user_id}")
        print(f"会话 ID：{thread_id}")
        print(memory_service.context(user_id))
        print(f"检查点数据库：{CHECKPOINT_PATH}")
        print(COMMAND_HELP)

        while True:
            try:
                question = input("\n你：").strip()
            except EOFError:  # 管道或 Ctrl+D 关闭输入时正常退出，而不是抛栈。
                print("\n输入已关闭，会话已保存。")
                break
            if question.lower() == "quit":
                print("会话已保存，程序退出。")
                break
            if not question:
                print("问题不能为空。")
                continue

            command = parse_control_command(question)
            if command is not None:
                command_name, payload = command
                if command_name == "memories":
                    print(memory_service.context(user_id))
                    continue
                if command_name == "unknown":
                    print(f"未知命令。可用{COMMAND_HELP}")
                    continue
                if not payload:
                    print("请在 /remember 后输入要记住的内容。")
                    continue
                handle_remember(memory_service, user_id, payload)
                continue

            try:
                # 长期记忆由 AgentService 在每次请求开始时重新读取，
                # 所以刚写入的记忆下一句话就生效，不需要重建图。
                result = agent_service.answer(user_id, thread_id, question)
            except OpenAIError as error:
                print(f"API 调用失败：{error}")
                continue
            except (RuntimeError, ValueError, ValidationError) as error:
                print(f"Agent 执行失败：{error}")
                continue

            print_answer(result)


if __name__ == "__main__":  # 使用 python -m 执行本模块时才启动交互。
    main()
