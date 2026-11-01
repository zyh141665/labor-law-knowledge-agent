"""演示候选提取、安全门控、人工确认和长期记忆写入。

业务流程全部交给 ``MemoryService``，本脚本只负责问一句、打印结果。
以前这段流程在这里又实现了一遍，和 CLI、HTTP 接口各写一份，
三处容易走偏；现在统一到服务层，演示脚本只做输入输出。
"""

from openai import OpenAI, OpenAIError  # 创建 DeepSeek 客户端并捕获 API 调用错误。
from pydantic import ValidationError  # 捕获结构化候选解析失败。
from projects.enterprise_knowledge_agent.app.memory_review import MemoryReviewError

from projects.enterprise_knowledge_agent.app.config import load_settings  # 读取模型配置。
from projects.enterprise_knowledge_agent.app.memory import open_memory_store  # 记忆存储。
from projects.enterprise_knowledge_agent.app.memory_service import (  # 记忆的统一入口。
    MemoryService,
)


USER_ID = "student-001"  # 本演示把经过确认的记忆保存到这个用户名下。


def main() -> None:
    """读取一句用户原话，只有通过规则且得到确认后才写入数据库。"""
    try:
        settings = load_settings()  # 从项目 .env 读取 DeepSeek 配置。
    except RuntimeError as error:
        print(f"配置错误：{error}")
        return

    user_text = input("请输入一句可能值得长期记住的话：").strip()
    if not user_text:  # 空文本不调用模型，避免浪费一次 API 请求。
        print("输入不能为空。")
        return

    client = OpenAI(
        api_key=settings.api_key,
        base_url=settings.base_url,
    )

    with open_memory_store() as store:  # 用户确认后才打开长期记忆数据库。
        service = MemoryService(client, settings.model, store)
        print("\n写入前已有的长期记忆：")
        print(service.context(USER_ID))

        try:
            proposal = service.propose(USER_ID, user_text)  # 提取 + 本地门控 + 判重。
        except (OpenAIError, ValidationError, MemoryReviewError) as error:
            print(f"候选记忆提取失败：{error}")
            return

        print("\n模型提出的候选：")
        print(proposal.candidate.model_dump())
        print(f"本地门控：{proposal.reason}")
        if not proposal.allowed:  # 未通过门控时模型没有机会写数据库。
            print("未写入长期记忆。")
            return
        if proposal.duplicate_of:
            print(f"判重：与已有记忆 {proposal.duplicate_of} 是同一件事，将合并为一条。")

        approval = input("是否确认保存这条长期记忆？(y/N)：").strip().lower()
        if approval != "y":  # 只有明确输入 y 才视为授权。
            print("用户未确认，未写入长期记忆。")
            return

        try:
            outcome = service.save(
                USER_ID,
                proposal.candidate.memory_key,
                proposal.candidate.category,
                proposal.candidate.content,
                review_id=proposal.review_id,
                confirmed=True,
            )
        except MemoryReviewError as error:
            print(f"未写入长期记忆：{error}")
            return
        print(f"\n写入结果：{outcome.model_dump()}")  # 动作、落在哪个 key 都要看得见。
        print("\n写入后：")
        print(service.context(USER_ID))


if __name__ == "__main__":  # 使用 python -m 运行本模块时才启动演示。
    main()
