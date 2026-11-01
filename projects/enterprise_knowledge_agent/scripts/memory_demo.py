"""演示长期记忆在不同 thread_id 之间仍能被同一用户读取。"""

from projects.enterprise_knowledge_agent.app.memory import (  # 导入长期记忆组件。
    get_memory,
    open_memory_store,
    save_memory,
)


USER_ID = "student-001"  # 用户身份在多个会话之间保持不变。
FIRST_THREAD_ID = "learning-chat-001"  # 模拟该用户的第一个会话。
SECOND_THREAD_ID = "learning-chat-002"  # 模拟同一用户新建的另一个会话。


def main() -> None:
    """在第一个会话写入目标，再模拟第二个会话读取该目标。"""
    with open_memory_store() as store:  # 打开同一份跨会话长期记忆数据库。
        save_memory(  # 模拟在第一个 thread 中确认并保存用户长期目标。
            store,
            USER_ID,
            memory_key="learning_goal",
            category="goal",
            content="学习 Agent 应用开发并寻找相关实习",
        )
        save_memory(  # 保存同一用户希望 Agent 采用的长期回答偏好。
            store,
            USER_ID,
            memory_key="answer_style",
            category="preference",
            content="回答尽量简洁，并优先解释结论",
        )
        restored = get_memory(  # 模拟另一个 thread 根据 user_id 读取长期目标。
            store,
            USER_ID,
            memory_key="learning_goal",
        )

    print(f"写入会话：{FIRST_THREAD_ID}")  # thread 不同只是为了说明跨会话场景。
    print(f"读取会话：{SECOND_THREAD_ID}")
    print(f"用户 ID：{USER_ID}")
    print(f"恢复的长期记忆：{restored}")


if __name__ == "__main__":  # 使用 python -m 运行本模块时才执行演示。
    main()
