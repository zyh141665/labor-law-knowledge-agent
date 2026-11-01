"""测试长期记忆的更新、用户隔离、提示词上下文以及写入前判重。"""

from langgraph.store.sqlite import SqliteStore  # 使用内存 SQLite 运行快速隔离测试。

from projects.enterprise_knowledge_agent.app.memory import (
    build_memory_context,
    char_overlap,
    get_memory,
    keys_related,
    list_memories,
    normalize_content,
    save_memory,
    save_memory_deduped,
)


def test_same_key_updates_memory_and_users_are_isolated() -> None:
    """同一用户和 key 应更新；另一用户不应读到这条记忆。"""
    with SqliteStore.from_conn_string(":memory:") as store:  # 测试结束后自动销毁数据库。
        store.setup()  # 为内存数据库建立 Store 所需的数据表。
        save_memory(store, "user-A", "answer_style", "preference", "详细")
        save_memory(store, "user-A", "answer_style", "preference", "简洁")

        user_a_memory = get_memory(store, "user-A", "answer_style")
        user_b_memory = get_memory(store, "user-B", "answer_style")

    assert user_a_memory is not None
    assert user_a_memory.content == "简洁"
    assert user_b_memory is None


def test_memory_context_contains_key_category_and_content() -> None:
    """模型上下文应保留记忆名称、类别和内容。"""
    with SqliteStore.from_conn_string(":memory:") as store:
        store.setup()
        save_memory(
            store,
            "user-A",
            "learning_goal",
            "goal",
            "学习 Agent 应用开发",
        )
        memories = list_memories(store, "user-A")

    context = build_memory_context(memories)

    assert "learning_goal（goal）" in context
    assert "学习 Agent 应用开发" in context


def test_normalize_content_ignores_punctuation_and_spaces() -> None:
    """标点和空白不携带语义，不该让同一句话变成两条记忆。"""
    assert normalize_content("回答尽量简洁。") == normalize_content("回答 尽量 简洁")


def test_keys_related_only_accepts_prefix_family() -> None:
    """只有“短 key 是长 key 词干”才算同一件事，普通的公共前缀不算。"""
    assert keys_related("answer_style", "answer_style_conclusion_first")
    assert keys_related("answer_style", "answer_style")
    assert not keys_related("answer_style", "answer_tone")  # 同前缀但不同分支。
    assert not keys_related("style", "answer_style")  # 词干不匹配。


def test_duplicate_key_family_merges_into_shorter_key() -> None:
    """同一个 key 家族只留一条，保留更短的 key 和更完整的内容。"""
    with SqliteStore.from_conn_string(":memory:") as store:
        store.setup()
        save_memory(
            store,
            "user-A",
            "answer_style",
            "preference",
            "回答尽量简洁，并优先解释结论",
        )
        outcome = save_memory_deduped(
            store,
            "user-A",
            "answer_style_conclusion_first",
            "preference",
            "用户偏好回答时先给结论，再解释原因。",
        )
        memories = list_memories(store, "user-A")

    assert outcome.action == "merge"
    assert outcome.memory_key == "answer_style"
    assert list(memories) == ["answer_style"]  # 没有多出第二条。
    assert memories["answer_style"].content.startswith("用户偏好回答时先给结论")


def test_identical_content_under_new_key_is_skipped() -> None:
    """内容一模一样、只是换了个 key 名字，不该在库里存两遍。"""
    with SqliteStore.from_conn_string(":memory:") as store:
        store.setup()
        save_memory(store, "user-A", "answer_style", "preference", "回答要简洁")
        outcome = save_memory_deduped(
            store,
            "user-A",
            "reply_style",
            "preference",
            "回答要简洁。",
        )
        memories = list_memories(store, "user-A")

    assert outcome.action == "skip"
    assert outcome.memory_key == "answer_style"
    assert len(memories) == 1


def test_new_shorter_key_replaces_longer_duplicate() -> None:
    """新 key 更短时合并到新 key，旧的冗余 key 被移除而不是并存。"""
    with SqliteStore.from_conn_string(":memory:") as store:
        store.setup()
        save_memory(
            store,
            "user-A",
            "answer_style_conclusion_first",
            "preference",
            "先给结论，再解释原因",
        )
        outcome = save_memory_deduped(
            store,
            "user-A",
            "answer_style",
            "preference",
            "先给结论再解释原因，尽量简洁",
        )
        memories = list_memories(store, "user-A")

    assert outcome.action == "merge"
    assert outcome.memory_key == "answer_style"
    assert outcome.removed_key == "answer_style_conclusion_first"
    assert list(memories) == ["answer_style"]
    assert memories["answer_style"].content.endswith("尽量简洁")


def test_unrelated_memory_is_inserted_even_with_same_prefix() -> None:
    """同类别但 key 不是前缀家族、内容也不同时，应当正常新增。"""
    with SqliteStore.from_conn_string(":memory:") as store:
        store.setup()
        save_memory(store, "user-A", "learning_goal", "goal", "找一份 Agent 实习")
        outcome = save_memory_deduped(
            store,
            "user-A",
            "target_city",
            "goal",
            "目标城市是杭州",
        )
        memories = list_memories(store, "user-A")

    assert outcome.action == "insert"
    assert len(memories) == 2


def test_key_prefix_alone_does_not_merge_unrelated_content() -> None:
    """goal 和 goal_city 也是前缀家族，但说的是两件事，绝不能合并。

    这是纯按 key 判重最危险的误伤：合并会直接覆盖掉其中一条的信息。
    """
    with SqliteStore.from_conn_string(":memory:") as store:
        store.setup()
        save_memory(
            store,
            "user-A",
            "goal",
            "goal",
            "学习 Agent 应用开发并寻找相关实习",
        )
        outcome = save_memory_deduped(
            store,
            "user-A",
            "goal_city",
            "goal",
            "杭州",
        )
        memories = list_memories(store, "user-A")

    assert outcome.action == "insert"
    assert len(memories) == 2  # 两条都还在，没有互相覆盖。
    assert memories["goal"].content.startswith("学习 Agent")


def test_char_overlap_is_zero_for_unrelated_sentences() -> None:
    """无关短句几乎不共用字，这是 key 前缀判据的第二道保险。"""
    related = char_overlap("回答尽量简洁，并优先解释结论", "用户偏好回答时先给结论，再解释原因。")
    unrelated = char_overlap("学习 Agent 应用开发并寻找相关实习", "杭州")

    assert related >= 0.2  # 真实重复的偏好对，重合度明显更高。
    assert unrelated == 0.0


def test_duplicate_detection_never_crosses_categories() -> None:
    """goal 和 preference 即使内容相同也不该互相覆盖。"""
    with SqliteStore.from_conn_string(":memory:") as store:
        store.setup()
        save_memory(store, "user-A", "answer_style", "preference", "回答要简洁")
        outcome = save_memory_deduped(
            store,
            "user-A",
            "target_city",
            "goal",
            "回答要简洁",
        )
        memories = list_memories(store, "user-A")

    assert outcome.action == "insert"
    assert len(memories) == 2
