"""检查长期记忆里的重复条目，默认只报告，不会改动任何数据。

为什么单独做成脚本而不是自动清理：

判重规则能覆盖“同一个 key 家族”和“内容完全相同”，但真正复杂的情况
（换了说法、key 完全不搭边）需要人来判断。历史数据是既成事实，
自动删改它比留两条重复更危险。所以默认只打印发现了什么、打算怎么合并，
确认无误后再加 ``--apply`` 真的执行。

用法：

    # 只看报告
    python -m projects.enterprise_knowledge_agent.scripts.dedup_memories

    # 只看某一个用户
    python -m projects.enterprise_knowledge_agent.scripts.dedup_memories --user student-001

    # 确认后真的合并
    python -m projects.enterprise_knowledge_agent.scripts.dedup_memories --apply
"""

import argparse  # 用 --apply / --user 控制行为，避免误改数据。
from collections import defaultdict
from collections.abc import Sequence

from projects.enterprise_knowledge_agent.app.memory import (  # 与运行时同一套判重规则。
    MemoryRecord,
    is_same_memory,
    open_memory_store,
    save_memory,
)


def group_duplicates(
    memories: dict[str, MemoryRecord],
) -> list[list[str]]:
    """把指向同一件事的记忆分成若干组，只返回成员数大于 1 的组。

    判据直接复用 ``is_same_memory``，保证“线上写入时的判重”和“历史清理”
    用的是同一套规则——两套规则是最容易埋雷的地方。用并查集处理传递性：
    A 像 B、B 像 C 时，三条应该合成一组，而不是只合并前两条。
    """
    keys = sorted(memories)
    parent = {key: key for key in keys}

    def find(key: str) -> str:
        while parent[key] != key:
            parent[key] = parent[parent[key]]  # 路径压缩，避免长链拖慢查询。
            key = parent[key]
        return key

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for index, left in enumerate(keys):
        for right in keys[index + 1 :]:
            if is_same_memory(left, memories[left], right, memories[right]):
                union(left, right)

    groups: dict[str, list[str]] = defaultdict(list)
    for key in keys:
        groups[find(key)].append(key)
    return [sorted(group) for group in groups.values() if len(group) > 1]


def plan_merge(
    group: list[str],
    memories: dict[str, MemoryRecord],
) -> tuple[str, str]:
    """决定这一组合并后保留哪个 key、用哪段内容。

    规则和线上写入保持一致：保留最短的 key（语义最原子，以后同类内容
    还能继续并到它身上），内容取信息量最大的那条。
    """
    keep_key = min(group, key=lambda key: (len(key), key))
    merged_content = max(
        (memories[key].content for key in group),
        key=len,
    )
    return keep_key, merged_content


def main(argv: Sequence[str] | None = None) -> int:
    """扫描全部用户的长期记忆，报告重复组；--apply 时才真的合并。"""
    parser = argparse.ArgumentParser(description="长期记忆重复项检查")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="真的执行合并；不加这个参数只打印报告",
    )
    parser.add_argument(
        "--user",
        default=None,
        help="只处理指定用户；默认扫描全部用户",
    )
    args = parser.parse_args(argv)

    with open_memory_store() as store:
        namespaces = list(store.list_namespaces(prefix=("users",)))
        user_ids = sorted(
            {
                namespace[1]  # 命名空间形如 ("users", user_id, "memories")。
                for namespace in namespaces
                if len(namespace) >= 2
            }
        )
        if args.user:
            user_ids = [user_id for user_id in user_ids if user_id == args.user]

        if not user_ids:
            print("没有找到任何用户的长期记忆。")
            return 0

        found_any = False
        for user_id in user_ids:
            memories = {
                item.key: MemoryRecord.model_validate(item.value)
                for item in store.search(("users", user_id, "memories"))
            }
            groups = group_duplicates(memories)
            if not groups:
                print(f"[{user_id}] {len(memories)} 条记忆，没有发现重复。")
                continue

            found_any = True
            print(f"\n[{user_id}] {len(memories)} 条记忆，发现 {len(groups)} 组重复：")
            for group in groups:
                keep_key, merged_content = plan_merge(group, memories)
                print(f"  - 重复组：{' / '.join(group)}")
                for key in group:
                    print(f"      {key}：{memories[key].content}")
                print(f"    合并后 -> {keep_key}：{merged_content}")
                removed = [key for key in group if key != keep_key]
                if removed:
                    print(f"    将被删除：{', '.join(removed)}")

                if args.apply:
                    save_memory(
                        store,
                        user_id,
                        keep_key,
                        memories[keep_key].category,
                        merged_content,
                    )
                    for key in removed:
                        store.delete(("users", user_id, "memories"), key)
                    print("    已执行。")

        if not found_any:
            print("\n没有发现重复记忆，无需处理。")
        elif not args.apply:
            print(
                "\n以上只是报告，没有改动任何数据。"
                "确认无误后加上 --apply 再执行一次。"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
