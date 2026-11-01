"""使用 LangGraph SqliteStore 保存跨会话的用户长期记忆。

除了最基础的读写，本模块还负责**写入前判重**。长期记忆的 key 由模型生成，
同一件事在不同时刻很容易得到不同的 key：只说“回答要简洁，先说结论”这件事，
先被存成 answer_style，下次又变成 answer_style_conclusion_first。两条内容
近似、key 不同的记忆同时进入提示词，模型就要自己去调和两份重复甚至矛盾的
偏好。判重因此放在写入前，而不是靠事后清理。
"""

import re  # 归一化记忆内容时去掉标点和空白。
from collections.abc import Iterator  # 表示 with 代码块会暂时获得一个 SqliteStore。
from contextlib import contextmanager  # 用上下文管理器确保 SQLite 连接最终被关闭。
from datetime import UTC, datetime  # 生成带 UTC 时区的记忆更新时间。
from pathlib import Path  # 以跨平台方式构造数据库文件路径。
from typing import Literal  # 限制记忆类别只能使用预先允许的值。

from langgraph.store.sqlite import SqliteStore  # LangGraph 的跨会话 SQLite 数据存储。
from pydantic import BaseModel, Field  # 定义并校验长期记忆的数据结构。


PROJECT_ROOT = Path(__file__).resolve().parent.parent  # 得到企业知识库 Agent 根目录。
MEMORY_PATH = PROJECT_ROOT / "data" / "memory.sqlite"  # 长期记忆使用独立数据库文件。

# 判重时先去掉的字符：空白与中英文标点。它们不携带语义，
# 留着会让“回答尽量简洁。”和“回答尽量简洁”变成两条不同的记忆。
NOISE_PATTERN = re.compile(r"[\s，。、；：！？…—·（）()《》【】\"'`~!@#$%^&*+=|\\/:;,.<>?-]+")

# 光看 key 前缀会误伤：goal 和 goal_city 也构成“前缀家族”，
# 但一个是“学习 Agent”，一个是“杭州”，显然是两件事。
# 所以 key 关系只作为“候选信号”，还要内容真的重合才判为同一件事。
# 0.2 是实测的分界：真实重复的偏好对约为 0.32，无关短句基本在 0.1 以下。
KEY_RELATION_MIN_OVERLAP = 0.2


class MemoryRecord(BaseModel):
    """一条经过结构校验、可以跨会话使用的长期事实。"""

    category: Literal["profile", "preference", "goal"]  # 事实、偏好或长期目标。
    content: str = Field(min_length=1)  # 真正需要长期保留的文字内容。
    updated_at: str  # 最近一次新增或更新该记忆的 UTC 时间。


class MemorySaveOutcome(BaseModel):
    """一次写入请求最终发生了什么，供界面如实展示，避免“已保存”掩盖跳过。

    - ``insert``：库里没有相近记忆，新增了一条；
    - ``update``：同一个 key 又写了一次，原地覆盖；
    - ``merge``：另一个 key 上已有近似记忆，合并过去，没有多出一条；
    - ``skip``：内容与已有记忆完全一致，只刷新时间戳或干脆不写。
    """

    action: Literal["insert", "update", "merge", "skip"]
    memory_key: str  # 最终这条记忆落在哪个 key 上。
    duplicate_of: str | None = None  # 被判定为重复的已有 key。
    removed_key: str | None = None  # 合并时被清掉的冗余 key；没有删除时为 None。
    reason: str  # 一句话说明为什么是这个动作。


def user_memory_namespace(user_id: str) -> tuple[str, ...]:
    """为每个用户生成相互隔离的长期记忆命名空间。"""
    if not user_id.strip():  # 空用户 ID 会导致不同用户的数据无法正确隔离。
        raise ValueError("user_id 不能为空。")
    return ("users", user_id, "memories")  # 同一 user_id 的所有 thread 共用该分区。


@contextmanager
def open_memory_store() -> Iterator[SqliteStore]:
    """打开长期记忆数据库，并在使用结束后自动关闭连接。"""
    MEMORY_PATH.parent.mkdir(  # 第一次运行时确保 data 目录存在。
        parents=True,  # 如果缺少多级父目录则一并创建。
        exist_ok=True,  # 目录已经存在时不报错。
    )
    with SqliteStore.from_conn_string(str(MEMORY_PATH)) as store:  # 打开 SQLite Store。
        store.setup()  # 创建 Store 首次运行所需的数据表和索引。
        yield store  # 把可用的 Store 暂时交给调用方。


def save_memory(
    store: SqliteStore,
    user_id: str,
    memory_key: str,
    category: Literal["profile", "preference", "goal"],
    content: str,
) -> None:
    """新增长期记忆；相同用户和 memory_key 再次写入时会更新旧值。"""
    record = MemoryRecord(  # 写数据库前先用 Pydantic 检查内容是否合法。
        category=category,
        content=content,
        updated_at=datetime.now(UTC).isoformat(),
    )
    namespace = user_memory_namespace(user_id)  # 找到当前用户独立的记忆分区。
    store.put(
        namespace,  # 决定把记忆存入哪个用户的分区。
        memory_key,  # 这条记忆的稳定名称，例如 learning_goal。
        record.model_dump(),  # 把 Pydantic 对象转换成可存储的普通字典。
    )


def normalize_content(text: str) -> str:
    """去掉标点和空白并统一大小写，让同一个意思在字面上可比。"""
    return NOISE_PATTERN.sub("", text).lower()


def keys_related(key_a: str, key_b: str) -> bool:
    """两个 key 是否指向同一件事：完全相等，或短的那个是长的词干。

    模型给同一件事实起名时，倾向于在旧名字后面追加限定词，例如
    ``answer_style`` → ``answer_style_conclusion_first``。这种“前缀 +
    下划线 + 补充说明”的关系是最强的重复信号，而且判定完全是确定性的。
    """
    if key_a == key_b:
        return True
    short_key, long_key = sorted((key_a, key_b), key=len)
    return long_key.startswith(f"{short_key}_")


def char_overlap(text_a: str, text_b: str) -> float:
    """两段文字的字集合 Jaccard 相似度（0~1）。

    用字而不是词，是因为这里只做粗筛：中文短句换个说法时，字面上的
    重合通常仍然可观，而无关的两句话几乎不共用字。
    """
    set_a = set(normalize_content(text_a))
    set_b = set(normalize_content(text_b))
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def is_same_memory(
    key_a: str,
    record_a: MemoryRecord,
    key_b: str,
    record_b: MemoryRecord,
) -> bool:
    """两条记忆是否指向同一件事。

    这是全项目唯一的判重定义，写入路径和清理脚本都调它，
    避免“线上判重一套规则、清理脚本另一套”这种最难查的偏差。

    两个条件**且**关系：key 属于同一前缀家族，并且内容确实有重合。
    只看 key 会把 goal / goal_city 这种误合并，只看内容又抓不到
    “换个说法重记一遍”的情况。
    """
    if record_a.category != record_b.category:
        return False
    if normalize_content(record_a.content) == normalize_content(record_b.content):
        return True
    if not keys_related(key_a, key_b):
        return False
    return char_overlap(record_a.content, record_b.content) >= KEY_RELATION_MIN_OVERLAP


def find_duplicate(
    store: SqliteStore,
    user_id: str,
    memory_key: str,
    category: str,
    content: str,
) -> tuple[str, MemoryRecord] | None:
    """写入前找出最可能与之重复的已有记忆。

    只在同一个 category 内比较：goal 和 preference 即使措辞相近，
    也不该互相覆盖。先比内容是否全等（确定性最强），再比 key 与内容组合。
    返回 ``(已有 key, 已有记忆)``；没有重复时返回 None。
    """
    probe = MemoryRecord(category=category, content=content, updated_at="")
    memories = list_memories(store, user_id)
    normalized = normalize_content(content)

    # 第一轮只认“内容归一化后完全相同”，这是确定性最强、最不该有争议的情况；
    # 放在前面，是为了让精确命中优先于后面基于 key 家族的推断。
    for key, record in sorted(memories.items()):
        if record.category != category:
            continue
        if normalize_content(record.content) == normalized:
            return key, record

    for key, record in sorted(memories.items()):
        if is_same_memory(key, record, memory_key, probe):
            return key, record
    return None


def save_memory_deduped(
    store: SqliteStore,
    user_id: str,
    memory_key: str,
    category: Literal["profile", "preference", "goal"],
    content: str,
) -> MemorySaveOutcome:
    """写入前先判重，尽量让同一件事只留一条记忆。

    三种处理方式：

    1. 没有相近记忆 → 正常新增；
    2. 同一个 key → 原地覆盖（与原 ``save_memory`` 行为一致）；
    3. 另一个 key 上是同一件事 → **合并**：保留更短的 key（语义更原子，
       以后同类内容还能继续并到它身上），内容取信息量更大的那条。

    只有“新 key 比旧 key 更短”时才会真的删掉旧 key，其余情况旧 key
    原地升级，不会凭空丢失数据。
    """
    duplicate = find_duplicate(store, user_id, memory_key, category, content)
    if duplicate is None:
        save_memory(store, user_id, memory_key, category, content)
        return MemorySaveOutcome(
            action="insert",
            memory_key=memory_key,
            reason="库里没有相近记忆，新增一条。",
        )

    duplicate_key, duplicate_record = duplicate
    if duplicate_key == memory_key:
        save_memory(store, user_id, memory_key, category, content)
        return MemorySaveOutcome(
            action="update",
            memory_key=memory_key,
            duplicate_of=memory_key,
            reason="同一个 key 已有记忆，用新内容覆盖。",
        )

    if normalize_content(duplicate_record.content) == normalize_content(content):
        # 内容一模一样、只是 key 换了个名字。此时不新增也不改写，
        # 让数据保持安静，避免同一句话在库里存两遍。
        return MemorySaveOutcome(
            action="skip",
            memory_key=duplicate_key,
            duplicate_of=duplicate_key,
            reason=f"与已有记忆 {duplicate_key} 内容一致，没有重复写入。",
        )

    keep_key = duplicate_key if len(duplicate_key) <= len(memory_key) else memory_key
    merged_content = (
        duplicate_record.content
        if len(duplicate_record.content) > len(content)
        else content
    )
    save_memory(store, user_id, keep_key, category, merged_content)

    removed_key: str | None = None
    if keep_key == memory_key:  # 新 key 更短，旧的长 key 属于冗余，合并后移除。
        store.delete(user_memory_namespace(user_id), duplicate_key)
        removed_key = duplicate_key

    return MemorySaveOutcome(
        action="merge",
        memory_key=keep_key,
        duplicate_of=duplicate_key,
        removed_key=removed_key,
        reason=f"与已有记忆 {duplicate_key} 指向同一件事，合并为一条。",
    )


def get_memory(
    store: SqliteStore,
    user_id: str,
    memory_key: str,
) -> MemoryRecord | None:
    """读取当前用户的一条指定长期记忆。"""
    namespace = user_memory_namespace(user_id)  # 使用 user_id 定位用户分区。
    item = store.get(namespace, memory_key)  # 使用稳定 key 查询某一类记忆。
    if item is None:  # 第一次使用或该记忆尚未建立时返回 None。
        return None
    return MemoryRecord.model_validate(item.value)  # 将数据库字典恢复成受校验对象。


def list_memories(
    store: SqliteStore,
    user_id: str,
) -> dict[str, MemoryRecord]:
    """读取某个用户的全部长期记忆，并保留每条记忆的稳定 key。"""
    namespace = user_memory_namespace(user_id)  # 只搜索当前用户的记忆分区。
    items = store.search(namespace)  # 不提供 query 时列出该命名空间中的记录。
    return {
        item.key: MemoryRecord.model_validate(item.value)  # 恢复并重新校验每条记忆。
        for item in items
    }


def build_memory_context(
    memories: dict[str, MemoryRecord],
) -> str:
    """把结构化长期记忆转换成模型能够阅读的上下文文本。"""
    if not memories:  # 新用户尚无长期记忆时返回明确说明。
        return "暂无用户长期记忆。"

    lines = ["用户长期记忆："]  # 标明下面内容来自长期记忆而不是企业政策。
    for memory_key, record in sorted(memories.items()):  # 排序使输出和测试保持稳定。
        lines.append(  # 同时保留 key、类别和内容，方便模型理解其含义。
            f"- {memory_key}（{record.category}）：{record.content}"
        )
    return "\n".join(lines)  # 将多条记忆拼成一段可注入提示词的文本。
