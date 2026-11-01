"""统一收口长期记忆的读取、提取、判重和写入。

为什么需要这一层：

长期记忆的完整流程是「模型提候选 → 本地规则门控 → 人工确认 → 判重写入」，
四步。以前这套流程在 CLI 和 HTTP 接口里各写了一遍，改动时要同步两处，
而且两边很容易走偏（例如 CLI 漏了判重）。现在只保留这里一份，
``app/main.py`` 和 ``app/api.py`` 都只负责输入输出。
"""

from typing import Literal

from langgraph.store.base import BaseStore  # 读取跨会话的用户长期记忆。
from openai import OpenAI  # 调用模型提取候选记忆。
from pydantic import BaseModel

from .memory import (  # 复用已验证的记忆读写与判重实现。
    MemoryRecord,
    MemorySaveOutcome,
    build_memory_context,
    find_duplicate,
    list_memories,
    save_memory_deduped,
)
from .memory_policy import (  # 候选提取与本地安全门控。
    MemoryCandidate,
    check_memory_candidate,
    extract_memory_candidate,
)
from .memory_review import MemoryReviewError, MemoryReviews  # 后端保存已审核候选，拒绝绕过与篡改。


class MemoryProposal(BaseModel):
    """一次「提取候选记忆」的全部结论，供界面展示后由用户决定是否保存。"""

    candidate: MemoryCandidate  # 模型给出的候选项。
    allowed: bool  # 本地规则是否放行；False 时界面不应给出保存按钮。
    reason: str  # 放行或拒绝的原因，直接展示给用户。
    duplicate_of: str | None = None  # 判定为重复的已有 key；None 表示是新记忆。
    review_id: str | None = None  # 仅放行的候选有一次性审核编号。


class MemoryService:
    """长期记忆的唯一业务入口，CLI 和 HTTP 接口共用同一份实现。

    ``client`` 允许为 None，因为「列记忆」和「写记忆」根本不需要模型——
    清理脚本、后台任务只需要这两件事，不该被迫先配一个 API Key。
    只有 ``propose``（要调模型提候选）会要求客户端存在。
    """

    def __init__(
        self,
        client: OpenAI | None,
        model: str,
        store: BaseStore,
        reviews: MemoryReviews | None = None,
    ) -> None:
        self.client = client  # 复用主流程的模型客户端，不额外创建连接。
        self.model = model  # 提取候选记忆时使用的模型名称。
        self.store = store  # 按 user_id 隔离的长期记忆存储。
        self.reviews = reviews if reviews is not None else MemoryReviews()  # HTTP 请求间共享同一审核表。

    def list(self, user_id: str) -> dict[str, MemoryRecord]:
        """读取某个用户的全部长期记忆。"""
        return list_memories(self.store, user_id)

    def context(self, user_id: str) -> str:
        """把长期记忆整理成可直接注入提示词的文本。"""
        return build_memory_context(self.list(user_id))

    def propose(self, user_id: str, text: str) -> MemoryProposal:
        """让模型提候选，再用确定性规则决定能否进入人工确认阶段。"""
        if self.client is None:  # 只有这一步真的需要模型。
            raise RuntimeError("当前 MemoryService 没有模型客户端，无法提取候选记忆。")

        existing = self.list(user_id)  # 已有记忆既用于提示词，也用于本地判重。
        candidate = extract_memory_candidate(
            self.client,
            self.model,
            text,
            existing=existing,
        )
        allowed, reason = check_memory_candidate(candidate, text)
        if not allowed:
            # 没通过门控就不必判重，避免给出“跟某条重复”这种会误导人的提示。
            return MemoryProposal(
                candidate=candidate,
                allowed=False,
                reason=reason,
            )

        return MemoryProposal(
            candidate=candidate,
            allowed=True,
            reason=reason,
            duplicate_of=self._resolve_duplicate(user_id, candidate, existing),
            review_id=self.reviews.issue(user_id, candidate, text),
        )

    def save(
        self,
        user_id: str,
        memory_key: str,
        category: Literal["profile", "preference", "goal"],
        content: str,
        *,
        review_id: str = "",
        confirmed: bool = False,
    ) -> MemorySaveOutcome:
        """必须对应后端审核过的原候选，明确确认后才判重写入。"""
        with self.reviews.confirm(
            review_id, user_id, memory_key, category, content, confirmed
        ) as reviewed:
            allowed, reason = check_memory_candidate(  # 最终写入仍再次执行本地规则。
                reviewed.candidate, reviewed.original_text
            )
            if not allowed:
                raise MemoryReviewError(reason)
            return save_memory_deduped(
                self.store, user_id, memory_key, category, content
            )

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    def _resolve_duplicate(
        self,
        user_id: str,
        candidate: MemoryCandidate,
        existing: dict[str, MemoryRecord],
    ) -> str | None:
        """确定这条候选究竟和哪条已有记忆重复。

        优先级：确定性规则 > 模型判断。规则能命中时以规则为准，因为它可复现；
        规则没命中时才看模型的 ``duplicate_of``，并且必须校验那个 key
        确实是该用户已有的记忆，防止模型凭空编一个 key 出来。
        """
        duplicate = find_duplicate(
            self.store,
            user_id,
            candidate.memory_key,
            candidate.category,
            candidate.content,
        )
        if duplicate is not None:
            return duplicate[0]

        claimed = candidate.duplicate_of.strip()
        if claimed and claimed in existing:
            return claimed
        return None
