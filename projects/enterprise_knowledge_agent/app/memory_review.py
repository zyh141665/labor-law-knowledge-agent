"""保存后端已审核的候选，防止客户端绕过审核、篡改或重放写入请求。"""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from secrets import token_urlsafe
from threading import RLock
from time import monotonic

from .memory_policy import MemoryCandidate


class MemoryReviewError(ValueError):
    """候选未审核、过期、被修改或未确认时拒绝写入。"""


@dataclass(frozen=True)
class ReviewedMemory:
    user_id: str  # 审核结果只对提出该候选的用户有效。
    candidate: MemoryCandidate  # 保存副本，不能信任客户端再次提交的内容。
    original_text: str  # 写入时用原始输入重新运行敏感信息规则。
    expires_at: float  # 单调时钟不受电脑校时影响。


class MemoryReviews:
    """单进程、限容量的临时审核记录；重启后需要重新提取候选。"""

    def __init__(self, ttl_seconds: float = 600, max_pending: int = 1024) -> None:
        self.ttl_seconds = ttl_seconds  # 默认十分钟内必须完成确认。
        self.max_pending = max_pending  # 不让未确认候选无限占用内存。
        self._items: dict[str, ReviewedMemory] = {}
        self._lock = RLock()  # 多个 HTTP 线程不能同时消费同一个审核编号。

    def issue(
        self, user_id: str, candidate: MemoryCandidate, original_text: str
    ) -> str:
        """只由服务层在规则放行后调用，不向客户端提供创建审核记录的接口。"""
        with self._lock:
            now = monotonic()
            self._items = {  # 每次签发顺带清理过期候选。
                key: item for key, item in self._items.items() if item.expires_at > now
            }
            if len(self._items) >= self.max_pending:
                raise MemoryReviewError("待确认记忆过多，请稍后重新提取。")
            review_id = token_urlsafe(32)  # 随机编号不能由用户、key 等可猜信息构造。
            self._items[review_id] = ReviewedMemory(
                user_id=user_id,
                candidate=candidate.model_copy(deep=True),
                original_text=original_text,
                expires_at=now + self.ttl_seconds,
            )
            return review_id

    @contextmanager
    def confirm(
        self,
        review_id: str,
        user_id: str,
        memory_key: str,
        category: str,
        content: str,
        confirmed: bool,
    ) -> Iterator[ReviewedMemory]:
        """验证后在锁内完成写入，成功才消费编号；数据库失败可安全重试。"""
        if confirmed is not True:  # 客户端必须明确提交确认，不做默认同意。
            raise MemoryReviewError("请明确确认保存后再提交。")
        with self._lock:
            reviewed = self._items.get(review_id)
            if reviewed is None or reviewed.user_id != user_id:
                raise MemoryReviewError("没有有效的候选审核记录，请先重新提取。")
            if reviewed.expires_at <= monotonic():
                del self._items[review_id]
                raise MemoryReviewError("候选审核已过期，请重新提取。")
            candidate = reviewed.candidate
            if (memory_key, category, content) != (
                candidate.memory_key,
                candidate.category,
                candidate.content,
            ):
                raise MemoryReviewError("候选内容已被修改，请重新提取并确认。")
            yield reviewed  # 调用者重新过规则并写数据库，异常时不消费编号。
            del self._items[review_id]  # 成功写入后不能再次重放这个请求。
