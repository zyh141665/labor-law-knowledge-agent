"""HTTP 接口的令牌认证。

三个关键取舍：

**令牌绑定用户，而不是只做一道门。**
如果只放一个全局共享令牌，任何拿到它的人把请求里的 ``user_id`` 改一下
就能读别人的聊天记录和长期记忆——认证做了等于没做。所以令牌同时也
决定了“你是谁”，后端从令牌推导身份，不信客户端自报。

**恒定时间比较。**
用 ``hmac.compare_digest`` 而不是 ``==``：普通字符串比较在第一个不同的
字符处就返回，攻击者能靠响应时间逐字节把令牌试出来。这里还刻意把
整张表遍历完再返回，不匹配到就提前 break。

**没配令牌就拒绝启动。**
“忘了配”导致认证静默失效，比启动时直接报错危险得多。宁可起不来，
也不要起一个看着有认证、实际上谁都能进的服务。
"""

import hmac  # 恒定时间比较，避免通过响应时间试探令牌。
import os  # 读取 AGENT_API_TOKENS 环境变量。
import re  # 校验用户 ID 的字符范围。

from pydantic import BaseModel

from .config import load_env  # 复用与主流程相同的 .env 读取规则。


ENV_NAME = "AGENT_API_TOKENS"  # 令牌表所在的环境变量名。
# 与 API 层 user_id 的校验规则保持一致：只允许字母、数字、下划线、连字符。
# 两处规则不一致会导致“令牌里的身份合法、但请求参数被 422 拒掉”这种怪事。
USER_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
MIN_TOKEN_LENGTH = 8  # 太短的令牌既容易被猜，也容易和别的变量撞上。
SEPARATOR = ":"  # 每个条目的格式是 token:user_id。


class AuthConfigError(RuntimeError):
    """令牌配置缺失或格式不合法；由启动流程抛出，阻止服务带病运行。"""


class Principal(BaseModel):
    """一次请求背后的身份，由令牌推导而来。"""

    user_id: str  # 该令牌绑定的用户，所有读写都以它为边界。
    token_label: str  # 令牌末四位的掩码，只用于日志排查，不泄露令牌本身。


def parse_token_map(raw: str) -> dict[str, str]:
    """把 ``token:user, token2:user2`` 解析成 令牌 → 用户 的映射。

    允许用换行代替逗号，因为把多个令牌塞进一行环境变量不好读。
    任何一处格式错误都直接报错，而不是跳过：静默跳过一条，结果是
    那个人怎么也登不上，却看不出原因。
    """
    mapping: dict[str, str] = {}
    entries = [chunk.strip() for chunk in raw.replace("\n", ",").split(",")]

    for entry in entries:
        if not entry:  # 结尾多一个逗号、或多一个换行，属于正常书写习惯。
            continue
        token, separator, user_id = entry.partition(SEPARATOR)
        token, user_id = token.strip(), user_id.strip()
        if not separator or not token or not user_id:
            raise AuthConfigError(
                f"令牌条目格式应为 token:user_id，实际是 {entry!r}。"
            )
        if len(token) < MIN_TOKEN_LENGTH:
            raise AuthConfigError(
                f"令牌长度不能少于 {MIN_TOKEN_LENGTH} 个字符（用户 {user_id}）。"
            )
        if not USER_ID_PATTERN.fullmatch(user_id):
            raise AuthConfigError(
                f"用户 ID 只能包含字母、数字、下划线和连字符：{user_id!r}。"
            )
        if token in mapping:
            raise AuthConfigError(f"令牌 {token[:4]}… 重复出现。")
        mapping[token] = user_id

    if not mapping:
        raise AuthConfigError(f"{ENV_NAME} 里没有任何有效令牌。")
    return mapping


def load_token_map() -> dict[str, str]:
    """从环境变量加载令牌表；没有配置时抛错，让后端启动失败。"""
    load_env()
    raw = os.getenv(ENV_NAME, "").strip()
    if not raw:
        raise AuthConfigError(
            f"没有配置 {ENV_NAME}，后端拒绝启动。"
            "请在 .env 中设置，格式为 token:user_id，多个用逗号分隔；"
            "例如 AGENT_API_TOKENS=my-secret-token-1:alice,my-secret-token-2:bob"
        )
    return parse_token_map(raw)


def parse_bearer(authorization: str | None) -> str | None:
    """从 ``Authorization: Bearer xxx`` 里取出令牌；格式不对时返回 None。

    scheme 比较不区分大小写：HTTP 头里 ``bearer`` 和 ``Bearer`` 都常见。
    """
    if not authorization:
        return None
    scheme, separator, credential = authorization.partition(" ")
    if not separator or scheme.strip().lower() != "bearer":
        return None
    credential = credential.strip()
    return credential or None


def authenticate(
    token_map: dict[str, str],
    authorization: str | None,
) -> Principal | None:
    """校验请求头，返回对应身份；不通过时返回 None。

    遍历完整张表而不是命中即返回，是为了让“命中第 1 条”和“命中第 5 条”
    耗时一致——否则可以从响应时间里猜出令牌表的规模和顺序。
    """
    presented = parse_bearer(authorization)
    if presented is None:
        return None

    matched_user: str | None = None
    matched_token = ""
    for token, user_id in token_map.items():
        if hmac.compare_digest(presented, token) and matched_user is None:
            matched_user = user_id
            matched_token = token
    if matched_user is None:
        return None
    return Principal(
        user_id=matched_user,
        token_label=f"…{matched_token[-4:]}",
    )
