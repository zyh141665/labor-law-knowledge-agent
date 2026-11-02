"""本机演示用账号与会话存储；密码慢哈希，令牌只存摘要，不复用共享身份。"""

import hashlib  # 密码用PBKDF2，随机会话令牌用SHA256摘要检索。
import hmac  # 校验密码摘要时使用恒定时间比较。
import re
import secrets  # 每个账号的盐与每次登录的令牌都独立随机生成。
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from threading import RLock
from uuid import uuid4

from .auth import Principal


ACCOUNT_PATH = Path(__file__).resolve().parent.parent / "data" / "accounts.sqlite"
PASSWORD_ITERATIONS = 600000  # PBKDF2-HMAC-SHA256工作因子；不使用快速SHA256存密码。
SESSION_SECONDS = 8 * 60 * 60
USERNAME_PATTERN = re.compile(r"^[a-z][a-z0-9_-]{2,31}$")


class AccountError(ValueError):
    def __init__(self, code, message, status=400):
        super().__init__(message)
        self.code, self.message, self.status = code, message, status


def normalize_username(username):
    normalized = username.strip().lower()  # 避免Alice/alice变成两个账号。
    if not USERNAME_PATTERN.fullmatch(normalized):
        raise AccountError(
            "invalid_username",
            "用户名须为3—32位字母、数字、下划线或连字符，且以字母开头。",
        )
    return normalized


def password_hash(password, salt, iterations=PASSWORD_ITERATIONS):
    if not 12 <= len(password) <= 128:
        raise AccountError(
            "invalid_password", "密码须为12—128个字符；请使用本项目专用密码。"
        )
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt), iterations
    ).hex()


class AccountStore:
    """每次操作单独连接；锁与SQL事务保护本进程并发，SQLite保存跨重启状态。"""

    def __init__(self, path=ACCOUNT_PATH, clock=time.time):
        self.path, self.clock, self.lock = Path(path), clock, RLock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._db() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS accounts (
                    user_id TEXT PRIMARY KEY, username TEXT NOT NULL UNIQUE,
                    salt TEXT NOT NULL, password_hash TEXT NOT NULL, iterations INTEGER NOT NULL,
                    created_at INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS account_sessions (
                    token_hash TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES accounts(user_id),
                    expires_at INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS session_expiry ON account_sessions(expires_at);
                CREATE TABLE IF NOT EXISTS auth_attempts (
                    bucket TEXT PRIMARY KEY, started_at INTEGER NOT NULL, count INTEGER NOT NULL
                );
            """)
        self.dummy_salt = secrets.token_hex(16)  # 不存在的用户也执行同等成本密码运算。

    @contextmanager
    def _db(self):
        db = sqlite3.connect(str(self.path), timeout=5)
        db.execute("PRAGMA foreign_keys=ON")
        db.row_factory = sqlite3.Row
        try:
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()  # Connection的with本身不会关闭连接，需要显式关闭。

    def _limit(self, scope, value, maximum):
        """记录所有尝试（含成功），按真实连接IP/用户名限制，不信任X-Forwarded-For。"""
        now = int(self.clock())
        bucket = scope + ":" + hashlib.sha256(value.encode()).hexdigest()
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")  # 多个连接不能同时越过计数上限。
            db.execute("DELETE FROM auth_attempts WHERE started_at <= ?", (now - 300,))
            row = db.execute(
                "SELECT count FROM auth_attempts WHERE bucket=?", (bucket,)
            ).fetchone()
            if row and row[0] >= maximum:
                raise AccountError(
                    "auth_rate_limited", "尝试过于频繁，请5分钟后再试。", 429
                )
            db.execute(
                "INSERT INTO auth_attempts VALUES (?,?,1) ON CONFLICT(bucket) DO UPDATE SET count=count+1",
                (bucket, now),
            )

    def register(self, username, password, peer="local"):
        username = normalize_username(username)
        with self.lock:
            self._limit("register-ip", peer, 10)
            salt = secrets.token_hex(16)
            digest = password_hash(password, salt)
            user_id = "acct_" + uuid4().hex  # 不能注册成旧令牌用户或指定别人的user_id。
            try:
                with self._db() as db:
                    db.execute(
                        "INSERT INTO accounts VALUES (?,?,?,?,?,?)",
                        (
                            user_id,
                            username,
                            salt,
                            digest,
                            PASSWORD_ITERATIONS,
                            int(self.clock()),
                        ),
                    )
            except sqlite3.IntegrityError:
                raise AccountError(
                    "username_unavailable", "该用户名不可用，请更换。", 409
                ) from None
            return {"user_id": user_id, "username": username}

    def login(self, username, password, peer="local"):
        username = normalize_username(username)
        with self.lock:
            self._limit("login-ip", peer, 60)
            self._limit("login-user", username, 10)
            with self._db() as db:
                row = db.execute(
                    "SELECT * FROM accounts WHERE username=?", (username,)
                ).fetchone()
                digest = password_hash(
                    password,
                    row["salt"] if row else self.dummy_salt,
                    row["iterations"] if row else PASSWORD_ITERATIONS,
                )
                if row is None or not hmac.compare_digest(digest, row["password_hash"]):
                    raise AccountError("invalid_credentials", "用户名或密码错误。", 401)
                now = int(self.clock())
                db.execute("DELETE FROM account_sessions WHERE expires_at<=?", (now,))
                # 每个账号保留最多20个活跃登录，防止成功登录无限堆积会话。
                db.execute(
                    "DELETE FROM account_sessions WHERE user_id=? AND token_hash NOT IN (SELECT token_hash FROM account_sessions WHERE user_id=? ORDER BY expires_at DESC LIMIT 19)",
                    (row["user_id"], row["user_id"]),
                )
                token = secrets.token_urlsafe(32)
                expires_at = now + SESSION_SECONDS
                db.execute(
                    "INSERT INTO account_sessions VALUES (?,?,?)",
                    (
                        hashlib.sha256(token.encode()).hexdigest(),
                        row["user_id"],
                        expires_at,
                    ),
                )
                return {
                    "access_token": token,
                    "token_type": "bearer",
                    "user_id": row["user_id"],
                    "username": row["username"],
                    "expires_at": expires_at,
                }

    def authenticate(self, token):
        if not token or not 20 <= len(token) <= 128:
            return None
        digest = hashlib.sha256(token.encode()).hexdigest()
        with self.lock, self._db() as db:
            row = db.execute(
                "SELECT user_id FROM account_sessions WHERE token_hash=? AND expires_at>?",
                (digest, int(self.clock())),
            ).fetchone()
        return Principal(user_id=row[0], token_label="…" + token[-4:]) if row else None

    def logout(self, token):
        with self.lock, self._db() as db:
            db.execute(
                "DELETE FROM account_sessions WHERE token_hash=?",
                (hashlib.sha256(token.encode()).hexdigest(),),
            )
