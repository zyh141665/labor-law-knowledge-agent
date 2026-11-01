"""配置带请求追踪 ID 和自动轮转的 UTF-8 应用日志。"""

import logging  # Python 标准日志系统。
from contextvars import ContextVar  # 在当前请求调用链中安全保存 request_id。
from logging.handlers import RotatingFileHandler  # 限制单个日志文件大小并保留备份。
from pathlib import Path  # 构造跨平台日志文件路径。


PROJECT_ROOT = Path(__file__).resolve().parent.parent  # 得到企业知识库 Agent 根目录。
LOG_PATH = PROJECT_ROOT / "logs" / "agent_api.log"  # 保存应用日志的 UTF-8 文件。
LOGGER_NAME = "enterprise_agent"  # 只配置本项目日志，不修改其他库的全局行为。
request_id_context: ContextVar[str] = ContextVar(  # 每个并发请求拥有独立的追踪 ID。
    "request_id",
    default="-",  # 不在 HTTP 请求中的日志使用短横线占位。
)


class RequestIdFilter(logging.Filter):
    """把当前上下文中的 request_id 注入每一条 LogRecord。"""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_context.get()  # Formatter 可以读取 request_id 字段。
        return True  # 返回 True 表示允许当前日志继续输出。


def setup_logging(log_path: Path = LOG_PATH) -> logging.Logger:
    """幂等配置项目日志；重复调用不会重复添加 Handler。"""
    logger = logging.getLogger(LOGGER_NAME)  # 获取本项目共享的父 Logger。
    if logger.handlers:  # Uvicorn reload 或测试重复加载时复用已有配置。
        return logger

    log_path.parent.mkdir(  # 首次启动时建立 logs 目录。
        parents=True,
        exist_ok=True,
    )
    handler = RotatingFileHandler(  # 文件达到上限后自动生成 .1、.2 备份。
        log_path,
        maxBytes=1_000_000,  # 单个日志文件上限约为 1 MB。
        backupCount=3,  # 最多保留三个历史日志文件。
        encoding="utf-8",  # 明确使用 UTF-8，避免中文日志乱码。
    )
    handler.addFilter(RequestIdFilter())  # 为每条文件日志附加请求追踪 ID。
    handler.setFormatter(  # 使用稳定、便于搜索的单行结构。
        logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | "
            "request_id=%(request_id)s | %(message)s"
        )
    )
    logger.addHandler(handler)  # 本项目日志写入轮转文件。
    logger.setLevel(logging.INFO)  # 记录正常请求、警告和错误，不记录调试噪声。
    logger.propagate = False  # 防止同一条日志被根 Logger 再输出一次。
    return logger


def get_logger(component: str) -> logging.Logger:
    """为不同代码模块创建同属 enterprise_agent 的子 Logger。"""
    return logging.getLogger(f"{LOGGER_NAME}.{component}")
