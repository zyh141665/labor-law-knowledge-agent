"""测试请求追踪 ID 能够进入 UTF-8 文件日志。"""

from pathlib import Path  # 标注 pytest 提供的临时目录类型。

from projects.enterprise_knowledge_agent.app.logging_config import (
    LOGGER_NAME,
    get_logger,
    request_id_context,
    setup_logging,
)


def test_request_id_is_written_to_log(tmp_path: Path) -> None:
    """同一调用链中的日志应带有当前 request_id。"""
    logger = setup_logging(tmp_path / "test.log")
    token = request_id_context.set("request-test-001")
    try:
        get_logger("test").info("test_event")
        for handler in logger.handlers:  # 强制刷新，确保断言前内容已经写入磁盘。
            handler.flush()
    finally:
        request_id_context.reset(token)

    log_text = (tmp_path / "test.log").read_text(encoding="utf-8")
    assert "request_id=request-test-001" in log_text
    assert "test_event" in log_text

    for handler in list(logger.handlers):  # 清理全局 Logger，避免影响后续测试或真实启动。
        handler.close()
        logger.removeHandler(handler)
    assert logger.name == LOGGER_NAME
