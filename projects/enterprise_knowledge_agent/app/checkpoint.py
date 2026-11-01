"""创建项目使用的 SQLite 检查点保存器。"""

from collections.abc import Iterator  # Iterator 表示 with 代码块暂时取得一个保存器。
from contextlib import contextmanager  # 把普通生成器包装成可安全关闭资源的 with 上下文。
from pathlib import Path  # 以跨平台方式构造 SQLite 数据库文件路径。

from langgraph.checkpoint.sqlite import SqliteSaver  # LangGraph 官方 SQLite 检查点实现。


PROJECT_ROOT = Path(__file__).resolve().parent.parent  # 得到 enterprise_knowledge_agent 根目录。
CHECKPOINT_PATH = PROJECT_ROOT / "data" / "checkpoints_v2.sqlite"  # v2 只保存普通 Python 数据。


@contextmanager
def open_checkpointer() -> Iterator[SqliteSaver]:
    """打开 SQLite 保存器，并在 with 代码块结束时自动关闭数据库连接。"""
    CHECKPOINT_PATH.parent.mkdir(  # 第一次运行时确保 data 目录已经存在。
        parents=True,  # 中间目录不存在时一并创建。
        exist_ok=True,  # 目录已经存在时不报错。
    )
    with SqliteSaver.from_conn_string(  # 使用数据库文件路径创建同步 SQLite 保存器。
        str(CHECKPOINT_PATH)
    ) as checkpointer:
        yield checkpointer  # 暂时把保存器交给 main.py，用完后连接会自动关闭。
