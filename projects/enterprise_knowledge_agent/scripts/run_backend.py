"""单独启动后端 FastAPI 服务。

这个脚本只负责后端。前端在另一个终端窗口用 run_frontend.py 启动：

    终端 1（后端）：
        python -m projects.enterprise_knowledge_agent.scripts.run_backend

    终端 2（前端）：
        python -m projects.enterprise_knowledge_agent.scripts.run_frontend

第一个终端会停在这里输出 uvicorn 的日志，属于正常现象，不要关。
关闭后端就是这个窗口按 Ctrl+C。

改端口用参数，前端要跟着改：
    终端 1：... run_backend --port 8010
    终端 2：... run_frontend --api http://127.0.0.1:8010
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent  # enterprise_knowledge_agent
LEARNING_ROOT = PROJECT_ROOT.parent.parent  # agent-learning，uvicorn 必须在这里启动


def parse_args() -> argparse.Namespace:
    """解析监听地址、端口和是否自动重载。"""
    parser = argparse.ArgumentParser(
        description="单独启动劳动法知识库 Agent 的后端服务",
    )
    parser.add_argument(
        "--host",
        default=os.getenv("AGENT_API_HOST", "127.0.0.1"),
        help="监听地址，默认 127.0.0.1；只有需要局域网访问时才改成 0.0.0.0",
    )
    parser.add_argument(
        "--port",
        default=os.getenv("AGENT_API_PORT", "8000"),
        help="监听端口，默认 8000",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="代码改动后自动重启，开发时用；首次加载模型会多花十几秒",
    )
    return parser.parse_args()


def build_command(args: argparse.Namespace) -> list[str]:
    """拼出 uvicorn 命令行。

    用 ``-m`` 而不是 import 后调用，是为了让 uvicorn 自己在子进程里
    管理应用生命周期；直接在当前进程 import 会让 Ctrl+C 行为变得难以预料。
    """
    command = [
        sys.executable,
        "-m",
        "uvicorn",
        "projects.enterprise_knowledge_agent.app.api:app",
        "--host",
        str(args.host),
        "--port",
        str(args.port),
    ]
    if args.reload:
        # 只监听 app/ 目录。整个仓库一起监听时，data/ 下 SQLite 的每次写入
        # 都会触发重启，导致刚问完一句话后端就自己关了。
        command += ["--reload", "--reload-dir", str(PROJECT_ROOT / "app")]
    return command


def main() -> int:
    args = parse_args()
    base_url = f"http://{args.host}:{args.port}"

    print(f"[后端] 正在启动，监听 {base_url}")
    print("[后端] 首次启动需要加载 Embedding 模型，看到 Uvicorn running 即就绪")
    print(f"[后端] 接口文档 {base_url}/docs")
    print("[后端] 前端请在另一个终端运行：")
    print(f"       python -m projects.enterprise_knowledge_agent.scripts.run_frontend --api {base_url}")
    print("[后端] 停止本服务：在这个窗口按 Ctrl+C")
    print("-" * 70, flush=True)

    try:
        process = subprocess.Popen(build_command(args), cwd=str(LEARNING_ROOT))
    except OSError as error:  # 极少数情况下解释器路径不对，给出可读提示而不是堆栈。
        print(f"[后端] 启动失败：{error}", file=sys.stderr)
        return 1

    try:
        return process.wait()
    except KeyboardInterrupt:
        # Windows 下 Ctrl+C 会同时发给同控制台的子进程，子进程通常已经在退出，
        # 这里再兜一次，避免留下占着 8000 端口的僵尸进程。
        print("\n[后端] 收到 Ctrl+C，正在停止 …")
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
        print("[后端] 已停止。")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
