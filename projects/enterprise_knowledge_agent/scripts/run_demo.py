"""一键同时启动后端和前端（便利脚本，可选）。

**日常开发请优先用两个终端分别启动**，这样两边的日志各自独立，
前端崩了不影响后端，后端也能单独开 --reload：

    终端 1：python -m projects.enterprise_knowledge_agent.scripts.run_backend
    终端 2：python -m projects.enterprise_knowledge_agent.scripts.run_frontend

这个脚本只是把上面两步合成一条命令，方便演示时少开一个窗口：

    python -m projects.enterprise_knowledge_agent.scripts.run_demo

顺序很重要：先起后端，等 /health 通过，再起前端。
否则前端一打开就是「无法连接后端」，得手动点重试。

按 Ctrl+C 会同时停掉两个进程。
"""

import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent  # enterprise_knowledge_agent
LEARNING_ROOT = PROJECT_ROOT.parent.parent  # agent-learning，uvicorn 必须在这里启动

API_HOST = os.getenv("AGENT_API_HOST", "127.0.0.1")
API_PORT = os.getenv("AGENT_API_PORT", "8000")
STREAMLIT_PORT = os.getenv("STREAMLIT_PORT", "8501")
HEALTH_URL = f"http://{API_HOST}:{API_PORT}/health"
STARTUP_TIMEOUT_SECONDS = 180  # 首次启动要加载 91MB Embedding 模型，给足时间。


def wait_for_backend() -> bool:
    """轮询 /health，直到后端可用或超时。

    只轮询健康检查，不解析日志：日志格式会变，HTTP 状态码不会。
    """
    deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(HEALTH_URL, timeout=3) as response:
                if response.status == 200:
                    return True
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            time.sleep(1.0)
    return False


def main() -> int:
    env = {**os.environ, "AGENT_API_BASE_URL": f"http://{API_HOST}:{API_PORT}"}

    print(f"[run_demo] 启动后端 uvicorn，监听 {API_HOST}:{API_PORT} …")
    backend = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "projects.enterprise_knowledge_agent.app.api:app",
            "--host",
            API_HOST,
            "--port",
            API_PORT,
        ],
        cwd=str(LEARNING_ROOT),
        env=env,
    )

    frontend: subprocess.Popen | None = None
    try:
        print(f"[run_demo] 等待后端就绪（最多 {STARTUP_TIMEOUT_SECONDS} 秒）…")
        if not wait_for_backend():
            print(
                "[run_demo] 后端没能在规定时间内启动。"
                "请单独运行下面的命令查看报错：\n"
                "  python -m projects.enterprise_knowledge_agent.scripts.run_backend",
                file=sys.stderr,
            )
            return 1

        print(f"[run_demo] 后端就绪，启动前端 streamlit，端口 {STREAMLIT_PORT} …")
        print(f"[run_demo] 浏览器打开 http://localhost:{STREAMLIT_PORT}")
        frontend = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(PROJECT_ROOT / "streamlit_app.py"),
                "--server.port",
                STREAMLIT_PORT,
                "--server.headless",
                "false",
            ],
            cwd=str(LEARNING_ROOT),
            env=env,
        )
        frontend.wait()
        return 0
    except KeyboardInterrupt:
        print("\n[run_demo] 收到 Ctrl+C，正在停止 …")
        return 0
    finally:
        for name, process in (("前端", frontend), ("后端", backend)):
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                print(f"[run_demo] {name}已停止。")


if __name__ == "__main__":
    raise SystemExit(main())
