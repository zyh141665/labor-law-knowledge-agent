"""单独启动前端 Streamlit 界面。

只启动界面，不含后端。后端在另一个终端用 run_backend.py 启动：

    终端 1（后端）：
        python -m projects.enterprise_knowledge_agent.scripts.run_backend

    终端 2（前端）：
        python -m projects.enterprise_knowledge_agent.scripts.run_frontend

启动前会带访问令牌探测后端 /info：
- 连得上：打印后端版本和法条数，再打开页面
- 连不上：**仍然启动界面**，只在终端提示。因为后端可能正在加载模型，
  或者你打算马上在第三个窗口把它起起来。界面本身有「重试连接」按钮。
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from projects.enterprise_knowledge_agent.app.api_client import AgentApiClient, AgentApiError


PROJECT_ROOT = Path(__file__).resolve().parent.parent  # enterprise_knowledge_agent
LEARNING_ROOT = PROJECT_ROOT.parent.parent  # agent-learning
DEFAULT_API_BASE_URL = "http://127.0.0.1:8000"


def parse_args() -> argparse.Namespace:
    """解析后端地址、界面端口和是否自动打开浏览器。"""
    parser = argparse.ArgumentParser(
        description="单独启动劳动法知识库 Agent 的 Streamlit 前端",
    )
    parser.add_argument(
        "--api",
        default=os.getenv("AGENT_API_BASE_URL", DEFAULT_API_BASE_URL),
        help=f"后端地址，默认 {DEFAULT_API_BASE_URL}；后端换了端口这里要跟着改",
    )
    parser.add_argument(
        "--port",
        default=os.getenv("STREAMLIT_PORT", "8501"),
        help="界面端口，默认 8501",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="不自动打开浏览器，只打印地址",
    )
    return parser.parse_args()


def probe_backend(base_url: str) -> dict[str, object] | None:
    """探一次后端 /info，返回后端自述；连不上返回 None。

    复用轻量 HTTP 客户端，以便加载同一份 .env、携带令牌并禁用本地代理。
    这个客户端不导入模型、检索器或数据库。
    """
    client = AgentApiClient(base_url=base_url, timeout=3.0)
    try:
        data = client.info()  # /info 需要认证，不能发送不带令牌的裸请求。
    except AgentApiError as error:
        print(f"[前端] 后端检查失败：{error.code}；请检查连接和访问令牌配置。")
        return None
    finally:
        client.close()  # 探测完成即释放连接，不与后续 Streamlit 进程共享。
    return data if isinstance(data, dict) else None


def main() -> int:
    args = parse_args()
    base_url = args.api.rstrip("/")

    print(f"[前端] 后端地址 {base_url}")
    info = probe_backend(base_url)
    if info is None:
        print("[前端] 暂未通过后端连接或认证检查，界面仍会打开并显示具体提示。")
        print("[前端] 如果还没启动后端，请在另一个终端运行：")
        print("       python -m projects.enterprise_knowledge_agent.scripts.run_backend")
        print("[前端] 如果后端正在加载模型，等它输出 Uvicorn running 后点重试即可。")
    else:
        model = info.get("model", "未知模型")
        chunks = info.get("knowledge_chunks", "?")
        version = info.get("version", "?")
        print(f"[前端] 后端已就绪：{model} · {chunks} 条法条 · v{version}")

    # 显式指定端口时若被占用，应换端口重启，不能承诺自动递增。
    print(f"[前端] 界面地址 http://localhost:{args.port}（端口被占用时请用 --port 换一个）")
    print("[前端] 停止本服务：在这个窗口按 Ctrl+C")
    print("-" * 70, flush=True)

    env = {**os.environ, "AGENT_API_BASE_URL": base_url}
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(PROJECT_ROOT / "streamlit_app.py"),
        "--server.port",
        str(args.port),
        "--server.headless",
        "true" if args.no_browser else "false",
    ]

    try:
        process = subprocess.Popen(command, cwd=str(LEARNING_ROOT), env=env)
    except OSError as error:
        print(f"[前端] 启动失败：{error}", file=sys.stderr)
        return 1

    try:
        return process.wait()
    except KeyboardInterrupt:
        print("\n[前端] 收到 Ctrl+C，正在停止 …")
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
        print("[前端] 已停止（后端如果还开着，记得一起去关）。")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
