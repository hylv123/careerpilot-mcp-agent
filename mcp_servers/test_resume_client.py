"""MCP stdio 客户端测试：连接 mcp_servers/resume_server.py 并列出工具。

运行方式（在 careerpilot 目录下）：
    .venv\\Scripts\\python.exe mcp_servers\\test_resume_client.py

验收标准：能连上并列出 get_resume、search_projects 两个工具。
（参照 MCP 官方教程第 20 章的 McpClient 思路：stdio_client + ClientSession）
"""

import asyncio
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters, stdio_client

SERVER_SCRIPT = Path(__file__).resolve().parent / "resume_server.py"


async def main() -> int:
    # 用当前解释器拉起服务器子进程（与调用方同一环境，避免解释器不一致）
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(SERVER_SCRIPT)],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            names = [t.name for t in tools.tools]
            print(f"共发现 {len(names)} 个工具: {names}")

            expected = {"get_resume", "search_projects"}
            missing = expected - set(names)
            if missing:
                print(f"[FAIL] 缺少工具: {sorted(missing)}")
                return 1
            print("[OK] get_resume、search_projects 均已列出")

            #验证 get_resume 能真正被调用（验收只要求列出，这一步是加分项）
            result = await session.call_tool("get_resume", {})
            text = getattr(result.content[0], "text", str(result.content[0]))
            print("get_resume() 调用结果:", text)
            return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
