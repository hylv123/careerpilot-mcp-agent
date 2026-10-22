"""Verify that all configured MCP tools can be discovered."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.mcp_tools import get_mcp_tools


EXPECTED_TOOLS = {
    "get_resume",
    "search_projects",
    "save_job",
    "update_status",
    "list_jobs",
    "generate_resume_md",
    "generate_interview_prep",
}


def main() -> int:
    tools = get_mcp_tools()
    names = {tool.name for tool in tools}
    missing = EXPECTED_TOOLS - names
    extra = names - EXPECTED_TOOLS

    print(f"发现 {len(names)} 个工具：{sorted(names)}")
    if missing:
        print(f"[FAIL] 缺少工具：{sorted(missing)}")
        return 1
    if extra:
        print(f"[INFO] 额外工具：{sorted(extra)}")
    print("[OK] MCP 工具发现通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
