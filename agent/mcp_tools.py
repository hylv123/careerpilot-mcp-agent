import asyncio
import json
import sys
from pathlib import Path
from typing import Any
from langchain_core.tools import StructuredTool as SyncStructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from time import perf_counter
from loguru import logger

_MCP_JSON = Path(__file__).resolve().parent.parent / "mcp.json"

_client: MultiServerMCPClient | None = None
_tools_cache: list | None = None

# 工具失败时返回的固定前缀。
# 为什么需要它：工具异常不往外抛（抛了会打断整次分析），而是降级成一句文本。
# 但"降级成文本"就等于调用方无法区分"工具真的成功了"和"工具失败了"——
# 保存岗位时会直接变成静默丢数据。所以给失败文本一个机器可判的前缀，
# 由 is_tool_error() 判定，调用方据此写结构化字段。
TOOL_ERROR_PREFIX = "MCP 工具调用失败："

# 失败文本的两种来源，必须都认：
#   1. TOOL_ERROR_PREFIX —— 本文件 _to_sync_tool 捕获的传输/协议异常；
#   2. "Error executing tool" —— FastMCP 在服务端把工具异常包成的 ToolError
#      （mcp/server/fastmcp/tools/base.py 里 f"Error executing tool {name}: {e}"）。
#      adapter 不抛异常、只把文本塞进 content blocks，isError 标记在这一层丢了，
#      所以只能按 SDK 的约定文本识别。__pycache__ 里也能看到同样的字符串。
TOOL_ERROR_MARKERS = (TOOL_ERROR_PREFIX, "Error executing tool")


def _build_client() -> MultiServerMCPClient:
    """从 mcp.json 构造 MultiServerMCPClient"""
    with open(_MCP_JSON, encoding="utf-8") as f:
        servers = json.load(f)["mcpServers"]

    connections = {}
    for name, cfg in servers.items():
        command = cfg["command"]
        # 保证子进程用当前 venv 的 python，而不是 PATH 上的任意 python
        if command in ("python", "python3"):
            command = sys.executable
        args = [
            # mcp.json 里的相对路径（如 mcp_servers/xxx.py）基于 mcp.json 所在目录解析
            str((_MCP_JSON.parent / a).resolve()) if not Path(a).is_absolute() else a
            for a in cfg.get("args", [])
        ]
        connections[name] = {
            "command": command,
            "args": args,
            "transport": cfg.get("transport", "stdio"),
        }

    return MultiServerMCPClient(connections)

def _summarize_payload(payload: Any, limit: int = 200) -> str:
    """限制日志中的工具参数长度，避免泄露或刷屏。"""

    text = repr(payload)

    if len(text) <= limit:
        return text

    return text[:limit] + "..."


def _normalize_tool_result(result: Any) -> str:
    """Convert MCP content blocks into concise text for LangChain and CLI use."""

    if isinstance(result, str):
        return result
    if isinstance(result, list):
        texts = [
            str(item.get("text", ""))
            for item in result
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        if texts:
            return "\n".join(texts)
    if isinstance(result, dict):
        content = result.get("content")
        if content is not None:
            return _normalize_tool_result(content)
    return str(result)

def is_tool_error(result: Any) -> bool:
    """判断一次工具调用是不是"失败降级成文本"。

    Args:
        result: 工具调用的返回值（归一化后的字符串，或原始 content blocks）。

    Returns:
        True 表示这次调用失败（调用方不该把它当成成功）。

    为什么不能用异常：`_to_sync_tool` 刻意把异常吞成文本，保证一次工具失败
    不会打断整次分析。代价就是调用方失去了"成功/失败"的信号，
    只能靠 TOOL_ERROR_MARKERS 把信号找回来。

    为什么还要处理 list：adapter 在服务端工具抛错时返回的是
    [{'type': 'text', 'text': 'Error executing tool ...'}] 这样的内容块，
    而不是抛异常；直接传进来时也必须能识别。
    """
    if isinstance(result, str):
        text = result
    elif isinstance(result, list):
        text = "\n".join(
            str(item.get("text", ""))
            for item in result
            if isinstance(item, dict) and item.get("type") == "text"
        )
    else:
        return False
    return text.lstrip().startswith(TOOL_ERROR_MARKERS)


def _to_sync_tool(tool):
    """
    MCP 工具只实现异步调用（_arun），ToolNode 同步执行时会报
    'StructuredTool does not support sync invocation'。
    这里包一个同步壳：保留原 args_schema，func 内部用 asyncio.run 跑异步调用。
    注意不能使用 langchain_core.tools.Tool —— 它是单输入工具，
    会把参数字典拆散成单个位置参数，导致 MCP 工具收到裸值。
    """
    def _run(*args, **kwargs):
        if args:
            payload = args[0] if len(args) == 1 else args
        else:
            payload = kwargs

        started = perf_counter()

        logger.info(
            "[mcp] 调用工具 {}，参数摘要={}",
            tool.name,
            _summarize_payload(payload),
        )

        try:
            result = asyncio.run(tool.ainvoke(payload))

            logger.info(
                "[mcp] 工具 {} 调用成功，耗时 {:.3f}s",
                tool.name,
                perf_counter() - started,
            )

            return _normalize_tool_result(result)

        except Exception as exc:
            logger.exception(
                "[mcp] 工具 {} 调用失败，耗时 {:.3f}s，"
                "错误类型={}，错误={}",
                tool.name,
                perf_counter() - started,
                type(exc).__name__,
                exc,
            )

            return (
                f"{TOOL_ERROR_PREFIX}{tool.name}："
                f"{type(exc).__name__}: {exc}"
            )
    return SyncStructuredTool(
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_schema,
        func=_run,
    )


def get_mcp_tools() -> list:
    """获取并缓存全部 MCP 工具。"""

    global _client, _tools_cache

    if _tools_cache is not None:
        logger.debug(
            "[mcp] 使用工具缓存，共 {} 个工具",
            len(_tools_cache),
        )
        return _tools_cache

    started = perf_counter()
    logger.info("[mcp] 开始连接 MCP Server 并发现工具")

    try:
        _client = _build_client()
        async_tools = asyncio.run(_client.get_tools())
        _tools_cache = [
            _to_sync_tool(tool)
            for tool in async_tools
        ]

        logger.info(
            "[mcp] 工具发现成功，共 {} 个，耗时 {:.3f}s，tools={}",
            len(_tools_cache),
            perf_counter() - started,
            [tool.name for tool in _tools_cache],
        )

        return _tools_cache

    except Exception:
        logger.exception(
            "[mcp] 工具发现失败，耗时 {:.3f}s",
            perf_counter() - started,
        )
        raise

if __name__ == "__main__":
    for t in get_mcp_tools():
        print(f"- {t.name}: {t.description.splitlines()[0] if t.description else ''}")