"""执行一次岗位分析的共享入口。

为什么要单独抽出来：
    API 进程和 worker 进程都要跑同一段"构造初始状态 → invoke → 组装结果"。
    如果两边各写一份，将来加一个字段就会漏改一处 —— 这种 bug 最难查，
    因为它只在某一条代码路径上出现。放在这里，两个进程 import 同一份实现。

    注意：这个文件【不】依赖 FastAPI。
    worker 只是一个后台进程，不该被迫引入 Web 框架及其依赖。
"""

from __future__ import annotations

from typing import Any

from langgraph.errors import GraphRecursionError

from agent.graph import build_graph

MAX_RECURSION_LIMIT = 50

# 进程内缓存已编译的工作流
_graph: Any | None = None


def get_graph() -> Any:
    """获取已编译的工作流，进程内只构建一次。

    为什么不能每个任务都 build_graph()：
        build_graph() 每次都要重新发现 MCP 工具、bind_tools、重建 ToolNode，
        这些是有成本的初始化动作，应该在进程生命周期内只做一次。

    为什么用懒加载而不是在服务启动时构建：
        构建过程会连接 MCP Server 并读取 API Key。如果启动时就构建，
        配置有误会导致整个服务起不来；懒加载让服务始终能正常启动，
        由具体请求来暴露配置问题。
        对 worker 来说同理 —— 它可以在 Redis 空转时正常待命。

    Returns:
        已编译的 LangGraph 应用。
    """
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def build_initial_state(jd: str) -> dict[str, Any]:
    """构造一次分析所需的初始状态。

    Args:
        jd: 岗位 JD 原文。

    Returns:
        符合 agent.state.State 定义的初始状态字典。
    """
    return {
        "user_input": jd,
        "analysis": {},
        "evidence": [],
        "report_text": "",
        "resume_bullets": [],
        "interview_questions": [],
        "messages": [],
        "tool_rounds": 0,
    }


def run_analysis(jd: str) -> dict[str, Any]:
    """执行一次完整分析，返回可序列化的结果字典。

    Args:
        jd: 岗位 JD 原文。

    Returns:
        字段与 api.schemas.AnalyzeResponse 一一对应的字典。
        这里刻意返回 dict 而不是 Pydantic 对象：worker 要把它 JSON 序列化
        存进 Redis，而 API 要把它塞进响应模型 —— dict 是两边都能用的最小公约数。

    Raises:
        GraphRecursionError: 工作流执行步数超限。
        Exception: 上游大模型或 MCP 工具不可用。
    """
    result = get_graph().invoke(
        build_initial_state(jd),
        config={"recursion_limit": MAX_RECURSION_LIMIT},
    )
    return {
        "status": "done",
        "analysis": result.get("analysis", {}),
        "evidence_count": len(result.get("evidence", [])),
        "report_text": result.get("report_text", ""),
        "resume_bullets": result.get("resume_bullets", []),
        "interview_questions": result.get("interview_questions", []),
    }
