"""Compile the graph without starting MCP subprocesses or calling the model."""

import sys
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import agent.graph as graph_module


def has_path(graph, source: str, target: str) -> bool:
    seen: set[str] = set()

    def walk(node: str) -> bool:
        if node == target:
            return True
        if node in seen:
            return False
        seen.add(node)
        return any(
            walk(edge.target)
            for edge in graph.edges
            if edge.source == node
        )

    return walk(source)


def has_direct_edge(graph, source: str, target: str) -> bool:
    """判断 source -> target 是否存在【直接】的边。

    Args:
        graph: LangGraph 静态图对象。
        source: 起点节点名。
        target: 终点节点名。

    Returns:
        存在直接边返回 True。

    为什么不能只用 has_path：
        has_path 只证明"可达"，中间可以绕好几跳。而条件边必须能被 LangGraph
        静态推导出来（靠 add_conditional_edges 传 path_map），
        只看可达性的话，"tools -> agent -> dispatch" 也会被误判成
        "tools 能直接去 dispatch"，这条断言就失去意义了。
    """
    return any(
        edge.source == source and edge.target == target for edge in graph.edges
    )


def build_static_graph():
    """编译图，但把 MCP 工具强制为空（不起子进程、不碰网络）。

    用 patch 而不是直接赋值 `graph_module.get_mcp_tools = ...`：
    直接赋值会污染同进程里后续的测试，patch 退出即恢复。
    """
    with patch.object(graph_module, "get_mcp_tools", lambda: []):
        return graph_module.build_graph()


def check_graph() -> list[tuple[str, bool]]:
    """编译图并返回 15 条结构断言的结果。

    抽成函数是为了能被测试直接调用（tests/test_graph_topology.py），
    这样"图断言"才真的在 CI 里跑，而不是只存在于一个手动脚本。
    """
    graph = build_static_graph().get_graph()

    return [
        ("START -> analyze", has_path(graph, "__start__", "analyze")),
        ("analyze -> match", has_path(graph, "analyze", "match")),
        ("match -> agent", has_path(graph, "match", "agent")),
        ("agent -> tools", has_path(graph, "agent", "tools")),
        ("tools -> agent", has_path(graph, "tools", "agent")),
        # 工具跑完不再是"无条件回 agent"：轮次用完时必须能直接去 dispatch
        ("tools -> agent（直接边）", has_direct_edge(graph, "tools", "agent")),
        ("tools -> dispatch（直接边）", has_direct_edge(graph, "tools", "dispatch")),
        # 并行扇出：agent -> dispatch -> 三个生成节点
        ("agent -> dispatch", has_path(graph, "agent", "dispatch")),
        ("dispatch -> report", has_path(graph, "dispatch", "report")),
        ("dispatch -> rewrite", has_path(graph, "dispatch", "rewrite")),
        ("dispatch -> questions", has_path(graph, "dispatch", "questions")),
        # 并行扇入：三个生成节点都汇聚到 save，且不再有 report -> rewrite 这类串行边
        ("report -> save", has_path(graph, "report", "save")),
        ("rewrite -> save", has_path(graph, "rewrite", "save")),
        ("questions -> save", has_path(graph, "questions", "save")),
        ("save -> END", has_path(graph, "save", "__end__")),
    ]


def main() -> int:
    checks = check_graph()
    for name, passed in checks:
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")
    return 0 if all(passed for _, passed in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())

