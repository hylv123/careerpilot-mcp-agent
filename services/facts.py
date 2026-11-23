"""事实台账：把易过期的数字从"各处手写"收敛成"从代码/产物推导"。

为什么做这件事：
    简历、项目描述、报告里散布着"3 个 MCP Server""9 个工具""124 例测试""30 片段"
    这类数字。它们每改一次代码就过时一次，而校验层只能保证"文案和证据一致"，
    保证不了"证据本身是新的"。这两周已经因此三次把过时数字写进投递材料。

做法：
    数字不手写，全部从真实产物推导——
      MCP Server 数 ← mcp.json
      MCP 工具数   ← 已注册 Server 里的 @mcp.tool()
      离线测试数   ← tests/ 下 test_* 方法（AST 静态统计，不跑测试）
      知识库片段数 ← index.faiss 的 ntotal
      图节点数     ← agent/graph.py 里的 add_node 调用
    然后用正则把文本里"声称的数字"抓出来比对，对不上就是 error。

有意不扫描教学文档与历史评测结果：它们记录的是当时的实现，改了反而失去对照价值。
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Pattern

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_CN_NUM = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}


def _to_int(token: str) -> int | None:
    if token.isdigit():
        return int(token)
    return _CN_NUM.get(token)


def _count_mcp_servers() -> tuple[int, str]:
    config = json.loads((PROJECT_ROOT / "mcp.json").read_text(encoding="utf-8"))
    return len(config.get("mcpServers", {})), "mcp.json"


def _count_mcp_tools() -> tuple[int, str]:
    config = json.loads((PROJECT_ROOT / "mcp.json").read_text(encoding="utf-8"))
    total = 0
    for entry in config.get("mcpServers", {}).values():
        for arg in entry.get("args", []):
            if not str(arg).endswith(".py"):
                continue
            path = PROJECT_ROOT / str(arg)
            if path.exists():
                total += path.read_text(encoding="utf-8").count("@mcp.tool()")
    return total, "mcp.json → @mcp.tool()"


def _count_tests() -> tuple[int, str]:
    total = 0
    for path in (PROJECT_ROOT / "tests").glob("test_*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
                total += 1
    return total, "tests/ 下 test_* 方法"


def _count_chunks() -> tuple[int, str]:
    try:
        import faiss

        return (
            int(faiss.read_index(str(PROJECT_ROOT / "knowledge" / "index" / "index.faiss")).ntotal),
            "knowledge/index/index.faiss",
        )
    except Exception:
        return 0, "index.faiss（未找到）"


def _count_graph_nodes() -> tuple[int, str]:
    text = (PROJECT_ROOT / "agent" / "graph.py").read_text(encoding="utf-8")
    return len(re.findall(r"g\.add_node\(", text)), "agent/graph.py 的 add_node"


@dataclass(frozen=True)
class Fact:
    id: str
    label: str
    expected: int
    source: str
    patterns: tuple[Pattern[str], ...]
    allow_at_least: bool = False

    def accepts(self, claimed: int, at_least: bool) -> bool:
        if claimed == self.expected:
            return True
        # 写成"230+ 例"时，只要不超过真实数量就算对（测试数只会涨）
        return self.allow_at_least and at_least and claimed <= self.expected


_FACTS: list[Fact] | None = None


def derived_facts() -> list[Fact]:
    """按真实产物推导事实（进程内缓存，重启即刷新）。"""
    global _FACTS
    if _FACTS is None:
        servers, servers_src = _count_mcp_servers()
        tools, tools_src = _count_mcp_tools()
        tests, tests_src = _count_tests()
        chunks, chunks_src = _count_chunks()
        nodes, nodes_src = _count_graph_nodes()
        _FACTS = [
            Fact(
                id="mcp.servers",
                label="MCP Server 数量",
                expected=servers,
                source=servers_src,
                patterns=(re.compile(r"([0-9]+|[一二两三四五六七八九十])\s*个\s*(?:stdio\s*)?(?:MCP\s*)?Server"),),
            ),
            Fact(
                id="mcp.tools",
                label="MCP 工具数量",
                expected=tools,
                source=tools_src,
                patterns=(re.compile(r"([0-9]+|[一二两三四五六七八九十])\s*个\s*(?:MCP\s*)?工具"),),
            ),
            Fact(
                id="tests.count",
                label="离线测试数量",
                expected=tests,
                source=tests_src,
                patterns=(re.compile(r"([0-9]+)(\+)?\s*例[^。\n]{0,8}?测试"),),
                allow_at_least=True,
            ),
            Fact(
                id="corpus.chunks",
                label="知识库片段数",
                expected=chunks,
                source=chunks_src,
                patterns=(
                    # 只认"声明知识库规模"的句式，避免把"30 个片段这个规模下……"这类
                    # 评测讨论里的历史数字当成现况
                    re.compile(r"([0-9]+)\s*(余)?\s*个?\s*片段\s*的?\s*知识库"),
                    re.compile(r"知识库[^。\n]{0,8}?([0-9]+)\s*(余)?\s*个?\s*片段"),
                ),
                allow_at_least=True,
            ),
            Fact(
                id="graph.nodes",
                label="工作流节点数",
                expected=nodes,
                source=nodes_src,
                patterns=(re.compile(r"([0-9]+)\s*个?\s*节点"),),
            ),
        ]
    return _FACTS


def reset_cache() -> None:
    global _FACTS
    _FACTS = None


# 这些目录里的数字是历史快照，不该跟着代码变
SKIP_DIRS = ("docs/teaching", "docs/walkthrough", "eval", "tests", ".venv", "logs", "tmp")

DEFAULT_TARGETS = ("knowledge/docs", "output", "README.md", "data/resume.local.json")


def _skip(path: Path) -> bool:
    rel = path.relative_to(PROJECT_ROOT).as_posix()
    return any(rel.startswith(prefix) or f"/{prefix}/" in f"/{rel}" for prefix in SKIP_DIRS)


def lint_text(text: str, *, origin: str = "<text>") -> list[dict]:
    """检查一段文本里"声称的数字"是否与事实台账一致。"""
    issues: list[dict] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        for fact in derived_facts():
            for pattern in fact.patterns:
                for match in pattern.finditer(line):
                    claimed = _to_int(match.group(1))
                    if claimed is None:
                        continue
                    at_least = bool(match.lastindex and match.lastindex >= 2 and match.group(2))
                    if fact.accepts(claimed, at_least):
                        continue
                    issues.append(
                        {
                            "origin": origin,
                            "line": lineno,
                            "fact_id": fact.id,
                            "label": fact.label,
                            "claimed": claimed,
                            "expected": fact.expected,
                            "source": fact.source,
                            "text": line.strip()[:120],
                        }
                    )
    return issues


def lint_paths(paths: list[Path]) -> list[dict]:
    issues: list[dict] = []
    for path in paths:
        if not path.is_file() or _skip(path):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        issues.extend(lint_text(text, origin=str(path.relative_to(PROJECT_ROOT))))
    return issues


def default_targets() -> list[Path]:
    files: list[Path] = []
    for target in DEFAULT_TARGETS:
        path = PROJECT_ROOT / target
        if path.is_dir():
            files.extend(sorted(path.rglob("*.md")))
            if path.name != "output":
                files.extend(sorted(path.rglob("*.json")))
        elif path.exists():
            files.append(path)
    return files


def lint_project() -> dict:
    """扫描默认范围，返回事实清单 + 问题列表。"""
    facts = [
        {"id": fact.id, "label": fact.label, "expected": fact.expected, "source": fact.source}
        for fact in derived_facts()
    ]
    targets = default_targets()
    issues = lint_paths(targets)
    return {"facts": facts, "issues": issues, "scanned": len(targets)}
