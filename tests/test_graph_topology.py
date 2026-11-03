"""图拓扑断言进 CI。

以前这些断言只存在于 `scripts/verify_graph.py`——一个手动脚本，CI 不跑它。
也就是说"图被改坏"和"CI 绿"可以同时成立。这里把同一个 `check_graph()`
接进 unittest，让 `discover -s tests` 就能覆盖。
"""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "verify_graph.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("_verify_graph", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_verify = _load_script()


class GraphTopologyTests(unittest.TestCase):
    def test_all_topology_checks_pass(self) -> None:
        failed = [name for name, ok in _verify.check_graph() if not ok]
        self.assertEqual(failed, [], f"图结构断言失败：{failed}")

    def test_expected_node_set(self) -> None:
        # 节点名是"接线约定"：多一个少一个都说明图被改过
        expected = {
            "analyze", "match", "agent", "tools", "dispatch",
            "report", "rewrite", "questions", "save",
        }
        graph = _verify.build_static_graph().get_graph()
        nodes = set(graph.nodes) - {"__start__", "__end__"}
        self.assertEqual(nodes, expected)

    def test_generation_nodes_are_parallel_not_chained(self) -> None:
        # 三个生成节点必须各自从 dispatch 扇出、各自汇入 save；
        # 一旦被改成 report -> rewrite 这种串行链，耗时会翻倍
        graph = _verify.build_static_graph().get_graph()
        edges = {(edge.source, edge.target) for edge in graph.edges}

        for node in ("report", "rewrite", "questions"):
            self.assertIn(("dispatch", node), edges, f"{node} 不是从 dispatch 扇出的")
            self.assertIn((node, "save"), edges, f"{node} 没有汇入 save")

        self.assertNotIn(("report", "rewrite"), edges)
        self.assertNotIn(("rewrite", "questions"), edges)


if __name__ == "__main__":
    unittest.main()
