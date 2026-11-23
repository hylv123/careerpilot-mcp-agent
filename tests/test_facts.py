"""事实台账测试。

锁三件事：
  1. 关键数字确实是从产物推导的（不是手写常量）；
  2. 过时数字能被抓出来（这几周真实踩过的四种）；
  3. 历史快照（教学文档、评测结果）不参与比对，避免把"当时的实现"当成 bug。
"""

from __future__ import annotations

import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from api.main import app
from services import facts


class DerivedFactsTests(unittest.TestCase):
    def setUp(self) -> None:
        facts.reset_cache()

    def test_facts_come_from_artifacts(self) -> None:
        by_id = {fact.id: fact for fact in facts.derived_facts()}
        self.assertEqual(by_id["mcp.servers"].expected, 2)
        self.assertEqual(by_id["mcp.tools"].expected, 4)
        self.assertEqual(by_id["graph.nodes"].expected, 9)
        self.assertGreater(by_id["tests.count"].expected, 100)
        self.assertGreaterEqual(by_id["corpus.chunks"].expected, 1)

    def test_each_fact_points_at_a_real_source(self) -> None:
        for fact in facts.derived_facts():
            self.assertTrue(fact.source)
            path = fact.source.split(" ")[0]
            if path.endswith((".json", ".py", ".faiss")):
                self.assertTrue((facts.PROJECT_ROOT / path).exists(), path)


class LintTextTests(unittest.TestCase):
    def setUp(self) -> None:
        facts.reset_cache()

    def test_detects_stale_server_and_tool_counts(self) -> None:
        issues = facts.lint_text("接入 3 个 stdio MCP Server，共 9 个工具")
        labels = {issue["fact_id"] for issue in issues}
        self.assertEqual(labels, {"mcp.servers", "mcp.tools"})

    def test_detects_stale_test_and_chunk_counts(self) -> None:
        issues = facts.lint_text("跑过 124 例离线测试，构建 30 片段的知识库")
        labels = {issue["fact_id"] for issue in issues}
        self.assertIn("tests.count", labels)
        self.assertIn("corpus.chunks", labels)

    def test_accepts_at_least_phrasing(self) -> None:
        # 测试数只会涨，"230+ 例"永远成立；"30 余个片段"同理
        self.assertEqual(facts.lint_text("230+ 例离线测试，构建 30 余个片段的知识库"), [])

    def test_accepts_chinese_numerals(self) -> None:
        self.assertEqual(facts.lint_text("接入两个 MCP Server"), [])

    def test_ignores_scale_discussion(self) -> None:
        # 评测讨论里的历史规模不是"当前事实"，不该报错
        text = "30 个片段这个规模下，BM25 单路明显更差。"
        self.assertEqual(facts.lint_text(text), [])

    def test_clean_text_has_no_issues(self) -> None:
        self.assertEqual(facts.lint_text("知识库共 31 个片段，9 个节点并行扇出。"), [])


class LintProjectTests(unittest.TestCase):
    def setUp(self) -> None:
        facts.reset_cache()

    def test_current_project_is_consistent(self) -> None:
        report = facts.lint_project()
        self.assertEqual(report["issues"], [], f"发现过时数字：{report['issues']}")
        self.assertGreater(report["scanned"], 0)

    def test_historical_docs_are_skipped(self) -> None:
        teaching = facts.PROJECT_ROOT / "docs" / "teaching" / "05-第4讲-工具层-MCP.md"
        if not teaching.exists():
            self.skipTest("教学文档不存在")
        self.assertEqual(facts.lint_paths([teaching]), [])


class FactsApiTests(unittest.TestCase):
    def test_facts_endpoint(self) -> None:
        client = TestClient(app)
        body = client.get("/facts").json()
        ids = {item["id"] for item in body["facts"]}
        self.assertEqual(ids, {"mcp.servers", "mcp.tools", "tests.count", "corpus.chunks", "graph.nodes"})

    def test_lint_endpoint(self) -> None:
        client = TestClient(app)
        body = client.get("/facts/lint").json()
        self.assertIn("facts", body)
        self.assertIn("issues", body)
        self.assertEqual(body["issues"], [])


if __name__ == "__main__":
    unittest.main()
