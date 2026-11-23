"""document MCP Server 的行为测试。

岗位相关的 MCP Server 已删除：岗位读写走原生服务 `services/jobs.py`，
由 `tests/test_job_workbench.py` 覆盖。这里只剩文档生成 Server。

加强点：
  - 断言完整返回值，而不是 `assertIn("AI Intern", saved)` 这种"包含就算过"；
  - 文档内容断言整段 Markdown，而不是只查一个词。
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent.mcp_tools import _normalize_tool_result
from mcp_servers import document_server


class DocumentServerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temp_dir.name) / "output"
        self.output_dir.mkdir(parents=True)
        self.output_patch = patch.object(document_server, "OUT", self.output_dir)
        self.output_patch.start()

    def tearDown(self) -> None:
        self.output_patch.stop()
        self.temp_dir.cleanup()

    def test_generate_resume_markdown(self) -> None:
        document_server.generate_resume_md("AI Intern", ["Built MCP tools", "Added RAG eval"])

        files = list(self.output_dir.glob("resume_*.md"))
        self.assertEqual(len(files), 1)
        self.assertEqual(
            files[0].read_text(encoding="utf-8"),
            "#AI Intern\n\n- Built MCP tools\n- Added RAG eval",
        )

    def test_generate_interview_prep(self) -> None:
        document_server.generate_interview_prep("AI Intern", ["Q one", "Q two"])
        files = list(self.output_dir.glob("interview_*.md"))
        self.assertEqual(len(files), 1)
        content = files[0].read_text(encoding="utf-8")
        self.assertIn("AI Intern 面试准备", content)
        self.assertIn("## Q1.Q one", content)
        self.assertIn("## Q2.Q two", content)

    def test_safe_filename(self) -> None:
        result = document_server.safe_filename('AI:Intern/2026?')
        for bad in (":", "/", "?"):
            self.assertNotIn(bad, result)


class MCPAdapterTests(unittest.TestCase):
    def test_normalizes_text_content_blocks(self) -> None:
        self.assertEqual(
            _normalize_tool_result([{"type": "text", "text": "已保存岗位"}]),
            "已保存岗位",
        )

    def test_passes_through_plain_string(self) -> None:
        self.assertEqual(_normalize_tool_result("原文"), "原文")

    def test_unknown_shape_falls_back_to_str(self) -> None:
        self.assertEqual(_normalize_tool_result(123), "123")


if __name__ == "__main__":
    unittest.main()
