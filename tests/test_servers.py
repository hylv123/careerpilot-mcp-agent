"""job / document 两个 MCP Server 的行为测试（加强版）。

加强点：
  - 断言直接查 SQLite 的**命名列**，不再用 `rows[0][3]` 这种位置索引
    （列序一变，旧断言要么静默错、要么报个看不懂的 IndexError）；
  - 断言完整返回值，而不是 `assertIn("AI Intern", saved)` 这种"包含就算过"；
  - 文档内容断言整段 Markdown，而不是只查一个词。
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent.mcp_tools import _normalize_tool_result
from mcp_servers import document_server, job_server


class JobServerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "jobs.db"
        self.db_patch = patch.object(job_server, "DB", self.db_path)
        self.db_patch.start()

    def tearDown(self) -> None:
        self.db_patch.stop()
        self.temp_dir.cleanup()

    def _rows(self) -> list[sqlite3.Row]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            return conn.execute(
                "SELECT id, title, company, jd, status FROM jobs ORDER BY id"
            ).fetchall()
        finally:
            conn.close()

    def test_save_job_writes_named_columns(self) -> None:
        message = job_server.save_job("AI Intern", "Demo Co", "Python RAG")
        self.assertEqual(message, "已保存岗位：AI Intern from Demo Co")

        rows = self._rows()
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["title"], "AI Intern")
        self.assertEqual(row["company"], "Demo Co")
        self.assertEqual(row["jd"], "Python RAG")
        self.assertEqual(row["status"], "未投", "status 的默认值应该由建表语句提供")

    def test_update_status_then_filter(self) -> None:
        job_server.save_job("AI Intern", "Demo Co", "jd")
        job_id = self._rows()[0]["id"]

        message = job_server.update_status(job_id, "面试")
        self.assertEqual(message, f"岗位 {job_id} 状态更新为 面试")
        self.assertEqual(self._rows()[0]["status"], "面试")

        filtered = json.loads(job_server.list_jobs("面试"))
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0], [job_id, "AI Intern", "Demo Co", "面试"])
        self.assertEqual(json.loads(job_server.list_jobs("已投")), [])

    def test_update_missing_job_reports_not_found(self) -> None:
        message = job_server.update_status(9999, "已投")
        self.assertEqual(message, "未找到岗位：9999")

    def test_rejects_invalid_status(self) -> None:
        job_server.save_job("AI Intern", "Demo Co", "jd")
        message = job_server.update_status(1, "未知状态")

        self.assertTrue(message.startswith("无效状态：未知状态"))
        for allowed in ("未投", "已投", "面试", "offer", "拒"):
            self.assertIn(allowed, message)
        # 状态没被改
        self.assertEqual(self._rows()[0]["status"], "未投")

    def test_list_jobs_from_company(self) -> None:
        job_server.save_job("A", "Acme", "jd1")
        job_server.save_job("B", "Other", "jd2")

        rows = json.loads(job_server.list_jobs_from_company("Acme"))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][1], "A")


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
