from __future__ import annotations

import json
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

    def test_job_crud(self) -> None:
        saved = job_server.save_job("AI Intern", "Demo Co", "Python RAG")
        self.assertIn("AI Intern", saved)

        jobs = json.loads(job_server.list_jobs())
        self.assertEqual(len(jobs), 1)
        job_id = jobs[0][0]

        updated = job_server.update_status(job_id, "面试")
        self.assertIn("面试", updated)
        filtered = json.loads(job_server.list_jobs("面试"))
        self.assertEqual(filtered[0][3], "面试")

    def test_rejects_invalid_status(self) -> None:
        job_server.save_job("AI Intern", "Demo Co", "Python RAG")
        message = job_server.update_status(1, "未知状态")
        self.assertIn("无效状态", message)


class DocumentServerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temp_dir.name) / "output"
        self.output_dir.mkdir(parents=True)
        self.output_patch = patch.object(
            document_server,
            "OUT",
            self.output_dir,
        )
        self.output_patch.start()

    def tearDown(self) -> None:
        self.output_patch.stop()
        self.temp_dir.cleanup()

    def test_generate_resume_markdown(self) -> None:
        document_server.generate_resume_md(
            "AI Intern",
            ["Built MCP tools", "Added RAG evaluation"],
        )
        files = list(self.output_dir.glob("resume_*.md"))
        self.assertEqual(len(files), 1)
        content = files[0].read_text(encoding="utf-8")
        self.assertIn("Built MCP tools", content)

    def test_safe_filename(self) -> None:
        result = document_server.safe_filename('AI:Intern/2026?')
        self.assertNotIn(":", result)
        self.assertNotIn("/", result)
        self.assertNotIn("?", result)


class MCPAdapterTests(unittest.TestCase):
    def test_normalizes_text_content_blocks(self) -> None:
        result = _normalize_tool_result(
            [{"type": "text", "text": "已保存岗位"}]
        )
        self.assertEqual(result, "已保存岗位")


if __name__ == "__main__":
    unittest.main()

