from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from services import jobs


class JobWorkbenchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.db = str(Path(self.tempdir.name) / "jobs.db")
        self.env = patch.dict(os.environ, {"CAREERPILOT_DB_PATH": self.db})
        self.env.start()

    def tearDown(self) -> None:
        self.env.stop()
        self.tempdir.cleanup()

    def test_create_update_and_list_job(self) -> None:
        job = jobs.create_job(title="AI 实习生", company="Demo", jd="招聘 AI 实习生，要求 Python")
        self.assertEqual(job["status"], "未投")
        updated = jobs.update_job(job["id"], status="已投", notes="已内推", deadline="2026-12-01")
        self.assertEqual(updated["notes"], "已内推")
        self.assertEqual(jobs.list_jobs(status="已投")[0]["id"], job["id"])

    def test_invalid_status_is_rejected(self) -> None:
        job = jobs.create_job(title="AI 实习生", company="Demo", jd="招聘 AI 实习生，要求 Python")
        with self.assertRaises(ValueError):
            jobs.update_job(job["id"], status="随便写")

    def test_summary_drops_header_lines_and_keeps_content(self) -> None:
        jd = "岗位名称：AI 实习生\n公司：Demo\n岗位职责\n负责 Agent 工具接入与提示词调优，参与效果评测。"
        summary = jobs.summarize_jd(jd)
        self.assertNotIn("岗位名称", summary)
        self.assertNotIn("公司：", summary)
        self.assertNotIn("岗位职责", summary)
        self.assertIn("Agent 工具接入", summary)

    def test_summary_is_truncated_with_ellipsis(self) -> None:
        summary = jobs.summarize_jd("负责" * 300)
        self.assertLessEqual(len(summary), jobs.JD_SUMMARY_LIMIT + 1)
        self.assertTrue(summary.endswith("…"))

    def test_records_carry_summary(self) -> None:
        job = jobs.create_job(title="AI 实习生", company="Demo", jd="招聘 AI 实习生，要求 Python 与 RAG。")
        self.assertIn("Python", job["jd_summary"])
        self.assertIn("jd_summary", jobs.list_jobs()[0])
        self.assertIn("jd_summary", jobs.get_job(job["id"]) or {})

    def test_delete_removes_record(self) -> None:
        job = jobs.create_job(title="AI 实习生", company="Demo", jd="招聘 AI 实习生，要求 Python")
        self.assertTrue(jobs.delete_job(job["id"]))
        self.assertIsNone(jobs.get_job(job["id"]))
        self.assertFalse(jobs.delete_job(job["id"]))
        self.assertEqual(jobs.list_jobs(), [])


if __name__ == "__main__":
    unittest.main()
