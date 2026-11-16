"""`run.py` 的入口一致性测试。

为什么需要它：
    CLI 以前是一套独立实现（自己 build_graph + 手写初始 state），
    和 API/worker 漂在了递归上限上（25 vs 50）。改成共用 `run_analysis` 之后，
    这里守住两件事：**它真的调用了共享入口**，以及**输出格式没变**。
"""

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import run


def _result(**overrides):
    base = {
        "status": "done",
        "analysis": {},
        "evidence_count": 0,
        "report_text": "",
        "resume_bullets": [],
        "interview_questions": [],
        "saved": False,
        "save_error": "",
    }
    base.update(overrides)
    return base


class RunCliTests(unittest.TestCase):
    def _invoke(self, argv, result):
        with patch.object(run, "setup_logging"), \
             patch.object(run, "run_analysis", return_value=result) as called, \
             patch.object(run.sys, "argv", argv), \
             redirect_stdout(io.StringIO()) as out:
            code = run.main()
        return code, out.getvalue(), called

    def test_uses_shared_entry_point(self) -> None:
        """必须走 tasks.runner.run_analysis，而不是自己 build_graph。"""
        code, _, called = self._invoke(["run.py", "--jd", "  招聘 RAG  "], _result())

        self.assertEqual(code, 0)
        called.assert_called_once_with("招聘 RAG")

    def test_prints_sections_in_order(self) -> None:
        result = _result(
            report_text="报告正文",
            resume_bullets=[{"bullet": "要点A", "evidence_source": "简历.md"}],
            interview_questions=[
                {"question": "Q1?", "answer_points": ["点1"], "evidence_source": "简历.md"}
            ],
            saved=True,
        )

        _, out, _ = self._invoke(["run.py", "--jd", "jd"], result)

        self.assertIn("报告正文", out)
        self.assertIn("要点A", out)
        self.assertIn("Q1?", out)
        self.assertIn("已写入 SQLite", out)
        self.assertLess(out.index("报告正文"), out.index("要点A"))

    def test_unsaved_reason_is_visible(self) -> None:
        result = _result(saved=False, save_error="MCP 工具不可用")

        _, out, _ = self._invoke(["run.py", "--jd", "jd"], result)

        self.assertIn("未保存", out)
        self.assertIn("MCP 工具不可用", out)

    def test_empty_jd_is_rejected_before_calling_model(self) -> None:
        with patch.object(run, "setup_logging"), \
             patch.object(run, "run_analysis") as called, \
             patch.object(run.sys, "argv", ["run.py", "--jd", "   "]):
            with self.assertRaises(SystemExit):
                run.main()

        called.assert_not_called()


if __name__ == "__main__":
    unittest.main()
