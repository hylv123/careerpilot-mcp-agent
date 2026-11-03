"""HTTP 契约的字段集合锁定。

`AnalyzeResponse` 是跨进程契约（worker 写 → API 读 → 客户端消费）。
加字段、删字段、改默认值，以前都不会有任何测试变红——只有线上客户端崩了才发现。
这里把字段名集合与必填项显式写下来。
"""

from __future__ import annotations

import unittest

from api.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    TaskAccepted,
    TaskStatusResponse,
)


class AnalyzeRequestContractTests(unittest.TestCase):
    def test_only_jd_field(self) -> None:
        self.assertEqual(set(AnalyzeRequest.model_fields), {"jd"})

    def test_jd_is_required_with_length_limits(self) -> None:
        field = AnalyzeRequest.model_fields["jd"]
        self.assertTrue(field.is_required())
        self.assertEqual(field.metadata[0].min_length, 10)


class AnalyzeResponseContractTests(unittest.TestCase):
    EXPECTED = {
        "status",
        "analysis",
        "evidence_count",
        "report_text",
        "resume_bullets",
        "interview_questions",
        "saved",
        "save_error",
    }

    def test_field_set_is_stable(self) -> None:
        self.assertEqual(set(AnalyzeResponse.model_fields), self.EXPECTED)

    def test_only_status_is_required(self) -> None:
        required = {name for name, f in AnalyzeResponse.model_fields.items() if f.is_required()}
        self.assertEqual(required, {"status"})

    def test_saved_defaults_to_false(self) -> None:
        # 默认必须是 False：没显式说成功，就不能当成成功
        self.assertIs(AnalyzeResponse(status="done").saved, False)
        self.assertEqual(AnalyzeResponse(status="done").save_error, "")


class TaskStatusResponseContractTests(unittest.TestCase):
    def test_field_set_is_stable(self) -> None:
        self.assertEqual(
            set(TaskStatusResponse.model_fields),
            {"task_id", "status", "progress", "error", "updated_at", "result"},
        )

    def test_result_is_optional(self) -> None:
        payload = TaskStatusResponse(task_id="t", status="pending")
        self.assertIsNone(payload.result)

    def test_result_nested_model_validation(self) -> None:
        payload = TaskStatusResponse(
            task_id="t",
            status="done",
            result={"status": "done", "saved": True},  # type: ignore[arg-type]
        )
        self.assertIsNotNone(payload.result)
        self.assertIs(payload.result.saved, True)


class TaskAcceptedContractTests(unittest.TestCase):
    def test_field_set_and_defaults(self) -> None:
        self.assertEqual(set(TaskAccepted.model_fields), {"task_id", "status", "reused"})
        accepted = TaskAccepted(task_id="task-1")
        self.assertEqual(accepted.status, "pending")
        self.assertIs(accepted.reused, False)


if __name__ == "__main__":
    unittest.main()
