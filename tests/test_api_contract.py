"""HTTP 契约测试：用 FastAPI TestClient 跑，不需要 uvicorn/Redis/模型。

覆盖四条最容易被改坏的路径：
  /health 探活、入参校验、同步接口的响应字段、Redis 不可用时的降级。
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

try:
    from fastapi.testclient import TestClient
except Exception:  # pragma: no cover - 缺 httpx 时 TestClient 不可用
    TestClient = None  # type: ignore[assignment]

import api.main as api_main
from tasks.store import RedisUnavailable

_NEED_CLIENT = unittest.skipUnless(TestClient is not None, "缺少 httpx，TestClient 不可用")


def _payload() -> dict:
    return {
        "status": "done",
        "analysis": {"job_title": "AI 实习生", "company": "Demo", "level": "实习生"},
        "evidence_count": 2,
        "report_text": "报告正文",
        "resume_bullets": [{"bullet": "b", "evidence_source": "resume.md"}],
        "interview_questions": [{"question": "q", "answer_points": ["a"], "evidence_source": "resume.md"}],
        "saved": False,
        "save_error": "MCP 工具调用失败：save_job：OperationalError",
    }


@_NEED_CLIENT
class HealthTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(api_main.app)

    def test_health_is_ok(self) -> None:
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"status": "ok"})

    def test_short_jd_rejected_by_pydantic(self) -> None:
        resp = self.client.post("/analyze", json={"jd": "太短"})
        self.assertEqual(resp.status_code, 422)

    def test_missing_jd_rejected(self) -> None:
        resp = self.client.post("/analyze", json={})
        self.assertEqual(resp.status_code, 422)


@_NEED_CLIENT
class SyncAnalyzeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(api_main.app)

    def test_sync_returns_new_saved_fields(self) -> None:
        with patch.object(api_main, "run_analysis", return_value=_payload()):
            resp = self.client.post("/analyze/sync", json={"jd": "招聘 AI 应用开发实习生，要求 Python"})

        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertIs(body["saved"], False)
        self.assertIn("save_job", body["save_error"])
        self.assertEqual(body["report_text"], "报告正文")

    def test_sync_wraps_failure_as_502(self) -> None:
        with patch.object(api_main, "run_analysis", side_effect=RuntimeError("模型超时")):
            resp = self.client.post("/analyze/sync", json={"jd": "招聘 AI 应用开发实习生，要求 Python"})

        self.assertEqual(resp.status_code, 502)
        self.assertIn("模型超时", resp.json()["detail"])


@_NEED_CLIENT
class DegradeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(api_main.app)

    def test_analyze_degrades_to_sync_when_redis_down(self) -> None:
        with patch.object(api_main, "get_store", side_effect=RedisUnavailable("conn refused")), \
             patch.object(api_main, "run_analysis", return_value=_payload()):
            resp = self.client.post("/analyze", json={"jd": "招聘 AI 应用开发实习生，要求 Python"})

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.headers.get("X-Analysis-Mode"), "sync-fallback")
        self.assertIs(resp.json()["saved"], False)

    def test_task_query_returns_503_when_store_down(self) -> None:
        with patch.object(api_main, "get_store", side_effect=RedisUnavailable("conn refused")):
            resp = self.client.get("/tasks/task-x")

        self.assertEqual(resp.status_code, 503)


if __name__ == "__main__":
    unittest.main()
