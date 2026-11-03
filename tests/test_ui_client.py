"""ui/api_client.py 的离线测试：不需要 uvicorn、Redis、模型。

用一个假 session 顶替 requests，逐条验证：
  202 受理 / 200 同步降级 / 429 限流 / 404 过期 / 轮询终止 / 轮询超时。
"""

from __future__ import annotations

import unittest

from ui import api_client as api


class _Resp:
    def __init__(self, status_code: int, payload=None, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("no json body")
        return self._payload


class _Session:
    """记录调用、按预设返回的假 requests 会话。"""

    def __init__(self, post: _Resp | None = None, get: _Resp | list[_Resp] | None = None) -> None:
        self.calls: list[tuple] = []
        self._post = post
        self._get = get

    def post(self, url, json=None, timeout=None):
        self.calls.append(("POST", url, json))
        return self._post

    def get(self, url, timeout=None):
        self.calls.append(("GET", url))
        if isinstance(self._get, list):
            # 最后一帧重复返回，模拟"状态一直不变"
            return self._get.pop(0) if len(self._get) > 1 else self._get[0]
        return self._get


class SubmitTests(unittest.TestCase):
    def test_202_returns_task_id_and_reused_flag(self) -> None:
        session = _Session(post=_Resp(202, {"task_id": "task-1", "status": "pending", "reused": True}))
        result = api.submit_analysis("招聘 AI 实习生，要求 Python", "http://api", session=session)

        self.assertEqual(result.task_id, "task-1")
        self.assertTrue(result.reused)
        self.assertIsNone(result.direct_result)
        method, url, payload = session.calls[0]
        self.assertEqual((method, url), ("POST", "http://api/analyze"))
        self.assertEqual(payload, {"jd": "招聘 AI 实习生，要求 Python"})

    def test_200_means_sync_fallback(self) -> None:
        session = _Session(post=_Resp(200, {"status": "done", "saved": True}))
        result = api.submit_analysis("jd", "http://api", session=session)

        self.assertEqual(result.task_id, "")
        self.assertEqual(result.direct_result["saved"], True)

    def test_429_raises_with_hint(self) -> None:
        session = _Session(post=_Resp(429, {"detail": "当前提交过于频繁"}))
        with self.assertRaises(api.ApiError) as ctx:
            api.submit_analysis("jd", "http://api", session=session)
        self.assertIn("限流", str(ctx.exception))

    def test_422_raises_with_detail(self) -> None:
        session = _Session(post=_Resp(422, {"detail": "jd 太短"}))
        with self.assertRaises(api.ApiError) as ctx:
            api.submit_analysis("jd", "http://api", session=session)
        self.assertIn("jd 太短", str(ctx.exception))


class FetchTests(unittest.TestCase):
    def test_404_raises_expired(self) -> None:
        session = _Session(get=_Resp(404, {"detail": "任务不存在或已过期"}))
        with self.assertRaises(api.ApiError) as ctx:
            api.fetch_task("task-x", "http://api", session=session)
        self.assertIn("过期", str(ctx.exception))

    def test_503_raises_storage(self) -> None:
        session = _Session(get=_Resp(503, {"detail": "任务存储暂时不可用"}))
        with self.assertRaises(api.ApiError) as ctx:
            api.fetch_task("task-x", "http://api", session=session)
        self.assertIn("存储", str(ctx.exception))

    def test_200_returns_payload(self) -> None:
        session = _Session(get=_Resp(200, {"task_id": "task-x", "status": "running"}))
        payload = api.fetch_task("task-x", "http://api", session=session)
        self.assertEqual(payload["status"], "running")


class WaitTests(unittest.TestCase):
    def test_stops_on_done_and_reports_progress(self) -> None:
        session = _Session(
            get=[
                _Resp(200, {"status": "pending", "progress": "已排队"}),
                _Resp(200, {"status": "running", "progress": "正在分析"}),
                _Resp(200, {"status": "done", "result": {"saved": True}}),
            ]
        )
        seen: list[str] = []
        payload = api.wait_for_result(
            "task-x",
            "http://api",
            session=session,
            on_update=lambda status, progress: seen.append(status),
            sleep=lambda _: None,
            clock=lambda: 0.0,
        )
        self.assertEqual(payload["status"], "done")
        self.assertEqual(seen, ["pending", "running", "done"])

    def test_times_out_instead_of_spinning_forever(self) -> None:
        session = _Session(get=_Resp(200, {"status": "running"}))

        class _Clock:
            def __init__(self) -> None:
                self.now = 0.0

            def __call__(self) -> float:
                self.now += 10.0
                return self.now

        with self.assertRaises(api.ApiError) as ctx:
            api.wait_for_result(
                "task-x",
                "http://api",
                session=session,
                sleep=lambda _: None,
                clock=_Clock(),
                max_wait=30.0,
            )
        self.assertIn("超时", str(ctx.exception))


class SyncTests(unittest.TestCase):
    def test_analyze_sync_returns_payload(self) -> None:
        session = _Session(post=_Resp(200, {"status": "done", "saved": False}))
        payload = api.analyze_sync("jd", "http://api", session=session)
        self.assertIs(payload["saved"], False)

    def test_analyze_sync_502_raises(self) -> None:
        session = _Session(post=_Resp(502, {"detail": "分析失败：模型超时"}))
        with self.assertRaises(api.ApiError) as ctx:
            api.analyze_sync("jd", "http://api", session=session)
        self.assertIn("模型超时", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
