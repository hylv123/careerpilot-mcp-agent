"""重试与退避测试。

这块是全项目"最贵"的一段：一次分析 8~9 次模型调用，重试层数或退避参数写错，
轻则慢一倍，重则像历史上那样单次分析跑到 560 秒。所以它必须有网。

覆盖：
  - _is_upstream_error：上游故障 vs 内容不合规怎么区分
  - _backoff_delay：指数、抖动边界、封顶、两类错误的不同基准
  - invoke_structured：3 次尝试、成功即返回、失败抛 RuntimeError 且保留 __cause__
  - safe_invoke：普通链的同款语义
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from pydantic import ValidationError

import agent.graph as graph
from agent.schemas import JobAnalysisGraph

VALID = {
    "job_title": "AI 实习生",
    "company": "Demo",
    "requirements": ["Python"],
    "keywords": ["RAG"],
    "level": "实习生",
}


class _Chain:
    """按脚本依次抛异常或返回的假链。"""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def invoke(self, inputs):
        self.calls += 1
        outcome = self.outcomes.pop(0) if len(self.outcomes) > 1 else self.outcomes[0]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class UpstreamErrorTests(unittest.TestCase):
    def test_timeout_is_upstream(self) -> None:
        self.assertTrue(graph._is_upstream_error(TimeoutError("Read timed out")))

    def test_rate_limit_is_upstream(self) -> None:
        self.assertTrue(graph._is_upstream_error(RuntimeError("Error 429 too many requests")))

    def test_validation_error_is_not_upstream(self) -> None:
        try:
            JobAnalysisGraph.model_validate({"job_title": "x"})
        except ValidationError as exc:
            self.assertFalse(graph._is_upstream_error(exc))
        else:  # pragma: no cover
            self.fail("构造 ValidationError 失败")

    def test_none_is_not_upstream(self) -> None:
        self.assertFalse(graph._is_upstream_error(None))


class BackoffTests(unittest.TestCase):
    def test_upstream_sequence_doubles(self) -> None:
        exc = TimeoutError("timeout")
        with patch.object(graph.random, "uniform", return_value=0.0):
            delays = [graph._backoff_delay(i, exc) for i in (1, 2, 3)]
        self.assertEqual(delays, [1.0, 2.0, 4.0])

    def test_content_error_waits_quarter(self) -> None:
        exc = ValueError("schema mismatch")
        with patch.object(graph.random, "uniform", return_value=0.0):
            self.assertAlmostEqual(graph._backoff_delay(1, exc), 0.25)

    def test_capped_at_max(self) -> None:
        exc = TimeoutError("timeout")
        with patch.object(graph.random, "uniform", return_value=0.0):
            self.assertEqual(graph._backoff_delay(10, exc), graph.RETRY_MAX_DELAY)

    def test_jitter_never_exceeds_cap(self) -> None:
        exc = TimeoutError("timeout")
        # 让抖动取到最大正偏移，仍不能突破上限
        with patch.object(graph.random, "uniform", return_value=100.0):
            self.assertLessEqual(graph._backoff_delay(10, exc), graph.RETRY_MAX_DELAY)

    def test_jitter_never_negative(self) -> None:
        exc = TimeoutError("timeout")
        with patch.object(graph.random, "uniform", return_value=-100.0):
            self.assertGreaterEqual(graph._backoff_delay(1, exc), 0.0)


class InvokeStructuredTests(unittest.TestCase):
    def test_retries_then_succeeds(self) -> None:
        chain = _Chain([TimeoutError("timed out"), TimeoutError("timed out"), VALID])
        sleeps: list[float] = []
        with patch.object(graph, "sleep", side_effect=sleeps.append), \
             patch.object(graph.random, "uniform", return_value=0.0):
            result = graph.invoke_structured(chain, {"user_input": "jd"}, JobAnalysisGraph)

        self.assertEqual(chain.calls, 3)
        self.assertEqual(sleeps, [1.0, 2.0])          # 上游错误：1s、2s
        self.assertEqual(result.job_title, "AI 实习生")

    def test_single_attempt_when_first_succeeds(self) -> None:
        chain = _Chain([VALID])
        with patch.object(graph, "sleep") as sleep:
            graph.invoke_structured(chain, {}, JobAnalysisGraph)
        self.assertEqual(chain.calls, 1)
        sleep.assert_not_called()

    def test_gives_up_after_three_attempts(self) -> None:
        chain = _Chain([TimeoutError("timed out")])
        with patch.object(graph, "sleep"), patch.object(graph.random, "uniform", return_value=0.0):
            with self.assertRaises(RuntimeError) as ctx:
                graph.invoke_structured(chain, {}, JobAnalysisGraph)

        self.assertEqual(chain.calls, 3)
        self.assertIsNotNone(ctx.exception.__cause__, "必须保留根因，否则日志里看不到真实错误")
        self.assertIsInstance(ctx.exception.__cause__, TimeoutError)

    def test_invalid_payload_is_retried(self) -> None:
        chain = _Chain([{"job_title": "x"}, {"job_title": "x"}, VALID])
        with patch.object(graph, "sleep"), patch.object(graph.random, "uniform", return_value=0.0):
            result = graph.invoke_structured(chain, {}, JobAnalysisGraph)
        self.assertEqual(chain.calls, 3)
        self.assertEqual(result.level, "实习生")


class SafeInvokeTests(unittest.TestCase):
    def test_returns_first_result(self) -> None:
        chain = _Chain(["answer"])
        with patch.object(graph, "sleep") as sleep:
            self.assertEqual(graph.safe_invoke(chain, {}, "report"), "answer")
        sleep.assert_not_called()

    def test_retries_once_then_succeeds(self) -> None:
        chain = _Chain([TimeoutError("timeout"), "ok"])
        sleeps: list[float] = []
        with patch.object(graph, "sleep", side_effect=sleeps.append), \
             patch.object(graph.random, "uniform", return_value=0.0):
            self.assertEqual(graph.safe_invoke(chain, {}, "agent"), "ok")
        self.assertEqual(chain.calls, 2)
        self.assertEqual(sleeps, [1.0])

    def test_raises_after_retries(self) -> None:
        chain = _Chain([ValueError("boom")])
        with patch.object(graph, "sleep"), patch.object(graph.random, "uniform", return_value=0.0):
            with self.assertRaises(RuntimeError) as ctx:
                graph.safe_invoke(chain, {}, "report")
        self.assertIn("report", str(ctx.exception))
        self.assertIsInstance(ctx.exception.__cause__, ValueError)


if __name__ == "__main__":
    unittest.main()
