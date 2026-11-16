"""`knowledge/router.py` 的单测：规模自适应的模式选择与分发。

为什么这层需要测试：
    它是"小语料直读、大语料检索"的唯一开关。选错模式不会报错，
    只会让召回悄悄变差（或让上下文爆掉），属于最难发现的那类问题。
"""

from __future__ import annotations

import json
import os
import unittest
from unittest.mock import patch

from knowledge import router


class ResolveModeTests(unittest.TestCase):
    def test_explicit_mode_wins(self) -> None:
        for mode in ("direct", "vector", "hybrid"):
            with patch.dict(os.environ, {router.MODE_ENV: mode}):
                self.assertEqual(router.resolve_mode(size=10_000), mode)

    def test_auto_small_corpus_goes_direct(self) -> None:
        with patch.dict(os.environ, {router.MODE_ENV: "auto"}, clear=False):
            self.assertEqual(router.resolve_mode(size=30), "direct")

    def test_auto_large_corpus_goes_vector(self) -> None:
        with patch.dict(os.environ, {router.MODE_ENV: "auto"}, clear=False):
            self.assertEqual(router.resolve_mode(size=5_000), "vector")

    def test_threshold_is_configurable(self) -> None:
        env = {router.MODE_ENV: "auto", router.THRESHOLD_ENV: "10"}
        with patch.dict(os.environ, env, clear=False):
            self.assertEqual(router.resolve_mode(size=30), "vector")

    def test_invalid_mode_falls_back_to_auto(self) -> None:
        with patch.dict(os.environ, {router.MODE_ENV: "magic"}, clear=False):
            self.assertEqual(router.resolve_mode(size=30), "direct")

    def test_invalid_threshold_falls_back_to_default(self) -> None:
        env = {router.MODE_ENV: "auto", router.THRESHOLD_ENV: "abc"}
        with patch.dict(os.environ, env, clear=False):
            self.assertEqual(router.resolve_mode(size=30), "direct")


class DispatchTests(unittest.TestCase):
    def test_direct_mode_skips_vector_search(self) -> None:
        fake = [{"content": "全量片段", "source": "a.md", "score": 0.0}]
        with patch.dict(os.environ, {router.MODE_ENV: "direct"}, clear=False), \
             patch("knowledge.direct.direct_search", return_value=fake) as direct, \
             patch.object(router, "retrieve_agentic") as agentic:
            hits = router.search_for_match(["RAG"], k=4)

        self.assertEqual(hits, fake)
        direct.assert_called_once()
        agentic.assert_not_called()

    def test_vector_mode_uses_agentic_retrieval(self) -> None:
        fake = [{"content": "向量片段", "source": "b.md", "score": 0.1}]
        with patch.dict(os.environ, {router.MODE_ENV: "vector"}, clear=False), \
             patch.object(router, "retrieve_agentic", return_value=fake) as agentic:
            hits = router.search_for_match(["RAG"], k=4)

        self.assertEqual(hits, fake)
        agentic.assert_called_once()

    def test_tool_search_respects_mode(self) -> None:
        fake = [{"content": "工具检索结果", "source": "c.md", "score": 0.0}]
        with patch.dict(os.environ, {router.MODE_ENV: "direct"}, clear=False), \
             patch("knowledge.direct.direct_search", return_value=fake):
            hits = router.search_for_tool("MCP", k=2)

        self.assertEqual(hits, fake)

    def test_search_evidence_returns_json_string(self) -> None:
        fake = [{"content": "片段", "source": "d.md", "score": 0.2}]
        with patch.object(router, "search_for_tool", return_value=fake):
            payload = router.search_evidence("query", k=4)

        self.assertIsInstance(payload, str)
        self.assertEqual(json.loads(payload), fake)


if __name__ == "__main__":
    unittest.main()
