"""`knowledge/bm25.py` 的单测：分词、RRF 融合、BM25 排序、混合检索。

为什么不测真实索引：
    这层要保证的是"算法和融合逻辑对不对"，不是"知识库里有什么"。
    真实索引由评测脚本（eval/ablation.py）负责，那里才有标注集。
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from knowledge import bm25


class TokenizeTests(unittest.TestCase):
    def test_tech_term_stays_whole(self) -> None:
        """LangGraph 不能被 jieba 切成 Lang + Graph，否则 BM25 会乱命中。"""
        tokens = bm25.tokenize("用 LangGraph 编排 Agent")

        self.assertIn("langgraph", tokens)
        self.assertIn("agent", tokens)

    def test_lowercased_and_punctuation_removed(self) -> None:
        tokens = bm25.tokenize("RAG，检索！")

        self.assertIn("rag", tokens)
        self.assertNotIn("，", tokens)


class RrfTests(unittest.TestCase):
    def test_item_in_both_lists_wins(self) -> None:
        fused = bm25.rrf_fuse([["A", "B"], ["B", "C"]], k=3)

        self.assertEqual(fused[0], "B")

    def test_respects_k(self) -> None:
        fused = bm25.rrf_fuse([["A", "B", "C"]], k=2)

        self.assertEqual(len(fused), 2)

    def test_empty_ranking_is_safe(self) -> None:
        self.assertEqual(bm25.rrf_fuse([[], []], k=3), [])


class Bm25SearchTests(unittest.TestCase):
    CORPUS = [
        ("做过 RAG 检索与查询改写", "项目.md"),
        ("实现 MCP 工具与 FastAPI 接口", "项目.md"),
        ("课程作业：数据结构与算法", "课程.md"),
    ]

    def test_rare_term_ranks_first(self) -> None:
        hits = bm25.bm25_search("MCP", k=2, corpus=self.CORPUS)

        self.assertTrue(hits)
        self.assertIn("MCP", hits[0]["content"])
        self.assertEqual(hits[0]["source"], "项目.md")

    def test_unmatched_query_returns_empty(self) -> None:
        """BM25 分数全为 0 时不该硬塞结果给调用方。"""
        hits = bm25.bm25_search("量子计算", k=2, corpus=self.CORPUS)

        self.assertEqual(hits, [])

    def test_score_is_numeric_and_descending(self) -> None:
        hits = bm25.bm25_search("检索", k=3, corpus=self.CORPUS)

        scores = [hit["score"] for hit in hits]
        self.assertEqual(scores, sorted(scores, reverse=True))


class HybridSearchTests(unittest.TestCase):
    def test_fusion_prefers_item_from_both_arms(self) -> None:
        vector_hits = [
            {"content": "A", "source": "a.md", "source_path": "a.md", "score": 0.1},
            {"content": "B", "source": "b.md", "source_path": "b.md", "score": 0.2},
        ]
        bm25_hits = [
            {"content": "B", "source": "b.md", "source_path": "b.md", "score": 3.0},
            {"content": "C", "source": "c.md", "source_path": "c.md", "score": 2.0},
        ]

        with patch.object(bm25, "retrieve", return_value=vector_hits), \
             patch.object(bm25, "bm25_search", return_value=bm25_hits):
            hits = bm25.hybrid_search("query", k=2)

        self.assertEqual(hits[0]["content"], "B", "两路都命中的片段应该排最前")
        self.assertEqual(len(hits), 2)
        # 元数据优先取向量那一路的（更完整）
        self.assertEqual(hits[0]["source"], "b.md")


if __name__ == "__main__":
    unittest.main()
