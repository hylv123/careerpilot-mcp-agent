from __future__ import annotations

import unittest

from services.matching import assess_match


class MatchAssessmentTests(unittest.TestCase):
    def test_reports_coverage_gaps_and_evidence_sources(self) -> None:
        result = assess_match(
            {"requirements": ["Python", "RAG", "FastAPI"]},
            [{"content": "使用 Python 构建 RAG 服务", "source": "项目经历.md"}],
        )
        self.assertEqual(result["score"], 67)
        self.assertEqual(result["covered_requirements"], ["Python", "RAG"])
        self.assertEqual(result["gaps"], ["FastAPI"])
        self.assertEqual(result["evidence_sources"], ["项目经历.md"])


if __name__ == "__main__":
    unittest.main()
