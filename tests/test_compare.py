"""多岗位聚类与投递聚焦：确定性算法，不花模型调用。

锁三件事：
  1. 技能识别稳定（中文要求里能认出技术词，不把"熟悉/要求"当技能）；
  2. 聚类结果确定（同一批岗位永远算出同一批簇，顺序也一致）；
  3. 覆盖率与缺口的定义不会悄悄变（覆盖率 = 有材料可写，不是能力判断）。
"""

from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from api.main import app
from services import compare


def _job(job_id: int, requirements: list[str], *, jd: str = "", title: str = "AI 实习生") -> dict:
    return {
        "id": job_id,
        "title": title,
        "company": "Demo",
        "jd": jd or "招聘 AI 实习生",
        "requirements": requirements,
        "preferences": [],
    }


class RequirementTokenTests(unittest.TestCase):
    def test_extracts_tech_terms_from_chinese(self) -> None:
        tokens = compare.requirement_tokens(["熟悉 Python 与 LangChain，了解 RAG 检索"])
        self.assertIn("python", tokens)
        self.assertIn("langchain", tokens)
        self.assertIn("rag", tokens)

    def test_ignores_generic_words(self) -> None:
        tokens = compare.requirement_tokens(["熟悉 了解 掌握 要求 经验 优先"])
        self.assertEqual(tokens, set())

    def test_domains_are_not_skills(self) -> None:
        tokens = compare.requirement_tokens(["来源 https://blog.csdn.net/x，要求 Java"])
        self.assertEqual(tokens, {"java"})

    def test_falls_back_to_jd_when_requirements_missing(self) -> None:
        job = {"id": 1, "jd": "要求熟悉 FastAPI 与 Redis", "requirements": [], "preferences": []}
        self.assertEqual(compare.job_tokens(job), {"fastapi", "redis"})


class ClusteringTests(unittest.TestCase):
    def test_similar_jobs_share_a_cluster(self) -> None:
        jobs = [
            _job(1, ["Python", "LangChain", "RAG"]),
            _job(2, ["Python", "LangChain", "RAG", "MCP"]),
            _job(3, ["Java", "Spring", "MySQL"]),
        ]
        clusters = compare.cluster_jobs(jobs)
        self.assertEqual(len(clusters), 2)
        sizes = sorted(len(cluster["jobs"]) for cluster in clusters)
        self.assertEqual(sizes, [1, 2])

    def test_clustering_is_deterministic(self) -> None:
        jobs = [
            _job(1, ["Python", "RAG"]),
            _job(2, ["Python", "RAG", "MCP"]),
            _job(3, ["Java", "Spring"]),
        ]
        first = compare.cluster_jobs(jobs)
        second = compare.cluster_jobs(list(reversed(jobs)))
        self.assertEqual(
            [(cluster["label"], [job["id"] for job in cluster["jobs"]]) for cluster in first],
            [(cluster["label"], [job["id"] for job in cluster["jobs"]]) for cluster in second],
        )

    def test_jobs_without_tokens_are_excluded(self) -> None:
        clusters = compare.cluster_jobs([{"id": 9, "jd": "无技术词", "requirements": [], "preferences": []}])
        self.assertEqual(clusters, [])


class CoverageTests(unittest.TestCase):
    def test_coverage_ratio(self) -> None:
        self.assertEqual(compare.coverage({"python", "rag", "mcp"}, {"python", "rag"}), 0.667)

    def test_empty_requirements_score_zero(self) -> None:
        self.assertEqual(compare.coverage(set(), {"python"}), 0.0)


class FocusReportTests(unittest.TestCase):
    EVIDENCE = "用 Python 和 LangChain 做了 RAG 检索，写过 FastAPI 接口。"

    def test_clusters_sorted_by_coverage(self) -> None:
        jobs = [
            _job(1, ["Python", "LangChain", "RAG"]),
            _job(2, ["Kubernetes", "Terraform", "Helm"]),
        ]
        report = compare.focus_report(jobs, self.EVIDENCE)
        self.assertEqual(len(report["clusters"]), 2)
        self.assertGreater(report["clusters"][0]["avg_coverage"], report["clusters"][1]["avg_coverage"])

    def test_gaps_aggregate_across_jobs(self) -> None:
        jobs = [
            _job(1, ["Python", "Kubernetes"]),
            _job(2, ["Python", "Kubernetes", "RAG"]),
        ]
        report = compare.focus_report(jobs, self.EVIDENCE)
        gaps = {item["token"]: item["jobs"] for item in report["gaps"]}
        self.assertEqual(gaps.get("kubernetes"), 2)
        self.assertNotIn("python", gaps)

    def test_per_job_coverage_present(self) -> None:
        report = compare.focus_report([_job(1, ["Python", "RAG"])], self.EVIDENCE)
        self.assertEqual(len(report["jobs"]), 1)
        self.assertEqual(report["jobs"][0]["coverage"], 1.0)


class InsightsApiTests(unittest.TestCase):
    def test_endpoint_shape(self) -> None:
        client = TestClient(app)
        body = client.get("/insights/clusters").json()
        for key in ("clusters", "gaps", "jobs", "evidence_tokens"):
            self.assertIn(key, body)
        self.assertIsInstance(body["clusters"], list)


if __name__ == "__main__":
    unittest.main()
