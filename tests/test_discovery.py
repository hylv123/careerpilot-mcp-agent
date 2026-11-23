"""岗位发现：适配器归一化、链接白名单、打分与入库。

全部离线：网络调用被替换成固定 JSON，保证测试不受对方接口变动影响。
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from api.main import app
from services import discovery, job_sources

GREENHOUSE_FIXTURE = {
    "jobs": [
        {
            "id": 101,
            "title": "AI Agent Intern",
            "absolute_url": "https://boards.greenhouse.io/acme/jobs/101",
            "location": {"name": "Shenzhen"},
            "updated_at": "2026-11-20T10:00:00Z",
            "content": "<p>Build LangGraph agents &amp; RAG pipelines</p>",
        },
        {
            "id": 102,
            "title": "Phishing Role",
            "absolute_url": "https://evil.example.com/jobs/102",
            "location": {"name": "Remote"},
            "content": "<p>fake</p>",
        },
    ]
}


class SourceAdapterTests(unittest.TestCase):
    def test_source_registry(self) -> None:
        ids = {item["id"] for item in job_sources.list_sources()}
        self.assertEqual(ids, {"greenhouse", "lever", "ashby", "remoteok", "arbeitnow"})
        for item in job_sources.list_sources():
            self.assertTrue(item["authority"])

    def test_greenhouse_normalizes_and_drops_foreign_links(self) -> None:
        with patch.object(job_sources, "http_get_json", return_value=GREENHOUSE_FIXTURE):
            rows = job_sources.fetch_source("greenhouse", board="acme")
        self.assertEqual(len(rows), 1, "非白名单域名的岗位必须被丢弃")
        self.assertEqual(rows[0]["title"], "AI Agent Intern")
        self.assertIn("LangGraph", rows[0]["description"])
        self.assertNotIn("<p>", rows[0]["description"])

    def test_remoteok_skips_legal_notice_row(self) -> None:
        payload = [
            {"legal": "terms of service"},
            {"id": "1", "position": "AI Engineer", "company": "Acme", "url": "https://remoteok.com/l/1", "tags": ["python"]},
        ]
        with patch.object(job_sources, "http_get_json", return_value=payload):
            rows = job_sources.fetch_source("remoteok")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["company"], "Acme")

    def test_unknown_source_is_rejected(self) -> None:
        with self.assertRaises(job_sources.SourceError):
            job_sources.fetch_source("dosomething-random")

    def test_non_https_is_rejected_before_network(self) -> None:
        with self.assertRaises(job_sources.SourceError):
            job_sources.http_get_json("http://example.com/jobs", source_id="x")


class DiscoveryLogicTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.env = patch.dict(os.environ, {"CAREERPILOT_DB_PATH": str(Path(self.tmp.name) / "jobs.db")})
        self.env.start()

    def tearDown(self) -> None:
        self.env.stop()
        self.tmp.cleanup()

    def test_query_filter_uses_tokens_and_substring(self) -> None:
        lead = {"title": "AI Agent 实习生", "company": "Acme", "description": "Python", "tokens": ["python"]}
        self.assertTrue(discovery._matches(lead, "AI Agent 实习", set()))
        self.assertTrue(discovery._matches(lead, "LangChain", {"langchain"}) is False)
        self.assertTrue(discovery._matches({"title": "Backend", "description": "", "tokens": ["java"]}, "python", {"python"}) is False)

    def test_score_prefers_evidence_coverage(self) -> None:
        lead = {"tokens": ["python", "rag", "mcp"]}
        self.assertGreater(
            discovery._score(lead, {"python", "rag", "mcp"}, set()),
            discovery._score(lead, {"python"}, set()),
        )

    def test_upsert_dedupes_by_source_and_url(self) -> None:
        lead = {
            "source": "remoteok",
            "external_id": "1",
            "url": "https://remoteok.com/l/1",
            "title": "AI Engineer",
            "company": "Acme",
            "tokens": ["python"],
            "score": 0.5,
        }
        discovery._upsert([lead])
        discovery._upsert([{**lead, "title": "AI Engineer (Updated)"}])
        rows = discovery.list_leads()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["title"], "AI Engineer (Updated)")

    def test_save_lead_creates_job_with_source_url(self) -> None:
        discovery._upsert(
            [
                {
                    "source": "greenhouse",
                    "external_id": "9",
                    "url": "https://boards.greenhouse.io/acme/jobs/9",
                    "title": "AI Agent Intern",
                    "company": "Acme",
                    "description": "Build agents with LangGraph",
                    "tokens": ["langgraph"],
                    "score": 0.8,
                }
            ]
        )
        lead_id = discovery.list_leads()[0]["id"]
        job = discovery.save_lead_to_jobs(lead_id)
        self.assertIsNotNone(job)
        self.assertEqual(job["source_url"], "https://boards.greenhouse.io/acme/jobs/9")
        self.assertIn("LangGraph", job["jd"])
        self.assertEqual(discovery.list_leads(status="new"), [])

    def test_ignore_lead(self) -> None:
        discovery._upsert(
            [{"source": "remoteok", "url": "https://remoteok.com/l/2", "title": "X", "tokens": [], "score": 0}]
        )
        lead_id = discovery.list_leads()[0]["id"]
        self.assertTrue(discovery.ignore_lead(lead_id))
        self.assertEqual(discovery.list_leads(status="new"), [])


class DiscoveryApiTests(unittest.TestCase):
    def test_sources_endpoint(self) -> None:
        client = TestClient(app)
        body = client.get("/discovery/sources").json()
        self.assertEqual(len(body["sources"]), 5)

    def test_search_can_fail_gracefully(self) -> None:
        # 只验证错误聚合结构：把抓取换成必然失败
        with patch.object(discovery, "fetch_source", side_effect=job_sources.SourceError("网络不可用")):
            result = discovery.search("python", sources=["remoteok"])
        self.assertEqual(result["saved"], 0)
        self.assertIn("网络不可用", result["errors"][0])


if __name__ == "__main__":
    unittest.main()
