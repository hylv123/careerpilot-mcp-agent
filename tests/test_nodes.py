"""节点级测试：只测"节点收什么、给什么、拼出来的 prompt 对不对"。

不调真实模型：用假 chain 拦在 `invoke_structured` / `retrieve_agentic` 前，
断言的是**节点自身的契约**，而不是模型输出质量。

覆盖：
  - analysis_node：把 user_input 传进链、把结构化结果落成 dict
  - match_evidence：关键词拼接、空结果的降级、注入的 messages 里是 JSON
  - rewrite_resume / gen_questions：analysis/evidence 必须是 JSON 字符串（与 report 一致）
"""

from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from langchain_core.runnables import RunnableLambda

import agent.graph as graph
from agent.schemas import (
    InterviewQuestion,
    InterviewQuestionList,
    JobAnalysisGraph,
    ResumeBullet,
    ResumeBulletList,
)

ANALYSIS = {
    "job_title": "AI 研发实习生",
    "company": "Demo",
    "requirements": ["Python"],
    "keywords": ["RAG", "LangChain"],
    "level": "实习生",
}
EVIDENCE = [
    {"content": "做过 RAG 项目", "source": "项目经历.md", "source_path": "/x/项目经历.md", "score": 0.8}
]


class _FakeModel:
    """`_get_llm()` 的替身：只需要能接到 `|` 右边。"""

    def with_structured_output(self, schema, method=None):
        return RunnableLambda(lambda _prompt: "")


def _state(**overrides):
    state = {
        "user_input": "招聘 AI 研发实习生，要求 RAG",
        "analysis": dict(ANALYSIS),
        "evidence": list(EVIDENCE),
        "report_text": "",
        "resume_bullets": [],
        "interview_questions": [],
        "saved": False,
        "save_error": "",
        "messages": [],
        "tool_rounds": 0,
    }
    state.update(overrides)
    return state


class AnalysisNodeTests(unittest.TestCase):
    def test_passes_user_input_and_returns_plain_dict(self) -> None:
        captured: dict = {}
        parsed = JobAnalysisGraph(**ANALYSIS)

        def fake_invoke_structured(chain, inputs, schema):
            captured.update(inputs)
            captured["schema"] = schema.__name__
            return parsed

        with patch.object(graph, "_get_llm", return_value=_FakeModel()), \
             patch.object(graph, "invoke_structured", side_effect=fake_invoke_structured):
            update = graph.analysis_node(_state())

        self.assertEqual(captured["user_input"], "招聘 AI 研发实习生，要求 RAG")
        self.assertEqual(captured["schema"], "JobAnalysisGraph")
        self.assertIsInstance(update["analysis"], dict, "State 里必须是 dict，不能是 Pydantic 对象")
        self.assertEqual(update["analysis"]["job_title"], "AI 研发实习生")


class MatchEvidenceTests(unittest.TestCase):
    def test_builds_query_from_keywords_and_returns_evidence(self) -> None:
        captured: dict = {}

        def fake_retrieve(keywords, k):
            captured.update({"keywords": keywords, "k": k})
            return list(EVIDENCE)

        with patch.object(graph, "retrieve_agentic", side_effect=fake_retrieve):
            update = graph.match_evidence(_state())

        self.assertEqual(captured["keywords"], ["RAG", "LangChain"])
        self.assertEqual(captured["k"], 4)
        self.assertEqual(update["evidence"], EVIDENCE)

    def test_message_contains_json_not_repr(self) -> None:
        with patch.object(graph, "retrieve_agentic", return_value=list(EVIDENCE)):
            update = graph.match_evidence(_state())

        role, content = update["messages"][0]
        self.assertEqual(role, "user")
        # 中文不转义 + 是合法 JSON 片段
        self.assertIn("做过 RAG 项目", content)
        self.assertNotIn("\\u", content)
        self.assertIn('"source": "项目经历.md"', content)

    def test_empty_retrieval_still_returns_message(self) -> None:
        # 检索为空要能继续走（否则下游 report 会拿到 None）
        with patch.object(graph, "retrieve_agentic", return_value=[]):
            update = graph.match_evidence(_state())

        self.assertEqual(update["evidence"], [])
        self.assertIn("[]", update["messages"][0][1])


class RewriteNodeTests(unittest.TestCase):
    def test_inputs_are_json_strings(self) -> None:
        captured: dict = {}
        parsed = ResumeBulletList(
            bullets=[ResumeBullet(bullet="用 RAG 做过检索", evidence_source="项目经历.md")]
        )

        def fake_invoke_structured(chain, inputs, schema):
            captured.update(inputs)
            return parsed

        with patch.object(graph, "_get_llm", return_value=_FakeModel()), \
             patch.object(graph, "invoke_structured", side_effect=fake_invoke_structured):
            update = graph.rewrite_resume(_state())

        self.assertIsInstance(captured["analysis"], str)
        self.assertIsInstance(captured["evidence"], str)
        self.assertEqual(json.loads(captured["analysis"])["job_title"], "AI 研发实习生")
        self.assertEqual(len(update["resume_bullets"]), 1)
        self.assertEqual(update["resume_bullets"][0]["evidence_source"], "项目经历.md")
        self.assertIn("来源", update["messages"][0][1])


class QuestionsNodeTests(unittest.TestCase):
    def test_inputs_are_json_and_five_questions_rendered(self) -> None:
        captured: dict = {}
        parsed = InterviewQuestionList(
            questions=[
                InterviewQuestion(question=f"问题{i}", answer_points=[f"要点{i}"], evidence_source="项目经历.md")
                for i in range(5)
            ]
        )

        def fake_invoke_structured(chain, inputs, schema):
            captured.update(inputs)
            return parsed

        with patch.object(graph, "_get_llm", return_value=_FakeModel()), \
             patch.object(graph, "invoke_structured", side_effect=fake_invoke_structured):
            update = graph.gen_questions(_state())

        self.assertIsInstance(captured["analysis"], str)
        self.assertIsInstance(captured["evidence"], str)
        self.assertEqual(len(update["interview_questions"]), 5)
        message = update["messages"][0][1]
        for i in range(1, 6):
            self.assertIn(f"### Q{i}:", message)


if __name__ == "__main__":
    unittest.main()
