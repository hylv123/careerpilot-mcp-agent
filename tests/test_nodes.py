"""节点级测试：只测"节点收什么、给什么、拼出来的 prompt 对不对"。

不调真实模型：用假 chain 拦在 `invoke_structured` / `search_for_match` 前，
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
    SkillGroup,
    TailoredBullet,
    TailoredResume,
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

        with patch.object(graph, "search_for_match", side_effect=fake_retrieve):
            update = graph.match_evidence(_state())

        self.assertEqual(captured["keywords"], ["RAG", "LangChain"])
        self.assertEqual(captured["k"], 4)
        self.assertEqual(update["evidence"], EVIDENCE)

    def test_message_contains_json_not_repr(self) -> None:
        with patch.object(graph, "search_for_match", return_value=list(EVIDENCE)):
            update = graph.match_evidence(_state())

        role, content = update["messages"][0]
        self.assertEqual(role, "user")
        # 中文不转义 + 是合法 JSON 片段
        self.assertIn("做过 RAG 项目", content)
        self.assertNotIn("\\u", content)
        self.assertIn('"source": "项目经历.md"', content)

    def test_empty_retrieval_still_returns_message(self) -> None:
        # 检索为空要能继续走（否则下游 report 会拿到 None）
        with patch.object(graph, "search_for_match", return_value=[]):
            update = graph.match_evidence(_state())

        self.assertEqual(update["evidence"], [])
        self.assertIn("[]", update["messages"][0][1])


class RewriteNodeTests(unittest.TestCase):
    def test_inputs_are_json_strings_and_output_is_tailored_resume(self) -> None:
        captured: dict = {}
        parsed = TailoredResume(
            headline="做过 RAG 检索链路的 AI 应用开发",
            summary="用 LangChain 搭过检索链路。",
            skill_groups=[SkillGroup(category="框架与工具", items=["LangChain"])],
            bullets=[
                TailoredBullet(
                    project="RAG 项目",
                    summary="检索链路",
                    bullet="用 RAG 做过检索",
                    jd_requirement="RAG",
                    evidence_source="项目经历.md",
                    evidence_span="做过 RAG 项目",
                )
            ],
        )

        def fake_invoke_structured(chain, inputs, schema):
            captured.update(inputs)
            return parsed

        state = _state(analysis={**ANALYSIS, "requirements": ["Python", "RAG"]})
        with patch.object(graph, "_get_llm", return_value=_FakeModel()), \
             patch.object(graph, "invoke_structured", side_effect=fake_invoke_structured):
            update = graph.rewrite_resume(state)

        self.assertIsInstance(captured["analysis"], str)
        self.assertIsInstance(captured["evidence"], str)
        self.assertEqual(json.loads(captured["analysis"])["job_title"], "AI 研发实习生")

        tailored = update["tailored_resume"]
        self.assertEqual(tailored["headline"], "做过 RAG 检索链路的 AI 应用开发")
        self.assertEqual(len(tailored["bullets"]), 1)
        self.assertEqual(tailored["bullets"][0]["project"], "RAG 项目")
        # 证据里没有 LangChain，技能词必须被过滤掉
        self.assertEqual(tailored["skill_groups"], [])

        # 兼容契约：resume_bullets 仍然只有 bullet/evidence_source 两个键
        self.assertEqual(len(update["resume_bullets"]), 1)
        self.assertEqual(
            set(update["resume_bullets"][0]), {"bullet", "evidence_source"}
        )
        self.assertIn("来源", update["messages"][0][1])


class AtomInjectionTests(unittest.TestCase):
    def test_approved_atoms_are_injected_and_counted(self) -> None:
        parsed = TailoredResume(
            headline="做过 RAG 检索链路",
            summary="用 LangChain 搭过检索链路。",
            skill_groups=[],
            bullets=[
                TailoredBullet(
                    project="RAG 项目",
                    summary="检索链路",
                    bullet="用 RAG 做过检索",
                    jd_requirement="RAG",
                    evidence_source="项目经历.md",
                    evidence_span="做过 RAG 项目",
                    atom_id="atom-1",
                )
            ],
        )
        captured: dict = {}

        def fake_invoke_structured(chain, inputs, schema):
            captured.update(inputs)
            return parsed

        atom = {"id": "atom-1", "project": "CareerPilot", "span": "做过 RAG 项目", "kind": "experience"}
        state = _state(analysis={**ANALYSIS, "requirements": ["Python", "RAG"]})
        with patch.object(graph, "_get_llm", return_value=_FakeModel()), \
             patch.object(graph, "invoke_structured", side_effect=fake_invoke_structured), \
             patch.object(graph, "atoms_service") as atoms_service:
            atoms_service.approved_experience_atoms.return_value = [atom]
            update = graph.rewrite_resume(state)

        self.assertIn("atom-1", captured["atoms_block"])
        self.assertEqual(update["tailored_resume"]["atom_stats"]["cited"], 1)
        self.assertEqual(update["tailored_resume"]["atom_stats"]["approved"], 1)

class PromptVariableTests(unittest.TestCase):
    """prompt 里加变量却忘在调用处传，只有真跑一次才会炸。

    2026-11-22 真实踩到过：给简历节点加 {style_correction} 时误改了面试题节点的
    human 模板，单测因为 patch 掉 invoke_structured 而全绿，真实运行直接
    KeyError。这里让 prompt 真的格式化一次（不 patch invoke_structured）。
    """

    class _EchoModel:
        def __init__(self, parsed):
            self.parsed = parsed

        def with_structured_output(self, schema, method=None):
            from langchain_core.runnables import RunnableLambda

            return RunnableLambda(lambda _prompt: self.parsed)

    def test_questions_prompt_variables_match_inputs(self) -> None:
        parsed = InterviewQuestionList(
            questions=[
                InterviewQuestion(
                    question=f"问题{i}",
                    answer_points=[f"要点{i}"],
                    evidence_source="项目经历.md",
                )
                for i in range(5)
            ]
        )
        with patch.object(graph, "_get_llm", return_value=self._EchoModel(parsed)):
            update = graph.gen_questions(_state())

        self.assertEqual(len(update["interview_questions"]), 5)

    def test_resume_prompt_variables_match_inputs(self) -> None:
        parsed = TailoredResume(
            headline="做过 RAG 检索链路",
            summary="用 LangChain 搭过检索链路。",
            skill_groups=[],
            bullets=[
                TailoredBullet(
                    project="RAG 项目",
                    summary="检索链路",
                    bullet="用 RAG 做过检索",
                    jd_requirement="RAG",
                    evidence_source="项目经历.md",
                    evidence_span="做过 RAG 项目",
                )
            ],
        )
        state = _state(analysis={**ANALYSIS, "requirements": ["Python", "RAG"]})
        with patch.object(graph, "_get_llm", return_value=self._EchoModel(parsed)):
            update = graph.rewrite_resume(state)

        self.assertEqual(len(update["tailored_resume"]["bullets"]), 1)


class StyleRetryTests(unittest.TestCase):
    """命中 AI 腔时只重写一次，且不改变事实链路。"""

    def _run(self, first, second=None):
        calls: list[dict] = []
        results = [first] if second is None else [first, second]

        def fake_invoke_structured(chain, inputs, schema):
            calls.append(inputs)
            return results[len(calls) - 1]

        state = _state(analysis={**ANALYSIS, "requirements": ["Python", "RAG"]})
        with patch.object(graph, "_get_llm", return_value=_FakeModel()), \
             patch.object(graph, "invoke_structured", side_effect=fake_invoke_structured):
            update = graph.rewrite_resume(state)
        return update, calls

    @staticmethod
    def _resume(headline: str, summary: str, bullet: str) -> TailoredResume:
        return TailoredResume(
            headline=headline,
            summary=summary,
            skill_groups=[],
            bullets=[
                TailoredBullet(
                    project="RAG 项目",
                    summary="检索链路",
                    bullet=bullet,
                    jd_requirement="RAG",
                    evidence_source="项目经历.md",
                    evidence_span="做过 RAG 项目",
                )
            ],
        )

    def test_clean_first_version_is_not_retried(self) -> None:
        update, calls = self._run(
            self._resume("做过 RAG 检索链路", "用 LangChain 搭过检索链路。", "用 RAG 做过检索")
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(update["tailored_resume"]["style"], {"issues": [], "retried": False})

    def test_buzzword_version_is_rewritten_once(self) -> None:
        dirty = self._resume("赋能业务闭环", "致力于打造生态，全链路助力。", "赋能检索闭环")
        clean = self._resume("做过 RAG 检索链路", "用 LangChain 搭过检索链路。", "用 RAG 做过检索")
        update, calls = self._run(dirty, clean)
        self.assertEqual(len(calls), 2)
        self.assertIn("赋能", calls[1]["style_correction"])
        self.assertEqual(update["tailored_resume"]["style"], {"issues": [], "retried": True})
        self.assertEqual(update["tailored_resume"]["headline"], "做过 RAG 检索链路")

    def test_retry_failure_keeps_first_version(self) -> None:
        dirty = self._resume("赋能业务闭环", "致力于打造生态。", "赋能检索闭环")
        calls: list[dict] = []

        def fake_invoke_structured(chain, inputs, schema):
            calls.append(inputs)
            if len(calls) == 1:
                return dirty
            raise RuntimeError("模型又挂了")

        state = _state(analysis={**ANALYSIS, "requirements": ["Python", "RAG"]})
        with patch.object(graph, "_get_llm", return_value=_FakeModel()), \
             patch.object(graph, "invoke_structured", side_effect=fake_invoke_structured):
            update = graph.rewrite_resume(state)

        self.assertEqual(len(calls), 2)
        self.assertIs(update["tailored_resume"]["style"]["retried"], False)
        self.assertTrue(update["tailored_resume"]["style"]["issues"])


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
