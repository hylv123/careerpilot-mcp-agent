"""JD 定制简历的真实性链路测试。

这组测试锁的是"简历能不能投出去"这件事，不是"文本好不好看"：
  1. 数字必须能在证据里找到（编造量化成果代价最大）；
  2. 项目名必须真实存在；
  3. 每条要点必须对应岗位分析里真实存在的要求或偏好；
  4. 来源必须可追溯；
  5. 渲染出来的 Markdown 要如实标注"哪些要求没有经历支撑"。
"""

from __future__ import annotations

import unittest

from agent.grounding import (
    evidence_corpus,
    filter_skills,
    sanitize_text,
    unverified_numbers,
    verify_bullets,
)
from agent.style import ANTI_AI_RULES, style_correction, style_issues
from api.schemas import AnalyzeResponse
from services.resume_md import render_resume_markdown, safe_filename

EVIDENCE = [
    {
        "content": "在 CareerPilot 项目中用 LangChain 搭建 RAG 检索链路，支持 3 轮工具调用。",
        "source": "knowledge/docs/private/project.md",
        "score": 0.9,
    },
    {
        "content": "负责 FastAPI 接口与 Redis 队列的幂等设计。",
        "source": "knowledge/docs/private/project2.md",
        "score": 0.7,
    },
]

ANALYSIS = {
    "job_title": "AI Agent 开发实习生",
    "company": "Demo",
    "requirements": ["熟悉 LangChain", "熟悉 RAG 检索"],
    "preferences": ["有 MCP 项目经验优先"],
    "keywords": ["RAG", "LangChain"],
    "level": "实习生",
}


def _bullet(**overrides) -> dict:
    base = {
        "project": "CareerPilot",
        "summary": "检索与评测",
        "bullet": "用 LangChain 搭建 RAG 检索链路，支持 3 轮工具调用",
        "jd_requirement": "熟悉 RAG 检索",
        "evidence_source": "knowledge/docs/private/project.md",
        "evidence_span": "用 LangChain 搭建 RAG 检索链路",
    }
    base.update(overrides)
    return base


class VerifyBulletsTests(unittest.TestCase):
    def test_grounded_bullet_is_kept(self) -> None:
        kept, dropped = verify_bullets([_bullet()], EVIDENCE, ANALYSIS)
        self.assertEqual(len(kept), 1)
        self.assertEqual(dropped, [])

    def test_fabricated_number_is_dropped(self) -> None:
        kept, dropped = verify_bullets([_bullet(bullet="把检索准确率提升 30%")], EVIDENCE, ANALYSIS)
        self.assertEqual(kept, [])
        self.assertIn("数字", " ".join(dropped[0][1]))

    def test_fabricated_project_is_dropped(self) -> None:
        kept, dropped = verify_bullets([_bullet(project="某大厂推荐系统")], EVIDENCE, ANALYSIS)
        self.assertEqual(kept, [])
        self.assertIn("项目名", " ".join(dropped[0][1]))

    def test_requirement_must_exist_in_analysis(self) -> None:
        kept, dropped = verify_bullets(
            [_bullet(jd_requirement="熟悉 Kubernetes 集群运维")], EVIDENCE, ANALYSIS
        )
        self.assertEqual(kept, [])
        self.assertIn("要求", " ".join(dropped[0][1]))

    def test_preference_counts_as_valid_requirement(self) -> None:
        kept, _ = verify_bullets([_bullet(jd_requirement="有 MCP 项目经验优先")], EVIDENCE, ANALYSIS)
        self.assertEqual(len(kept), 1)

    def test_unknown_source_is_dropped(self) -> None:
        kept, dropped = verify_bullets([_bullet(evidence_source="resume.md")], EVIDENCE, ANALYSIS)
        self.assertEqual(kept, [])
        self.assertIn("来源", " ".join(dropped[0][1]))

    def test_missing_span_is_dropped(self) -> None:
        kept, dropped = verify_bullets([_bullet(evidence_span="")], EVIDENCE, ANALYSIS)
        self.assertEqual(kept, [])
        self.assertIn("证据片段", " ".join(dropped[0][1]))

    def test_span_must_be_verbatim(self) -> None:
        # 大意相同但不是原文，一样不认
        kept, dropped = verify_bullets(
            [_bullet(evidence_span="用 LangChain 搭建了 RAG 检索")], EVIDENCE, ANALYSIS
        )
        self.assertEqual(kept, [])
        self.assertIn("不是原文", " ".join(dropped[0][1]))

    def test_strength_escalation_is_dropped(self) -> None:
        kept, dropped = verify_bullets(
            [_bullet(bullet="主导 RAG 检索链路搭建")], EVIDENCE, ANALYSIS
        )
        self.assertEqual(kept, [])
        self.assertIn("主张强度", " ".join(dropped[0][1]))

    def test_strength_matching_evidence_is_kept(self) -> None:
        evidence = [
            {
                "content": "在 CareerPilot 中独立负责 RAG 检索链路的搭建，支持 3 轮工具调用。",
                "source": "knowledge/docs/private/project.md",
            }
        ]
        kept, _ = verify_bullets(
            [
                _bullet(
                    bullet="独立负责 RAG 检索链路搭建，支持 3 轮工具调用",
                    evidence_span="独立负责 RAG 检索链路的搭建，支持 3 轮工具调用",
                )
            ],
            evidence,
            ANALYSIS,
        )
        self.assertEqual(len(kept), 1)

    def test_atom_citation_must_match_approved_atom(self) -> None:
        atoms = {"atom-1": "用 LangChain 搭建 RAG 检索链路"}
        kept, _ = verify_bullets([_bullet(atom_id="atom-1")], EVIDENCE, ANALYSIS, approved_atoms=atoms)
        self.assertEqual(len(kept), 1)

        kept, dropped = verify_bullets(
            [_bullet(atom_id="atom-fake")], EVIDENCE, ANALYSIS, approved_atoms=atoms
        )
        self.assertEqual(kept, [])
        self.assertIn("原子", " ".join(dropped[0][1]))

    def test_atom_span_must_match_atom_text(self) -> None:
        atoms = {"atom-1": "负责 FastAPI 接口与 Redis 队列的幂等设计"}
        kept, dropped = verify_bullets(
            [_bullet(atom_id="atom-1")], EVIDENCE, ANALYSIS, approved_atoms=atoms
        )
        self.assertEqual(kept, [])
        self.assertIn("不一致", " ".join(dropped[0][1]))

    def test_bullet_without_atom_still_allowed(self) -> None:
        # 兜底路径：没引用原子时仍可用现有 span 校验通过
        kept, _ = verify_bullets([_bullet(atom_id="")], EVIDENCE, ANALYSIS, approved_atoms={})
        self.assertEqual(len(kept), 1)

    def test_mixed_batch_keeps_only_valid(self) -> None:
        kept, dropped = verify_bullets(
            [_bullet(), _bullet(bullet="提升 50% 效率")], EVIDENCE, ANALYSIS
        )
        self.assertEqual(len(kept), 1)
        self.assertEqual(len(dropped), 1)


class TextSanitizingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.corpus = evidence_corpus(EVIDENCE)

    def test_unverified_numbers_are_detected(self) -> None:
        self.assertEqual(unverified_numbers("支持 3 轮工具调用", self.corpus), [])
        self.assertEqual(unverified_numbers("提升 30%", self.corpus), ["30"])

    def test_sentence_with_fabricated_number_is_removed(self) -> None:
        text = "熟悉 Agent 应用开发。曾把延迟降低 40%。"
        cleaned = sanitize_text(text, self.corpus)
        self.assertIn("熟悉 Agent 应用开发", cleaned)
        self.assertNotIn("40", cleaned)

    def test_all_unverified_falls_back(self) -> None:
        cleaned = sanitize_text("把转化率提升 80%。", self.corpus, fallback="默认定位")
        self.assertEqual(cleaned, "默认定位")

    def test_skill_with_punctuation_variant_is_kept(self) -> None:
        # 证据写"MCP（FastMCP，3 个 stdio Server）"，模型摘成"MCP（FastMCP）"
        groups = [{"category": "AI 应用", "items": ["MCP（FastMCP）"]}]
        evidence = [{"content": "用 MCP（FastMCP，3 个 stdio Server）暴露工具。", "source": "x.md"}]
        self.assertEqual(filter_skills(groups, evidence), groups)

    def test_skills_not_in_evidence_are_removed(self) -> None:
        groups = [
            {"category": "框架与工具", "items": ["LangChain", "TensorFlow"]},
            {"category": "通用技能", "items": ["Kubernetes"]},
        ]
        kept = filter_skills(groups, EVIDENCE)
        self.assertEqual(kept, [{"category": "框架与工具", "items": ["LangChain"]}])


class ResumeMarkdownTests(unittest.TestCase):
    def setUp(self) -> None:
        self.resume = {
            "headline": "做过 RAG 检索链路的 AI 应用开发",
            "summary": "用 LangChain 搭建过检索链路，并负责接口与队列的幂等设计。",
            "skill_groups": [{"category": "框架与工具", "items": ["LangChain", "FastAPI"]}],
            "bullets": [
                _bullet(),
                _bullet(
                    project="CareerPilot",
                    bullet="负责 FastAPI 接口与 Redis 队列的幂等设计",
                    jd_requirement="熟悉 LangChain",
                    evidence_source="project2.md",
                ),
            ],
        }
        self.markdown = render_resume_markdown(
            self.resume, ANALYSIS, candidate_name="张三", generated_at="2026-11-22"
        )

    def test_contains_all_sections(self) -> None:
        for heading in ("# 张三", "## 一句话定位", "## 概述", "## 技能专长", "## 项目与经历"):
            self.assertIn(heading, self.markdown)

    def test_body_has_no_meta_annotations(self) -> None:
        """正文里不能出现"对应岗位要求 / 来源"这类机器注解。"""
        body = self.markdown.split("## 附：自查区")[0]
        self.assertNotIn("对应岗位", body)
        self.assertNotIn("来源", body)
        self.assertIn("- **检索与评测**：用 LangChain 搭建 RAG 检索链路，支持 3 轮工具调用", body)

    def test_appendix_keeps_traceability(self) -> None:
        self.assertIn("### 事实出处", self.markdown)
        self.assertIn("用 LangChain 搭建 RAG 检索链路", self.markdown)
        # 绝对路径不能出现在简历里
        self.assertNotIn("knowledge/docs/private", self.markdown)

    def test_coverage_table_marks_missing_experience(self) -> None:
        self.assertIn("| 偏好：有 MCP 项目经验优先 | 暂无可核实的经历 |", self.markdown)

    def test_states_truthfulness(self) -> None:
        self.assertIn("未新增项目、技术或量化数字", self.markdown)


    def test_skills_use_fixed_category_order_and_dot_separator(self) -> None:
        resume = {
            "headline": "h",
            "summary": "s",
            "skill_groups": [
                {"category": "通用技能", "items": ["技术文档撰写", "团队协作"]},
                {"category": "编程语言", "items": ["Python", "SQL"]},
                {"category": "AI核心技术", "items": ["RAG", "Prompt 工程"]},
            ],
            "bullets": [_bullet()],
        }
        markdown = render_resume_markdown(resume, ANALYSIS, generated_at="2026-11-22")
        self.assertIn("**编程语言**\\\nPython · SQL", markdown)
        self.assertLess(markdown.index("**编程语言**"), markdown.index("**AI核心技术**"))
        self.assertLess(markdown.index("**AI核心技术**"), markdown.index("**通用技能**"))

    def test_skill_parentheses_are_stripped(self) -> None:
        resume = {
            "headline": "h",
            "summary": "s",
            "skill_groups": [
                {"category": "框架与工具", "items": ["Redis（消费组 / PEL 回收）", "Docker"]}
            ],
            "bullets": [_bullet()],
        }
        markdown = render_resume_markdown(resume, ANALYSIS, generated_at="2026-11-22")
        self.assertIn("Redis · Docker", markdown)
        self.assertNotIn("PEL 回收", markdown)

    def test_requirement_covered_by_skills_section(self) -> None:
        resume = {
            "headline": "h",
            "summary": "s",
            "skill_groups": [{"category": "编程语言", "items": ["Python"]}],
            "bullets": [_bullet(jd_requirement="熟悉 RAG 检索")],
        }
        analysis = {**ANALYSIS, "requirements": [*ANALYSIS["requirements"], "Python"]}
        markdown = render_resume_markdown(resume, analysis, generated_at="2026-11-22")
        self.assertIn("| 要求：Python | 有真实经历支撑 | 技能栏 |", markdown)

    def test_preference_covered_by_keyword_in_bullet(self) -> None:
        """模型把 MCP 经历挂到"工具接入"这条要求上时，MCP 偏好不应被标成没支撑。"""
        resume = {
            "headline": "h",
            "summary": "s",
            "skill_groups": [],
            "bullets": [
                _bullet(
                    bullet="用 MCP（FastMCP）把查简历、记投递做成独立工具，由模型按协议调用",
                    jd_requirement="工具/Function Calling 接入",
                )
            ],
        }
        analysis = {**ANALYSIS, "preferences": [*ANALYSIS["preferences"], "有 Kubernetes 集群经验"]}
        markdown = render_resume_markdown(resume, analysis, generated_at="2026-11-22")
        self.assertIn("| 偏好：有 MCP 项目经验优先 | 有真实经历支撑 |", markdown)
        # 完全没写过的偏好不能被凑出来
        self.assertIn("| 偏好：有 Kubernetes 集群经验 | 暂无可核实的经历 |", markdown)

    def test_filename_is_safe(self) -> None:
        self.assertEqual(safe_filename("Demo/AI: 实习生?"), "Demo_AI_实习生")
        self.assertEqual(safe_filename("???"), "resume")


class StyleTests(unittest.TestCase):
    def test_detects_buzzwords(self) -> None:
        self.assertEqual(style_issues(["通过赋能业务实现闭环"]), ["赋能", "闭环"])

    def test_clean_engineering_text_has_no_issues(self) -> None:
        text = "用 FastAPI 重写接口，POST /analyze 立即返回 task_id，客户端轮询取结果。"
        self.assertEqual(style_issues([text]), [])

    def test_repeated_sentence_frame_is_detected(self) -> None:
        hits = style_issues(["基于 A 实现检索。基于 B 实现队列。基于 C 实现评测。"])
        self.assertTrue(any("基于" in hit for hit in hits))

    def test_correction_names_offending_terms(self) -> None:
        message = style_correction(["赋能"])
        self.assertIn("赋能", message)
        self.assertIn("保持所有事实", message)

    def test_rules_text_lists_forbidden_words(self) -> None:
        self.assertIn("赋能", ANTI_AI_RULES)
        self.assertIn("45", ANTI_AI_RULES)


class RunnerIntegrationTests(unittest.TestCase):
    def test_markdown_is_rendered_into_api_payload(self) -> None:
        from tasks import runner

        class _StubGraph:
            def invoke(self, state, config=None):
                return {
                    **state,
                    "analysis": ANALYSIS,
                    "evidence": EVIDENCE,
                    "report_text": "报告",
                    "resume_bullets": [{"bullet": "用 LangChain 搭建 RAG", "evidence_source": "knowledge/docs/private/project.md"}],
                    "interview_questions": [],
                    "tailored_resume": {
                        "headline": "做过 RAG 检索链路的 AI 应用开发",
                        "summary": "用 LangChain 搭建过检索链路。",
                        "skill_groups": [{"category": "框架", "items": ["LangChain"]}],
                        "bullets": [_bullet()],
                    },
                    "saved": True,
                    "save_error": "",
                }

        original = runner._graph
        runner._graph = _StubGraph()
        try:
            result = runner.run_analysis("招聘 AI Agent 开发实习生")
        finally:
            runner._graph = original

        self.assertIn("markdown", result["tailored_resume"])
        self.assertIn("## 一句话定位", result["tailored_resume"]["markdown"])
        # 没有定制简历时不应强行渲染空文档
        response = AnalyzeResponse(**result)
        self.assertTrue(response.tailored_resume["bullets"])

    def test_empty_tailored_resume_has_no_markdown(self) -> None:
        from tasks import runner

        class _StubGraph:
            def invoke(self, state, config=None):
                return {**state, "analysis": ANALYSIS, "evidence": [], "tailored_resume": {}}

        original = runner._graph
        runner._graph = _StubGraph()
        try:
            result = runner.run_analysis("jd")
        finally:
            runner._graph = original

        self.assertEqual(result["tailored_resume"], {})


if __name__ == "__main__":
    unittest.main()
