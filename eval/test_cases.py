"""CareerPilot 20-case evaluation set.

Run from the ``careerpilot`` directory::

    .venv/Scripts/python.exe -m eval.test_cases

Use ``--limit 1`` for a cheap smoke test. MCP tools are disabled by default so
evaluation cannot accidentally write a document or job record; pass
``--with-tools`` only when the complete MCP path also needs to be evaluated.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import agent.graph as graph_module
from agent.schemas import (
    InterviewQuestionList,
    JobAnalysisGraph,
    ResumeBulletList,
)
from agent.state import State

# Windows 控制台默认 GBK：错误信息里出现 ¥ 这类字符会让 print 抛 UnicodeEncodeError，
# 进而中断整轮评测并丢掉结果。统一成 UTF-8 + 替换字符。
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


CASES: list[dict[str, Any]] = [
    {
        "id": "real-01",
        "kind": "real",
        "source_title": "AI & Technical Program Management Intern - AI Fund",
        "source_url": "https://jobs.lever.co/AIFund/66bd2284-3a9b-4c58-bae1-efc78638ec64",
        "jd": "AI Fund 招聘 AI 与技术项目管理实习生，要求理解 LLM、RAG 和 Agent 的应用，能够使用 Python 编写基础脚本，并接触过 LangChain 等生成式 AI 工具。",
        "expect_skills": ["Python", "RAG", "LangChain"],
    },
    {
        "id": "real-02",
        "kind": "real",
        "source_title": "Engineering Intern - Fluency",
        "source_url": "https://jobs.ashbyhq.com/fluency/2aced4e2-485b-4525-802c-763e62c91e88",
        "jd": "Fluency 招聘工程实习生，AI 方向需要使用 LLM 和 Agent 系统，设计 Prompt 与评测，并参与 Agent 编排、可观测性和数据流水线建设，主要语言为 Python 或 TypeScript。",
        "expect_skills": ["LLM", "Agent", "Python"],
    },
    {
        "id": "real-03",
        "kind": "real",
        "source_title": "Data Engineer Intern - Malt",
        "source_url": "https://jobs.lever.co/malt/473946fd-c520-4892-a851-c486af9ed851",
        "jd": "Malt 招聘数据工程实习生，要求使用 SQL、dbt 和 Python，并在数据工程代码库中构建 AI Agent；希望候选人了解 LLM、RAG、OpenAI API 或 LangChain。",
        "expect_skills": ["Python", "SQL", "RAG"],
    },
    {
        "id": "real-04",
        "kind": "real",
        "source_title": "GTM AI Engineering Intern - G2",
        "source_url": "https://jobs.ashbyhq.com/G2/c61dbaff-87e5-4481-82a7-0636e8aea415",
        "jd": "G2 招聘 GTM AI 工程实习生，参与 AI Agent 的开发部署与自动化工作流建设，要求具备 Python 编程能力、API 与 SDK 使用经验，并熟悉 LLM 平台和 Prompt Engineering。",
        "expect_skills": ["Agent", "Python", "API"],
    },
    {
        "id": "real-05",
        "kind": "real",
        "source_title": "AI Engineering Intern - QuantCo",
        "source_url": "https://jobs.lever.co/quantco-/da68e273-8f7e-47a7-a7d6-bf658a52561b",
        "jd": "QuantCo 招聘 AI 工程实习生，负责设计、测试与部署 AI 系统，建设数据和推理流水线，并编排集成 RAG 等 LLM 组件；岗位要求具备 Python 编程经验。",
        "expect_skills": ["Python", "RAG", "LLM"],
    },
    {
        "id": "real-06",
        "kind": "real",
        "source_title": "Software Engineering Intern - Auctor",
        "source_url": "https://jobs.ashbyhq.com/auctor/9d67c881-3ca7-4616-a059-e94c03e19607/",
        "jd": "Auctor 招聘软件工程实习生，使用 Next.js 与 Python 开发产品功能，原型化和评测 LLM Agent 工作流，并针对大型企业数据建设检索流水线；希望候选人了解向量数据库。",
        "expect_skills": ["Python", "Agent", "检索"],
    },
    {
        "id": "real-07",
        "kind": "real",
        "source_title": "AI Intern - Infinite Machine",
        "source_url": "https://jobs.ashbyhq.com/infinite-machine/265d75f2-8d42-4d6b-9ecd-13d65a4b5d1a/",
        "jd": "Infinite Machine 招聘 AI 实习生，使用 LLM Agent、脚本和工作流自动化增强公司业务，建设内部工具、Dashboard 与自动化能力；加分技能包括 Python 和 JavaScript/TypeScript。",
        "expect_skills": ["LLM", "Agent", "Python"],
    },
    {
        "id": "real-08",
        "kind": "real",
        "source_title": "Software Engineer (Agent Platform) Intern - Netic",
        "source_url": "https://jobs.ashbyhq.com/netic/b0ea7aab-8eea-4d31-96f9-278364180ae7",
        "jd": "Netic 招聘 Agent Platform 软件工程实习生，参与多 Agent 工作流编排、Agent 评测循环、Guardrail 与回滚机制；希望候选人熟悉 Python、FastAPI、LLM API、RAG、LangChain 或 MCP。",
        "expect_skills": ["Python", "FastAPI", "MCP"],
    },
    {
        "id": "real-09",
        "kind": "real",
        "source_title": "AI FS Engineer Intern - Intelligent Internet",
        "source_url": "https://jobs.ashbyhq.com/Intelligent.Internet/4f9bdc2c-8333-4654-82fe-7392069e4174/",
        "jd": "Intelligent Internet 招聘 AI 全栈工程实习生，使用 Python、FastAPI 或 Flask 构建 Agent 工作流后端，开发基于 Qdrant、Pinecone 或 FAISS 的 RAG 组件，并要求熟悉 API、Git 与 Docker。",
        "expect_skills": ["Python", "FastAPI", "FAISS"],
    },
    {
        "id": "real-10",
        "kind": "real",
        "source_title": "AI/LLM Engineering Intern - Mercura",
        "source_url": "https://jobs.ashbyhq.com/mercura/24a9e5dc-384c-4351-a208-4b2a961d1d23/",
        "jd": "Mercura 招聘 AI/LLM 工程实习生，端到端建设 Agentic AI 系统，开发 RAG、混合搜索和结构化检索，并构建评测、监控与反馈循环；岗位要求 Python，FastAPI 为加分项。",
        "expect_skills": ["Python", "RAG", "评测"],
    },
    {
        "id": "variant-01",
        "kind": "variant",
        "source_case": "real-01",
        "jd": "希望你用检索增强生成技术建设个人资料问答能力，完成语义召回并利用编排框架生成有依据的回答。",
        "expect_skills": ["检索", "语义", "编排"],
    },
    {
        "id": "variant-02",
        "kind": "variant",
        "source_case": "real-02",
        "jd": "岗位需要让语言模型按标准协议使用外部能力，并能够执行带状态的多步决策流程。",
        "expect_skills": ["协议", "工具", "工作流"],
    },
    {
        "id": "variant-03",
        "kind": "variant",
        "source_case": "real-03",
        "jd": "需要编写服务端程序处理业务数据，提供接口并操作关系型数据，代码要使用版本控制管理。",
        "expect_skills": ["接口", "数据", "版本控制"],
    },
    {
        "id": "variant-04",
        "kind": "variant",
        "source_case": "real-04",
        "jd": "负责把长文档拆成合适片段，将文本映射为稠密表示，并从本地索引中找出语义接近的内容。",
        "expect_skills": ["文本", "向量", "索引"],
    },
    {
        "id": "variant-05",
        "kind": "variant",
        "source_case": "real-05",
        "jd": "将校园业务表格中的重复记录合并，按类别整理数据，并通过命令行脚本输出清单。",
        "expect_skills": ["表格", "去重", "脚本"],
    },
    {
        "id": "variant-06",
        "kind": "variant",
        "source_case": "real-06",
        "jd": "研究文字内容的数值化表示，使含义相近的句子在表示空间中距离更近，并验证匹配效果。",
        "expect_skills": ["文本", "向量", "相似"],
    },
    {
        "id": "variant-07",
        "kind": "variant",
        "source_case": "real-07",
        "jd": "提升AI流水线的可靠性，需要记录每一步执行轨迹，对失败进行兜底，并用测试样例衡量输出质量。",
        "expect_skills": ["日志", "容错", "评测"],
    },
    {
        "id": "variant-08",
        "kind": "variant",
        "source_case": "real-08",
        "jd": "快速制作一个可交互的求职助手演示页面，让用户提交岗位描述并查看模型生成结果。",
        "expect_skills": ["交互", "页面", "模型"],
    },
    {
        "id": "variant-09",
        "kind": "variant",
        "source_case": "real-09",
        "jd": "为岗位跟踪功能设计轻量级本地存储，支持保存、筛选和更新申请状态。",
        "expect_skills": ["存储", "筛选", "状态"],
    },
    {
        "id": "variant-10",
        "kind": "variant",
        "source_case": "real-10",
        "jd": "生成内容必须能够追溯到候选人的原始材料，需要检测无依据陈述并对输出格式做自动检查。",
        "expect_skills": ["来源", "幻觉", "格式"],
    },
]


PLACEHOLDER_PATTERNS = (
    "不存在的项目名",
    "某某项目",
    "xxx项目",
    "虚构项目",
)


def _normalize_source(value: str) -> str:
    """Normalize a generated source for tolerant path comparison."""
    normalized = value.strip().strip("`'\"《》").replace("\\", "/").casefold()
    return re.sub(r"/+", "/", normalized)


def _source_matches(candidate: str, allowed_sources: set[str]) -> bool:
    normalized = _normalize_source(candidate)
    if not normalized:
        return False
    return any(
        normalized == allowed
        or normalized.endswith("/" + allowed)
        or allowed.endswith("/" + normalized)
        for allowed in allowed_sources
    )


def _collect_message_text(output: dict[str, Any]) -> str:
    parts: list[str] = []
    for message in output.get("messages", []):
        content = getattr(message, "content", "")
        if isinstance(content, str):
            parts.append(content)
        elif content:
            parts.append(str(content))
    return "\n".join(parts)


def _format_exception_chain(exc: BaseException) -> str:
    """Keep wrapped validation/model errors visible in the JSON report."""
    parts: list[str] = []
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        parts.append(f"{type(current).__name__}: {current}")
        current = current.__cause__ or current.__context__
    return " <- ".join(parts)


# 环境级故障（TLS/连接/超时）与用例本身的失败分开处理：
# 前者重跑该用例，后者计入质量指标。
ENV_ERROR_HINTS = (
    "ssl",
    "certificate",
    "connection",
    "timed out",
    "timeout",
    "eof",
    "reset by peer",
    "temporarily unavailable",
    "name resolution",
)
ENV_RETRIES = 2
ENV_RETRY_DELAYS = (5.0, 15.0)


def _is_env_error(exc: BaseException) -> bool:
    """判断异常是否由网络/TLS 这类环境问题引起。"""
    text = _format_exception_chain(exc).lower()
    return any(hint in text for hint in ENV_ERROR_HINTS)


def evaluate_output(case: dict[str, Any], output: dict[str, Any]) -> dict[str, Any]:
    """Evaluate one completed graph state without another LLM call."""
    validation_errors: list[str] = []

    try:
        analysis = JobAnalysisGraph.model_validate(output.get("analysis", {}))
    except Exception as exc:
        analysis = None
        validation_errors.append(f"JobAnalysisGraph: {exc}")

    try:
        bullets = ResumeBulletList.model_validate(
            {"bullets": output.get("resume_bullets", [])}
        )
    except Exception as exc:
        bullets = None
        validation_errors.append(f"ResumeBulletList: {exc}")

    try:
        questions = InterviewQuestionList.model_validate(
            {"questions": output.get("interview_questions", [])}
        )
    except Exception as exc:
        questions = None
        validation_errors.append(f"InterviewQuestionList: {exc}")

    generated_sources: list[str] = []
    if bullets is not None:
        generated_sources.extend(item.evidence_source for item in bullets.bullets)
    if questions is not None:
        generated_sources.extend(item.evidence_source for item in questions.questions)

    evidence_sources = {
        _normalize_source(str(item.get("source", "")))
        for item in output.get("evidence", [])
        if str(item.get("source", "")).strip()
    }
    citation_ok = bool(generated_sources) and all(
        source.strip() for source in generated_sources
    )
    citation_truth_ok = (
        citation_ok
        and bool(evidence_sources)
        and all(_source_matches(source, evidence_sources) for source in generated_sources)
    )

    structured_text = json.dumps(
        {
            "analysis": output.get("analysis", {}),
            "resume_bullets": output.get("resume_bullets", []),
            "interview_questions": output.get("interview_questions", []),
        },
        ensure_ascii=False,
    )
    expected_skills = case["expect_skills"]
    skill_coverage_ok = all(
        skill.casefold() in structured_text.casefold() for skill in expected_skills
    )

    combined_text = structured_text + "\n" + _collect_message_text(output)
    placeholder_hit = next(
        (pattern for pattern in PLACEHOLDER_PATTERNS if pattern.casefold() in combined_text.casefold()),
        None,
    )
    hallucination_risk = not citation_truth_ok or placeholder_hit is not None

    return {
        "id": case["id"],
        "kind": case["kind"],
        "source_title": case.get("source_title"),
        "source_url": case.get("source_url"),
        "source_case": case.get("source_case"),
        "jd": case["jd"],
        "expect_skills": expected_skills,
        "format_ok": not validation_errors,
        "skill_coverage_ok": skill_coverage_ok,
        "citation_ok": citation_ok,
        "citation_truth_ok": citation_truth_ok,
        "hallucination_risk": hallucination_risk,
        "placeholder_hit": placeholder_hit,
        "validation_errors": validation_errors,
        "evidence_sources": sorted(evidence_sources),
        "generated_sources": generated_sources,
        "analysis": analysis.model_dump() if analysis is not None else None,
        "resume_bullets": (
            [item.model_dump() for item in bullets.bullets]
            if bullets is not None
            else output.get("resume_bullets", [])
        ),
        "interview_questions": (
            [item.model_dump() for item in questions.questions]
            if questions is not None
            else output.get("interview_questions", [])
        ),
    }


def _initial_state(jd: str) -> State:
    return {
        "user_input": jd,
        "analysis": {},
        "evidence": [],
        "resume_bullets": [],
        "interview_questions": [],
        "messages": [],
        "tool_rounds": 0,
    }


def _print_summary(results: list[dict[str, Any]], elapsed_seconds: float) -> None:
    total = len(results)
    if total == 0:
        print("没有执行任何评测用例。")
        return

    def passed(field: str) -> int:
        return sum(result.get(field) is True for result in results)

    risk_count = sum(result.get("hallucination_risk") is True for result in results)
    failed_count = sum("runtime_error" in result for result in results)
    env_retries = sum(int(result.get("env_retries", 0)) for result in results)
    env_failed = sum(1 for result in results if result.get("env_error"))

    print("\n" + "=" * 64)
    print(f"评测完成：{total} 例，运行异常：{failed_count} 例，耗时：{elapsed_seconds:.1f}s")
    print(f"格式通过率：{passed('format_ok')}/{total} ({passed('format_ok') / total:.1%})")
    print(f"关键词覆盖率：{passed('skill_coverage_ok')}/{total} ({passed('skill_coverage_ok') / total:.1%})")
    print(f"引用存在率：{passed('citation_ok')}/{total} ({passed('citation_ok') / total:.1%})")
    print(f"引用真实性：{passed('citation_truth_ok')}/{total} ({passed('citation_truth_ok') / total:.1%})")
    print(f"明显幻觉风险：{risk_count}/{total} ({risk_count / total:.1%})")
    if env_retries or env_failed:
        print(f"环境级故障：重跑 {env_retries} 次，重跑后仍失败 {env_failed} 例")
    print("=" * 64)


def run_eval(
    cases: list[dict[str, Any]],
    output_path: Path,
    *,
    with_tools: bool = False,
) -> list[dict[str, Any]]:
    """Run selected cases and persist detailed machine-readable results."""
    if not with_tools:
        graph_module.get_mcp_tools = lambda: []

    app = graph_module.build_graph()
    results: list[dict[str, Any]] = []
    started = time.perf_counter()

    for index, case in enumerate(cases, start=1):
        print(f"[{index:02d}/{len(cases):02d}] {case['id']} 开始")
        case_started = time.perf_counter()
        env_retries = 0
        while True:
            try:
                output = app.invoke(_initial_state(case["jd"]))
                result = evaluate_output(case, output)
                result["elapsed_seconds"] = round(time.perf_counter() - case_started, 3)
                if env_retries:
                    result["env_retries"] = env_retries
                print(
                    f"  format={result['format_ok']} "
                    f"citation={result['citation_ok']} "
                    f"truth={result['citation_truth_ok']} "
                    f"risk={result['hallucination_risk']}"
                )
                break
            except Exception as exc:
                if _is_env_error(exc) and env_retries < ENV_RETRIES:
                    delay = ENV_RETRY_DELAYS[min(env_retries, len(ENV_RETRY_DELAYS) - 1)]
                    env_retries += 1
                    print(
                        f"  ENV-RETRY {env_retries}/{ENV_RETRIES}: "
                        f"{type(exc).__name__}，{delay:.0f}s 后重跑该用例"
                    )
                    time.sleep(delay)
                    continue
                result = {
                    "id": case["id"],
                    "kind": case["kind"],
                    "source_title": case.get("source_title"),
                    "source_url": case.get("source_url"),
                    "source_case": case.get("source_case"),
                    "jd": case["jd"],
                    "expect_skills": case["expect_skills"],
                    "format_ok": False,
                    "skill_coverage_ok": False,
                    "citation_ok": False,
                    "citation_truth_ok": False,
                    "hallucination_risk": True,
                    "runtime_error": _format_exception_chain(exc),
                    "env_error": _is_env_error(exc),
                    "env_retries": env_retries,
                    "elapsed_seconds": round(time.perf_counter() - case_started, 3),
                }
                print(f"  ERROR: {result['runtime_error']}")
                break
        results.append(result)

    elapsed = time.perf_counter() - started
    report = {
        "summary": {
            "total": len(results),
            "format_passed": sum(item.get("format_ok") is True for item in results),
            "skill_coverage_passed": sum(
                item.get("skill_coverage_ok") is True for item in results
            ),
            "citation_passed": sum(item.get("citation_ok") is True for item in results),
            "citation_truth_passed": sum(
                item.get("citation_truth_ok") is True for item in results
            ),
            "hallucination_risk_count": sum(
                item.get("hallucination_risk") is True for item in results
            ),
            "runtime_error_count": sum("runtime_error" in item for item in results),
            "env_error_count": sum(1 for item in results if item.get("env_error")),
            "env_retry_count": sum(int(item.get("env_retries", 0)) for item in results),
            "elapsed_seconds": round(elapsed, 3),
            "with_tools": with_tools,
        },
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _print_summary(results, elapsed)
    print(f"详细结果：{output_path.resolve()}")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 CareerPilot 20 例评测集")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="只运行前 N 例，用于低成本冒烟测试",
    )
    parser.add_argument(
        "--with-tools",
        action="store_true",
        help="启用真实 MCP 工具；默认关闭以避免评测产生外部写操作",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("results.json"),
        help="详细 JSON 结果文件路径",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="只列出20个用例，不调用模型",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.list:
        for case in CASES:
            print(f"{case['id']} [{case['kind']}] {case['jd']}")
        print(f"共 {len(CASES)} 例")
        return 0

    selected = CASES if args.limit is None else CASES[: max(args.limit, 0)]
    run_eval(selected, args.output, with_tools=args.with_tools)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
