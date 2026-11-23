"""把结构化的定制简历渲染成一份可直接投递的 Markdown。

为什么单独一层：Agent 只负责"写什么"（结构化事实 + 对应关系），
排版是确定性工作，用模板渲染而不是让模型自由发挥——
模型排版每次都不一样，还会顺手加一句没有出处的自我评价。

渲染内容全部来自 `TailoredResume`，这里不新增任何事实。
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any

_SAFE_NAME = re.compile(r"[^\w\u4e00-\u9fff.-]+")
_PARENS = re.compile(r"\s*[（(][^）)]*[）)]")

SKILL_CATEGORY_ORDER = ("编程语言", "AI核心技术", "框架与工具", "通用技能")


def _plain_skill(item: Any) -> str:
    """技能词去掉括号补充说明，并按 · 分隔的写法清理空白。

    简历里的"Redis Stream（消费组 / PEL 任务回收 / 幂等去重 / 限流）"
    是笔记写法，不是技能名；括号里的细节留给项目经历去讲。
    """
    return _PARENS.sub("", str(item or "")).strip()


def safe_filename(text: str, *, fallback: str = "resume") -> str:
    cleaned = _SAFE_NAME.sub("_", str(text or "")).strip("._")
    return cleaned or fallback


_ASCII_TERM = re.compile(r"[A-Za-z][A-Za-z0-9+#.]{1,}")
_CJK_RUN = re.compile(r"[\u4e00-\u9fff]{3,}")


def _requirement_terms(requirement: str) -> set[str]:
    """从一条要求里抽出用于比对的关键词。

    为什么要抽词而不是直接比整句：模型常把"有 MCP 项目经验优先"这条偏好
    落到"工具/Function Calling 接入"这条要求上（语义对得上，字面对不上），
    只做字符串包含会把明明写了的经历标成"没支撑"，反而误导用户。
    这里取 ASCII 技术词 + 中文三元组，命中即算覆盖，宁可放宽也不漏报。
    """
    terms = {token.casefold() for token in _ASCII_TERM.findall(requirement)}
    for run in _CJK_RUN.findall(requirement):
        terms.update(run[i : i + 3] for i in range(len(run) - 2))
    return {term for term in terms if len(term) >= 2}


def _bullet_covers(bullet: dict[str, Any], requirement: str) -> bool:
    stated = str(bullet.get("jd_requirement", "")).strip()
    if stated and (
        stated == requirement or requirement in stated or stated in requirement
    ):
        return True
    haystack = f"{bullet.get('bullet', '')} {bullet.get('project', '')}".casefold()
    return any(term in haystack for term in _requirement_terms(requirement))


def _coverage_rows(resume: dict[str, Any], analysis: dict[str, Any]) -> list[tuple[str, str, str]]:
    """把"岗位要求 → 是否有真实经历支撑"整理成表格行。

    这是这份简历最该被用户检查的地方：没被覆盖的要求就是真实的短板，
    不应该在简历里假装有。
    """
    bullets = resume.get("bullets") or []
    rows: list[tuple[str, str, str]] = []
    requirements = [str(item) for item in analysis.get("requirements", []) if str(item).strip()]
    preferences = [str(item) for item in analysis.get("preferences", []) if str(item).strip()]

    skills_text = " ".join(
        str(item)
        for group in resume.get("skill_groups", [])
        for item in group.get("items", [])
    ).casefold()

    for label, items in (("要求", requirements), ("偏好", preferences)):
        for requirement in items:
            hits = [bullet for bullet in bullets if _bullet_covers(bullet, requirement)]
            in_skills = any(term in skills_text for term in _requirement_terms(requirement))
            if hits:
                status = "有真实经历支撑"
                evidence = "；".join(
                    sorted({_short_source(str(hit.get("project", ""))) for hit in hits})
                )
            elif in_skills:
                # 技能区写过也算有落点，只是没有对应到具体项目
                status = "有真实经历支撑"
                evidence = "技能栏"
            else:
                status = "暂无可核实的经历"
                evidence = "—"
            rows.append((f"{label}：{requirement}", status, evidence))
    return rows


def _short_source(source: str) -> str:
    """只保留文件名：绝对路径会泄露本机目录，也不适合出现在要投出去的简历里。"""
    cleaned = str(source or "").replace("\\", "/")
    return Path(cleaned).name or cleaned


def render_resume_markdown(
    resume: dict[str, Any],
    analysis: dict[str, Any],
    *,
    candidate_name: str = "",
    generated_at: str | None = None,
) -> str:
    """渲染定制简历 Markdown。

    Args:
        resume: `TailoredResume` 的 dict 形态（已通过真实性校验）。
        analysis: 岗位分析（用于覆盖对照表与标题）。
        candidate_name: 候选人姓名，留空则用"候选人"。
        generated_at: 生成时间文本，默认取当前时间。

    Returns:
        可直接下载/投递的 Markdown 文本。
    """
    job_title = str(analysis.get("job_title") or "目标岗位")
    company = str(analysis.get("company") or "")
    name = candidate_name.strip() or "候选人"
    stamp = generated_at or datetime.now().strftime("%Y-%m-%d")

    lines: list[str] = []
    title_suffix = f"（{company} · {job_title}）" if company else f"（{job_title}）"
    lines.append(f"# {name} — 定制简历{title_suffix}")
    lines.append("")
    lines.append(
        f"> 针对「{company + ' ' if company else ''}{job_title}」改写；"
        "内容仅取自已核实的真实经历，未新增项目、技术或量化数字。"
    )
    lines.append("")

    headline = str(resume.get("headline") or "").strip()
    if headline:
        lines.append("## 一句话定位")
        lines.append("")
        lines.append(headline)
        lines.append("")

    summary = str(resume.get("summary") or "").strip()
    if summary:
        lines.append("## 概述")
        lines.append("")
        lines.append(summary)
        lines.append("")

    skill_groups = resume.get("skill_groups") or []
    if skill_groups:
        lines.append("## 技能专长")
        lines.append("")
        by_category = {str(group.get("category", "")): group for group in skill_groups}
        for category in SKILL_CATEGORY_ORDER:
            group = by_category.get(category)
            if not group:
                continue
            items = [_plain_skill(item) for item in group.get("items", [])]
            items = [item for item in items if item]
            if not items:
                continue
            # 分类单独一行、条目用 · 分隔：一眼能扫完，也不会让人以为在列清单
            lines.append(f"**{category}**\\")
            lines.append(" · ".join(items))
            lines.append("")

    bullets = resume.get("bullets") or []
    if bullets:
        lines.append("## 项目与经历")
        lines.append("")
        by_project: dict[str, list[dict[str, Any]]] = {}
        for bullet in bullets:
            by_project.setdefault(str(bullet.get("project", "")), []).append(bullet)
        for project, items in by_project.items():
            lines.append(f"### {project}")
            lines.append("")
            for item in items:
                # 正文只留简历句子本身：对应要求和来源放进文末自查区，
                # 否则每条下面挂两行注解，读者一眼就看出这是机器拼的。
                summary = str(item.get("summary") or "").strip()
                bullet = str(item.get("bullet", "")).strip()
                lines.append(f"- **{summary}**：{bullet}" if summary else f"- {bullet}")
            lines.append("")

    rows = _coverage_rows(resume, analysis)
    sources = _source_rows(bullets)
    style = resume.get("style") or {}
    lines.append("## 附：自查区（投递前请删除本节）")
    lines.append("")

    if rows:
        lines.append("### 岗位要求对照")
        lines.append("")
        lines.append("| 岗位要求 / 偏好 | 是否有真实经历支撑 | 对应项目 |")
        lines.append("| --- | --- | --- |")
        for requirement, status, evidence in rows:
            lines.append(f"| {requirement} | {status} | {evidence} |")
        lines.append("")

    if sources:
        lines.append("### 事实出处（逐条对照原文）")
        lines.append("")
        lines.append("| 项目 | 证据片段（原文） | 原子 | 来源文件 |")
        lines.append("| --- | --- | --- | --- |")
        for project, span, atom, source in sources:
            lines.append(f"| {project} | {span} | {atom} | {source} |")
        lines.append("")

    if style:
        status = "未发现模板腔" if not style.get("issues") else f"仍有：{'、'.join(style['issues'])}"
        rewritten = "，已自动重写一次" if style.get("retried") else ""
        lines.append(f"### 文风检查\n\n{status}{rewritten}。")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(
        f"生成时间：{stamp}　生成方式：CareerPilot（JD 分析 → 真实经历检索 → 定制改写）。"
        "正文只保留真实经历；出处、要求覆盖与文风检查见文末自查区。"
    )
    lines.append("")
    return "\n".join(lines)


def _source_rows(bullets: list[dict[str, Any]]) -> list[tuple[str, str, str, str]]:
    """逐条列出"要点 ← 哪段原文 ← 哪份文件"。

    正文不标来源，但核对必须能做到句子级：文件级引用太粗，
    一篇文章里哪一句才是支撑，只有贴出原文片段才说得清。
    """
    rows: list[tuple[str, str, str, str]] = []
    for bullet in bullets:
        project = str(bullet.get("project", "")).strip() or "未命名项目"
        span = str(bullet.get("evidence_span", "")).strip() or "—"
        if len(span) > 60:
            span = span[:57] + "…"
        atom = str(bullet.get("atom_id", "")).strip() or "—"
        rows.append((project, span, atom, _short_source(str(bullet.get("evidence_source", "")))))
    return rows


def _is_preference(requirement: str, analysis: dict[str, Any]) -> bool:
    preferences = {str(item).strip() for item in analysis.get("preferences", [])}
    return requirement.strip() in preferences
