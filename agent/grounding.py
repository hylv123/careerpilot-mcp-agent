"""定制简历的真实性校验。

为什么需要这一层（而不是只靠 prompt 里写"不要编造"）：
    简历是**会被投出去**的东西。模型编一个"提升 30% 转化"，用户很可能
    自己都没注意到就发出去了；这类数字面试官一问就穿，代价由候选人承担。
    所以真实性必须是**代码判定**的，不能是提示词里的君子协定。

四道校验，任一不过就丢掉这条要点：
    1. 来源可追溯：evidence_source 必须能在本轮证据里找到（复用 safety 的规则）；
    2. 项目真实存在：project 名称必须在本轮证据正文里出现过；
    3. 数字可核对：要点里出现的每个阿拉伯数字都必须能在证据里找到，
       否则视为编造量化成果（"提升 30%"这种最危险，必须逐字有出处）；
    4. 对应要求有效：jd_requirement 必须是岗位分析里真实存在的要求或偏好，
       避免"为了显得匹配而虚构一条岗位要求"。

同一套规则也用于清洗 headline / summary / 技能列表：
    技能里出现证据中不存在的技术栈，等于凭空多会一门技术。
"""

from __future__ import annotations

import re
from typing import Any, Iterable

from loguru import logger

from .safety import source_in_evidence

_NUMBER = re.compile(r"\d+(?:\.\d+)?")

# 主张强度阶梯：简历里最容易被"升级"的就是这类动词。
# 证据只写"参与"，要点写成"主导"，事实没变但性质变了——这比编数字更隐蔽。
STRONG_CLAIMS = ("主导", "独立负责", "独立完成", "独立开发", "从 0 到 1", "从零", "牵头", "独自主导")
_SENTENCE_SPLIT = re.compile(r"(?<=[。！？；;])")
_WS = re.compile(r"\s+")


def _norm(text: Any) -> str:
    return _WS.sub("", str(text or "")).casefold()


def _squash(text: Any) -> str:
    """去掉空白与标点后再比较。

    为什么技能/项目名要用它：证据里写的是"MCP（FastMCP，3 个 stdio Server）"，
    模型摘成"MCP（FastMCP）"——只去空格仍不相等，会把真实技能误判成编造。
    这里只放宽"是否出现过"的判定，数字校验仍走严格匹配。
    """
    return re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]", "", str(text or "")).casefold()


def evidence_corpus(evidence: Iterable[dict[str, Any]]) -> str:
    """把本轮证据正文拼成一段归一化文本，用来判断某个事实是否真的出现过。"""
    return _norm("\n".join(str(item.get("content", "")) for item in evidence))


def unverified_numbers(text: str, corpus: str) -> list[str]:
    """返回文本里出现、但证据正文中没有的数字。"""
    return [number for number in _NUMBER.findall(str(text or "")) if number not in corpus]


def span_in_evidence(span: str, evidence: Iterable[dict[str, Any]]) -> bool:
    """证据片段必须能在本轮证据正文里逐字找到（忽略空白与大小写）。"""
    target = _norm(span)
    if len(target) < 4:
        return False
    return target in evidence_corpus(evidence)


def strength_escalated(bullet: str, span: str) -> bool:
    """要点是否把"参与"级别的证据写成了"主导"级别。"""
    span_text = str(span or "")
    for claim in STRONG_CLAIMS:
        if claim in str(bullet or "") and claim not in span_text:
            return True
    return False


def _requirement_matches(requirement: str, allowed: set[str]) -> bool:
    """判断模型给出的 jd_requirement 是否真的来自岗位分析。

    允许包含关系（模型常把"熟悉 Python"缩成"Python"），但只在两个串
    都足够长时才放宽，避免"Python" 误配到"Python 与 Java"这种半句。
    """
    target = _norm(requirement)
    if not target:
        return True
    for candidate in allowed:
        if target == candidate:
            return True
        if len(target) >= 4 and len(candidate) >= 4 and (target in candidate or candidate in target):
            return True
    return False


def verify_bullets(
    bullets: Iterable[dict[str, Any]],
    evidence: Iterable[dict[str, Any]],
    analysis: dict[str, Any],
    approved_atoms: dict[str, str] | None = None,
) -> tuple[list[dict[str, Any]], list[tuple[dict[str, Any], list[str]]]]:
    """按四道校验过滤定制简历要点。

    Args:
        approved_atoms: 已确认原子的 {id: span}。传了就校"引用的原子是否真实存在且对得上"；
            不传则只做语料级校验（兼容旧调用）。

    Returns:
        (保留的要点, [(被丢弃的要点, 原因列表)])。
    """
    materialized = list(evidence)
    corpus = evidence_corpus(materialized)
    squeezed = _squash(corpus)
    allowed = {
        _norm(item)
        for item in [*analysis.get("requirements", []), *analysis.get("preferences", [])]
        if str(item).strip()
    }

    kept: list[dict[str, Any]] = []
    dropped: list[tuple[dict[str, Any], list[str]]] = []

    for item in bullets:
        reasons: list[str] = []
        if not source_in_evidence(str(item.get("evidence_source", "")), materialized):
            reasons.append("来源不在本轮证据里")

        project = _squash(item.get("project", ""))
        if not project or project not in squeezed:
            reasons.append(f"项目名未在证据中出现：{item.get('project', '')!r}")

        missing = unverified_numbers(str(item.get("bullet", "")), corpus)
        if missing:
            reasons.append(f"数字在证据里找不到：{missing}")

        span = str(item.get("evidence_span", "")).strip()
        if not span:
            reasons.append("缺少证据片段（evidence_span）")
        elif not span_in_evidence(span, materialized):
            reasons.append(f"证据片段不是原文：{span[:30]!r}")
        elif strength_escalated(str(item.get("bullet", "")), span):
            reasons.append("主张强度高于证据原文（如 参与→主导）")

        atom_id = str(item.get("atom_id", "")).strip()
        if atom_id and approved_atoms is not None:
            atom_span = approved_atoms.get(atom_id)
            if atom_span is None:
                reasons.append(f"引用的原子不存在或未确认：{atom_id}")
            elif _norm(span) not in _norm(atom_span):
                reasons.append(f"证据片段与该原子不一致：{atom_id}")

        if not _requirement_matches(str(item.get("jd_requirement", "")), allowed):
            reasons.append(f"对应要求在岗位分析里不存在：{item.get('jd_requirement', '')!r}")

        if reasons:
            dropped.append((item, reasons))
        else:
            kept.append(item)

    if dropped:
        logger.warning(
            "[grounding] 丢弃 {} 条无法核实的简历要点：{}",
            len(dropped),
            [f"{item.get('project', '')}: {reasons}" for item, reasons in dropped],
        )
    return kept, dropped


def sanitize_text(text: str, corpus: str, *, fallback: str = "") -> str:
    """删掉含未核实数字的句子；整段都不可信时返回 fallback。

    为什么按句子删而不是整段丢：概述里通常一半是真实身份描述、
    一半是量化成绩；整段丢会把可用信息也一起扔掉。
    """
    cleaned = str(text or "").strip()
    if not cleaned:
        return fallback
    kept = [
        sentence
        for sentence in _SENTENCE_SPLIT.split(cleaned)
        if sentence.strip() and not unverified_numbers(sentence, corpus)
    ]
    result = "".join(kept).strip()
    return result or fallback


def filter_skills(
    skill_groups: Iterable[dict[str, Any]],
    evidence: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """删掉证据里不存在的技能词与空分组。"""
    corpus = _squash(evidence_corpus(evidence))
    kept: list[dict[str, Any]] = []
    dropped: list[str] = []
    for group in skill_groups:
        items = [str(item) for item in group.get("items", []) if str(item).strip()]
        valid = [item for item in items if _squash(item) in corpus]
        dropped.extend(item for item in items if item not in valid)
        if valid:
            kept.append({"category": str(group.get("category", "")), "items": valid})
    if dropped:
        logger.warning("[grounding] 丢弃证据里没有的技能词：{}", dropped)
    return kept
