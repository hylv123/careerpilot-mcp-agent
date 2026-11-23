"""Deterministic, evidence-backed job-match assessment.

This complements the LLM narrative; it does not claim semantic equivalence.
Each covered requirement is visible and traceable to retrieved material.
"""

from __future__ import annotations

from typing import Any


def assess_match(analysis: dict[str, Any], evidence: list[dict[str, Any]]) -> dict[str, Any]:
    requirements = [str(item).strip() for item in analysis.get("requirements", []) if str(item).strip()]
    corpus = "\n".join(str(item.get("content", "")) for item in evidence).casefold()
    covered = [item for item in requirements if item.casefold() in corpus]
    gaps = [item for item in requirements if item not in covered]
    score = round(100 * len(covered) / len(requirements)) if requirements else 0
    return {
        "score": score,
        "method": "需求关键词在本轮检索证据中的字面覆盖率；不等同于能力结论。",
        "covered_requirements": covered,
        "gaps": gaps,
        "evidence_sources": sorted({str(item.get("source", "")) for item in evidence if item.get("source")}),
    }
