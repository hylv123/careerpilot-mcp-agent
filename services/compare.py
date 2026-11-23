"""多岗位聚类与投递聚焦：把岗位库变成"该重点投哪类"的判断。

为什么是确定性算法而不是让模型总结：
    "哪类岗位我更有胜算"这个问题，答案是可计算的——把每条 JD 的岗位要求
    摊成技能集合，按集合相似度聚类，再用"这些技能在我的证据库里覆盖率"排序。
    模型来写只会给出一段听起来合理、但每次都不一样的话。
    这里输出的是同一份数据永远算出同一个结果，能解释、能复核。

覆盖率的定义（故意保守）：
    某项要求如果只是 JD 里提到、而我的证据里从没出现过，就算未覆盖。
    它衡量的是"有没有材料可写"，不是"我会不会"——两者不能混为一谈。
"""

from __future__ import annotations

import re
from typing import Any, Iterable

# 技术词表：只用来从中文要求里认出"技能点"。
# 不追求全，追求稳——认错一个词会污染整个聚类。
TECH_TERMS: tuple[str, ...] = (
    "python", "java", "go", "golang", "c++", "javascript", "typescript", "sql",
    "pytorch", "tensorflow", "langchain", "langgraph", "llamaindex", "rag",
    "mcp", "fastmcp", "faiss", "milvus", "redis", "postgresql", "sqlite",
    "fastapi", "flask", "django", "docker", "kubernetes", "k8s", "git", "linux",
    "streamlit", "pydantic", "prompt", "agent", "llm", "transformer",
    "向量检索", "结构化输出", "提示词", "微调", "多模态", "评测", "召回",
    "数据库", "消息队列", "异步", "并发", "缓存", "分布式", "检索增强",
)

_ASCII_TOKEN = re.compile(r"[a-z][a-z0-9+#.]{1,}")
# JD 里常夹着来源链接，域名不是技能点
_DOMAIN_LIKE = re.compile(r"(?:[a-z0-9-]+\.)+(?:com|net|org|cn|io|dev|me)$")
_STOP = {
    "熟悉", "了解", "掌握", "要求", "优先", "经验", "能力", "负责", "参与",
    "岗位", "职责", "任职", "加分", "相关", "以上", "具备", "良好",
}


def requirement_tokens(texts: Iterable[str]) -> set[str]:
    """把岗位要求/偏好（或 JD 原文）摊成技能集合。"""
    joined = " ".join(str(item) for item in texts).casefold()
    tokens: set[str] = set()

    for term in TECH_TERMS:
        if term in joined:
            tokens.add(term)

    for token in _ASCII_TOKEN.findall(joined):
        token = token.strip(".")
        if len(token) >= 2 and token not in _STOP and not _DOMAIN_LIKE.match(token):
            # 只保留技术味够的：要么在词表里，要么带数字/符号（c++, gpt-4）
            if any(term in token or token in term for term in TECH_TERMS) or re.search(r"[0-9+#.]", token):
                tokens.add(token)
    return tokens


def job_tokens(job: dict[str, Any]) -> set[str]:
    """优先用分析时存下来的 requirements/preferences，老记录回退到解析 JD。"""
    stored = [*job.get("requirements", []), *job.get("preferences", [])]
    if stored:
        return requirement_tokens(stored) or requirement_tokens([job.get("jd", "")])
    return requirement_tokens([job.get("jd", "")])


def evidence_tokens(evidence_text: str) -> set[str]:
    """证据库里出现过的技能集合（同样是词表匹配，保证与 job_tokens 可比）。"""
    return requirement_tokens([evidence_text])


def coverage(tokens: set[str], evidence: set[str]) -> float:
    if not tokens:
        return 0.0
    return round(len(tokens & evidence) / len(tokens), 3)


def jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 0.0
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def cluster_jobs(jobs: list[dict[str, Any]], *, threshold: float = 0.34) -> list[dict[str, Any]]:
    """按技能集合相似度做单链聚类（结果确定：输入顺序固定 → 输出固定）。"""
    prepared = []
    for job in sorted(jobs, key=lambda item: int(item.get("id", 0))):
        tokens = job_tokens(job)
        if tokens:
            prepared.append((job, tokens))

    clusters: list[dict[str, Any]] = []
    for job, tokens in prepared:
        for cluster in clusters:
            if any(jaccard(tokens, other) >= threshold for other in cluster["token_sets"]):
                cluster["jobs"].append(job)
                cluster["token_sets"].append(tokens)
                break
        else:
            clusters.append({"jobs": [job], "token_sets": [tokens]})

    result: list[dict[str, Any]] = []
    for cluster in clusters:
        ranked = sorted(_top_tokens(cluster["token_sets"]), key=lambda item: (-item[1], item[0]))
        label = " · ".join(token for token, _count in ranked[:3]) or "未识别技能"
        result.append(
            {
                "label": label,
                "jobs": cluster["jobs"],
                "tokens": sorted(set().union(*cluster["token_sets"])),
            }
        )
    return result


def _top_tokens(token_sets: list[set[str]]) -> list[tuple[str, int]]:
    counts: dict[str, int] = {}
    for tokens in token_sets:
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1
    return list(counts.items())


def focus_report(jobs: list[dict[str, Any]], evidence_text: str, *, threshold: float = 0.34) -> dict[str, Any]:
    """输出"按簇的平均覆盖率排序" + "全库缺口"两件事。"""
    evidence = evidence_tokens(evidence_text)
    clusters = cluster_jobs(jobs, threshold=threshold)

    rows = []
    for cluster in clusters:
        scores = [coverage(job_tokens(job), evidence) for job in cluster["jobs"]]
        rows.append(
            {
                "label": cluster["label"],
                "size": len(cluster["jobs"]),
                "avg_coverage": round(sum(scores) / len(scores), 3) if scores else 0.0,
                "job_ids": [int(job["id"]) for job in cluster["jobs"]],
                "tokens": cluster["tokens"],
            }
        )
    rows.sort(key=lambda row: (-row["avg_coverage"], -row["size"], row["label"]))

    gaps: dict[str, int] = {}
    for job in jobs:
        for token in job_tokens(job) - evidence:
            gaps[token] = gaps.get(token, 0) + 1

    per_job = [
        {
            "id": int(job["id"]),
            "title": job.get("title", ""),
            "company": job.get("company", ""),
            "tokens": sorted(job_tokens(job)),
            "coverage": coverage(job_tokens(job), evidence),
        }
        for job in sorted(jobs, key=lambda item: int(item.get("id", 0)))
    ]

    return {
        "clusters": rows,
        "gaps": [
            {"token": token, "jobs": count}
            for token, count in sorted(gaps.items(), key=lambda item: (-item[1], item[0]))[:10]
        ],
        "jobs": per_job,
        "evidence_tokens": len(evidence),
    }
