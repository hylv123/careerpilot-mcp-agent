"""岗位发现：抓取 → 过滤 → 打分 → 入候选池 → 一键导入岗位库。

打分是确定性的：用候选岗位的技能词与"我的证据库"比覆盖率，再叠加关键词命中。
不调用模型——"这条岗位值不值得看"必须先能解释，再谈智能。
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from typing import Any, Iterable

from loguru import logger

from services import compare, jobs
from services.job_sources import SOURCES, SourceError, fetch_source

LEAD_STATUSES = ("new", "saved", "ignored")


def _migrate(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS job_leads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            external_id TEXT NOT NULL DEFAULT '',
            url TEXT NOT NULL,
            title TEXT NOT NULL,
            company TEXT NOT NULL DEFAULT '',
            location TEXT NOT NULL DEFAULT '',
            description TEXT NOT NULL DEFAULT '',
            posted_at TEXT NOT NULL DEFAULT '',
            tokens TEXT NOT NULL DEFAULT '[]',
            score REAL NOT NULL DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'new',
            fetched_at TEXT NOT NULL DEFAULT (datetime('now')),
            UNIQUE(source, url)
        )
        """
    )


def _connect() -> sqlite3.Connection:
    path = jobs.database_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(path))
    connection.row_factory = sqlite3.Row
    _migrate(connection)
    return connection


def _row_to_lead(row: sqlite3.Row) -> dict[str, Any]:
    lead = dict(row)
    try:
        lead["tokens"] = json.loads(lead.get("tokens") or "[]")
    except json.JSONDecodeError:
        lead["tokens"] = []
    return lead


def list_leads(*, status: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
    connection = _connect()
    try:
        if status:
            rows = connection.execute(
                "SELECT * FROM job_leads WHERE status = ? ORDER BY score DESC, id DESC LIMIT ?",
                (status, limit),
            ).fetchall()
        else:
            rows = connection.execute(
                "SELECT * FROM job_leads ORDER BY score DESC, id DESC LIMIT ?", (limit,)
            ).fetchall()
    finally:
        connection.close()
    return [_row_to_lead(row) for row in rows]


def _query_tokens(query: str) -> set[str]:
    return compare.requirement_tokens([query])


def _matches(lead: dict[str, Any], query: str, query_tokens: set[str]) -> bool:
    if not query.strip():
        return True
    haystack = f"{lead.get('title', '')} {lead.get('company', '')} {lead.get('description', '')[:600]}".casefold()
    if query.casefold().strip() in haystack:
        return True
    return bool(query_tokens & set(lead.get("tokens", [])))


def _score(lead: dict[str, Any], evidence: set[str], query_tokens: set[str]) -> float:
    tokens = set(lead.get("tokens", []))
    coverage = compare.coverage(tokens, evidence)
    query_hit = 1.0 if (query_tokens & tokens) else 0.0
    return round(0.75 * coverage + 0.25 * query_hit, 3)


def search(
    query: str,
    *,
    sources: Iterable[str] | None = None,
    boards: dict[str, str] | None = None,
    limit_per_source: int = 30,
) -> dict[str, Any]:
    """抓取岗位并写入候选池，返回本次结果。

    Args:
        query: 关键词（中文/英文都行，用于过滤与命中打分）。
        sources: 要抓的来源 id；默认全部。
        boards: 需要 board slug 的来源参数，例如 {"greenhouse": "anthropic"}。
        limit_per_source: 每个来源最多取多少条。
    """
    from knowledge.direct import all_chunks

    evidence = compare.evidence_tokens(
        "\n".join(str(chunk.get("content", "")) for chunk in all_chunks())
    )
    query_tokens = _query_tokens(query)
    chosen = list(sources) if sources else list(SOURCES)

    fetched = 0
    kept: list[dict[str, Any]] = []
    errors: list[str] = []

    for source_id in chosen:
        board = (boards or {}).get(source_id)
        try:
            rows = fetch_source(source_id, board=board, limit=limit_per_source)
        except SourceError as exc:
            errors.append(str(exc))
            logger.warning("[discovery] 来源 {} 失败：{}", source_id, exc)
            continue

        fetched += len(rows)
        for raw in rows:
            tokens = sorted(compare.requirement_tokens([raw.get("title", ""), raw.get("description", "")[:800]]))
            lead = {
                **raw,
                "company": raw.get("company") or (board or raw.get("source_name", "")),
                "tokens": tokens,
            }
            if not _matches(lead, query, query_tokens):
                continue
            lead["score"] = _score(lead, evidence, query_tokens)
            kept.append(lead)

    saved = _upsert(kept)
    return {
        "fetched": fetched,
        "kept": len(kept),
        "saved": saved,
        "errors": errors,
        "leads": sorted(kept, key=lambda item: item["score"], reverse=True)[:50],
    }


def _upsert(leads: list[dict[str, Any]]) -> int:
    if not leads:
        return 0
    connection = _connect()
    try:
        for lead in leads:
            connection.execute(
                """
                INSERT INTO job_leads
                    (source, external_id, url, title, company, location, description, posted_at, tokens, score, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'new')
                ON CONFLICT(source, url) DO UPDATE SET
                    title = excluded.title,
                    company = excluded.company,
                    location = excluded.location,
                    description = excluded.description,
                    posted_at = excluded.posted_at,
                    tokens = excluded.tokens,
                    score = excluded.score,
                    fetched_at = datetime('now')
                """,
                (
                    lead.get("source", ""),
                    lead.get("external_id", ""),
                    lead.get("url", ""),
                    lead.get("title", ""),
                    lead.get("company", ""),
                    lead.get("location", ""),
                    lead.get("description", ""),
                    lead.get("posted_at", ""),
                    json.dumps(lead.get("tokens", []), ensure_ascii=False),
                    float(lead.get("score", 0)),
                ),
            )
        connection.commit()
    finally:
        connection.close()
    return len(leads)


def save_lead_to_jobs(lead_id: int) -> dict[str, Any] | None:
    """把候选岗位导入岗位库（不自动跑分析，避免顺手烧掉模型额度）。"""
    connection = _connect()
    try:
        row = connection.execute("SELECT * FROM job_leads WHERE id = ?", (lead_id,)).fetchone()
        if row is None:
            return None
        lead = _row_to_lead(row)
        connection.execute("UPDATE job_leads SET status = 'saved' WHERE id = ?", (lead_id,))
        connection.commit()
    finally:
        connection.close()

    jd = f"{lead['title']}\n\n{lead['description']}".strip()
    return jobs.create_job(
        title=lead["title"] or "未命名岗位",
        company=lead["company"] or lead.get("source_name", ""),
        jd=jd[:10000],
        source_url=lead["url"],
        notes=f"来源：{lead.get('source', '')}｜发现时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}",
    )


def ignore_lead(lead_id: int) -> bool:
    connection = _connect()
    try:
        cursor = connection.execute("UPDATE job_leads SET status = 'ignored' WHERE id = ?", (lead_id,))
        connection.commit()
        return cursor.rowcount > 0
    finally:
        connection.close()
