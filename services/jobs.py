"""Persistent job-application workbench data.

The Agent may create a job record, but users manage it through this service.
Keeping SQLite access here prevents the HTTP API and MCP server from growing
different schemas and update rules.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "jobs.db"
ALLOWED_STATUSES = ("未投", "已投", "笔试", "面试", "offer", "拒绝", "已归档")
JD_SUMMARY_LIMIT = 140

# JD 抓下来常带"岗位名称：/公司：/薪资："这类表头行，它们和岗位内容无关，
# 单独成行时先剔掉，概括描述才不会被表头占满。
_JD_HEADER = re.compile(
    r"^\s*(岗位名称|职位名称|岗位|职位|公司名称|公司|招聘方|薪资|薪资范围|"
    r"工作地点|地点|发布时间|更新时间|来源)\s*[:：]"
)
# 章节标题同样是噪声：它在段落里起分隔作用，但单独抽出来当"概括"读不通。
_JD_SECTION = re.compile(
    r"^(岗位职责|工作职责|工作内容|职位描述|岗位描述|任职要求|岗位要求|职位要求|加分项|优先条件)$"
)


def summarize_jd(jd: str, limit: int = JD_SUMMARY_LIMIT) -> str:
    """把 JD 压成一句能扫读的概括，供岗位列表直接展示。

    这里刻意不用大模型：岗位列表要秒开、要零成本、要可复现。
    做法是"去表头 → 合并空白 → 按标点截断"，宁可朴素也不要幻觉。
    """
    lines = [re.sub(r"\s+", " ", line).strip() for line in str(jd or "").replace("\r", "").split("\n")]
    lines = [line for line in lines if line]
    content = [line for line in lines if not _JD_HEADER.match(line) and not _JD_SECTION.match(line)]
    text = " ".join(content or lines)
    if len(text) <= limit:
        return text
    window = text[:limit]
    for sep in ("。", "；", ";", "！", "!", "，", ","):
        cut = window.rfind(sep)
        if cut >= limit // 2:
            return window[: cut + 1].rstrip() + "…"
    return window.rstrip() + "…"


def database_path() -> Path:
    return Path(os.getenv("CAREERPILOT_DB_PATH", str(DEFAULT_DB_PATH))).resolve()


@contextmanager
def session() -> Iterator[sqlite3.Connection]:
    path = database_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(path))
    connection.row_factory = sqlite3.Row
    try:
        _migrate(connection)
        with connection:
            yield connection
    finally:
        connection.close()


def _migrate(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            company TEXT NOT NULL,
            jd TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT '未投',
            source_url TEXT NOT NULL DEFAULT '',
            deadline TEXT NOT NULL DEFAULT '',
            notes TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
        """
    )
    existing = {row[1] for row in connection.execute("PRAGMA table_info(jobs)")}
    for name, definition in (
        ("source_url", "TEXT NOT NULL DEFAULT ''"),
        ("deadline", "TEXT NOT NULL DEFAULT ''"),
        ("notes", "TEXT NOT NULL DEFAULT ''"),
        ("updated_at", "TEXT NOT NULL DEFAULT ''"),
        # 分析时的岗位要求/偏好存 JSON：多岗位聚类靠它，而不是靠再解析一遍 JD
        ("requirements", "TEXT NOT NULL DEFAULT ''"),
        ("preferences", "TEXT NOT NULL DEFAULT ''"),
    ):
        if name not in existing:
            connection.execute(f"ALTER TABLE jobs ADD COLUMN {name} {definition}")


def _record(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    record = dict(row)
    # 概括描述按 jd 现算，不落库：JD 一旦被 PATCH 修改，列表里也不会留下旧摘要。
    record["jd_summary"] = summarize_jd(record.get("jd", ""))
    for field in ("requirements", "preferences"):
        raw = record.get(field) or "[]"
        try:
            parsed = json.loads(raw)
            record[field] = parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            record[field] = []
    return record


def create_job(
    *,
    title: str,
    company: str,
    jd: str,
    source_url: str = "",
    deadline: str = "",
    notes: str = "",
    requirements: list[str] | None = None,
    preferences: list[str] | None = None,
) -> dict[str, Any]:
    with session() as connection:
        cursor = connection.execute(
            """INSERT INTO jobs(title, company, jd, source_url, deadline, notes, requirements, preferences)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                title,
                company,
                jd,
                source_url,
                deadline,
                notes,
                json.dumps(list(requirements or []), ensure_ascii=False),
                json.dumps(list(preferences or []), ensure_ascii=False),
            ),
        )
        row = connection.execute("SELECT * FROM jobs WHERE id = ?", (cursor.lastrowid,)).fetchone()
    return _record(row) or {}


def list_jobs(*, status: str | None = None) -> list[dict[str, Any]]:
    with session() as connection:
        if status:
            rows = connection.execute("SELECT * FROM jobs WHERE status = ? ORDER BY updated_at DESC, id DESC", (status,)).fetchall()
        else:
            rows = connection.execute("SELECT * FROM jobs ORDER BY updated_at DESC, id DESC").fetchall()
    return [record for row in rows if (record := _record(row)) is not None]


def get_job(job_id: int) -> dict[str, Any] | None:
    with session() as connection:
        row = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    return _record(row)


def update_job(job_id: int, **changes: str) -> dict[str, Any] | None:
    allowed = {"title", "company", "jd", "status", "source_url", "deadline", "notes"}
    updates = {name: value for name, value in changes.items() if name in allowed and value is not None}
    if "status" in updates and updates["status"] not in ALLOWED_STATUSES:
        raise ValueError(f"无效状态：{updates['status']}")
    if not updates:
        return get_job(job_id)
    assignments = ", ".join(f"{name} = ?" for name in updates)
    values = [*updates.values(), job_id]
    with session() as connection:
        cursor = connection.execute(
            f"UPDATE jobs SET {assignments}, updated_at = datetime('now') WHERE id = ?", values
        )
        if cursor.rowcount == 0:
            return None
        row = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    return _record(row)


def delete_job(job_id: int) -> bool:
    """物理删除一条岗位记录。

    为什么不做软删除：投递库里的记录是用户自己的操作痕迹，删错了可以重录；
    而"已归档"状态已经承担了"留着但不再跟进"的语义。再叠一层软删除只会让
    列表查询到处漏 `AND deleted_at IS NULL`，反而更容易出错。
    """
    with session() as connection:
        cursor = connection.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
    return cursor.rowcount > 0
