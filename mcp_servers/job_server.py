import json
import os
import sqlite3
from pathlib import Path
from contextlib import contextmanager
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("JobServer")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB = Path(
    os.getenv("CAREERPILOT_DB_PATH", str(PROJECT_ROOT / "data" / "jobs.db"))
).resolve()
ALLOWED_STATUSES = {"未投", "已投", "面试", "offer", "拒"}

def _conn():
    DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS jobs(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT, company TEXT, jd TEXT,
        status TEXT DEFAULT '未投', created_at TEXT DEFAULT(datetime('now')))
        """
    )
    return conn


@contextmanager
def _session():
    """一次工具调用 = 一个连接，用完一定关闭。

    为什么不能写成 `with _conn() as conn:`：
        sqlite3.Connection 的上下文管理器只做事务（正常 commit / 异常 rollback），
        **不会关闭连接**。每次工具调用泄漏一个连接，worker 长跑后句柄耗尽，
        最终 `too many open files`（Linux）或句柄上限（Windows）。
        save_job 原本用了 closing() 是对的，其余四个工具漏了。

    为什么还要套一层 `with conn:`：
        保留原来的事务语义：异常时回滚，不留半条记录。
    """
    conn = _conn()
    try:
        with conn:
            yield conn
    finally:
        conn.close()

@mcp.tool()
def save_job(title : str, company: str, jd: str) -> str:
    with _session() as conn:
        conn.execute(
            "INSERT INTO jobs(title, company, jd) VALUES(?,?,?)",
            (title, company, jd,),
        )
    return f"已保存岗位：{title} from {company}"

@mcp.tool()
def update_status(job_id: int, status: str) -> str:
    """更新投递状态,status: 未投/已投/面试/offer/拒"""
    if status not in ALLOWED_STATUSES:
        return f"无效状态：{status}，可选值为：{', '.join(sorted(ALLOWED_STATUSES))}"

    with _session() as conn:
        cursor = conn.execute(
            "UPDATE jobs SET status=? WHERE id=?",
            (status, job_id),
        )
        # rowcount 必须在连接关闭前读：关闭后再访问 Cursor 会抛
        # ProgrammingError: Cannot operate on a closed database
        affected = cursor.rowcount
    if affected == 0:
        return f"未找到岗位：{job_id}"
    return f"岗位 {job_id} 状态更新为 {status}"

@mcp.tool()
def list_jobs(status: str = "") -> str:
    """列出岗位，可按照状态筛选"""
    with _session() as conn:
        rows = conn.execute(
            "SELECT id, title, company, status FROM jobs WHERE ?='' OR status=?",
            (status, status),
        ).fetchall()
    return json.dumps(rows, ensure_ascii=False)

@mcp.tool()
def get_job(id: int):
    with _session() as conn:
        rows = conn.execute(
            "SELECT * FROM jobs WHERE id = ?",
            (id,),
        ).fetchall()
    return json.dumps(rows, ensure_ascii=False)

@mcp.tool()
def list_jobs_from_company(company_title: str):
    with _session() as conn:
        res = conn.execute(
            "SELECT * FROM jobs WHERE company = ?",
            (company_title,),
        ).fetchall()
    return json.dumps(res, ensure_ascii=False)



if __name__ == "__main__":
    mcp.run(transport = "stdio")

