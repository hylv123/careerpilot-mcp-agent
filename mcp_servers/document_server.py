import os
import re
from pathlib import Path

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("DocumentServer")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT = Path(
    os.getenv("CAREERPILOT_OUTPUT_DIR", str(PROJECT_ROOT / "output"))
).resolve()
OUT.mkdir(parents=True, exist_ok=True)


def safe_filename(text: str) -> str:
    """把字符串转为安全的文件名，防止非法字符"""
    return re.sub(r'[<>:"/\\|?!*]', '_', text.replace(' ', '_'))


@mcp.tool()
def generate_resume_md(title: str, bullets: list[str]) -> str:
    """生成定制简历 Markdown 并保存， bullets：简历要点列表"""
    content = f"#{title}\n\n" + "\n".join(f"- {b}" for b in bullets)
    safe_title = safe_filename(title)
    path = OUT / f"resume_{safe_title}.md"
    with path.open("w", encoding="utf-8") as f:
        f.write(content)
    return f"已生成：{path}"

@mcp.tool()
def generate_interview_prep(job_title: str, questions: list[str]) -> str:
    """生成面试准备文档，questions:面试题列表"""
    content = f"#{job_title} 面试准备\n\n" + "\n".join(f"## Q{i+1}.{q}\n\n" for i, q in enumerate(questions))
    safe_title = safe_filename(job_title)
    path = OUT / f"interview_{safe_title}.md"
    with path.open("w", encoding="utf-8") as f:
        f.write(content)
    return f"已生成: {path}"

if __name__ == "__main__":
    #当前版本，stdio子进程由客户端拉起，待扩展http
    mcp.run(transport="stdio")
