import json
from pathlib import Path

from mcp.server.fastmcp import FastMCP

# 真实简历优先从本地私有文件读取（已被 .gitignore 忽略）；
# 文件不存在时回退到脱敏演示数据，保证仓库克隆下来仍可直接运行。
LOCAL_RESUME_PATH = Path(__file__).resolve().parent.parent / "data" / "resume.local.json"

DEMO_RESUME_DATA = {
    "profile_type": "sanitized_demo",
    "skills": ["Python", "LangGraph", "LangChain", "RAG", "MCP"],
    "projects": [
        {
            "name": "CareerPilot MCP Agent",
            "desc": "基于 MCP、LangGraph 和 FAISS 的实习求职助手",
        },
    ],
    "education": "脱敏演示字段，请在本地替换为真实教育经历",
}


def _load_resume_data() -> dict:
    if LOCAL_RESUME_PATH.exists():
        try:
            return json.loads(LOCAL_RESUME_PATH.read_text(encoding="utf-8"))
        except Exception:
            # 本地文件格式错误时不阻塞 Server 启动，退回演示数据。
            pass
    return DEMO_RESUME_DATA


RESUME_DATA = _load_resume_data()


mcp = FastMCP("ResumeServer")

@mcp.tool()
def get_resume() -> str:
    """获取候选人简历信息（本地真实资料优先，缺失时回退脱敏演示数据）。"""
    return json.dumps(RESUME_DATA, ensure_ascii=False)

@mcp.tool()
def search_projects(keyword: str) -> str:
    """按关键词搜索项目经历。keyword: 技术关键词，如 'RAG'"""
    hits = [p for p in RESUME_DATA["projects"] if keyword.lower() in p["name"].lower() or keyword.lower() in p["desc"].lower()]
    return json.dumps(hits, ensure_ascii=False) if hits else "未找到相关项目"

if __name__ == "__main__":
    mcp.run(transport="stdio")
