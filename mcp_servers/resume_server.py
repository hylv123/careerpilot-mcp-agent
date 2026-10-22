import json
from mcp.server.fastmcp import FastMCP

RESUME_DATA = {
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


mcp = FastMCP("ResumeServer")

@mcp.tool()
def get_resume() -> str:
    """获取脱敏的演示简历信息。"""
    return json.dumps(RESUME_DATA, ensure_ascii=False)

@mcp.tool()
def search_projects(keyword: str) -> str:
    """按关键词搜索项目经历。keyword: 技术关键词，如 'RAG'"""
    hits = [p for p in RESUME_DATA["projects"] if keyword.lower() in p["name"].lower() or keyword.lower() in p["desc"].lower()]
    return json.dumps(hits, ensure_ascii=False) if hits else "未找到相关项目"

if __name__ == "__main__":
    mcp.run(transport="stdio")
