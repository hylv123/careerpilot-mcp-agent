import sys
from pathlib import Path

# 把项目根目录（careerpilot）加入 Python 路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.dashscope import DashScopeEmbeddings
from agent.llm import get_ds_chat_model
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from loguru import logger
import os
load_dotenv()

INDEX_DIR = Path(__file__).resolve().parent / "index"

_embeddings = None
_store = None


def _get_embeddings():
    global _embeddings
    if _embeddings is None:
        api_key = (
            os.getenv("DASHSCOPE_API_KEY")
            or os.getenv("aliQwen-api")
        )
        if not api_key:
            raise RuntimeError(
                "缺少 Embedding API Key，请在 .env 中填写 DASHSCOPE_API_KEY"
            )
        _embeddings = DashScopeEmbeddings(
            model=os.getenv("EMBEDDING_MODEL", "text-embedding-v3"),
            dashscope_api_key=api_key,
        )
    return _embeddings


def _get_store():
    global _store
    if _store is None:
        if not INDEX_DIR.exists():
            raise RuntimeError(
                "知识库索引不存在，请先运行 python -m knowledge.build_index"
            )
        _store = FAISS.load_local(
            str(INDEX_DIR),
            _get_embeddings(),
            allow_dangerous_deserialization=True,
        )
    return _store


def retrieve(query: str, k: int = 4) -> list[dict]:
    docs = _get_store().similarity_search_with_score(query, k=k)
    return [
        {
            "content": d.page_content, 
            "source": Path(str(d.metadata.get("source", ""))).name, 
            "source_path": str(d.metadata.get("source", "")), 
            "score": float(s)
        }
        for d, s in docs
    ]


def rewrite_query(jd_keywords: list[str], llm) -> str:
    """查询改写JD术语变为简历检索词"""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是一名招聘分析助手，负责把岗位描述中的专业术语改写成简历检索词。只输出检索词"),
            ("human", "关键词：{keywords}")
        ]
    )
    return (prompt | llm).invoke({"keywords": ",".join(jd_keywords[:5])}).content


# 「这条算不算高相关」的距离上限。**唯一来源**：scripts/inspect_scores.py
# 直接引用它，避免脚本里另写一个数字、改了生产却忘了改脚本。
# 注意它是 L2 距离（越小越相似），且与知识库/Embedding 模型强相关，
# 换数据后必须重新校准。
DEFAULT_MAX_DISTANCE = 1.02


def retrieve_agentic(
    jd_keywords: list[str],
    k: int = 4,
    min_hits: int = 2,
    max_distance: float = DEFAULT_MAX_DISTANCE,
) -> list[dict]:
    """
    先检索原字，当高相关结果不足时改写查询并补充检索。
    判断检索质量，因为索引中有足够文档时通常总会返回k条。
    """
    hits = retrieve(" ".join(jd_keywords[:5]), k=k)
    relevant_hits = [
        hit for hit in hits
        if hit.get("score", float("inf")) <= max_distance
    ]
    if len(relevant_hits) >= min_hits:
        return hits

    try:
        rewritten_query = rewrite_query(jd_keywords, get_ds_chat_model())
    except Exception as exc:
        logger.warning("[RAG] 查询改写失败，保留第一轮结果：{}", exc)
        return hits

    logger.info("[RAG] 高相关结果不足，执行补充检索，query={}", rewritten_query)
    second_hits = retrieve(rewritten_query, k=k)
    seen = {h["content"] for h in hits}
    merged = hits + [
        hit for hit in second_hits
        if hit["content"] not in seen
    ]
    return sorted(merged, key=lambda item: item.get("score", float("inf")))[:k]
