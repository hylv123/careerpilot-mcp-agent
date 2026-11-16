"""BM25 稀疏检索 + RRF 融合。

为什么要有这一路：
    向量检索擅长"意思相近"，但对**专有名词**不敏感——JD 里写 MCP、LangGraph、
    XAUTOCLAIM，它可能召回一段语义泛化、但一个字都没提的片段。
    BM25 正好相反：只认词面，稀有名一命中就加分。
    两者互补，所以做混合检索。

为什么融合用 RRF 而不是加权分数：
    向量返回的是 L2 距离（越小越好），BM25 返回的是相关度分数（越大越好），
    量纲完全不同，直接加权必然出错。RRF 只吃名次，天然规避这个问题。

为什么中文要先分词：
    BM25 按词计数，不分词等于按整句计数，等于废掉。
    jieba 默认会把 LangGraph 切成 Lang + Graph，所以要先把技术词加进词典。
"""

from __future__ import annotations

import re
from typing import Any, Iterable

import jieba
from loguru import logger
from rank_bm25 import BM25Okapi

from knowledge.retrieve import _get_store, retrieve

# 技术词不能被 jieba 切碎：切成两半之后，BM25 反而会命中一堆无关片段。
_TECH_TERMS = (
    "LangGraph", "LangChain", "LangChain4j", "LlamaIndex", "MCP", "FastMCP",
    "FAISS", "Pydantic", "FastAPI", "Redis", "Stream", "XAUTOCLAIM", "PEL",
    "BM25", "RRF", "RAG", "Agent", "ToolNode", "Docker", "Compose", "SQLite",
    "Embedding", "Recall", "Precision", "MRR", "DeepSeek", "DashScope",
)
for _term in _TECH_TERMS:
    jieba.add_word(_term)

# 标点与空白不参与计分，先替成空格再切词。
_PUNCT = re.compile(
    r"[\s\u3000，。；：、！？（）【】《》“”‘’·—…,.;:!?()\[\]{}<>\"/\\|+=*&^%$#@~`]+"
)


def tokenize(text: str) -> list[str]:
    """中文分词并做小写归一。"""
    cleaned = _PUNCT.sub(" ", str(text))
    return [
        token
        for token in (piece.strip().lower() for piece in jieba.cut(cleaned))
        if token
    ]


def rrf_fuse(
    rankings: Iterable[Iterable[str]],
    *,
    k: int = 4,
    c: int = 60,
) -> list[str]:
    """Reciprocal Rank Fusion：只按名次融合多路召回。

    Args:
        rankings: 多路召回，每路是按相关度降序排列的"片段正文"列表。
        k: 返回条数。
        c: 平滑常数，经验值 60。

    Returns:
        融合后的片段正文列表（按融合分数降序）。

    为什么用正文当 key：
        两路召回来自不同的打分体系，唯一可靠的公共标识就是片段正文本身；
        编号只对当前索引有效，重启/重建就会变。
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, key in enumerate(ranking, start=1):
            scores[key] = scores.get(key, 0.0) + 1.0 / (c + rank)
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [content for content, _ in ordered[:k]]


_corpus_cache: list[tuple[str, str]] | None = None
_bm25_cache: BM25Okapi | None = None


def _load_corpus() -> list[tuple[str, str]]:
    """从当前 FAISS 索引里取出 (片段正文, 来源) 列表。

    为什么复用 FAISS 的 docstore 而不是重新切一遍文档：
        重新切分依赖相同的文件、相同的切块参数；只要有一处不同，
        BM25 的片段就和向量检索的片段对不上，融合会变成两套语料混在一起。
        直接读索引里的片段，能保证两路检索面对的是同一批文本。
    """
    global _corpus_cache
    if _corpus_cache is None:
        store = _get_store()
        docs = list(store.docstore._dict.values())
        _corpus_cache = [
            (doc.page_content, str(doc.metadata.get("source", "")))
            for doc in docs
        ]
        logger.info("[BM25] 语料加载完成，共 {} 个片段", len(_corpus_cache))
    return _corpus_cache


def _get_bm25(corpus: list[tuple[str, str]]) -> BM25Okapi:
    global _bm25_cache
    if _bm25_cache is None:
        _bm25_cache = BM25Okapi([tokenize(text) for text, _ in corpus])
    return _bm25_cache


def bm25_search(
    query: str,
    k: int = 4,
    *,
    corpus: list[tuple[str, str]] | None = None,
) -> list[dict[str, Any]]:
    """BM25 检索，返回按相关度降序的片段。

    Args:
        query: 查询文本（会用同一套分词规则处理）。
        k: 返回条数。
        corpus: 可选，测试时注入小语料，避免依赖真实索引。

    Returns:
        每项含 content / source / source_path / score（分数越大越相关）。
    """
    items = corpus if corpus is not None else _load_corpus()
    index = _get_bm25(items) if corpus is None else BM25Okapi(
        [tokenize(text) for text, _ in items]
    )

    scores = index.get_scores(tokenize(query))
    order = sorted(range(len(items)), key=lambda i: (-scores[i], i))[:k]

    return [
        {
            "content": items[i][0],
            "source": items[i][1],
            "source_path": items[i][1],
            "score": float(scores[i]),
        }
        for i in order
        if scores[i] > 0
    ]


def hybrid_search(
    query: str,
    k: int = 4,
    *,
    pool: int = 10,
    c: int = 60,
) -> list[dict[str, Any]]:
    """向量 + BM25 两路召回，用 RRF 融合后取前 k 条。

    Args:
        query: 查询文本。
        k: 最终返回条数。
        pool: 每路各召回多少条再融合（比 k 大，才有多余候选可排）。
        c: RRF 平滑常数。

    Returns:
        与 retrieve() 同构的片段列表。
    """
    vector_hits = retrieve(query, k=pool)
    bm25_hits = bm25_search(query, k=pool)

    by_content: dict[str, dict[str, Any]] = {}
    for hit in bm25_hits:
        by_content.setdefault(hit["content"], hit)
    for hit in vector_hits:
        # 向量结果的元数据更完整（含原始 source_path），优先保留。
        by_content[hit["content"]] = hit

    fused = rrf_fuse(
        ([hit["content"] for hit in vector_hits], [hit["content"] for hit in bm25_hits]),
        k=k,
        c=c,
    )
    return [by_content[content] for content in fused if content in by_content]
