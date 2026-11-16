"""检索模式路由：按语料规模自动在「直读 / 向量 / 混合」之间选择。

为什么需要这一层：
    同一条链路要服务两种完全不同的规模——
        · 单用户：几十个片段、约六千 token，全量塞进上下文更准（检索只会丢证据）；
        · 多用户/大库：几千上万个片段，才轮到检索出场。
    把选择权写成环境变量 + 规模阈值，而不是写死在代码里，
    这样"为什么用 RAG"就变成一个可配置、可对照验证的工程决策。

环境变量：
    CAREERPILOT_RETRIEVAL_MODE         auto（默认）/ direct / vector / hybrid
    CAREERPILOT_DIRECT_READ_THRESHOLD  auto 模式下走直读的最大片段数（默认 200）
"""

from __future__ import annotations

import json
import os
from typing import Any

from loguru import logger

from knowledge.retrieve import retrieve, retrieve_agentic

MODE_ENV = "CAREERPILOT_RETRIEVAL_MODE"
THRESHOLD_ENV = "CAREERPILOT_DIRECT_READ_THRESHOLD"
VALID_MODES = ("auto", "direct", "vector", "hybrid")
DEFAULT_MODE = "auto"
DEFAULT_THRESHOLD = 200


def _threshold() -> int:
    raw = os.getenv(THRESHOLD_ENV, "").strip()
    if not raw:
        return DEFAULT_THRESHOLD
    try:
        return int(raw)
    except ValueError:
        logger.warning("[retrieval] 阈值 {} 不是整数，回退 {}", raw, DEFAULT_THRESHOLD)
        return DEFAULT_THRESHOLD


def configured_mode() -> str:
    """读取配置的模式；非法值回退 auto。"""
    mode = (os.getenv(MODE_ENV) or DEFAULT_MODE).strip().lower()
    if mode not in VALID_MODES:
        logger.warning("[retrieval] 未知模式 {}，回退 auto", mode)
        return DEFAULT_MODE
    return mode


def resolve_mode(size: int | None = None) -> str:
    """算出实际生效的模式（direct / vector / hybrid）。

    Args:
        size: 片段总数；不传则读取当前索引（会触发一次索引加载）。

    Returns:
        生效模式。
    """
    mode = configured_mode()
    if mode != "auto":
        return mode

    if size is None:
        from knowledge.direct import corpus_size

        size = corpus_size()

    return "direct" if size <= _threshold() else "vector"


def search_for_match(keywords: list[str], k: int = 4) -> list[dict[str, Any]]:
    """match 节点用的检索入口。

    为什么直读模式下不跑查询改写：
        改写是为了"第一轮没召回到"时补救；直读模式下全部片段都在上下文里，
        没有"没召回到"这回事，改写纯属多烧一次模型调用。
    """
    mode = resolve_mode()
    if mode == "direct":
        from knowledge.direct import direct_search

        logger.info("[retrieval] 模式=direct（语料小，全量进上下文）")
        return direct_search()

    if mode == "hybrid":
        from knowledge.bm25 import hybrid_search

        logger.info("[retrieval] 模式=hybrid（向量 + BM25 融合）")
        return hybrid_search(" ".join(keywords[:5]), k=k, pool=max(k * 3, 10))

    logger.info("[retrieval] 模式=vector（语料较大，走向量检索 + 按需改写）")
    return retrieve_agentic(keywords, k=k)


def search_for_tool(query: str, k: int = 4) -> list[dict[str, Any]]:
    """agent 工具用的检索入口（模型自己组织 query）。"""
    mode = resolve_mode()
    if mode == "direct":
        from knowledge.direct import direct_search

        return direct_search()

    if mode == "hybrid":
        from knowledge.bm25 import hybrid_search

        return hybrid_search(query, k=k, pool=max(k * 3, 10))

    return retrieve(query, k=k)


def search_evidence(query: str, k: int = 4) -> str:
    """把检索结果包装成模型可读的 JSON 字符串（工具返回值）。

    不做查询改写：工具调用本身就是模型组织的 query，再改写一次
    既多一次模型调用，也可能把 query 带偏。
    """
    hits = search_for_tool(query, k=k)
    logger.info("[RAG] 工具检索 query={}，命中 {} 条", query, len(hits))
    return json.dumps(hits, ensure_ascii=False)
