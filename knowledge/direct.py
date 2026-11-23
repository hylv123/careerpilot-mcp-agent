"""直读模式：小语料不检索，整包进上下文。

为什么要这个模式：
    一个人的简历 + 项目 + 课程笔记，切完只有几十个片段、约六千 token，
    比模型上下文窗口小两个数量级。这时候检索是纯负收益：
    它只会**丢证据**（召回不到就永远进不了上下文），不会带来任何节省。

    检索真正有价值的前提是"库比上下文大一个数量级"，这个前提在
    单用户场景下根本不成立。所以把它做成按规模自适应，而不是默认开。

为什么直接读 FAISS 的 docstore：
    和检索模式用同一批片段。如果这里重新切一遍文档，两种模式看到的
    语料就不一样，A/B 对比也就没有意义了。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger

from knowledge.retrieve import _get_store

# 直读模式下最多塞多少片段。超过就说明这个模式用错了（该走检索），
# 这里的上限只是防止有人把阈值调到离谱之后把上下文撑爆。
MAX_DIRECT_CHUNKS = 800

_chunks_cache: list[dict[str, Any]] | None = None

# 语料目录与索引目录：只用于"索引是否落后于文档"的检查。
_KNOWLEDGE_DIR = Path(__file__).resolve().parent
_DOCS_DIR = _KNOWLEDGE_DIR / "docs"
_INDEX_DIR = _KNOWLEDGE_DIR / "index"


def warn_if_index_stale() -> None:
    """文档比索引新时告警。

    为什么需要这条：直读模式读的是 **FAISS docstore**，不是实时文件。
    改完 knowledge/docs 下的简历/项目描述后如果不重建索引，
    检索和直读拿到的还是旧文本——表现为"事实明明改了，产出里还是旧数字"。
    这种静默不一致比报错更难查，所以每次加载语料时对比一次 mtime。
    """
    try:
        doc_files = list(_DOCS_DIR.rglob("*.md"))
        index_files = [_INDEX_DIR / "index.faiss", _INDEX_DIR / "index.pkl"]
        if not doc_files or not all(path.exists() for path in index_files):
            return
        newest_doc = max(path.stat().st_mtime for path in doc_files)
        oldest_index = min(path.stat().st_mtime for path in index_files)
        if newest_doc > oldest_index:
            logger.warning(
                "[direct] knowledge/docs 有文件比索引新，读取的仍是旧片段；"
                "请运行 python knowledge/build_index.py 重建索引"
            )
    except Exception as exc:  # 检查失败不该影响检索
        logger.debug("[direct] 索引新鲜度检查跳过：{}", exc)


def all_chunks() -> list[dict[str, Any]]:
    """取出索引里的全部片段（带来源），进程内缓存。"""
    global _chunks_cache
    if _chunks_cache is None:
        warn_if_index_stale()
        store = _get_store()
        docs = list(store.docstore._dict.values())
        _chunks_cache = [
            {
                "content": doc.page_content,
                "source": str(doc.metadata.get("source", "")),
                "source_path": str(doc.metadata.get("source", "")),
                # 直读没有相似度排序，score 统一记 0.0，避免下游误读成"最相关"。
                "score": 0.0,
            }
            for doc in docs
        ]
        logger.info("[direct] 语料加载完成，共 {} 个片段", len(_chunks_cache))
    return _chunks_cache


def reset_cache() -> None:
    """清空直读缓存（重建索引或修改资料后调用）。"""
    global _chunks_cache
    _chunks_cache = None


def corpus_size() -> int:
    """当前索引的片段总数。"""
    return len(all_chunks())


def direct_search(k: int | None = None) -> list[dict[str, Any]]:
    """返回全部片段（或前 k 个），不做相似度筛选。

    Args:
        k: 可选上限；不传表示全量。

    Returns:
        与 retrieve() 同构的片段列表。
    """
    chunks = all_chunks()
    if len(chunks) > MAX_DIRECT_CHUNKS:
        logger.warning(
            "[direct] 片段数 {} 超过直读上限 {}，只取前 {} 条；"
            "这个规模应该走检索模式",
            len(chunks),
            MAX_DIRECT_CHUNKS,
            MAX_DIRECT_CHUNKS,
        )
    selected = chunks[:k] if k else chunks[:MAX_DIRECT_CHUNKS]
    return list(selected)
