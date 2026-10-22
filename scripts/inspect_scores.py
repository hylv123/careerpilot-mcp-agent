"""打印知识库检索的距离分布，用于校准 max_distance 阈值。

为什么要写这个脚本：

    FAISS 的 similarity_search_with_score 返回的是「距离」而不是「相似度」，
    数值越小代表越相似。knowledge/retrieve.py 的 retrieve_agentic 里用
    max_distance=0.95 判断"这条算不算高相关"，但 0.95 是当初拍的经验值。

    只有看到真实查询下分数的分布，才能判断这个阈值到底是松了还是紧了。
    这个分布是校准阈值的唯一依据——结论必须是跑出来的，而不是凭感觉定的。

用法：

    python scripts/inspect_scores.py                      # 跑内置多组查询
    python scripts/inspect_scores.py --query "RAG 检索"    # 只跑一条
    python scripts/inspect_scores.py --threshold 1.2      # 换一个阈值看效果
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from knowledge.retrieve import retrieve  # noqa: E402

# 前四组是知识库里大概率有的主题，最后一组是知识库里大概率没有的主题。
# 最后一组很重要：它能告诉你「真正不相关时分数会高到多少」，
# 有了这个对照，0.95 到底合不合理才有判断基准。
DEFAULT_QUERIES: list[str] = [
    "Python LangChain RAG 检索",
    "向量数据库 FAISS Embedding",
    "MCP 工具调用 Agent",
    "大模型 Prompt 结构化输出",
    "Kubernetes 集群调度 运维",
]

DEFAULT_K = 5
DEFAULT_THRESHOLD = 0.95


def inspect(query: str, k: int, threshold: float) -> dict:
    """对单条查询检索并打印分布。

    Args:
        query: 检索词。
        k: 返回条数。
        threshold: 判定「高相关」的距离上限。

    Returns:
        本条查询的统计信息。

    Raises:
        RuntimeError: 索引不存在或缺少 Embedding API Key。
    """
    hits = retrieve(query, k=k)

    print("=" * 78)
    print(f"查询: {query}")
    print(f"阈值: {threshold}   返回条数: {len(hits)}")
    print("-" * 78)
    print(f"{'排名':<6}{'距离':<10}{'判定':<10}{'来源':<22}内容预览")
    print("-" * 78)

    scores: list[float] = []
    hit_count = 0

    for rank, item in enumerate(hits, start=1):
        score = float(item.get("score", float("inf")))
        scores.append(score)
        is_relevant = score <= threshold
        if is_relevant:
            hit_count += 1

        source = item.get("source", "")[:20]
        preview = item.get("content", "").replace("\n", " ")[:28]
        print(
            f"{rank:<6}{score:<10.4f}"
            f"{'高相关' if is_relevant else '--':<10}"
            f"{source:<22}{preview}"
        )

    print("-" * 78)
    if scores:
        print(
            f"最小 {min(scores):.4f} / 最大 {max(scores):.4f} / "
            f"均值 {sum(scores) / len(scores):.4f} / 高相关 {hit_count} 条"
        )
    print()

    return {
        "query": query,
        "min": min(scores) if scores else 0.0,
        "max": max(scores) if scores else 0.0,
        "hit_count": hit_count,
        "total": len(hits),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="检视知识库检索距离分布")
    parser.add_argument("--query", help="只跑指定的一条查询")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="返回条数")
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="高相关判定阈值",
    )
    args = parser.parse_args()

    queries = [args.query] if args.query else DEFAULT_QUERIES

    try:
        stats = [inspect(q, args.k, args.threshold) for q in queries]
    except RuntimeError as exc:
        # 常见原因：索引没构建（先跑 python -m knowledge.build_index）
        # 或 .env 里缺 DASHSCOPE_API_KEY
        print(f"检索失败：{exc}")
        return 1

    print("=" * 78)
    print("汇总")
    print("=" * 78)
    print(f"{'查询':<30}{'最小距离':<12}{'最大距离':<12}高相关")
    print("-" * 78)
    for item in stats:
        print(
            f"{item['query'][:28]:<30}{item['min']:<12.4f}"
            f"{item['max']:<12.4f}{item['hit_count']}/{item['total']}"
        )

    print()
    print("怎么读这张表：")
    print("  1. 多数查询的高相关数都等于 k   -> 阈值太松，等于没过滤，")
    print("     retrieve_agentic 的查询改写永远不会触发")
    print("  2. 多数查询的高相关数都是 0     -> 阈值太紧，每次都触发改写，")
    print("     白白多一次 LLM 调用")
    print("  3. 相关与不相关的查询分数能明显区分 -> 阈值合理")
    print()
    print("把结果记下来，这就是「0.95 是怎么定的」这个问题的答案。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
