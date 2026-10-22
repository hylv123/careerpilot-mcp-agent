"""检索质量评测：Recall@k / MRR / Precision@k。

为什么这套指标和 eval/test_cases.py 是两套：
    test_cases.py 测的是【端到端产出】：格式对不对、引用真不真、关键词覆盖没有。
    它回答的是"最终给用户的东西能不能用"。
    本脚本测的是【检索这一环】：给定一份 JD，应该被召回的片段有没有真的被召回。
    它回答的是"系统的地基牢不牢"。

    两者不可互相替代：端到端全绿但召回很差，说明生成环节在靠模型硬编；
    召回很好但端到端翻车，说明是后处理的问题。分开测才知道该改哪里。

为什么要对比 plain 与 agentic 两种策略：
    项目里有"高相关结果不足时改写查询并补充检索"的功能，但它一直没有被证明过有效。
    本脚本把两种策略放在同一批标注上跑，直接看 Recall 有没有提升 ——
    如果没提升，那就是个徒增复杂度的功能，应该删掉而不是写进简历。

运行：
    python -m eval.recall                     # 两种策略都跑
    python -m eval.recall --no-agentic        # 只跑基础检索（省钱，不调模型）
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

from knowledge.retrieve import _get_store, retrieve, retrieve_agentic

GOLDEN_PATH = Path(__file__).with_name("retrieval_golden.json")
DEFAULT_K = 4


def build_content_index() -> dict[str, int]:
    """建立"片段正文 -> 片段编号"的映射。

    Returns:
        正文到编号的字典。

    为什么要这个映射：
        检索接口返回的是片段内容，而标注集里写的是编号。
        没有这层映射，就无法判断"召回的第 3 条是不是标注里的第 25 号"。
    """
    store = _get_store()
    docs = list(store.docstore._dict.values())
    return {doc.page_content: index for index, doc in enumerate(docs)}


def recall_at(retrieved: list[int], relevant: list[int], k: int) -> float:
    """前 k 条结果里召回了多少比例的应该召回项。

    Args:
        retrieved: 按相似度排序的片段编号列表。
        relevant: 人工标注的应该召回的编号列表。
        k: 取前几条。

    Returns:
        0.0 ~ 1.0。
    """
    if not relevant:
        return 0.0
    hit = len(set(retrieved[:k]) & set(relevant))
    return hit / len(relevant)


def precision_at(retrieved: list[int], relevant: list[int], k: int) -> float:
    """前 k 条结果里有多少比例是真正相关的。

    Args:
        retrieved: 按相似度排序的片段编号列表。
        relevant: 人工标注的应该召回的编号列表。
        k: 取前几条。

    Returns:
        0.0 ~ 1.0。

    为什么精确率要和召回率一起看：
        只优化召回率有个作弊办法——把 k 调到等于全库大小，召回率必然 100%，
        但返回的东西全是噪声。精确率就是防止这种自欺。
    """
    top = retrieved[:k]
    if not top:
        return 0.0
    hit = len(set(top) & set(relevant))
    return hit / len(top)


def reciprocal_rank(retrieved: list[int], relevant: list[int]) -> float:
    """第一个命中项排在第几位（取倒数）。

    Args:
        retrieved: 按相似度排序的片段编号列表。
        relevant: 人工标注的应该召回的编号列表。

    Returns:
        1/名次，一个都没命中返回 0.0。

    为什么看这个：
        Recall@k 只关心"有没有进来"，不关心"排在第几"。
        而下游是把检索结果塞进 prompt 的，排在第一条和第四条，
        对模型注意力的影响完全不同。
    """
    for rank, index in enumerate(retrieved, start=1):
        if index in relevant:
            return 1.0 / rank
    return 0.0


def strategy_plain(
    case: dict[str, Any], content_index: dict[str, int], k: int
) -> list[int]:
    """基础检索：直接用 JD 关键词查一次。

    Args:
        case: 单个标注用例。
        content_index: 正文到编号的映射。
        k: 取前几条。

    Returns:
        召回的片段编号列表。
    """
    query = " ".join(case["keywords"][:5])
    hits = retrieve(query, k=k)
    return [content_index[h["content"]] for h in hits if h["content"] in content_index]


def strategy_agentic(
    case: dict[str, Any], content_index: dict[str, int], k: int
) -> list[int]:
    """Agentic 检索：相关结果不足时改写查询并补充检索。

    Args:
        case: 单个标注用例。
        content_index: 正文到编号的映射。
        k: 取前几条。

    Returns:
        召回的片段编号列表。

    注意：这个策略可能调用大模型改写查询，会花 token，
    所以用 --no-agentic 可以跳过它。
    """
    hits = retrieve_agentic(case["keywords"], k=k)
    return [content_index[h["content"]] for h in hits if h["content"] in content_index]


def strategy_forced(
    case: dict[str, Any], content_index: dict[str, int], k: int
) -> list[int]:
    """强制触发查询改写与补充检索。

    Args:
        case: 单个标注用例。
        content_index: 正文到编号的映射。
        k: 取前几条。

    Returns:
        召回的片段编号列表。

    为什么要有这个策略：
        默认参数下 min_hits=2、max_distance=1.02，实测几乎每条查询第一轮就
        "命中 2 条以上"，于是改写查询的分支【从来没有被触发过】。
        也就是说，那个功能在真实运行里是个摆设 —— 但它到底有没有价值，
        不触发就永远不知道。这里把 min_hits 设成 99 强制它改写，
        用同一批标注对比：改写了召回是变好还是变坏。
        如果变坏，这个功能就该删掉，而不是继续写进简历里充数。
    """
    hits = retrieve_agentic(case["keywords"], k=k, min_hits=99)
    return [content_index[h["content"]] for h in hits if h["content"] in content_index]


def evaluate(
    cases: list[dict[str, Any]],
    content_index: dict[str, int],
    k: int = DEFAULT_K,
    with_agentic: bool = True,
) -> dict[str, Any]:
    """跑完整评测。

    Args:
        cases: 标注用例列表。
        content_index: 正文到编号的映射。
        k: 取前几条。
        with_agentic: 是否跑 agentic 策略。

    Returns:
        含逐条结果与汇总的字典。
    """
    rows: list[dict[str, Any]] = []
    for case in cases:
        relevant = case["relevant_chunks"]
        plain = strategy_plain(case, content_index, k)

        row: dict[str, Any] = {
            "id": case["id"],
            "match_level": case["match_level"],
            "relevant_count": len(relevant),
            "plain_recall@1": recall_at(plain, relevant, 1),
            "plain_recall@2": recall_at(plain, relevant, 2),
            "plain_recall@k": recall_at(plain, relevant, k),
            "plain_precision@k": precision_at(plain, relevant, k),
            "plain_mrr": reciprocal_rank(plain, relevant),
            "plain_retrieved": plain,
        }

        if with_agentic:
            agentic = strategy_agentic(case, content_index, k)
            forced = strategy_forced(case, content_index, k)
            row.update(
                {
                    "agentic_recall@1": recall_at(agentic, relevant, 1),
                    "agentic_recall@k": recall_at(agentic, relevant, k),
                    "agentic_precision@k": precision_at(agentic, relevant, k),
                    "agentic_mrr": reciprocal_rank(agentic, relevant),
                    "agentic_retrieved": agentic,
                    "forced_recall@k": recall_at(forced, relevant, k),
                    "forced_precision@k": precision_at(forced, relevant, k),
                    "forced_mrr": reciprocal_rank(forced, relevant),
                    "forced_retrieved": forced,
                }
            )
        rows.append(row)
        print(
            f"  [{case['id']}] {case['match_level']:4s} |R|={len(relevant):2d}  "
            f"plain R@{k}={row[f'plain_recall@k']:.2f}  MRR={row['plain_mrr']:.2f}"
        )

    def average(key: str) -> float:
        return sum(row[key] for row in rows) / len(rows) if rows else 0.0

    summary: dict[str, Any] = {
        "cases": len(rows),
        "k": k,
        "plain_recall@1": average("plain_recall@1"),
        "plain_recall@2": average("plain_recall@2"),
        f"plain_recall@{k}": average(f"plain_recall@k"),
        f"plain_precision@{k}": average("plain_precision@k"),
        "plain_mrr": average("plain_mrr"),
    }
    if with_agentic:
        summary.update(
            {
                "agentic_recall@1": average("agentic_recall@1"),
                f"agentic_recall@{k}": average("agentic_recall@k"),
                f"agentic_precision@{k}": average("agentic_precision@k"),
                "agentic_mrr": average("agentic_mrr"),
                f"forced_recall@{k}": average("forced_recall@k"),
                f"forced_precision@{k}": average("forced_precision@k"),
                "forced_mrr": average("forced_mrr"),
            }
        )

    # 高匹配与低匹配分开统计：如果两组数字差不多，说明这套检索根本没有区分能力
    for level in ("high", "low"):
        subset = [row for row in rows if row["match_level"] == level]
        if subset:
            summary[f"{level}_match_recall@{k}"] = sum(
                row[f"plain_recall@k"] for row in subset
            ) / len(subset)

    return {"summary": summary, "rows": rows}


def main() -> int:
    """命令行入口。

    Returns:
        退出码。
    """
    parser = argparse.ArgumentParser(description="检索质量评测 Recall@k / MRR")
    parser.add_argument(
        "--golden",
        type=Path,
        default=GOLDEN_PATH,
        help="标注集路径",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("recall_results.json"),
        help="结果输出路径",
    )
    parser.add_argument(
        "-k",
        type=int,
        default=DEFAULT_K,
        help="取前几条结果",
    )
    parser.add_argument(
        "--no-agentic",
        action="store_true",
        help="跳过 agentic 策略（不调用模型，省 token）",
    )
    args = parser.parse_args()

    golden = json.loads(args.golden.read_text(encoding="utf-8"))
    cases = golden["cases"]
    print(f"标注集：{len(cases)} 条，知识库 {golden.get('library_size')} 个片段，k={args.k}")
    print()

    content_index = build_content_index()
    report = evaluate(
        cases, content_index, k=args.k, with_agentic=not args.no_agentic
    )

    print()
    print("=" * 60)
    print(f"汇总（{report['summary']['cases']} 条，k={args.k}）")
    print("=" * 60)
    for key, value in report["summary"].items():
        if key in ("cases", "k"):
            continue
        print(f"  {key:28s} {value:.3f}")

    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print()
    print(f"详细结果：{args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
