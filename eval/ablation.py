"""检索消融实验：向量 / BM25 / RRF 混合，三组对比。

为什么要做这个脚本：
    "加 BM25 会有好处"是一句很容易说、也很容易翻车的话。
    没有对照实验，它只是个人观点；有了这三组数字，它才是结论——
    哪怕结论是"在我这个语料规模下没有提升"，那也是可以写进复盘的事实。

为什么三组要用同一批标注、同一个 k：
    检索指标对标注和 k 极其敏感。换一套标注或改一个 k，
    数字就没有可比性，实验也就失去意义。

运行：
    python -m eval.ablation
    python -m eval.ablation -k 4 --golden eval/retrieval_golden_v2.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

from eval.recall import (
    build_content_index,
    precision_at,
    recall_at,
    reciprocal_rank,
)
from knowledge.bm25 import bm25_search, hybrid_search
from knowledge.retrieve import retrieve

DEFAULT_GOLDEN = Path(__file__).with_name("retrieval_golden_v2.json")
DEFAULT_OUTPUT = Path(__file__).with_name("ablation_results.json")
DEFAULT_K = 4


def _hits_to_ids(hits: list[dict[str, Any]], content_index: dict[str, int]) -> list[int]:
    """把检索结果映射回片段编号（标注集用的是编号）。"""
    return [
        content_index[hit["content"]]
        for hit in hits
        if hit["content"] in content_index
    ]


def strategies(k: int) -> dict[str, Callable[[str], list[dict[str, Any]]]]:
    """三组策略。三组的 pool 保持一致，避免胜在"看得更多"。"""
    pool = max(k * 3, 10)
    return {
        "vector": lambda query: retrieve(query, k=k),
        "bm25": lambda query: bm25_search(query, k=k),
        "rrf": lambda query: hybrid_search(query, k=k, pool=pool),
    }


def evaluate(
    cases: list[dict[str, Any]],
    content_index: dict[str, int],
    k: int = DEFAULT_K,
) -> dict[str, Any]:
    """跑完三组策略，返回逐条结果与汇总。"""
    arms = strategies(k)
    rows: list[dict[str, Any]] = []

    for case in cases:
        query = " ".join(case["keywords"][:5])
        relevant = case["relevant_chunks"]
        row: dict[str, Any] = {
            "id": case["id"],
            "match_level": case["match_level"],
            "query": query,
            "relevant_count": len(relevant),
        }

        for name, search in arms.items():
            ids = _hits_to_ids(search(query), content_index)
            row[name] = {
                "ids": ids,
                f"recall@{k}": recall_at(ids, relevant, k),
                f"precision@{k}": precision_at(ids, relevant, k),
                "mrr": reciprocal_rank(ids, relevant),
            }

        rows.append(row)
        print(
            f"  [{row['id']:14s}] |R|={row['relevant_count']:2d}  "
            + "  ".join(
                f"{name} R@{k}={row[name][f'recall@{k}']:.2f}" for name in arms
            )
        )

    summary: dict[str, Any] = {"cases": len(rows), "k": k}
    for name in arms:
        for metric in (f"recall@{k}", f"precision@{k}", "mrr"):
            values = [row[name][metric] for row in rows]
            summary[f"{name}_{metric}"] = sum(values) / len(values) if values else 0.0

    return {"summary": summary, "rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description="检索消融：向量 vs BM25 vs RRF")
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("-k", type=int, default=DEFAULT_K)
    args = parser.parse_args()

    golden = json.loads(args.golden.read_text(encoding="utf-8"))
    cases = golden["cases"]
    print(f"标注集：{args.golden.name}（{len(cases)} 条，k={args.k}）")
    print()

    content_index = build_content_index()
    print(f"当前索引：{len(content_index)} 个片段")
    print()

    report = evaluate(cases, content_index, k=args.k)

    print()
    print("=" * 62)
    print(f"汇总（{len(cases)} 条，k={args.k}）")
    print("=" * 62)
    header = f"{'指标':<22}" + "".join(f"{name:>12}" for name in ("vector", "bm25", "rrf"))
    print(header)
    for metric in (f"recall@{args.k}", f"precision@{args.k}", "mrr"):
        line = f"{metric:<22}"
        for name in ("vector", "bm25", "rrf"):
            line += f"{report['summary'][f'{name}_{metric}']:>12.3f}"
        print(line)

    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print()
    print(f"详细结果：{args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
