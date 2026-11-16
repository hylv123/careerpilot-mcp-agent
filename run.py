"""Command-line entry point for CareerPilot MCP Agent.

命令行、HTTP 接口、worker 共用同一个执行入口 `tasks.runner.run_analysis`。
为什么不各自 build_graph()：
    以前 CLI 自己建图 + 手写一份初始 state，API/worker 走 build_initial_state，
    两边很快就漂了——最典型的是递归上限：接口显式给 50，CLI 用默认的 25，
    同一个 JD 可能"命令行跑不通、接口能跑通"。现在只有一条路径。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from agent.logging_setup import setup_logging
from tasks.runner import run_analysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="分析岗位 JD，并生成匹配报告、简历要点和面试题",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--jd", help="直接传入岗位描述文本")
    source.add_argument("--jd-file", type=Path, help="从 UTF-8 文本文件读取岗位描述")
    return parser.parse_args()


def main() -> int:
    setup_logging()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    jd = args.jd
    if args.jd_file is not None:
        jd = args.jd_file.read_text(encoding="utf-8").strip()
    if not jd or not jd.strip():
        raise SystemExit("岗位描述不能为空")

    # 与 API / worker 共用同一个执行入口：
    # 初始 state、递归上限、结果投影都只有一份定义，不会再出现两条路径行为不一致。
    result = run_analysis(jd.strip())

    # 不再遍历 messages 打印：report / rewrite / questions 是并行执行的，
    # messages 里这三条的先后次序不确定，直接打印会导致每次输出顺序都不一样。
    # 这里读各自的字段，保证输出顺序永远是 报告 -> 简历要点 -> 面试题。
    report_text = result.get("report_text", "")
    if report_text.strip():
        print(report_text.strip())
        print()

    bullets = result.get("resume_bullets", [])
    if bullets:
        bullet_text = "\n".join(
            f"- {item.get('bullet', '')}\n  来源: {item.get('evidence_source', '')}"
            for item in bullets
        )
        print(f"## 定制简历要点：\n{bullet_text}")
        print()

    questions = result.get("interview_questions", [])
    if questions:
        sections: list[str] = []
        for idx, ques in enumerate(questions, start=1):
            answer_points = ques.get("answer_points", [])
            ans_text = "\n".join(f"- {point}" for point in answer_points)
            sections.append(
                f"### Q{idx}: {ques.get('question', '')}\n\n"
                f"{ans_text}\n\n"
                f"来源：{ques.get('evidence_source', '')}"
            )
        print("## 面试准备题\n\n" + "\n\n".join(sections))

    # 岗位是否落库必须显式说一句：以前它只藏在 messages 里，
    # 工具失败时终端看起来和成功完全一样。
    if result.get("saved"):
        print("\n岗位记录：已写入 SQLite")
    else:
        print(f"\n岗位记录：未保存 —— {result.get('save_error') or '未知原因'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
