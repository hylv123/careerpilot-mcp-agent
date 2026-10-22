"""Command-line entry point for CareerPilot MCP Agent."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from agent.graph import build_graph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="分析岗位 JD，并生成匹配报告、简历要点和面试题",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--jd", help="直接传入岗位描述文本")
    source.add_argument("--jd-file", type=Path, help="从 UTF-8 文本文件读取岗位描述")
    return parser.parse_args()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    jd = args.jd
    if args.jd_file is not None:
        jd = args.jd_file.read_text(encoding="utf-8").strip()
    if not jd or not jd.strip():
        raise SystemExit("岗位描述不能为空")

    app = build_graph()
    result = app.invoke(
        {
            "user_input": jd.strip(),
            "analysis": {},
            "evidence": [],
            "report_text": "",
            "resume_bullets": [],
            "interview_questions": [],
            "messages": [],
            "tool_rounds": 0,
        }
    )

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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
