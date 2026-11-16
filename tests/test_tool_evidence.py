"""`agent/graph.py::_tool_messages_to_evidence` 的单测。

为什么单独测这个纯函数：
    它是"工具结果能不能进入报告"的唯一通道。三条生成节点只读 analysis + evidence，
    所以这里少一条、错一条，都会让 agent/tools 两个节点白跑，
    而表面上任务依然成功——正是那种"看不出坏"的退化。
"""

from __future__ import annotations

import json
import unittest

from langchain_core.messages import AIMessage, ToolMessage

from agent.graph import (
    SEARCH_TOOL_NAME,
    TOOL_EVIDENCE_MAX_CHARS,
    TOOL_EVIDENCE_PREFIX,
    _build_search_tool,
    _tool_messages_to_evidence,
)
from agent.mcp_tools import TOOL_ERROR_PREFIX


def _call(name: str, call_id: str) -> dict:
    return {"name": name, "args": {}, "id": call_id}


def _tool_msg(content, call_id: str = "c1") -> ToolMessage:
    return ToolMessage(content=content, tool_call_id=call_id)


class ToolEvidenceTests(unittest.TestCase):
    def test_success_result_becomes_evidence(self) -> None:
        calls = [_call("search_projects", "c1")]
        msgs = [_tool_msg('[{"name": "CareerPilot"}]', "c1")]

        added = _tool_messages_to_evidence(calls, msgs, [])

        self.assertEqual(len(added), 1)
        self.assertEqual(added[0]["source"], f"{TOOL_EVIDENCE_PREFIX}search_projects")
        self.assertIn("CareerPilot", added[0]["content"])
        self.assertEqual(added[0]["score"], 0.0)

    def test_error_text_is_not_evidence(self) -> None:
        """失败降级成的文本不是事实依据，进 evidence 会污染引用来源。"""
        calls = [_call("save_job", "c1")]
        msgs = [_tool_msg(f"{TOOL_ERROR_PREFIX}save_job：timeout", "c1")]

        self.assertEqual(_tool_messages_to_evidence(calls, msgs, []), [])

    def test_duplicate_content_is_dropped(self) -> None:
        calls = [_call("get_resume", "c1"), _call("get_resume", "c2")]
        msgs = [_tool_msg("同一份简历", "c1"), _tool_msg("同一份简历", "c2")]

        added = _tool_messages_to_evidence(calls, msgs, [])

        self.assertEqual(len(added), 1)

    def test_existing_evidence_participates_in_dedup(self) -> None:
        calls = [_call("get_resume", "c1")]
        msgs = [_tool_msg("已经检索到的片段", "c1")]
        existing = [{"content": "已经检索到的片段", "source": "resume.md", "score": 0.2}]

        self.assertEqual(_tool_messages_to_evidence(calls, msgs, existing), [])

    def test_long_result_is_truncated(self) -> None:
        calls = [_call("get_resume", "c1")]
        msgs = [_tool_msg("x" * (TOOL_EVIDENCE_MAX_CHARS + 500), "c1")]

        added = _tool_messages_to_evidence(calls, msgs, [])

        self.assertEqual(len(added[0]["content"]), TOOL_EVIDENCE_MAX_CHARS)

    def test_non_tool_messages_are_ignored(self) -> None:
        calls = [_call("get_resume", "c1")]
        msgs = [AIMessage(content="我只是模型的一句回复")]

        self.assertEqual(_tool_messages_to_evidence(calls, msgs, []), [])

    def test_block_list_content_is_joined(self) -> None:
        calls = [_call("get_resume", "c1")]
        msgs = [_tool_msg([{"type": "text", "text": "第一段"}, {"type": "text", "text": "第二段"}], "c1")]

        added = _tool_messages_to_evidence(calls, msgs, [])

        self.assertIn("第一段", added[0]["content"])
        self.assertIn("第二段", added[0]["content"])


class SearchToolTests(unittest.TestCase):
    """检索工具（本地工具，不走 MCP）的注册与返回值解析。"""

    def test_tool_is_registered_with_expected_name(self) -> None:
        tool = _build_search_tool()

        self.assertEqual(tool.name, SEARCH_TOOL_NAME)
        self.assertIn("检索", tool.description)

    def test_hits_keep_original_file_source(self) -> None:
        """检索结果要逐条拆开，才能保住"结论来自哪个文件"。"""
        payload = json.dumps(
            [
                {
                    "content": "用 LangGraph 编排了九节点工作流",
                    "source": "项目经历.md",
                    "source_path": "/kb/docs/项目经历.md",
                    "score": 0.42,
                }
            ],
            ensure_ascii=False,
        )
        added = _tool_messages_to_evidence(
            [_call(SEARCH_TOOL_NAME, "c1")], [_tool_msg(payload, "c1")], []
        )

        self.assertEqual(len(added), 1)
        self.assertEqual(added[0]["source"], "项目经历.md")
        self.assertAlmostEqual(added[0]["score"], 0.42)

    def test_non_json_output_yields_nothing(self) -> None:
        """工具返回纯文本（比如降级文案）时不能崩，也不应产出证据。"""
        added = _tool_messages_to_evidence(
            [_call(SEARCH_TOOL_NAME, "c1")], [_tool_msg("不是 JSON", "c1")], []
        )

        self.assertEqual(added, [])


if __name__ == "__main__":
    unittest.main()
