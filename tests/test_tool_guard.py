"""`agent/mcp_tools.py::validate_tool_calls` 的单测。

为什么单独测这个函数：
    它是模型输出进入执行层之前的最后一道闸门，而模型输出是不可信输入——
    工具名是生成出来的、args 是生成出来的、tool_call_id 也可能缺失。
    这里每个分支都对应一种会把整条链路打断的真实故障，
    所以不需要连 Redis、不需要起 MCP Server，纯函数测干净。
"""

from __future__ import annotations

import unittest

from agent.mcp_tools import TOOL_REJECT_PREFIX, validate_tool_calls

ALLOWED = {"search_projects", "save_job"}


def _call(name: str = "search_projects", args: dict | None = None, call_id: str = "call_1") -> dict:
    call = {"name": name, "args": args if args is not None else {"q": "RAG"}}
    if call_id:
        call["id"] = call_id
    return call


class ValidateToolCallsTests(unittest.TestCase):
    def test_valid_call_passes_through(self) -> None:
        valid, rejected = validate_tool_calls([_call()], ALLOWED)
        self.assertEqual(len(valid), 1)
        self.assertEqual(rejected, [])

    def test_empty_input(self) -> None:
        for payload in (None, []):
            valid, rejected = validate_tool_calls(payload, ALLOWED)
            self.assertEqual((valid, rejected), ([], []))

    def test_unknown_tool_name_is_rejected(self) -> None:
        valid, rejected = validate_tool_calls([_call(name="delete_database")], ALLOWED)
        self.assertEqual(valid, [])
        self.assertEqual(len(rejected), 1)
        self.assertIn("工具名非法", rejected[0][1])

    def test_missing_tool_call_id_is_rejected(self) -> None:
        """没有 id 就没法回填 ToolMessage，下一轮协议会断，必须拦下来。"""
        valid, rejected = validate_tool_calls([_call(call_id="")], ALLOWED)
        self.assertEqual(valid, [])
        self.assertIn("tool_call_id", rejected[0][1])

    def test_non_object_args_is_rejected(self) -> None:
        broken = {"name": "search_projects", "args": "q=RAG", "id": "call_9"}
        valid, rejected = validate_tool_calls([broken], ALLOWED)
        self.assertEqual(valid, [])
        self.assertIn("JSON 对象", rejected[0][1])

    def test_mixed_calls_are_split(self) -> None:
        calls = [
            _call(call_id="ok-1"),
            _call(name="hallucinated_tool", call_id="bad-1"),
            _call(call_id="ok-2", args={"q": "MCP"}),
        ]
        valid, rejected = validate_tool_calls(calls, ALLOWED)
        self.assertEqual([c["id"] for c in valid], ["ok-1", "ok-2"])
        self.assertEqual([c["id"] for c, _ in rejected], ["bad-1"])

    def test_reject_prefix_is_stable(self) -> None:
        """前缀是给模型看的固定信号，改动会悄悄影响提示词里的约定。"""
        self.assertEqual(TOOL_REJECT_PREFIX, "工具调用被拒绝：")


if __name__ == "__main__":
    unittest.main()
