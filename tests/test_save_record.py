"""save 节点的结构化结果测试。

背景：`save_job` 工具失败时，`_to_sync_tool` 会吞掉异常、返回一句文本。
如果 `save` 节点只把这句文本塞进 messages，接口层就分不清
"写库成功"和"写库失败"——任务照样是 done，用户以为记录上了。
这组测试锁住三件事：
  1. 成功 → saved=True 且 save_error 为空；
  2. 失败 → saved=False 且 save_error 带真实原因；
  3. 工具缺失 → saved=False 且原因写明是工具不可用。
"""

from __future__ import annotations

import unittest

from agent.graph import save_record_impl
from agent.mcp_tools import TOOL_ERROR_PREFIX, is_tool_error
from api.schemas import AnalyzeResponse


class _FakeTool:
    """记录入参、返回预设结果的假工具，替代真实 MCP 调用。"""

    def __init__(self, result: str) -> None:
        self.result = result
        self.payload = None

    def invoke(self, payload):
        self.payload = payload
        return self.result


def _state() -> dict:
    return {
        "user_input": "招聘 AI 应用开发实习生，要求 Python 与 RAG",
        "analysis": {"job_title": "AI 应用开发实习生", "company": "Demo Co"},
    }


class SaveRecordTests(unittest.TestCase):
    def test_success_sets_saved_true(self) -> None:
        tool = _FakeTool("已保存岗位：AI 应用开发实习生 from Demo Co")
        update = save_record_impl(_state(), tool)
        self.assertIs(update["saved"], True)
        self.assertEqual(update["save_error"], "")
        self.assertEqual(len(update["messages"]), 1)

    def test_payload_uses_analysis_fields(self) -> None:
        tool = _FakeTool("已保存岗位：x from y")
        state = _state()
        save_record_impl(state, tool)
        self.assertEqual(tool.payload["title"], "AI 应用开发实习生")
        self.assertEqual(tool.payload["company"], "Demo Co")
        self.assertEqual(tool.payload["jd"], state["user_input"])

    def test_payload_falls_back_when_analysis_empty(self) -> None:
        tool = _FakeTool("已保存岗位：x from y")
        save_record_impl({"user_input": "jd", "analysis": {}}, tool)
        self.assertEqual(tool.payload["title"], "未识别岗位")
        self.assertEqual(tool.payload["company"], "未识别公司")

    def test_tool_error_sets_saved_false(self) -> None:
        error_text = f"{TOOL_ERROR_PREFIX}save_job：OperationalError: database is locked"
        update = save_record_impl(_state(), _FakeTool(error_text))
        self.assertIs(update["saved"], False)
        self.assertEqual(update["save_error"], error_text)

    def test_missing_tool_sets_saved_false(self) -> None:
        update = save_record_impl(_state(), None)
        self.assertIs(update["saved"], False)
        self.assertIn("save_job", update["save_error"])
        self.assertIn("不可用", update["save_error"])


class ToolErrorSignalTests(unittest.TestCase):
    def test_detects_error_prefix(self) -> None:
        self.assertTrue(is_tool_error(f"{TOOL_ERROR_PREFIX}job：boom"))

    def test_detects_fastmcp_server_error(self) -> None:
        # 服务端工具抛异常时 FastMCP 返回的文本（真实踩到过：数据库只读）
        self.assertTrue(
            is_tool_error("Error executing tool save_job: attempt to write a readonly database")
        )

    def test_detects_error_in_content_blocks(self) -> None:
        blocks = [{"type": "text", "text": "Error executing tool save_job: boom"}]
        self.assertTrue(is_tool_error(blocks))

    def test_normal_text_is_not_error(self) -> None:
        self.assertFalse(is_tool_error("已保存岗位：AI 实习生 from Demo Co"))

    def test_non_string_is_not_error(self) -> None:
        self.assertFalse(is_tool_error(None))
        self.assertFalse(is_tool_error(["已保存"]))


class AnalyzeResponseContractTests(unittest.TestCase):
    def test_new_fields_round_trip(self) -> None:
        response = AnalyzeResponse(
            status="done",
            saved=False,
            save_error="boom",
        )
        self.assertIs(response.saved, False)
        self.assertEqual(response.save_error, "boom")
        self.assertEqual(response.model_dump()["saved"], False)

    def test_new_fields_default(self) -> None:
        response = AnalyzeResponse(status="done")
        self.assertIs(response.saved, False)
        self.assertEqual(response.save_error, "")


class RunAnalysisContractTests(unittest.TestCase):
    """锁住"图 → run_analysis → AnalyzeResponse"这条契约链。"""

    def test_result_carries_saved_fields(self) -> None:
        from tasks import runner

        class _StubGraph:
            def invoke(self, state, config=None):
                return {
                    **state,
                    "analysis": {"job_title": "t", "company": "c"},
                    "report_text": "r",
                    "saved": False,
                    "save_error": "boom",
                }

        original = runner._graph
        runner._graph = _StubGraph()
        try:
            result = runner.run_analysis("jd")
        finally:
            runner._graph = original

        self.assertIs(result["saved"], False)
        self.assertEqual(result["save_error"], "boom")
        # 必须能直接被接口模型吃下，否则降级/同步路径会 500
        response = AnalyzeResponse(**result)
        self.assertIs(response.saved, False)


if __name__ == "__main__":
    unittest.main()
