"""save 节点的结构化结果测试。

背景：岗位写入以前走 MCP 的 `save_job` 工具，失败时适配层会吞掉异常、
只回一句文本；如果 `save` 节点只把这句文本塞进 messages，接口层就分不清
"写库成功"和"写库失败"——任务照样是 done，用户以为记录上了。

2026-11-22 起这一步改成原生调用 `services.jobs.create_job`：
不再经过 MCP 子进程，失败会直接抛异常、当场转成 saved=False。
这组测试锁住四件事：
  1. 成功 → saved=True 且 save_error 为空；
  2. 入参来自 analysis（缺失时用"未识别岗位/公司"兜底）；
  3. 写库抛异常 → saved=False 且 save_error 带真实原因；
  4. 结构化字段能被接口模型吃下。
"""

from __future__ import annotations

import sqlite3
import unittest
from unittest.mock import patch

from agent.graph import save_record_impl
from agent.mcp_tools import TOOL_ERROR_PREFIX, is_tool_error
from api.schemas import AnalyzeResponse


def _state() -> dict:
    return {
        "user_input": "招聘 AI 应用开发实习生，要求 Python 与 RAG",
        "analysis": {"job_title": "AI 应用开发实习生", "company": "Demo Co"},
    }


class SaveRecordTests(unittest.TestCase):
    def test_success_sets_saved_true(self) -> None:
        with patch("agent.graph.create_job", return_value={"id": 7, "title": "AI 应用开发实习生"}):
            update = save_record_impl(_state())
        self.assertIs(update["saved"], True)
        self.assertEqual(update["save_error"], "")
        self.assertEqual(len(update["messages"]), 1)
        self.assertIn("#7", update["messages"][0][1])

    def test_payload_uses_analysis_fields(self) -> None:
        state = _state()
        with patch("agent.graph.create_job", return_value={"id": 1}) as create:
            save_record_impl(state)
        create.assert_called_once_with(
            title="AI 应用开发实习生",
            company="Demo Co",
            jd=state["user_input"],
            requirements=[],
            preferences=[],
        )

    def test_payload_falls_back_when_analysis_empty(self) -> None:
        with patch("agent.graph.create_job", return_value={"id": 1}) as create:
            save_record_impl({"user_input": "jd", "analysis": {}})
        create.assert_called_once_with(
            title="未识别岗位",
            company="未识别公司",
            jd="jd",
            requirements=[],
            preferences=[],
        )

    def test_requirements_are_persisted_for_clustering(self) -> None:
        state = {
            "user_input": "招聘 AI 实习生",
            "analysis": {
                "job_title": "AI 实习生",
                "company": "Demo",
                "requirements": ["熟悉 Python"],
                "preferences": ["有 MCP 项目经验优先"],
            },
        }
        with patch("agent.graph.create_job", return_value={"id": 3}) as create:
            save_record_impl(state)
        kwargs = create.call_args.kwargs
        self.assertEqual(kwargs["requirements"], ["熟悉 Python"])
        self.assertEqual(kwargs["preferences"], ["有 MCP 项目经验优先"])

    def test_write_failure_sets_saved_false(self) -> None:
        with patch("agent.graph.create_job", side_effect=sqlite3.OperationalError("database is locked")):
            update = save_record_impl(_state())
        self.assertIs(update["saved"], False)
        self.assertIn("database is locked", update["save_error"])
        self.assertIn("未保存", update["save_error"])
        self.assertEqual(len(update["messages"]), 1)

    def test_native_path_does_not_use_mcp(self) -> None:
        """回归保护：save 节点不能再依赖 MCP 工具是否可用。"""
        with patch("agent.graph.create_job", return_value={"id": 1}):
            update = save_record_impl(_state())
        self.assertIs(update["saved"], True)
        self.assertNotIn("MCP", update["save_error"])


class ToolErrorSignalTests(unittest.TestCase):
    """其余 MCP 工具（简历/文档）仍靠这段信号判定失败，逻辑不能退化。"""

    def test_detects_error_prefix(self) -> None:
        self.assertTrue(is_tool_error(f"{TOOL_ERROR_PREFIX}job：boom"))

    def test_detects_fastmcp_server_error(self) -> None:
        # 服务端工具抛异常时 FastMCP 返回的文本（真实踩到过：数据库只读）
        self.assertTrue(
            is_tool_error("Error executing tool get_resume: attempt to write a readonly database")
        )

    def test_detects_error_in_content_blocks(self) -> None:
        blocks = [{"type": "text", "text": "Error executing tool get_resume: boom"}]
        self.assertTrue(is_tool_error(blocks))

    def test_normal_text_is_not_error(self) -> None:
        self.assertFalse(is_tool_error("已读取简历：data/resume.local.json"))

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
