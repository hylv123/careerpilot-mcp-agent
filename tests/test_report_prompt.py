"""report 节点的 prompt 输入回归：analysis / evidence 必须先序列化成 JSON。

以前直接传 dict/list，占位符里得到的是 Python repr（单引号、True/None），
不是合法 JSON；模型会把 None 当字符串读。rewrite/questions 一直是对的。
"""

from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

import agent.graph as graph


class ReportPromptTests(unittest.TestCase):
    def test_inputs_are_json_strings(self) -> None:
        state = {
            "user_input": "jd",
            "analysis": {"job_title": "AI 实习生", "level": "实习生", "keywords": ["RAG"]},
            "evidence": [{"content": "做过 RAG 项目", "source": "项目经历.md", "score": 0.8, "flag": None}],
        }
        captured: dict = {}

        def fake_safe_invoke(*, chain, inputs, operation_name):
            captured.update(inputs)
            return AIMessage(content="报告正文")

        fake_llm = RunnableLambda(lambda _prompt: AIMessage(content="x"))
        with patch.object(graph, "_get_llm", return_value=fake_llm), \
             patch.object(graph, "safe_invoke", side_effect=fake_safe_invoke):
            update = graph.report(state)

        self.assertEqual(update["report_text"], "报告正文")

        analysis = captured["analysis"]
        evidence = captured["evidence"]
        self.assertIsInstance(analysis, str, "analysis 应该是 JSON 字符串，不是 dict")
        self.assertIsInstance(evidence, str, "evidence 应该是 JSON 字符串，不是 list")

        # 能解析回原数据，且是合法 JSON（repr 会在这里炸）
        self.assertEqual(json.loads(analysis)["job_title"], "AI 实习生")
        self.assertIsNone(json.loads(evidence)[0]["flag"])
        # 中文不转义，避免 prompt 里出现 \uXXXX
        self.assertIn("AI 实习生", analysis)
        self.assertNotIn("\\u", analysis)
        # 报告正文只来自模型输出
        self.assertNotIn("'", analysis.split('"job_title"')[0] or "")


if __name__ == "__main__":
    unittest.main()
