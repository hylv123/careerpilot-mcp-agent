"""页面脚本冒烟测试：用 Streamlit 官方的 AppTest 无头执行 ui/app.py。

它验证的不是"好不好看"，而是三件更基础的事：
  1. 脚本能完整跑完，没有异常（语法/导入/API 变更都会在这里暴露）；
  2. 关键控件都在（JD 输入框、开始按钮、侧边栏配置）；
  3. 岗位工作台确实渲染了岗位记录，并且"同步"选项真的调用了同步接口。

这些用例必须与外部环境隔离：本机 8000 端口上可能正跑着 API、data/jobs.db 里
可能有真实岗位。若让页面真的去请求它，控件数量就会随数据库内容变化——
测试会在"有数据时红、没数据时绿"之间随机摇摆。所以下面统一 patch 掉网络调用。
"""

from __future__ import annotations

import sys
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

UI_DIR = Path(__file__).resolve().parent.parent / "ui"
if str(UI_DIR) not in sys.path:
    sys.path.insert(0, str(UI_DIR))

import api_client  # noqa: E402

try:
    from streamlit.testing.v1 import AppTest

    _HAS_STREAMLIT = True
except Exception:  # pragma: no cover - 没装界面依赖时走这里
    _HAS_STREAMLIT = False

APP_PATH = Path(__file__).resolve().parent.parent / "ui" / "app.py"
SYNC_LABEL = "同步（调试用，一次等到底）"

FAKE_JOB = {
    "id": 1,
    "title": "AI Agent 开发实习生",
    "company": "Demo 公司",
    "jd": "招聘 AI Agent 开发实习生，要求熟悉 Python 与 RAG。",
    "jd_summary": "招聘 AI Agent 开发实习生，要求熟悉 Python 与 RAG。",
    "status": "已投",
    "source_url": "https://example.com/job/1",
    "deadline": "2026-12-01",
    "notes": "已内推",
    "created_at": "2026-11-20 10:00:00",
    "updated_at": "2026-11-21 10:00:00",
}

FAKE_RESULT = {
    "status": "done",
    "analysis": {"job_title": "AI Agent 开发实习生", "company": "Demo 公司", "level": "实习"},
    "evidence_count": 2,
    "report_text": "这是一段匹配报告。",
    "resume_bullets": [{"bullet": "做过 RAG 检索链路", "evidence_source": "project.md"}],
    "interview_questions": [
        {"question": "如何评估 RAG 效果？", "answer_points": ["命中率"], "evidence_source": "project.md"}
    ],
    "tailored_resume": {
        "headline": "做过 RAG 检索链路的 AI 应用开发",
        "summary": "用 LangChain 搭建过检索链路。",
        "skill_groups": [{"category": "框架", "items": ["LangChain"]}],
        "bullets": [
            {
                "project": "project.md",
                "bullet": "用 LangChain 搭建 RAG 检索链路",
                "jd_requirement": "Python",
                "evidence_source": "project.md",
            }
        ],
        "markdown": "# 候选人 — 定制简历（Demo 公司 · AI Agent 开发实习生）\n\n## 一句话定位\n做过 RAG 检索链路的 AI 应用开发\n",
    },
    "saved": True,
    "save_error": "",
    "match_assessment": {
        "score": 50,
        "method": "需求关键词在本轮检索证据中的字面覆盖率；不等同于能力结论。",
        "covered_requirements": ["Python"],
        "gaps": ["RAG"],
        "evidence_sources": ["project.md"],
    },
}


class _offline:
    """让所有需要网络的调用失败，等价于"本机没有起 API"。

    页面现在有四个页签，任何一个偷偷打了真实网络请求，控件数量就会随本机
    数据变化——测试必须在"有数据/没数据"下都稳定，所以统一挡掉。
    """

    OFFLINE = ("list_jobs", "list_materials", "index_status", "job_sources", "discovery_leads")

    def __enter__(self):
        self._stack = ExitStack()
        for name in self.OFFLINE:
            self._stack.enter_context(
                patch(f"api_client.{name}", side_effect=api_client.ApiError("测试环境未连接 API"))
            )
        return self

    def __exit__(self, *exc_info):
        return self._stack.__exit__(*exc_info)


@unittest.skipUnless(_HAS_STREAMLIT, "未安装 streamlit，跳过界面冒烟")
class StreamlitAppSmokeTests(unittest.TestCase):
    def test_script_runs_without_exception(self) -> None:
        with _offline():
            app = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
        self.assertEqual(list(app.exception), [])

    def test_key_widgets_exist(self) -> None:
        with _offline():
            app = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
        jd_areas = [area for area in app.text_area if area.label == "岗位 JD"]
        self.assertEqual(len(jd_areas), 1)
        self.assertIn("开始分析", [button.label for button in app.button])
        self.assertEqual([box.label for box in app.sidebar.text_input], ["API 地址"])
        self.assertEqual([radio.label for radio in app.sidebar.radio], ["调用方式"])


class ProfileTabTests(unittest.TestCase):
    MATERIAL = {
        "name": "简历",
        "filename": "简历.md",
        "size": 42,
        "updated_at": "2026-11-22 20:00:00",
        "versions": 2,
        "indexed": False,
        "content": "# 简历\n\n做过 RAG 检索。",
    }

    def test_profile_tab_warns_when_index_is_stale(self) -> None:
        status = {
            "materials": 1,
            "index_ready": True,
            "index_updated_at": "2026-11-22 19:00:00",
            "stale": True,
            "chunks": 31,
        }
        with _offline(), patch("api_client.index_status", return_value=status), patch(
            "api_client.list_materials", return_value=[self.MATERIAL]
        ), patch("api_client.get_material", return_value=self.MATERIAL), patch(
            "api_client.list_material_versions", return_value=[]
        ):
            app = AppTest.from_file(str(APP_PATH), default_timeout=60).run()

        self.assertEqual(list(app.exception), [])
        warnings = " ".join(str(item.value) for item in app.warning)
        self.assertIn("索引尚未重建", warnings)
        self.assertIn("候选人资料", [str(item.value) for item in app.subheader])
        self.assertIn("做过 RAG 检索", app.text_area(key="material_content").value)

    def test_profile_tab_shows_fact_ledger(self) -> None:
        status = {"materials": 0, "index_ready": True, "index_updated_at": "2026-11-22 20:00:00", "stale": False, "chunks": 31}
        fact_list = [{"id": "mcp.tools", "label": "MCP 工具数量", "expected": 4, "source": "mcp.json"}]
        with _offline(), patch("api_client.index_status", return_value=status), patch(
            "api_client.list_materials", return_value=[]
        ), patch("api_client.facts", return_value=fact_list):
            app = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
        self.assertEqual(list(app.exception), [])
        self.assertIn("事实校验（易过期的数字）", [str(item.label) for item in app.expander])
        self.assertGreaterEqual(len(app.dataframe), 1)

    def test_profile_tab_empty_state_still_works(self) -> None:
        status = {"materials": 0, "index_ready": False, "index_updated_at": "", "stale": False, "chunks": 0}
        with _offline(), patch("api_client.index_status", return_value=status), patch(
            "api_client.list_materials", return_value=[]
        ):
            app = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
        self.assertEqual(list(app.exception), [])
        self.assertEqual(app.warning, [])


@unittest.skipUnless(_HAS_STREAMLIT, "未安装 streamlit，跳过界面冒烟")
class JobWorkbenchUITests(unittest.TestCase):
    def test_job_rows_render_with_editable_status_and_notes(self) -> None:
        with patch("api_client.list_jobs", return_value=[FAKE_JOB]):
            app = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
        self.assertEqual(list(app.exception), [])
        labels = [area.label for area in app.text_area]
        self.assertIn("备注 / 下一步行动", labels)
        self.assertIn("保存备注", [button.label for button in app.button])
        # 有岗位时不再提示"还没有岗位记录"
        self.assertEqual(len(app.info), 0)

    def test_status_lives_in_header_and_persists_immediately(self) -> None:
        """状态下拉框在卡片头部（和标题同一行），改完立刻落库。"""
        with patch("api_client.list_jobs", return_value=[FAKE_JOB]), patch("api_client.update_job") as update:
            app = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
            status_box = app.selectbox(key="status_1")
            self.assertEqual(status_box.value, "已投", "应显示记录当前状态")
            status_box.set_value("面试").run()
        self.assertEqual(list(app.exception), [])
        update.assert_called_once_with(1, {"status": "面试"}, api_client.DEFAULT_BASE_URL)

    def test_notes_save_sends_only_notes(self) -> None:
        with patch("api_client.list_jobs", return_value=[FAKE_JOB]), patch("api_client.update_job") as update:
            app = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
            app.text_area(key="notes_1").set_value("已内推，等回复").run()
            app.button(key="save_1").click().run()
        self.assertEqual(list(app.exception), [])
        update.assert_called_once_with(1, {"notes": "已内推，等回复"}, api_client.DEFAULT_BASE_URL)

    def test_status_change_failure_is_visible(self) -> None:
        with patch("api_client.list_jobs", return_value=[FAKE_JOB]), patch(
            "api_client.update_job", side_effect=api_client.ApiError("数据库被锁")
        ):
            app = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
            app.selectbox(key="status_1").set_value("面试").run()
        errors = [str(item.value) for item in app.error]
        self.assertTrue(any("数据库被锁" in text for text in errors))

    def test_empty_workbench_shows_hint(self) -> None:
        with patch("api_client.list_jobs", return_value=[]):
            app = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
        self.assertEqual(list(app.exception), [])
        captions = [str(item.value) for item in app.caption]
        self.assertTrue(any("还没有岗位记录" in text for text in captions))

    def test_jd_summary_visible_without_expanding(self) -> None:
        with patch("api_client.list_jobs", return_value=[FAKE_JOB]):
            app = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
        rendered = " ".join(str(item.value) for item in [*app.markdown, *app.text])
        self.assertIn(FAKE_JOB["jd_summary"], rendered)

    def test_delete_needs_confirmation_then_calls_api(self) -> None:
        with patch("api_client.list_jobs", return_value=[FAKE_JOB]), patch("api_client.delete_job") as delete:
            app = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
            self.assertNotIn("确认删除", [button.label for button in app.button])
            app.button(key="delete_1").click().run()
            self.assertIn("确认删除", [button.label for button in app.button])
            delete.assert_not_called()
            app.button(key="confirm_delete_yes_1").click().run()
        self.assertEqual(list(app.exception), [])
        delete.assert_called_once_with(1, api_client.DEFAULT_BASE_URL)

    def test_delete_cancel_keeps_record(self) -> None:
        with patch("api_client.list_jobs", return_value=[FAKE_JOB]), patch("api_client.delete_job") as delete:
            app = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
            app.button(key="delete_1").click().run()
            app.button(key="confirm_delete_no_1").click().run()
        self.assertEqual(list(app.exception), [])
        delete.assert_not_called()
        self.assertNotIn("确认删除", [button.label for button in app.button])


@unittest.skipUnless(_HAS_STREAMLIT, "未安装 streamlit，跳过界面冒烟")
class SyncModeWiringTests(unittest.TestCase):
    """回归保护：侧边栏"同步"选项必须真的调用 /analyze/sync。

    v0.2.0 曾出现"单选项存在但永远走异步提交"的假功能，用户无法察觉。
    """

    def test_sync_mode_calls_sync_endpoint(self) -> None:
        with _offline(), patch("api_client.analyze_sync", return_value=FAKE_RESULT) as sync:
            app = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
            app.sidebar.radio[0].set_value(SYNC_LABEL).run()
            app.button[0].click().run()
        self.assertEqual(list(app.exception), [])
        sync.assert_called_once()
        self.assertTrue(sync.call_args.args[0].strip())


    def test_tailored_resume_is_rendered_with_download(self) -> None:
        fake_submit = api_client.SubmitResult(task_id="task-abc", reused=False)
        with _offline(), patch("api_client.submit_analysis", return_value=fake_submit), patch(
            "api_client.wait_for_result",
            return_value={"status": "done", "result": FAKE_RESULT},
        ):
            app = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
            app.button[0].click().run()
        self.assertEqual(list(app.exception), [])
        body = " ".join(str(item.value) for item in app.markdown)
        self.assertIn("JD 定制简历", [str(item.value) for item in app.subheader])
        self.assertIn("## 一句话定位", body)

    def test_async_mode_calls_submit_endpoint(self) -> None:
        fake_submit = api_client.SubmitResult(task_id="task-abc", reused=False)
        with _offline(), patch("api_client.submit_analysis", return_value=fake_submit) as submit, patch(
            "api_client.wait_for_result",
            return_value={"status": "done", "result": FAKE_RESULT},
        ):
            app = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
            app.button[0].click().run()
        self.assertEqual(list(app.exception), [])
        submit.assert_called_once()


if __name__ == "__main__":
    unittest.main()
