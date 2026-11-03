"""CareerPilot 最小网页界面（Streamlit）。

启动：
    pip install -r requirements-ui.txt
    streamlit run ui/app.py

它只做三件事：填 JD → 提交并轮询 → 分块渲染结果。
所有 HTTP 交互都在 ui/api_client.py 里，这里只负责画页面。

为什么要界面：
    这个项目的交付物是三个结构化字段，命令行只能把它们打成一段文本；
    而报告/简历要点/面试题本来就是"分块看"的东西。
"""

from __future__ import annotations

import sys
from pathlib import Path

# `streamlit run ui/app.py` 时脚本所在目录不一定在 sys.path 上，
# 显式加进去，保证 api_client 能被导入（也能让 `python ui/app.py` 报出可读的错）。
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st  # noqa: E402

import api_client as api  # noqa: E402

SAMPLE_JD = (
    "招聘 AI 应用开发实习生，要求熟悉 Python、LangChain、RAG，"
    "有 MCP 或 Agent 项目经验优先。"
)

st.set_page_config(page_title="CareerPilot", page_icon="🧭", layout="wide")


def render_saved_banner(result: dict) -> None:
    """把"岗位到底存没存进 SQLite"显式画出来。

    这是"保存失败必须可见"的界面侧闭环：以前保存失败只藏在 messages 里，
    用户看到的是"分析成功"，实际上投递记录没写进去。
    """
    if result.get("saved"):
        st.success("岗位记录已写入投递库（SQLite）")
    else:
        st.warning(f"岗位记录未保存：{result.get('save_error') or '未知原因'}")


def render_result(result: dict) -> None:
    analysis = result.get("analysis") or {}
    left, right = st.columns([1, 2])
    with left:
        st.subheader("岗位分析")
        st.write(
            {
                "岗位": analysis.get("job_title", "—"),
                "公司": analysis.get("company", "—"),
                "级别": analysis.get("level", "—"),
            }
        )
        st.caption(f"命中经历 {result.get('evidence_count', 0)} 条")
    with right:
        st.subheader("匹配报告")
        st.markdown(result.get("report_text") or "_（空）_")

    st.divider()
    bullets = result.get("resume_bullets") or []
    st.subheader(f"定制简历要点（{len(bullets)} 条）")
    if not bullets:
        st.info("没有生成简历要点")
    for item in bullets:
        st.markdown(f"- {item.get('bullet', '')}")
        st.caption(f"来源：{item.get('evidence_source', '')}")

    st.divider()
    questions = result.get("interview_questions") or []
    st.subheader(f"面试准备题（{len(questions)} 道）")
    for index, question in enumerate(questions, start=1):
        with st.expander(f"Q{index}. {question.get('question', '')}"):
            for point in question.get("answer_points") or []:
                st.markdown(f"- {point}")
            st.caption(f"来源：{question.get('evidence_source', '')}")

    with st.expander("原始 JSON（排障用）"):
        st.json(result)


def main() -> None:
    st.title("🧭 CareerPilot")
    st.caption("输入一段岗位 JD，产出匹配报告、定制简历要点与面试题，并把岗位写进投递库。")

    with st.sidebar:
        st.header("连接设置")
        base_url = st.text_input("API 地址", value=api.DEFAULT_BASE_URL)
        mode = st.radio(
            "调用方式",
            ["异步（推荐，先返回 task_id 再轮询）", "同步（调试用，一次等到底）"],
            index=0,
        )
        poll_interval = st.slider("轮询间隔（秒）", 1.0, 10.0, 2.0, 0.5)
        st.caption("异步要求 API + Redis + worker 都在跑；同步只要求 API。")

    jd = st.text_area("岗位 JD", value=SAMPLE_JD, height=180)
    submitted = st.button("开始分析", type="primary", use_container_width=True)

    if not submitted:
        return
    if len(jd.strip()) < 10:
        st.error("JD 太短了（至少 10 个字符）")
        return

    status_box = st.status("正在提交…", expanded=True)
    try:
        submit = api.submit_analysis(jd.strip(), base_url)
        if submit.direct_result is not None:
            status_box.update(label="Redis 不可用，已同步跑完", state="complete")
            render_result(submit.direct_result)
            render_saved_banner(submit.direct_result)
            return

        hint = "（命中幂等，复用已有任务）" if submit.reused else ""
        status_box.write(f"已受理：`{submit.task_id}` {hint}")

        def on_update(status: str, progress: str) -> None:
            status_box.write(f"状态：`{status}` {progress}")

        payload = api.wait_for_result(
            submit.task_id,
            base_url,
            poll_interval=poll_interval,
            on_update=on_update,
        )
        status_box.update(label=f"任务结束：{payload.get('status')}", state="complete")

        if payload.get("status") == "failed":
            st.error(f"分析失败：{payload.get('error') or '未知原因'}")
            return

        result = payload.get("result") or {}
        render_result(result)
        render_saved_banner(result)
    except api.ApiError as exc:
        status_box.update(label="调用失败", state="error")
        st.error(str(exc))


main()
