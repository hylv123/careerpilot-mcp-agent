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

import re
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
JOB_STATUSES = ["未投", "已投", "笔试", "面试", "offer", "拒绝", "已归档"]

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


def _resume_filename(analysis: dict) -> str:
    """给下载的简历起一个不会炸掉文件系统的名字。"""
    parts = [
        str(analysis.get("company") or "").strip(),
        str(analysis.get("job_title") or "").strip(),
    ]
    raw = "_".join(part for part in parts if part) or "resume"
    safe = re.sub(r"[^\w\u4e00-\u9fff.-]+", "_", raw).strip("._")
    return f"{safe or 'resume'}.md"


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

    assessment = result.get("match_assessment") or {}
    if assessment:
        st.subheader("可解释匹配结论")
        st.metric("证据覆盖分", f"{assessment.get('score', 0)} / 100")
        st.caption(assessment.get("method", ""))
        covered, gaps = st.columns(2)
        with covered:
            st.write("已在本轮证据中覆盖")
            st.write(assessment.get("covered_requirements") or ["暂无"])
        with gaps:
            st.write("待补强 / 待确认")
            st.write(assessment.get("gaps") or ["暂无"])

    st.divider()
    tailored = result.get("tailored_resume") or {}
    if tailored.get("bullets"):
        st.subheader("JD 定制简历")
        head, download = st.columns([3, 1])
        head.caption(tailored.get("headline") or "已按岗位要求与偏好重写，内容全部来自真实经历")
        stats = tailored.get("atom_stats") or {}
        if stats:
            head.caption(
                f"原子引用 {stats.get('cited', 0)}/{stats.get('total', 0)} 条"
                f"（已确认原子 {stats.get('approved', 0)} 条）"
            )
        download.download_button(
            "下载 Markdown",
            tailored.get("markdown") or "",
            file_name=_resume_filename(analysis),
            mime="text/markdown",
            key="download_resume",
        )
        st.markdown(tailored.get("markdown") or "")
        with st.expander("结构化字段（排障用）"):
            st.json(tailored)
    else:
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


def _persist_status(job_id: int, base_url: str) -> None:
    """状态下拉框改动后立即落库。

    做成 on_change 回调而不是"再点一次保存"：状态是高频、单字段的动作，
    每改一次都要额外点一下按钮，最后一定会有人忘了点。
    回调里不能直接画错误提示（这时还没进入本次渲染），所以把结果暂存到
    session_state，由渲染阶段取出来显示。
    """
    new_status = st.session_state.get(f"status_{job_id}", "")
    try:
        api.update_job(job_id, {"status": new_status}, base_url)
    except api.ApiError as exc:
        st.session_state[f"status_error_{job_id}"] = str(exc)
    else:
        st.session_state[f"status_flash_{job_id}"] = new_status


def render_job_workbench(base_url: str) -> None:
    """Render the user-managed half of the product: a job application board."""
    st.subheader("岗位工作台")
    selected_status = st.selectbox("按投递状态筛选", ["全部", *JOB_STATUSES], key="job_filter")
    try:
        jobs = api.list_jobs(base_url, status="" if selected_status == "全部" else selected_status)
    except api.ApiError as exc:
        st.info(f"暂时无法读取岗位工作台：{exc}")
        return
    if not jobs:
        st.caption("还没有岗位记录。完成一次 JD 分析后会自动生成记录。")
        return

    st.caption(f"当前筛选下共 {len(jobs)} 条记录")

    with st.expander("投递聚焦（按技能聚类，确定性计算）"):
        try:
            report = api.clusters(base_url)
        except api.ApiError as exc:
            st.caption(f"暂时无法计算：{exc}")
        else:
            rows = report.get("clusters") or []
            if not rows:
                st.caption("还没有具备技能标签的岗位记录。")
            else:
                st.dataframe(
                    [
                        {
                            "岗位簇": row["label"],
                            "岗位数": row["size"],
                            "证据覆盖率": row["avg_coverage"],
                            "岗位 id": ", ".join(str(item) for item in row["job_ids"]),
                        }
                        for row in rows
                    ],
                    width="stretch",
                    hide_index=True,
                )
            gaps = report.get("gaps") or []
            if gaps:
                st.caption("全库缺口（JD 提到、但我的证据里没有）：")
                st.write(
                    " · ".join(f"{item['token']}（{item['jobs']} 个岗位）" for item in gaps[:8])
                )
            st.caption("覆盖率 = 岗位技能里有多少在我的证据库出现过；衡量有没有材料可写，不是能力结论。")
    for job in jobs:
        job_id = job["id"]
        # 用带边框的容器而不是 expander：JD 概括必须不展开就能看到，
        # 否则列表退化成"只有公司名和状态"，用户仍要逐条点开才知道是什么岗位。
        with st.container(border=True):
            # 状态和标题同一行：看列表时扫一眼就能改，不用先进详情。
            title_col, status_col = st.columns([4, 1], vertical_alignment="center")
            title_col.markdown(f"**#{job_id} · {job['company']} · {job['title']}**")
            status_col.selectbox(
                "投递状态",
                JOB_STATUSES,
                index=JOB_STATUSES.index(job["status"]) if job["status"] in JOB_STATUSES else 0,
                key=f"status_{job_id}",
                on_change=_persist_status,
                args=(job_id, base_url),
                label_visibility="collapsed",
            )
            flash = st.session_state.pop(f"status_flash_{job_id}", None)
            if flash:
                status_col.caption(f"✓ 已改为 {flash}")
            error = st.session_state.pop(f"status_error_{job_id}", None)
            if error:
                st.error(f"状态更新失败：{error}")

            st.caption(
                f"截止：{job.get('deadline') or '未填写'}　更新：{job.get('updated_at') or '—'}"
            )
            st.write(job.get("jd_summary") or "（该记录没有 JD 内容）")

            with st.expander("详情与操作"):
                if job.get("source_url"):
                    st.link_button("打开职位链接", job["source_url"])
                notes = st.text_area("备注 / 下一步行动", value=job.get("notes", ""), key=f"notes_{job_id}")
                actions = st.columns([1, 1, 2])
                if actions[0].button("保存备注", key=f"save_{job_id}"):
                    try:
                        api.update_job(job_id, {"notes": notes}, base_url)
                    except api.ApiError as exc:
                        st.error(str(exc))
                    else:
                        st.success("备注已保存")
                        st.rerun()
                if actions[1].button("删除记录", key=f"delete_{job_id}"):
                    st.session_state[f"confirm_delete_{job_id}"] = True

                if st.session_state.get(f"confirm_delete_{job_id}"):
                    st.warning(f"确认删除 #{job_id}「{job['company']} · {job['title']}」？删除后无法恢复。")
                    confirm, cancel = st.columns(2)
                    if confirm.button("确认删除", type="primary", key=f"confirm_delete_yes_{job_id}"):
                        try:
                            api.delete_job(job_id, base_url)
                        except api.ApiError as exc:
                            st.error(str(exc))
                        else:
                            st.session_state.pop(f"confirm_delete_{job_id}", None)
                            st.success("已删除")
                            st.rerun()
                    if cancel.button("取消", key=f"confirm_delete_no_{job_id}"):
                        st.session_state.pop(f"confirm_delete_{job_id}", None)
                        st.rerun()


def render_profile_tab(base_url: str) -> None:
    """候选人资料管理：编辑 Markdown 资料、留版本、重建索引。"""
    st.subheader("候选人资料")
    try:
        status = api.index_status(base_url)
    except api.ApiError as exc:
        st.info(f"暂时无法读取资料状态：{exc}")
        return

    head, action = st.columns([3, 1])
    head.caption(
        f"资料 {status.get('materials', 0)} 份 · 索引 {status.get('chunks', 0)} 片段"
        f" · 索引更新于 {status.get('index_updated_at') or '—'}"
    )
    if status.get("stale"):
        st.warning("资料有改动但索引尚未重建，检索拿到的仍是旧内容。")
    if action.button("重建索引", key="reindex_btn"):
        try:
            with st.spinner("重建中（约十几秒）…"):
                result = api.reindex(base_url)
        except api.ApiError as exc:
            st.error(str(exc))
        else:
            if result.get("ok"):
                st.success(f"索引已重建：{result.get('chunks')} 片段，用时 {result.get('seconds')}s")
                st.rerun()
            else:
                st.error(result.get("output") or "重建失败")

    try:
        materials = api.list_materials(base_url)
    except api.ApiError as exc:
        st.info(f"暂时无法读取资料列表：{exc}")
        return

    names = [item["name"] for item in materials]
    if names:
        choice = st.selectbox("选择资料", names, key="material_choice")
        if st.session_state.get("material_loaded") != choice:
            try:
                detail = api.get_material(choice, base_url)
            except api.ApiError as exc:
                st.error(str(exc))
            else:
                st.session_state["material_content"] = detail.get("content", "")
                st.session_state["material_loaded"] = choice

        st.text_area("正文（Markdown）", key="material_content", height=320)
        note = st.text_input("这次保存的备注", key="material_note", placeholder="例如：补充 MCP 工具数修正")
        save_col, delete_col = st.columns([1, 1])
        if save_col.button("保存资料", key="save_material", type="primary"):
            try:
                api.save_material(choice, st.session_state["material_content"], base_url, note=note)
            except api.ApiError as exc:
                st.error(str(exc))
            else:
                st.success("已保存（旧内容已存为历史版本）。索引需要重建才会生效。")
                st.rerun()
        if delete_col.button("删除资料", key="delete_material"):
            st.session_state["confirm_delete_material"] = True
        if st.session_state.get("confirm_delete_material"):
            st.warning(f"确认删除「{choice}」？删除前会自动留一份历史版本。")
            yes, no = st.columns(2)
            if yes.button("确认删除", key="confirm_delete_material_yes", type="primary"):
                try:
                    api.delete_material(choice, base_url)
                except api.ApiError as exc:
                    st.error(str(exc))
                else:
                    st.session_state.pop("confirm_delete_material", None)
                    st.session_state.pop("material_loaded", None)
                    st.success("已删除")
                    st.rerun()
            if no.button("取消", key="confirm_delete_material_no"):
                st.session_state.pop("confirm_delete_material", None)
                st.rerun()

        try:
            versions = api.list_material_versions(choice, base_url)
        except api.ApiError:
            versions = []
        with st.expander(f"历史版本（{len(versions)}）"):
            if not versions:
                st.caption("还没有历史版本。每次保存都会留下上一版。")
            for item in versions[:12]:
                row = st.columns([3, 1])
                row[0].caption(
                    f"{item['created_at']} · {item['size']} 字节"
                    + (f" · {item['note']}" if item.get("note") else "")
                )
                if row[1].button("恢复", key=f"restore_{item['version']}"):
                    try:
                        api.restore_material_version(choice, item["version"], base_url)
                    except api.ApiError as exc:
                        st.error(str(exc))
                    else:
                        st.session_state.pop("material_loaded", None)
                        st.success("已恢复为当前内容")
                        st.rerun()

    with st.expander("经历原子（真实性闸门）"):
        try:
            atom_payload = api.atoms(base_url)
        except api.ApiError as exc:
            st.caption(f"暂时无法读取原子库：{exc}")
        else:
            stats = atom_payload.get("stats", {})
            st.caption(
                f"共 {stats.get('total', 0)} 条（经历 {stats.get('experience', 0)} / "
                f"知识 {stats.get('knowledge', 0)}）· 已确认 {stats.get('approved', 0)} · "
                f"待确认 {stats.get('pending', 0)}"
            )
            st.caption(
                "原子是从索引语料里逐字切出来的原文片段；「知识」类（课程/面试笔记）只用于面试准备，"
                "不能写进简历说“我做过”。"
            )
            if st.button("从资料中抽取候选原子", key="extract_atoms"):
                try:
                    result = api.extract_atoms(base_url)
                except api.ApiError as exc:
                    st.error(str(exc))
                else:
                    st.success(f"新增 {result.get('added', 0)} 条，跳过重复 {result.get('skipped', 0)} 条")
                    st.rerun()
            bulk_col, kind_col = st.columns([2, 2])
            if bulk_col.button("确认全部「经历」原子", key="approve_all_experience"):
                try:
                    result = api.approve_atoms_bulk("experience", base_url)
                except api.ApiError as exc:
                    st.error(str(exc))
                else:
                    st.success(f"已确认 {result.get('approved', 0)} 条")
                    st.rerun()
            kind_filter = kind_col.selectbox(
                "只看", ["经历", "知识", "全部"], key="atom_kind_filter", label_visibility="collapsed"
            )
            wanted = {"经历": "experience", "知识": "knowledge", "全部": None}[kind_filter]
            pending = [
                atom
                for atom in atom_payload.get("atoms", [])
                if not atom.get("approved")
                and (wanted is None or atom.get("kind", "experience") == wanted)
            ]
            for atom in pending[:8]:
                row = st.columns([4, 1, 1])
                row[0].caption(f"{atom['project']}：{atom['span'][:90]}")
                if row[1].button("确认", key=f"approve_{atom['id']}"):
                    try:
                        api.set_atom_approved(atom["id"], True, base_url)
                    except api.ApiError as exc:
                        st.error(str(exc))
                    else:
                        st.rerun()
                if row[2].button("删除", key=f"drop_{atom['id']}"):
                    try:
                        api.delete_atom(atom["id"], base_url)
                    except api.ApiError as exc:
                        st.error(str(exc))
                    else:
                        st.rerun()

    with st.expander("事实校验（易过期的数字）"):
        try:
            fact_list = api.facts(base_url)
        except api.ApiError as exc:
            st.caption(f"暂时无法读取事实台账：{exc}")
        else:
            st.dataframe(
                [
                    {"事实": item["label"], "当前值": item["expected"], "来源": item["source"]}
                    for item in fact_list
                ],
                width="stretch",
                hide_index=True,
            )
        if st.button("校验资料与产物", key="fact_lint_btn"):
            try:
                report = api.facts_lint(base_url)
            except api.ApiError as exc:
                st.error(str(exc))
            else:
                issues = report.get("issues") or []
                if not issues:
                    st.success(f"已扫描 {report.get('scanned', 0)} 个文件，未发现过时数字。")
                else:
                    st.error(f"发现 {len(issues)} 处过时数字：")
                    st.dataframe(
                        [
                            {
                                "文件": item["origin"],
                                "行": item["line"],
                                "内容": item["text"],
                                "声称": item["claimed"],
                                "实际": item["expected"],
                            }
                            for item in issues
                        ],
                        width="stretch",
                        hide_index=True,
                    )

    with st.expander("新建资料 / 上传文件"):
        new_name = st.text_input("资料名（不含扩展名）", key="new_material_name")
        uploaded = st.file_uploader("上传 Markdown / 文本", type=["md", "txt"], key="new_material_file")
        new_content = st.text_area("正文", key="new_material_content", height=200)
        if st.button("创建资料", key="create_material"):
            payload = new_content
            if uploaded is not None and not payload.strip():
                payload = uploaded.getvalue().decode("utf-8", errors="replace")
            try:
                api.create_material(new_name, payload, base_url)
            except api.ApiError as exc:
                st.error(str(exc))
            else:
                st.success("已创建。记得重建索引后才会被检索到。")
                st.rerun()


def render_discovery_tab(base_url: str) -> None:
    """岗位发现：只接官方 ATS / 公开 API，链接必须落在来源域名白名单内。"""
    st.subheader("岗位发现")
    st.caption(
        "只调用招聘方官方 ATS（Greenhouse / Lever / Ashby）与公开 Job Board API，"
        "不抓网页；每条结果都带原始链接，域名不在来源白名单的会被丢弃。"
    )
    try:
        sources = api.job_sources(base_url)
    except api.ApiError as exc:
        st.info(f"暂时无法读取来源：{exc}")
        return

    labels = {item["id"]: f"{item['name']}（{item['authority']}）" for item in sources}
    default_sources = [item["id"] for item in sources if not item["needs_board"]]
    chosen = st.multiselect(
        "来源", options=list(labels), default=default_sources, format_func=lambda key: labels[key], key="discovery_sources"
    )
    boards: dict[str, str] = {}
    board_sources = [item for item in sources if item["needs_board"] and item["id"] in chosen]
    if board_sources:
        cols = st.columns(len(board_sources))
        for column, item in zip(cols, board_sources):
            boards[item["id"]] = column.text_input(
                f"{item['name']} 的 board slug", key=f"board_{item['id']}", placeholder="例如 anthropic"
            )
    query = st.text_input("关键词", value="AI Agent 实习", key="discovery_query")
    if st.button("搜索岗位", key="discovery_search", type="primary"):
        try:
            with st.spinner("抓取中…"):
                result = api.discovery_search(
                    query, sources=chosen or None, boards={k: v for k, v in boards.items() if v}, base_url=base_url
                )
        except api.ApiError as exc:
            st.error(str(exc))
        else:
            st.success(f"抓取 {result.get('fetched', 0)} 条，按关键词保留 {result.get('kept', 0)} 条")
            for error in result.get("errors", []):
                st.warning(error)

    try:
        leads = api.discovery_leads(base_url)
    except api.ApiError as exc:
        st.info(f"暂时无法读取候选岗位：{exc}")
        return
    if not leads:
        st.caption("候选池为空。选好来源后点「搜索岗位」。")
        return

    st.dataframe(
        [
            {
                "标题": lead["title"],
                "公司": lead.get("company") or "—",
                "地点": lead.get("location") or "—",
                "匹配分": lead.get("score", 0),
                "链接": lead.get("url", ""),
            }
            for lead in leads
        ],
        width="stretch",
        hide_index=True,
        column_config={"链接": st.column_config.LinkColumn("链接", display_text="打开")},
    )
    options = {lead["id"]: f'#{lead["id"]} {lead["title"][:40]}（{lead.get("score", 0)}）' for lead in leads}
    picked = st.selectbox("选择一条处理", options=list(options), format_func=lambda key: options[key], key="lead_pick")
    import_col, ignore_col = st.columns(2)
    if import_col.button("导入岗位库", key="lead_import"):
        try:
            job = api.save_lead(picked, base_url)
        except api.ApiError as exc:
            st.error(str(exc))
        else:
            st.success(f'已导入岗位 #{job.get("id")}：{job.get("title")}（可去「岗位工作台」分析）')
            st.rerun()
    if ignore_col.button("忽略", key="lead_ignore"):
        try:
            api.ignore_lead(picked, base_url)
        except api.ApiError as exc:
            st.error(str(exc))
        else:
            st.rerun()


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
        sync_mode = mode.startswith("同步")

    analyze_tab, jobs_tab, discovery_tab, profile_tab = st.tabs(
        ["岗位分析", "岗位工作台", "岗位发现", "候选人资料"]
    )
    with jobs_tab:
        render_job_workbench(base_url)
    with discovery_tab:
        render_discovery_tab(base_url)
    with profile_tab:
        render_profile_tab(base_url)

    with analyze_tab:
        jd = st.text_area("岗位 JD", value=SAMPLE_JD, height=180)
        submitted = st.button("开始分析", type="primary", width="stretch")

    if not submitted:
        return
    if len(jd.strip()) < 10:
        st.error("JD 太短了（至少 10 个字符）")
        return

    if sync_mode:
        # 同步模式必须真的走 /analyze/sync：否则侧边栏这个单选项只是装饰，
        # 用户以为自己在调同步链路，实际仍在排队。
        try:
            with st.spinner("同步执行中（一次分析约十几秒到两分钟）…"):
                result = api.analyze_sync(jd.strip(), base_url)
        except api.ApiError as exc:
            st.error(str(exc))
            return
        render_result(result)
        render_saved_banner(result)
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
