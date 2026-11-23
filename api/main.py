"""CareerPilot HTTP 服务入口。

接口：POST /analyze（202 + task_id）、GET /tasks/{task_id}、POST /analyze/sync、GET /health。
一次分析要十几秒到两分钟，可能超过网关与浏览器常见超时，所以主路径走异步；同步接口用于调试与 Redis 不可用时的降级。
"""
from __future__ import annotations

import os
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import JSONResponse
from loguru import logger

from api.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    JobCreateRequest,
    JobUpdateRequest,
    AtomBulkRequest,
    AtomUpdateRequest,
    DiscoverySearchRequest,
    MaterialCreateRequest,
    MaterialUpdateRequest,
    TaskAccepted,
    TaskStatusResponse,
)
from services import atoms as atoms_service
from services import compare
from services import discovery
from services import facts as facts_service
from services import profile
from services.jobs import (
    ALLOWED_STATUSES,
    create_job,
    delete_job,
    get_job,
    list_jobs,
    update_job,
)
from agent.logging_setup import setup_logging
from tasks.runner import run_analysis
from tasks.store import (
    STATUS_PENDING,
    RedisUnavailable,
    get_store,
    jd_fingerprint,
)

app = FastAPI(
    title="CareerPilot Agent API",
    description="输入岗位 JD，输出结构化匹配分析、简历要点与面试题",
    version="0.2.0",
)

# 进程入口显式初始化日志（原来在 agent/graph.py 的导入期做，只读目录会炸）
setup_logging()

# 每分钟最多受理多少个新任务。
# 限的是"入口"而不是"worker 能力"：队列里的任务都是已经答应了用户要做的，
# 做不完比直接拒绝更难看，所以宁可在入口就挡掉。
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "20"))


@app.get("/health")
def health() -> dict[str, str]:
    """健康检查接口。

    为什么需要它：部署之后，负载均衡、容器 healthcheck、监控探活
    都需要一个不带参数、能立刻返回的接口。这是工程化的最小标志。

    注意它【不】检查 Redis：Redis 挂了 API 仍然可用（会走同步降级）。
    探活失败会让容器被判定为不健康、反复重启，反而把降级能力也弄没了。
    """
    return {"status": "ok"}


def _submit(req: AnalyzeRequest) -> TaskAccepted:
    """把一个分析任务放进队列。

    Args:
        req: 已通过 Pydantic 校验的请求体。

    Returns:
        任务受理回执。

    Raises:
        RedisUnavailable: Redis 不可用，由调用方决定降级。
        HTTPException: 429，触发限流。
    """
    store = get_store()
    store.ensure_group()

    fingerprint = jd_fingerprint(req.jd)
    task_id = f"task-{uuid4().hex}"

    # 先判幂等，再判限流，而且这两步在同一个 Lua 脚本里完成：
    #   1. 命中幂等说明这份 JD 已经有任务了，它不该消耗新的配额，
    #      否则用户重复提交同一份 JD 会被自己的历史记录挡在门外；
    #   2. 拆成两次调用的话，"占位"和"限流拒绝后释放"之间存在窗口，
    #      释放动作可能删掉别人刚抢到的键。合并成原子操作后，
    #      被限流时根本不会产生占位，也就不存在"退还"。
    existing, accepted = store.try_reserve_with_quota(
        fingerprint, task_id, limit=RATE_LIMIT_PER_MINUTE
    )
    if existing:
        logger.info("[api] 命中幂等，复用已有任务 {}", existing)
        return TaskAccepted(task_id=existing, status=STATUS_PENDING, reused=True)

    if not accepted:
        raise HTTPException(status_code=429, detail="当前提交过于频繁，请稍后再试")

    store.enqueue(task_id, req.jd)
    store.set_status(task_id, STATUS_PENDING, progress="已排队，等待 worker 领取")
    logger.info("[api] 任务已受理 task_id={}", task_id)
    return TaskAccepted(task_id=task_id, status=STATUS_PENDING, reused=False)


@app.post("/analyze", response_model=TaskAccepted, status_code=202)
def analyze(req: AnalyzeRequest) -> TaskAccepted | JSONResponse:
    """受理一个岗位分析任务。

    正常路径：返回 202 + task_id，客户端用 GET /tasks/{task_id} 轮询结果。
    降级路径：Redis 不可用时直接同步执行，返回 200 + 完整结果，
              并在响应头 X-Analysis-Mode 标明走了降级——
              客户端既能拿到结果，监控上也能看出 Redis 出问题了。

    Args:
        req: 已经过 Pydantic 校验的请求体。

    Returns:
        TaskAccepted（202）或降级时的完整结果（200）。
    """
    try:
        return _submit(req)
    except RedisUnavailable as exc:
        # 为什么不直接抛 503：
        #   Redis 只是"任务调度用"，不是唯一数据源。它挂了最坏的结果是"慢"，
        #   不该是"不可用"—— 同步跑一次仍然能给出正确答案。
        logger.warning("[api] Redis 不可用，本次降级为同步执行：{}", exc)
        payload = run_analysis(req.jd)
        return JSONResponse(
            status_code=200,
            content=payload,
            headers={"X-Analysis-Mode": "sync-fallback"},
        )


@app.get("/tasks/{task_id}", response_model=TaskStatusResponse)
def get_task(task_id: str) -> TaskStatusResponse:
    """查询任务状态与结果。

    Args:
        task_id: 受理时返回的编号。

    Returns:
        任务状态；status 为 done 时 result 里有完整结果。

    Raises:
        HTTPException: 404 任务不存在或已过期；503 存储不可用。
    """
    try:
        payload = get_store().get_status(task_id)
    except RedisUnavailable as exc:
        raise HTTPException(
            status_code=503, detail=f"任务存储暂时不可用：{exc}"
        ) from exc

    if payload is None:
        # 结果只保留 24 小时，过期等同不存在。
        # 错误信息里说明这一点，免得用户以为系统把任务弄丢了。
        raise HTTPException(
            status_code=404, detail="任务不存在或已过期（结果保留 24 小时）"
        )
    return TaskStatusResponse(**payload)


@app.post("/analyze/sync", response_model=AnalyzeResponse)
def analyze_sync(req: AnalyzeRequest) -> AnalyzeResponse:
    """同步执行一次分析，直接返回完整结果。

    为什么保留它：
        1. 本地调试和冒烟测试不想先起 Redis；
        2. 它就是 Redis 不可用时的降级实现 ——
           降级不是另写一套，而是同一个 run_analysis 换个入口。
           这样"降级路径"永远和"正常路径"保持同样的正确性，
           不会出现"平时好好的、降级时结果不一样"这种最难查的 bug。

    Args:
        req: 已经过 Pydantic 校验的请求体。

    Returns:
        结构化分析结果。

    Raises:
        HTTPException: 上游大模型或 MCP 工具不可用时抛出 502。
    """
    try:
        return AnalyzeResponse(**run_analysis(req.jd))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"分析失败：{exc}") from exc


@app.get("/jobs")
def get_jobs(status: str | None = None) -> list[dict]:
    if status is not None and status not in ALLOWED_STATUSES:
        raise HTTPException(status_code=422, detail=f"无效状态：{status}")
    return list_jobs(status=status)


@app.post("/jobs", status_code=201)
def post_job(req: JobCreateRequest) -> dict:
    return create_job(**req.model_dump())


@app.get("/jobs/{job_id}")
def get_job_detail(job_id: int) -> dict:
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="岗位不存在")
    return job


@app.patch("/jobs/{job_id}")
def patch_job(job_id: int, req: JobUpdateRequest) -> dict:
    try:
        job = update_job(job_id, **req.model_dump(exclude_unset=True))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if job is None:
        raise HTTPException(status_code=404, detail="岗位不存在")
    return job


@app.delete("/jobs/{job_id}", status_code=204)
def remove_job(job_id: int) -> Response:
    """删除一条岗位记录。

    返回 204 而不是把被删对象回吐：调用方已经知道 id，回吐只会让"删成功"
    和"删了个不存在的"看起来一样。删不到就明确 404。
    """
    if not delete_job(job_id):
        raise HTTPException(status_code=404, detail="岗位不存在")
    return Response(status_code=204)


# ---------------------------------------------------------------- 候选人资料

@app.get("/profile/materials")
def get_materials() -> list[dict]:
    """列出 knowledge/docs/private 下的全部资料。"""
    return profile.list_materials()


@app.post("/profile/materials", status_code=201)
def post_material(req: MaterialCreateRequest) -> dict:
    try:
        return profile.save_material(req.name, req.content, note=req.note)
    except profile.ProfileError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/profile/materials/{name}")
def get_material(name: str) -> dict:
    try:
        return profile.read_material(name)
    except profile.ProfileError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.put("/profile/materials/{name}")
def put_material(name: str, req: MaterialUpdateRequest) -> dict:
    try:
        return profile.save_material(name, req.content, note=req.note)
    except profile.ProfileError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.delete("/profile/materials/{name}", status_code=204)
def remove_material(name: str) -> Response:
    try:
        profile.delete_material(name)
    except profile.ProfileError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return Response(status_code=204)


@app.get("/profile/materials/{name}/versions")
def get_material_versions(name: str) -> list[dict]:
    try:
        return profile.list_versions(name)
    except profile.ProfileError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/profile/materials/{name}/versions/{version}/restore")
def restore_material_version(name: str, version: str) -> dict:
    try:
        return profile.restore_version(name, version)
    except profile.ProfileError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/profile/index-status")
def get_index_status() -> dict:
    """索引是否落后于资料，界面用它决定要不要提示重建。"""
    return profile.index_status()


@app.post("/profile/reindex")
def post_reindex() -> dict:
    """重建知识库索引。失败时也返回 200 + ok=false，让界面能展示原因。"""
    try:
        result = profile.rebuild_index()
    except profile.ReindexBusy as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if result.get("ok"):
        # 索引变了，片段数这类事实跟着变；顺手校验一遍，别让过时数字混进材料
        facts_service.reset_cache()
        result["fact_issues"] = facts_service.lint_project()["issues"]
    return result


# ---------------------------------------------------------------- 事实台账

@app.get("/facts")
def get_facts() -> dict:
    """当前事实台账：易过期数字从代码/产物实时推导，不是手写常量。"""
    return {
        "facts": [
            {"id": fact.id, "label": fact.label, "expected": fact.expected, "source": fact.source}
            for fact in facts_service.derived_facts()
        ]
    }


@app.get("/facts/lint")
def get_facts_lint() -> dict:
    """把知识库资料、对外产物、README 里的数字与台账比对。"""
    return facts_service.lint_project()


# ---------------------------------------------------------------- 经历原子

@app.get("/profile/atoms")
def get_atoms(approved: bool | None = None) -> dict:
    """经历原子：默认返回全部，可按是否已确认过滤。"""
    return {"stats": atoms_service.stats(), "atoms": atoms_service.list_atoms(approved=approved)}


@app.post("/profile/atoms/extract")
def post_atoms_extract() -> dict:
    """从私有资料里切出候选原子（不改写，纯切分）。"""
    try:
        return atoms_service.extract_candidates()
    except atoms_service.AtomError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/profile/atoms/approve-bulk")
def post_atoms_approve_bulk(req: AtomBulkRequest) -> dict:
    """批量确认（例如"确认全部经历类原子"），逐条复核片段是否仍在原文里。"""
    changed = atoms_service.approve_bulk(kind=req.kind, source=req.source)
    return {"approved": changed, "stats": atoms_service.stats()}


@app.patch("/profile/atoms/{atom_id}")
def patch_atom(atom_id: str, req: AtomUpdateRequest) -> dict:
    """确认/撤回一条原子；确认前会再核对一次片段是否与原文一致。"""
    try:
        return atoms_service.set_approved(atom_id, req.approved)
    except atoms_service.AtomError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.delete("/profile/atoms/{atom_id}", status_code=204)
def delete_atom(atom_id: str) -> Response:
    try:
        atoms_service.delete_atom(atom_id)
    except atoms_service.AtomError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return Response(status_code=204)


# ---------------------------------------------------------------- 多岗位洞察

@app.get("/insights/clusters")
def get_insights_clusters() -> dict:
    """按技能集合聚类岗位，并用证据库算每簇的平均覆盖率与缺口。"""
    from knowledge.direct import all_chunks

    jobs = list_jobs()
    evidence_text = "\n".join(str(chunk.get("content", "")) for chunk in all_chunks())
    return compare.focus_report(jobs, evidence_text)


# ---------------------------------------------------------------- 岗位发现

@app.get("/discovery/sources")
def get_discovery_sources() -> dict:
    """可用的岗位来源（只含官方 ATS / 公开 API），带权威性说明。"""
    from services import job_sources

    return {"sources": job_sources.list_sources()}


@app.post("/discovery/search")
def post_discovery_search(req: DiscoverySearchRequest) -> dict:
    """抓取候选岗位并按证据覆盖率打分；抓取失败会逐来源报告原因。"""
    return discovery.search(
        req.query,
        sources=req.sources,
        boards=req.boards,
        limit_per_source=req.limit,
    )


@app.get("/discovery/leads")
def get_discovery_leads(status: str = "new", limit: int = 50) -> list[dict]:
    if status and status not in discovery.LEAD_STATUSES:
        raise HTTPException(status_code=422, detail=f"无效状态：{status}")
    return discovery.list_leads(status=status or None, limit=limit)


@app.post("/discovery/leads/{lead_id}/save")
def post_discovery_save(lead_id: int) -> dict:
    job = discovery.save_lead_to_jobs(lead_id)
    if job is None:
        raise HTTPException(status_code=404, detail="候选岗位不存在")
    return job


@app.post("/discovery/leads/{lead_id}/ignore")
def post_discovery_ignore(lead_id: int) -> dict:
    if not discovery.ignore_lead(lead_id):
        raise HTTPException(status_code=404, detail="候选岗位不存在")
    return {"ignored": lead_id}
