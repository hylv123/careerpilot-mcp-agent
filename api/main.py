"""CareerPilot HTTP 服务入口。

接口：POST /analyze（202 + task_id）、GET /tasks/{task_id}、POST /analyze/sync、GET /health。
一次分析要 50~120 秒，超过网关与浏览器常见超时，所以主路径走异步；同步接口用于调试与 Redis 不可用时的降级。
"""
from __future__ import annotations

import os
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from loguru import logger

from api.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    TaskAccepted,
    TaskStatusResponse,
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

    # 先判幂等，再判限流。顺序不能反：
    #   命中幂等说明这份 JD 已经有任务了，它不该消耗新的配额，
    #   否则用户重复提交同一份 JD 会被自己的历史记录挡在门外。
    existing = store.try_reserve(fingerprint, task_id)
    if existing:
        logger.info("[api] 命中幂等，复用已有任务 {}", existing)
        return TaskAccepted(task_id=existing, status=STATUS_PENDING, reused=True)

    if not store.try_consume_quota(RATE_LIMIT_PER_MINUTE):
        # 被限流拒绝时必须退还刚才占的位，
        # 否则这个 JD 在 24 小时内都无法重新提交。
        store.release_reservation(fingerprint)
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
