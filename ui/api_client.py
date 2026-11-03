"""界面层访问 HTTP API 的唯一入口。

为什么单独一个模块而不是全写在 app.py 里：
    Streamlit 每次交互都会把脚本整体重跑一遍，"发请求 + 轮询 + 错误映射"
    和"画页面"混在一起就没法测。抽出来之后，可以用假的 session 做离线单测，
    不需要起 uvicorn、不需要 Redis、更不需要模型 Key。

为什么只用 requests 不用 httpx：
    界面层只发 3 个请求（提交 / 查询 / 同步分析），同步阻塞模型最简单直白；
    httpx 的异步能力在这里没有收益，反而多一层概念。
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Callable

import requests

DEFAULT_BASE_URL = os.getenv("CAREERPILOT_API_URL", "http://127.0.0.1:8000")

# 终态：到了这两个状态就不可能再变，轮询必须停。
TERMINAL_STATUSES = ("done", "failed")


class ApiError(RuntimeError):
    """调用 API 失败：网络不通、限流、任务过期、服务端 5xx。"""


@dataclass
class SubmitResult:
    """一次提交的结果。

    Attributes:
        task_id: 异步任务的编号；同步降级时为空串。
        reused: 是否命中幂等（同一份 JD 已有任务）。
        direct_result: 非空表示 API 走了同步降级，结果已经在手上。
    """

    task_id: str = ""
    reused: bool = False
    direct_result: dict[str, Any] | None = None


def _detail(resp: Any) -> str:
    """从错误响应里尽量取出一句人能看懂的话。"""
    try:
        body = resp.json()
    except ValueError:
        return str(getattr(resp, "text", ""))[:200]
    if isinstance(body, dict):
        return str(body.get("detail") or body)[:200]
    return str(body)[:200]


def _http(session: Any) -> Any:
    """返回可用的 HTTP 客户端。

    测试里传假 session（有 post/get 两个方法即可），生产里传 requests 模块本身。
    """
    return requests if session is None else session


def submit_analysis(
    jd: str,
    base_url: str = DEFAULT_BASE_URL,
    *,
    timeout: float = 15.0,
    session: Any = None,
) -> SubmitResult:
    """提交一次岗位分析。

    Args:
        jd: 岗位 JD 原文。
        base_url: API 根地址。
        timeout: 单次 HTTP 超时（秒）。
        session: 可选，替身用。

    Returns:
        202 时返回 task_id；200 时表示 Redis 不可用、API 已同步跑完。

    Raises:
        ApiError: 网络错误、限流、参数不合法、服务端错误。
    """
    http = _http(session)
    url = f"{base_url.rstrip('/')}/analyze"
    try:
        resp = http.post(url, json={"jd": jd}, timeout=timeout)
    except requests.RequestException as exc:
        raise ApiError(f"连不上 API（{url}）：{exc}") from exc

    if resp.status_code == 202:
        body = resp.json()
        return SubmitResult(task_id=str(body["task_id"]), reused=bool(body.get("reused")))
    if resp.status_code == 200:
        # 降级路径：Redis 不可用时 API 直接同步跑完并返回完整结果
        return SubmitResult(direct_result=resp.json())
    if resp.status_code == 429:
        raise ApiError("提交过于频繁（入口限流 20 次/分钟），稍后再试")
    if resp.status_code == 422:
        raise ApiError(f"请求不合法：{_detail(resp)}")
    raise ApiError(f"提交失败（HTTP {resp.status_code}）：{_detail(resp)}")


def fetch_task(
    task_id: str,
    base_url: str = DEFAULT_BASE_URL,
    *,
    timeout: float = 15.0,
    session: Any = None,
) -> dict[str, Any]:
    """查询一次任务状态。

    Raises:
        ApiError: 404 任务不存在/过期、503 存储不可用、网络错误。
    """
    http = _http(session)
    url = f"{base_url.rstrip('/')}/tasks/{task_id}"
    try:
        resp = http.get(url, timeout=timeout)
    except requests.RequestException as exc:
        raise ApiError(f"查询任务失败（{url}）：{exc}") from exc

    if resp.status_code == 404:
        raise ApiError("任务不存在或已过期（结果只保留 24 小时）")
    if resp.status_code == 503:
        raise ApiError(f"任务存储暂时不可用：{_detail(resp)}")
    if resp.status_code != 200:
        raise ApiError(f"查询失败（HTTP {resp.status_code}）：{_detail(resp)}")
    return resp.json()


def wait_for_result(
    task_id: str,
    base_url: str = DEFAULT_BASE_URL,
    *,
    poll_interval: float = 2.0,
    max_wait: float = 600.0,
    session: Any = None,
    on_update: Callable[[str, str], None] | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """轮询直到任务进入终态。

    Args:
        task_id: 任务编号。
        base_url: API 根地址。
        poll_interval: 两次查询之间的间隔（秒）。
        max_wait: 最长等待（秒），超了抛错而不是无限转圈。
        session: 可选，替身用。
        on_update: 每次拿到新状态时回调 (status, progress)。
        clock: 单调时钟，测试里可以注入假时钟。
        sleep: 睡眠函数，测试里可以注入空函数。

    Returns:
        终态的任务 payload（含 result）。

    Raises:
        ApiError: 查询失败或等待超时。
    """
    started = clock()
    while True:
        payload = fetch_task(task_id, base_url, session=session)
        status = str(payload.get("status", ""))
        if on_update is not None:
            on_update(status, str(payload.get("progress", "")))
        if status in TERMINAL_STATUSES:
            return payload
        if clock() - started > max_wait:
            raise ApiError(f"等待超时（{max_wait:.0f} 秒），任务仍处于 {status}")
        sleep(poll_interval)


def analyze_sync(
    jd: str,
    base_url: str = DEFAULT_BASE_URL,
    *,
    timeout: float = 300.0,
    session: Any = None,
) -> dict[str, Any]:
    """同步分析：直接拿完整结果，用于调试和没有 Redis 的场景。

    timeout 默认 300 秒，因为一次分析实测 50~120 秒，给足余量。
    """
    http = _http(session)
    url = f"{base_url.rstrip('/')}/analyze/sync"
    try:
        resp = http.post(url, json={"jd": jd}, timeout=timeout)
    except requests.RequestException as exc:
        raise ApiError(f"连不上 API（{url}）：{exc}") from exc

    if resp.status_code == 200:
        return resp.json()
    raise ApiError(f"分析失败（HTTP {resp.status_code}）：{_detail(resp)}")
