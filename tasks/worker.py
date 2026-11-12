"""常驻任务消费者：从 Redis Stream 取任务并执行。

独立进程的原因：一次分析十几秒到两分钟，放在 API 进程里会占满请求线程；独立后可单独扩缩容。

启动方式：
    python -m tasks.worker
"""

from __future__ import annotations

import os
import signal
import socket
import sys
from pathlib import Path
from time import sleep

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 必须在 import agent.graph 之前设置：
#   日志文件是在 agent.graph 被 import 的那一刻就决定好的，
#   晚一步设置就还是和 API 抢同一个文件。
os.environ.setdefault("CAREERPILOT_ROLE", "worker")

from loguru import logger

from agent.logging_setup import setup_logging
from tasks.runner import run_analysis
from tasks.store import (
    STATUS_DONE,
    STATUS_FAILED,
    STATUS_RUNNING,
    RedisUnavailable,
    TaskMessage,
    get_store,
    jd_fingerprint,
)

# 认领滞留任务前要求的最小闲置时间：2 分钟。
# 设太短会误抢"正在正常执行"的任务（旧模型下单任务实测约 100 秒，新模型 10~20 秒，取最坏情况）。
STALE_IDLE_MS = int(os.getenv("WORKER_STALE_IDLE_MS", "120000"))
# 没任务时的阻塞等待时长。设短一点是为了让 SIGTERM 能被及时响应。
CLAIM_BLOCK_MS = int(os.getenv("WORKER_CLAIM_BLOCK_MS", "2000"))
RETRY_SLEEP_SECONDS = float(os.getenv("WORKER_RETRY_SLEEP", "5"))

_running = True


def _request_stop(signum: int, _frame: object) -> None:
    """收到停止信号时把循环标志置否，而不是直接 sys.exit。

    为什么要优雅退出：
        直接退出的话，正在执行的那条任务会变成一个"取出但未确认"的孤儿，
        要等 2 分钟后才被别的 worker 认领 —— 用户白白多等两分钟。
        置标志位让当前任务跑完、ACK 之后再退出。
    """
    global _running
    logger.info("[worker] 收到信号 {}，当前任务处理完毕后退出", signum)
    _running = False


def _handle(message: TaskMessage) -> None:
    """处理单条任务。

    Args:
        message: 从队列取出的任务消息。
    """
    store = get_store()
    store.set_status(
        message.task_id, STATUS_RUNNING, progress="正在分析岗位并生成材料"
    )

    try:
        result = run_analysis(message.jd)
        store.set_status(
            message.task_id, STATUS_DONE, progress="已完成", result=result
        )
        logger.info("[worker] 任务完成 task_id={}", message.task_id)
    except Exception as exc:
        # 失败也要确认这条消息。否则它会一直躺在 PEL 里被反复认领，
        # 每次重试都要再烧一遍 8~9 次模型调用的钱。
        # 正确的做法是：如实记为 failed，由客户端看到后自己决定要不要重来。
        store.set_status(
            message.task_id,
            STATUS_FAILED,
            error=f"{type(exc).__name__}: {exc}",
        )
        # 释放幂等占用，否则用户在 24 小时内无法原样重新提交这个 JD。
        # 必须带上自己的 task_id：只删自己那把锁，别把别人重新抢到的删掉。
        store.release_reservation(jd_fingerprint(message.jd), message.task_id)
        logger.exception("[worker] 任务失败 task_id={}", message.task_id)
    finally:
        store.ack(message.message_id)


def _drain(messages: list[TaskMessage], *, sleep_fn=sleep) -> None:
    """逐条处理本轮取到的消息，把"确认失败"和"任务失败"分开对待。

    为什么要单独包一层：
        `_handle` 的 finally 里会 `store.ack(...)`，而 ack 失败会抛
        RedisUnavailable。原来 `for message in messages: _handle(message)`
        写在 main 的 try 之外，这个异常会一路冒出 main → 进程退出。
        可"没确认成功"不代表"任务失败"：消息还留在 PEL 里，120s 后会被
        别的 worker 用 XAUTOCLAIM 认领，语义本来就是"未完成"。
        为一个可恢复的确认失败杀掉常驻进程，代价是：崩溃 + 用户白等 +
        整条分析重做一遍（重复烧模型调用、重复写库）。

    为什么不在这里重试 ack：
        绝大多数失败是一次性 socket 超时，交给 PEL 的 120s 重认领更简单，
        也不需要在本函数里引入"重试几次、退避多久"的新参数。

    Args:
        messages: 本轮 claim/reclaim 到的消息。
        sleep_fn: 睡眠函数，测试里可以注入空实现。
    """
    for message in messages:
        try:
            _handle(message)
        except RedisUnavailable as exc:
            logger.warning(
                "[worker] 确认/回写失败，消息留在 PEL 等待重新认领：{}", exc
            )
            sleep_fn(RETRY_SLEEP_SECONDS)


def main() -> int:
    """进程入口。

    Returns:
        退出码，正常退出为 0。
    """
    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)

    # 角色（CAREERPILOT_ROLE=worker）已在文件顶部设好，这里按角色落不同的日志文件
    setup_logging()

    store = get_store()
    store.ensure_group()
    consumer = f"worker-{socket.gethostname()}-{os.getpid()}"
    logger.info("[worker] 已启动 consumer={}", consumer)

    while _running:
        try:
            # 先看有没有别人崩掉留下的孤儿任务，再去领新任务。
            # 顺序不能反：否则新任务不断进来，滞留任务永远轮不到。
            messages = store.reclaim_stale(
                consumer, idle_ms=STALE_IDLE_MS, count=1
            )
            if not messages:
                messages = store.claim(consumer, count=1, block_ms=CLAIM_BLOCK_MS)
        except RedisUnavailable as exc:
            logger.warning(
                "[worker] Redis 暂时不可用，{}s 后重试：{}", RETRY_SLEEP_SECONDS, exc
            )
            sleep(RETRY_SLEEP_SECONDS)
            continue

        _drain(messages)

    logger.info("[worker] 已优雅退出")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
