"""常驻任务消费者。

为什么它是一个独立的进程，而不是在 API 进程里开线程：
    1. 分析要跑 50~120 秒，是 CPU/IO 混合的重活。放在处理 HTTP 的进程里，
       几个并发就能把请求线程占满，连 /health 都响应不了。
    2. 独立进程可以单独扩缩容 —— 任务积压时多起几个 worker 容器即可，
       API 那边完全不用动。这就是"队列"带来的解耦价值。

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
# 设太短会误抢"正在正常执行"的任务（一个任务本来就要跑 100 秒）。
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
        # 释放幂等占用，否则用户在 24 小时内无法原样重新提交这个 JD
        store.release_reservation(jd_fingerprint(message.jd))
        logger.exception("[worker] 任务失败 task_id={}", message.task_id)
    finally:
        store.ack(message.message_id)


def main() -> int:
    """进程入口。

    Returns:
        退出码，正常退出为 0。
    """
    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)

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

        for message in messages:
            _handle(message)

    logger.info("[worker] 已优雅退出")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
