"""任务存储层。

对外只暴露 TaskStore 以及与任务相关的数据结构，业务代码从这里 import 即可，
不需要知道底层用的是 Redis 的哪种数据结构。
"""

from .store import (
    CONSUMER_GROUP,
    DEFAULT_DEDUP_TTL,
    DEFAULT_RATE_WINDOW,
    DEFAULT_RESULT_TTL,
    QUEUE_KEY,
    STATUS_DONE,
    STATUS_FAILED,
    STATUS_PENDING,
    STATUS_RUNNING,
    RedisUnavailable,
    TaskMessage,
    TaskStore,
    get_store,
    jd_fingerprint,
)

__all__ = [
    "CONSUMER_GROUP",
    "DEFAULT_DEDUP_TTL",
    "DEFAULT_RATE_WINDOW",
    "DEFAULT_RESULT_TTL",
    "QUEUE_KEY",
    "STATUS_DONE",
    "STATUS_FAILED",
    "STATUS_PENDING",
    "STATUS_RUNNING",
    "RedisUnavailable",
    "TaskMessage",
    "TaskStore",
    "get_store",
    "jd_fingerprint",
]
