"""Redis 任务存储层。

为什么要有这一层：
    Redis 在本项目里承担四种角色（任务队列 / 状态存储 / 幂等去重 / 并发限流）。
    把它们全部关在一个文件里，业务代码（API 与 worker）永远不直接碰
    redis-py 的原始命令。好处有三个：
      1. key 的命名规则只有一处定义，不会东写一个西写一个；
      2. 将来要做"Redis 不可用时的降级"，只需要改这一个入口；
      3. 单元测试可以 mock 掉这一整层，不用真起 Redis。

为什么【不】用它做热点缓存：
    这是一个单次请求要跑 50~120 秒的重计算服务，QPS 极低，缓存命中率趋近于 0。
    硬加缓存，面试官问一句"命中率多少"就穿帮。
    Redis 在这里的价值是【削峰、异步、幂等、限流】，不是加速读取。

为什么用 Stream 而不是 List：
    List 的 BRPOP 一旦把任务取出来，Redis 里就再没有这条记录了 ——
    worker 崩溃，任务永久丢失，而且没人知道它曾经存在过。
    Stream 有消费组和 ACK：任务投递后留在 PEL（待确认列表）里，
    worker 跑完必须显式 XACK；若它挂了，用 XPENDING 能查出
    "哪些任务投递了但没人应答"，再用 XCLAIM 转给别人重做。
    这正是"任务丢了怎么办"的标准答案。
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from typing import Any

import redis
from redis.exceptions import RedisError

# ---------- key 命名 ----------
# 统一前缀：将来一个 Redis 上跑多个环境时，可以按前缀整体清理，
# 也不会和别人的 key 撞车。
KEY_PREFIX = "careerpilot"
QUEUE_KEY = f"{KEY_PREFIX}:queue"        # Stream：任务队列
TASK_KEY = f"{KEY_PREFIX}:task:{{}}"     # Hash：任务状态与结果
DEDUP_KEY = f"{KEY_PREFIX}:dedup:{{}}"   # String：JD 指纹 → task_id
RATE_KEY = f"{KEY_PREFIX}:rate:{{}}"     # String：限流计数
CONSUMER_GROUP = "analyzers"             # 消费组名，所有 worker 共享

# ---------- 任务状态 ----------
STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_FAILED = "failed"

# ---------- 默认时间 ----------
# 结果保留 24 小时。为什么必须设 TTL：
#   每条结果里都有一份完整的报告文本，几百条就能吃掉几百 MB。
#   它是"可重算的临时物"，不是"唯一副本"，到期就让它自己消失。
DEFAULT_RESULT_TTL = 24 * 3600
DEFAULT_DEDUP_TTL = 24 * 3600
DEFAULT_RATE_WINDOW = 60


class RedisUnavailable(RuntimeError):
    """Redis 不可用。

    由 API 层捕获，捕获后退回"同步执行"模式，保证服务不整体瘫痪。
    把它做成独立的异常类型，是为了让调用方能明确区分
    "Redis 挂了（可以降级）"和"业务代码出错（不该重试）"。
    """


@dataclass(frozen=True)
class TaskMessage:
    """从队列里取出的一条任务。

    Attributes:
        message_id: Stream 的消息 ID。必须靠它做 XACK，
            没有它就无法确认这条任务被处理完了。
        task_id: 业务侧的任务编号，用于查状态。
        jd: 岗位 JD 原文。
    """

    message_id: str
    task_id: str
    jd: str


def jd_fingerprint(jd: str) -> str:
    """计算 JD 的指纹，用于幂等去重。

    Args:
        jd: 岗位 JD 原文。

    Returns:
        十六进制的 sha256 摘要。

    为什么要 strip 之后再算：
        用户从网页上复制 JD，前后常常带一堆空格和换行。
        直接算哈希的话，"同一份 JD"会因为不可见字符不同而算成两个任务，
        幂等就失效了。
    """
    return hashlib.sha256(jd.strip().encode("utf-8")).hexdigest()


class TaskStore:
    """Redis 任务存储的统一入口。

    Examples:
        store = TaskStore.from_env()
        store.ensure_group()
        store.enqueue("task-1", "招聘 AI 应用开发实习生...")
        for msg in store.claim("worker-1"):
            store.set_status(msg.task_id, "running")
            store.ack(msg.message_id)
    """

    def __init__(
        self,
        client: redis.Redis,
        *,
        result_ttl: int = DEFAULT_RESULT_TTL,
        dedup_ttl: int = DEFAULT_DEDUP_TTL,
    ) -> None:
        """初始化。

        Args:
            client: 已经连好的 redis 客户端（decode_responses=True）。
            result_ttl: 任务结果的保留秒数。
            dedup_ttl: 幂等标记的保留秒数。
        """
        self._client = client
        self._result_ttl = result_ttl
        self._dedup_ttl = dedup_ttl

    @classmethod
    def from_env(cls) -> "TaskStore":
        """从环境变量 REDIS_URL 建立连接。

        Returns:
            一个新的 TaskStore 实例。

        Raises:
            RedisUnavailable: 连接失败。
        """
        url = os.getenv("REDIS_URL", "redis://redis:6379/0")
        try:
            client = redis.Redis.from_url(
                url,
                decode_responses=True,
                socket_connect_timeout=3,
                socket_timeout=5,
            )
        except RedisError as exc:
            raise RedisUnavailable(f"无法创建 Redis 连接：{exc}") from exc
        return cls(client)

    # ---------------------------------------------------------------- 健康
    def ping(self) -> bool:
        """探活。

        Returns:
            Redis 正常返回 True。
        """
        try:
            return bool(self._client.ping())
        except RedisError:
            return False

    # ---------------------------------------------------------------- 队列
    def ensure_group(self) -> None:
        """创建消费组，已存在则忽略。

        MKSTREAM 的作用：队列还没被写入过时，顺手把 Stream 建出来，
        否则 XGROUP CREATE 会因为"key 不存在"直接报错。
        """
        try:
            self._client.xgroup_create(
                QUEUE_KEY, CONSUMER_GROUP, id="0", mkstream=True
            )
        except RedisError as exc:
            # BUSYGROUP 表示组已存在，这是幂等操作，不算错误
            if "BUSYGROUP" not in str(exc):
                raise RedisUnavailable(f"创建消费组失败：{exc}") from exc

    def enqueue(self, task_id: str, jd: str) -> str:
        """把一个任务放进队列。

        Args:
            task_id: 任务编号。
            jd: 岗位 JD 原文。

        Returns:
            Stream 返回的消息 ID。

        Raises:
            RedisUnavailable: 写入失败。
        """
        try:
            return self._client.xadd(QUEUE_KEY, {"task_id": task_id, "jd": jd})
        except RedisError as exc:
            raise RedisUnavailable(f"任务入队失败：{exc}") from exc

    def claim(
        self,
        consumer: str,
        *,
        count: int = 1,
        block_ms: int = 5000,
    ) -> list[TaskMessage]:
        """取一批新任务。

        Args:
            consumer: 消费者名字，通常是 worker 的主机名 + 进程号。
            count: 最多取几条。
            block_ms: 没有任务时阻塞等待多少毫秒。

        Returns:
            取到的任务列表；为空表示这一轮没等到。

        Raises:
            RedisUnavailable: 读取失败。

        用 `>` 这个特殊 ID 的含义：
            "只给我从来没投递给任何人的新消息"。
            已投递但未确认的（PEL 里的）不会被这里取走，
            它们只能靠 reclaim_stale 认领，避免重复消费。
        """
        try:
            result = self._client.xreadgroup(
                CONSUMER_GROUP,
                consumer,
                {QUEUE_KEY: ">"},
                count=count,
                block=block_ms,
            )
        except RedisError as exc:
            raise RedisUnavailable(f"读取任务失败：{exc}") from exc

        messages: list[TaskMessage] = []
        for _stream, entries in result or []:
            for message_id, fields in entries:
                messages.append(
                    TaskMessage(
                        message_id=message_id,
                        task_id=fields.get("task_id", ""),
                        jd=fields.get("jd", ""),
                    )
                )
        return messages

    def ack(self, message_id: str) -> None:
        """确认一条任务已处理完。

        Args:
            message_id: claim 时拿到的消息 ID。

        Raises:
            RedisUnavailable: 确认失败。
        """
        try:
            self._client.xack(QUEUE_KEY, CONSUMER_GROUP, message_id)
        except RedisError as exc:
            raise RedisUnavailable(f"确认任务失败：{exc}") from exc

    def reclaim_stale(
        self,
        consumer: str,
        *,
        idle_ms: int = 120_000,
        count: int = 10,
    ) -> list[TaskMessage]:
        """认领"投递过但长时间没确认"的任务。

        Args:
            consumer: 认领方名字。
            idle_ms: 闲置超过多少毫秒的任务才认领。
            count: 最多认领几条。

        Returns:
            认领到的任务列表。

        Raises:
            RedisUnavailable: 操作失败。

        这一条是 Stream 相对 List 的核心优势：
            worker 取了任务之后进程崩了，那条消息会一直躺在 PEL 里没人管。
            别的 worker 定期跑一次这个函数，就能把"孤儿任务"接过来重做，
            并且因为它还在 PEL 里，不会被 claim() 当成新任务再发一次。
        """
        try:
            _next, entries, *_ = self._client.xautoclaim(
                QUEUE_KEY,
                CONSUMER_GROUP,
                consumer,
                min_idle_time=idle_ms,
                start_id="0-0",
                count=count,
            )
        except RedisError as exc:
            raise RedisUnavailable(f"认领滞留任务失败：{exc}") from exc

        return [
            TaskMessage(
                message_id=message_id,
                task_id=fields.get("task_id", ""),
                jd=fields.get("jd", ""),
            )
            for message_id, fields in entries or []
        ]

    # ------------------------------------------------------------ 状态结果
    def set_status(
        self,
        task_id: str,
        status: str,
        *,
        progress: str = "",
        result: dict[str, Any] | None = None,
        error: str = "",
    ) -> None:
        """写入任务状态。

        Args:
            task_id: 任务编号。
            status: 见 STATUS_* 常量。
            progress: 人类可读的进度描述，给前端展示用。
            result: 完成时的完整结果字典。
            error: 失败时的错误信息。

        Raises:
            RedisUnavailable: 写入失败。

        为什么状态写入和 TTL 设置放在一个 pipeline 里：
            分两次发的话，写完状态、还没设 TTL 时进程崩了，
            这条记录就会永久留在 Redis 里（没有过期时间），
            成为永远不会被清理的垃圾。用 pipeline 保证两步原子完成。
        """
        mapping: dict[str, str] = {
            "status": status,
            "progress": progress,
            "error": error,
            "updated_at": str(int(time.time())),
        }
        if result is not None:
            mapping["result"] = json.dumps(result, ensure_ascii=False)

        key = TASK_KEY.format(task_id)
        try:
            pipe = self._client.pipeline()
            pipe.hset(key, mapping=mapping)
            pipe.expire(key, self._result_ttl)
            pipe.execute()
        except RedisError as exc:
            raise RedisUnavailable(f"写入任务状态失败：{exc}") from exc

    def get_status(self, task_id: str) -> dict[str, Any] | None:
        """读取任务状态。

        Args:
            task_id: 任务编号。

        Returns:
            状态字典，任务不存在或已过期返回 None。

        Raises:
            RedisUnavailable: 读取失败。
        """
        key = TASK_KEY.format(task_id)
        try:
            raw = self._client.hgetall(key)
        except RedisError as exc:
            raise RedisUnavailable(f"读取任务状态失败：{exc}") from exc

        if not raw:
            return None

        payload: dict[str, Any] = {
            "task_id": task_id,
            "status": raw.get("status", STATUS_PENDING),
            "progress": raw.get("progress", ""),
            "error": raw.get("error", ""),
            "updated_at": int(raw.get("updated_at", 0)),
        }
        if raw.get("result"):
            try:
                payload["result"] = json.loads(raw["result"])
            except json.JSONDecodeError:
                # 结果字段坏了不该让整个查询 500，如实降级成"结果不可用"
                payload["result"] = None
                payload["error"] = "结果反序列化失败"
        return payload

    # ---------------------------------------------------------------- 幂等
    def try_reserve(self, fingerprint: str, task_id: str) -> str | None:
        """抢占一个 JD 的执行权。

        Args:
            fingerprint: jd_fingerprint() 算出的指纹。
            task_id: 本次任务的编号。

        Returns:
            None 表示抢到了，本任务应当执行；
            否则返回【已存在的那个 task_id】，调用方应直接复用它。

        Raises:
            RedisUnavailable: 操作失败。

        为什么用 SET NX EX 一条命令而不是"先 GET 再 SET"：
            两步操作之间存在竞态 —— 两个请求同时 GET 到"没人占用"，
            就会双双认为自己是第一个，去重失效。
            SET 的 NX 选项由 Redis 单线程保证原子性。

        为什么这个项目里幂等是【正确性要求】而不是优化：
            一次分析要调 8~9 次大模型（花钱），而且有副作用 ——
            会通过 MCP 把岗位记录写进 SQLite。
            用户手抖点两次，不去重就会多一条重复记录。
        """
        key = DEDUP_KEY.format(fingerprint)
        try:
            ok = self._client.set(
                key, task_id, nx=True, ex=self._dedup_ttl
            )
        except RedisError as exc:
            raise RedisUnavailable(f"抢占任务失败：{exc}") from exc

        if ok:
            return None
        try:
            return self._client.get(key)
        except RedisError as exc:
            raise RedisUnavailable(f"读取已占用任务失败：{exc}") from exc

    def release_reservation(self, fingerprint: str) -> None:
        """释放执行权，让同一个 JD 可以被重新提交。

        Args:
            fingerprint: jd_fingerprint() 算出的指纹。

        Raises:
            RedisUnavailable: 操作失败。

        什么时候需要：任务最终失败时。如果不释放，
        用户在 24 小时内无法通过"再提交一次"来重试。
        """
        try:
            self._client.delete(DEDUP_KEY.format(fingerprint))
        except RedisError as exc:
            raise RedisUnavailable(f"释放任务占用失败：{exc}") from exc

    # ---------------------------------------------------------------- 限流
    def try_consume_quota(
        self,
        limit: int,
        *,
        window_seconds: int = DEFAULT_RATE_WINDOW,
    ) -> bool:
        """尝试消耗一个配额。

        Args:
            limit: 一个时间窗口内允许的最大任务数。
            window_seconds: 时间窗口长度（秒）。

        Returns:
            True 表示放行，False 表示本窗口配额已用完。

        Raises:
            RedisUnavailable: 操作失败。

        为什么用 INCR + EXPIRE 而不是 INCR + 永久 key：
            计数 key 若不过期，内存里会堆积无数个"上一分钟"的 key。
            过期时间比窗口多给 1 秒，是为了避免刚好跨窗口时
            新窗口的 key 还没建立就被清掉。

        为什么限流放在这里而不是网关：
            worker 数量有限、上游模型也有配额。入口不限的话，
            瞬间涌入的请求会把队列撑爆，而队列里的任务都是
            "已经答应了用户要做的"，做不完比直接拒绝更难看。
        """
        key = RATE_KEY.format(int(time.time()) // window_seconds)
        try:
            pipe = self._client.pipeline()
            pipe.incr(key)
            pipe.expire(key, window_seconds + 1)
            count = pipe.execute()[0]
        except RedisError as exc:
            raise RedisUnavailable(f"限流计数失败：{exc}") from exc

        return int(count) <= limit


_STORE: TaskStore | None = None


def get_store() -> TaskStore:
    """获取进程内共享的 TaskStore 实例。

    Returns:
        TaskStore 单例。

    为什么要单例：
        redis-py 的连接是池化的，每个请求 new 一个会不断建连，
        端口和内存都吃不消。进程内共享一份即可。
    """
    global _STORE
    if _STORE is None:
        _STORE = TaskStore.from_env()
    return _STORE
