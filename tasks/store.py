"""Redis 任务存储层：任务队列、状态、幂等去重与限流。

用 Stream + 消费组而不是 List：List 的 BRPOP 取出即删除，worker 崩溃会丢任务；
Stream 的未确认消息留在 PEL，可以用 XAUTOCLAIM 转给其他 worker 重做。
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

# ---------- Lua 脚本 ----------
# 为什么这些操作必须写成 Lua：
#   "查幂等 → 查限流 → 占位 → 计数"是四步，客户端分开调的话，任何两步之间
#   都可能有别的请求挤进来：两个请求同时读到"没人占位"就会双双执行；
#   一个请求释放占位时也可能把别人刚抢到的键删掉。
#   Redis 执行 Lua 期间是单线程的，外部看到的只有"全做完"或"全没做"。
_RELEASE_LUA = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
    return redis.call('DEL', KEYS[1])
end
return 0
"""

# 返回值的三种情况（用类型区分，避免再引入一个布尔值）：
#   字符串 —— 幂等命中，值是已有的 task_id，调用方直接复用
#   1      —— 抢占成功且扣到了配额
#   0      —— 配额用尽，本次被限流（且没有留下任何占位）
_RESERVE_WITH_QUOTA_LUA = """
local existing = redis.call('GET', KEYS[1])
if existing then
    return existing
end
local count = redis.call('INCR', KEYS[2])
if count == 1 then
    redis.call('EXPIRE', KEYS[2], tonumber(ARGV[3]))
end
if count > tonumber(ARGV[2]) then
    redis.call('DECR', KEYS[2])
    return 0
end
redis.call('SET', KEYS[1], ARGV[1], 'EX', tonumber(ARGV[4]))
return 1
"""


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

    def release_reservation(self, fingerprint: str, task_id: str) -> bool:
        """释放执行权，让同一个 JD 可以被重新提交。

        Args:
            fingerprint: jd_fingerprint() 算出的指纹。
            task_id: 认为自己持有这份占位的任务编号。

        Returns:
            True 表示确实释放了自己的占位；False 表示这个键已经不属于
            调用方（被别人重新抢占了，或者早已过期），本次什么都没删。

        Raises:
            RedisUnavailable: 操作失败。

        为什么必须带上 task_id、并且用 Lua 比较后再删：
            原实现是 `DELETE(key)`，不看值。设想这个时序——
                ① 请求 A 抢到 JD-x 的占位，进入限流判断；
                ② A 被限流拒绝，走到释放；
                ③ 在 A 执行 DELETE 之前，B 刚好重新抢到了 JD-x；
            这时 A 的 DELETE 会**把 B 刚写进去的键删掉**，B 的任务就此失去
            幂等保护，同一个 JD 能被重复提交、重复扣费、重复写库。
            比较 + 删除必须在同一个原子操作里完成，所以用 Lua。

        什么时候需要：任务最终失败时。如果不释放，
        用户在 24 小时内无法通过"再提交一次"来重试。
        """
        key = DEDUP_KEY.format(fingerprint)
        try:
            released = self._client.eval(_RELEASE_LUA, 1, key, task_id)
        except RedisError as exc:
            raise RedisUnavailable(f"释放任务占用失败：{exc}") from exc
        return bool(released)

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

    def try_reserve_with_quota(
        self,
        fingerprint: str,
        task_id: str,
        *,
        limit: int,
        window_seconds: int = DEFAULT_RATE_WINDOW,
    ) -> tuple[str | None, bool]:
        """原子完成"幂等判重 + 限流计数 + 占位"。

        Args:
            fingerprint: jd_fingerprint() 算出的指纹。
            task_id: 本次任务的编号，抢占成功后写进幂等键的值里。
            limit: 一个时间窗口内允许受理的最大任务数。
            window_seconds: 限流窗口长度（秒）。

        Returns:
            (existing_task_id, accepted)：
                ("task-xxx", ...) —— 幂等命中，复用已有任务，不消耗配额
                (None, True)       —— 抢占成功，调用方应入队
                (None, False)      —— 配额用尽，调用方应返回 429

        Raises:
            RedisUnavailable: 操作失败。

        为什么要把三步合成一个 Lua：
            分开调的时候有三个真实的窗口——
                ① 先查后占：两个请求同时读到"没人占位"，双双执行；
                ② 占了位再查限流：被拒后要退位，退位又可能是误删；
                ③ 退位和下一个请求的占位交错：把别人的键删掉。
            合成之后这些窗口全部消失：被限流时**根本没有产生占位**，
            也就不存在"退还"这个动作，"退还时误删"自然不可能发生。

        为什么限流拒绝时要把计数减回去：
            原来用 INCR 先加再比较，被拒的请求也会把计数顶上去，
            等于"被拒绝一次就更难通过一次"，窗口内的额度会被自己的
            重试吃掉。现在只有真正受理的任务才计数。
        """
        dedup_key = DEDUP_KEY.format(fingerprint)
        rate_key = RATE_KEY.format(int(time.time()) // window_seconds)
        try:
            raw = self._client.eval(
                _RESERVE_WITH_QUOTA_LUA,
                2,
                dedup_key,
                rate_key,
                task_id,
                limit,
                window_seconds + 1,
                self._dedup_ttl,
            )
        except RedisError as exc:
            raise RedisUnavailable(f"抢占任务失败：{exc}") from exc

        if isinstance(raw, str) and raw:
            return raw, False
        return None, raw == 1


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
