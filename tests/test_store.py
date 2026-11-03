"""`tasks/store.py` 的 Redis 层测试。

为什么这块最重要：幂等抢占、限流窗口、PEL 认领、状态 TTL 全在这 342 行里，
而它们直接决定"会不会重复扣费 / 会不会丢任务 / 会不会限错人"。

用真 Redis（不是 mock）：PEL、XAUTOCLAIM、SET NX 这些语义 mock 不出来，
mock 出来的"通过"没有意义。默认连本机 db 15，跑前会 FLUSHDB。

    python -m unittest tests.test_store -v

没有 Redis 时整组 skip（CI 里由 service 容器提供）。
"""

from __future__ import annotations

import json
import os
import unittest

import redis

from tasks.store import (
    CONSUMER_GROUP,
    DEDUP_KEY,
    QUEUE_KEY,
    TASK_KEY,
    STATUS_DONE,
    RedisUnavailable,
    TaskStore,
    jd_fingerprint,
)

TEST_REDIS_URL = os.getenv("CAREERPILOT_TEST_REDIS_URL", "redis://127.0.0.1:6379/15")


def _connect(url: str = TEST_REDIS_URL) -> redis.Redis:
    return redis.Redis.from_url(
        url, decode_responses=True, socket_connect_timeout=1, socket_timeout=2
    )


def _redis_available() -> bool:
    try:
        _connect().ping()
        return True
    except Exception:
        return False


REDIS_AVAILABLE = _redis_available()


@unittest.skipUnless(REDIS_AVAILABLE, "本机没有可用 Redis（默认 redis://127.0.0.1:6379/15）")
class StoreTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self.client = _connect()
        self.client.flushdb()      # 只清 db 15，不碰别的库
        self.store = TaskStore(self.client)

    def tearDown(self) -> None:
        self.client.flushdb()
        self.client.close()


class FingerprintTests(unittest.TestCase):
    def test_strip_makes_same_fingerprint(self) -> None:
        self.assertEqual(jd_fingerprint(" 招聘 RAG 实习生 \n"), jd_fingerprint("招聘 RAG 实习生"))

    def test_different_text_differs(self) -> None:
        self.assertNotEqual(jd_fingerprint("Java"), jd_fingerprint("Python"))

    def test_is_sha256_hex(self) -> None:
        fp = jd_fingerprint("jd")
        self.assertEqual(len(fp), 64)
        int(fp, 16)  # 非法十六进制会抛 ValueError


class ReserveTests(StoreTestBase):
    def test_first_reserve_wins_second_reuses(self) -> None:
        self.assertIsNone(self.store.try_reserve("fp1", "task-a"))
        self.assertEqual(self.store.try_reserve("fp1", "task-b"), "task-a")

    def test_release_allows_resubmit(self) -> None:
        self.store.try_reserve("fp1", "task-a")
        self.store.release_reservation("fp1")
        self.assertIsNone(self.store.try_reserve("fp1", "task-c"))

    def test_reservation_has_ttl(self) -> None:
        self.store.try_reserve("fp1", "task-a")
        ttl = self.client.ttl(DEDUP_KEY.format("fp1"))
        self.assertGreater(ttl, 0, "幂等标记必须有 TTL，否则永远占着")


class QuotaTests(StoreTestBase):
    def test_three_calls_with_limit_two(self) -> None:
        self.assertTrue(self.store.try_consume_quota(2))
        self.assertTrue(self.store.try_consume_quota(2))
        self.assertFalse(self.store.try_consume_quota(2))

    def test_counter_has_ttl(self) -> None:
        self.store.try_consume_quota(5)
        keys = [k for k in self.client.keys("careerpilot:rate:*")]
        self.assertEqual(len(keys), 1)
        self.assertGreater(self.client.ttl(keys[0]), 0)


class QueueTests(StoreTestBase):
    def test_enqueue_then_claim(self) -> None:
        self.store.ensure_group()
        message_id = self.store.enqueue("task-1", "招聘 RAG")
        self.assertTrue(message_id)

        messages = self.store.claim("worker-1", count=1, block_ms=100)
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].task_id, "task-1")
        self.assertEqual(messages[0].jd, "招聘 RAG")

    def test_claim_is_exclusive_after_delivery(self) -> None:
        self.store.ensure_group()
        self.store.enqueue("task-1", "jd")
        self.assertEqual(len(self.store.claim("worker-1", block_ms=50)), 1)
        # 同一 consumer 再读 ">"：已投递但未 ack 的不会被再发一次
        self.assertEqual(self.store.claim("worker-1", block_ms=50), [])

    def test_ack_removes_from_pending(self) -> None:
        self.store.ensure_group()
        self.store.enqueue("task-1", "jd")
        message = self.store.claim("worker-1", block_ms=100)[0]
        self.store.ack(message.message_id)

        pending = self.client.xpending(QUEUE_KEY, CONSUMER_GROUP)["pending"]
        self.assertEqual(pending, 0)

    def test_reclaim_takes_over_unacked_message(self) -> None:
        # 模拟 worker 崩溃：取走但不 ack，另一个 worker 用 XAUTOCLAIM 接管
        self.store.ensure_group()
        self.store.enqueue("task-1", "jd")
        first = self.store.claim("worker-crashed", block_ms=100)[0]

        reclaimed = self.store.reclaim_stale("worker-alive", idle_ms=0, count=1)
        self.assertEqual([m.task_id for m in reclaimed], ["task-1"])
        self.assertEqual(reclaimed[0].message_id, first.message_id)

    def test_ensure_group_is_idempotent(self) -> None:
        self.store.ensure_group()
        self.store.ensure_group()  # BUSYGROUP 不应被当成错误


class StatusTests(StoreTestBase):
    def test_roundtrip_done_with_result(self) -> None:
        payload = {"status": "done", "analysis": {"job_title": "AI 实习生"}, "saved": True}
        self.store.set_status("task-1", STATUS_DONE, progress="已完成", result=payload)

        got = self.store.get_status("task-1")
        self.assertEqual(got["status"], STATUS_DONE)
        self.assertEqual(got["progress"], "已完成")
        self.assertEqual(got["result"]["analysis"]["job_title"], "AI 实习生")
        self.assertIs(got["result"]["saved"], True)
        self.assertGreater(got["updated_at"], 0)

    def test_status_has_ttl(self) -> None:
        self.store.set_status("task-1", STATUS_DONE)
        self.assertGreater(self.client.ttl(TASK_KEY.format("task-1")), 0)

    def test_missing_task_returns_none(self) -> None:
        self.assertIsNone(self.store.get_status("task-not-exist"))

    def test_broken_result_json_degrades(self) -> None:
        self.client.hset(
            TASK_KEY.format("task-1"),
            mapping={"status": STATUS_DONE, "result": "{not json", "updated_at": "1"},
        )
        got = self.store.get_status("task-1")
        self.assertIsNone(got["result"])
        self.assertIn("反序列化失败", got["error"])

    def test_chinese_is_not_escaped(self) -> None:
        self.store.set_status("task-1", STATUS_DONE, result={"report_text": "中文报告"})
        raw = self.client.hget(TASK_KEY.format("task-1"), "result")
        self.assertIn("中文报告", raw)
        self.assertNotIn("\\u", raw)


class UnavailableTests(unittest.TestCase):
    def test_bad_endpoint_raises_redis_unavailable(self) -> None:
        # 这里不连真 Redis：端口 1 必然连不上，验证异常被翻译成 RedisUnavailable
        client = redis.Redis.from_url(
            "redis://127.0.0.1:1/0", decode_responses=True,
            socket_connect_timeout=0.2, socket_timeout=0.2,
        )
        store = TaskStore(client)
        with self.assertRaises(RedisUnavailable):
            store.ensure_group()
        with self.assertRaises(RedisUnavailable):
            store.get_status("x")


if __name__ == "__main__":
    unittest.main()
