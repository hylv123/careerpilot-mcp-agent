"""worker 主循环测试：`main()` 的顺序、信号退出、Redis 故障退避。

不连真 Redis：用假 store 记录调用顺序，把 `_running` 在拿完一批消息后置 False，
让循环跑一轮就退出——这样测的是**主循环的编排**，不是 Redis 本身
（Redis 语义由 tests/test_store.py 覆盖）。
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from tasks import worker
from tasks.store import RedisUnavailable, TaskMessage


class _FakeStore:
    """记录调用顺序的假 store。

    `stop_after` 控制第几轮循环后把 `worker._running` 置 False ——
    主循环是 `while _running`，不给出口测试就会挂住（这条注释是踩过之后补的）。
    计数放在 `reclaim_stale` 里，因为它是每轮第一个被调用的方法。
    """

    def __init__(self, *, reclaim=None, claim=None, fail_first_claim=False, stop_after=1) -> None:
        self.calls: list[str] = []
        self._reclaim = reclaim if reclaim is not None else []
        self._claim = claim if claim is not None else []
        self._fail_first = fail_first_claim
        self._stop_after = stop_after
        self.iterations = 0
        self.acked: list[str] = []
        self.statuses: list[tuple[str, str]] = []

    def ensure_group(self) -> None:
        self.calls.append("ensure_group")

    def reclaim_stale(self, consumer, idle_ms=0, count=1):
        self.calls.append("reclaim_stale")
        self.iterations += 1
        if self.iterations >= self._stop_after:
            worker._running = False
        return list(self._reclaim)

    def claim(self, consumer, count=1, block_ms=0):
        self.calls.append("claim")
        if self._fail_first:
            self._fail_first = False
            raise RedisUnavailable("conn reset")
        return list(self._claim)

    def set_status(self, task_id, status, **kwargs) -> None:
        self.statuses.append((task_id, status))

    def ack(self, message_id) -> None:
        self.acked.append(message_id)

    def release_reservation(self, fingerprint) -> None:
        self.calls.append("release_reservation")


def _msg(task_id="task-1"):
    return TaskMessage(message_id=f"{task_id}-0", task_id=task_id, jd="jd")


class WorkerMainTests(unittest.TestCase):
    def setUp(self) -> None:
        worker._running = True

    def tearDown(self) -> None:
        worker._running = True

    def test_happy_path_order_and_ack(self) -> None:
        store = _FakeStore(claim=[_msg()])
        with patch.object(worker, "get_store", return_value=store), \
             patch.object(worker, "run_analysis", return_value={"saved": True}), \
             patch.object(worker, "setup_logging"):
            code = worker.main()

        self.assertEqual(code, 0)
        # 先回收孤儿、再领新任务；顺序反了会饿死滞留任务
        self.assertEqual(store.calls[:3], ["ensure_group", "reclaim_stale", "claim"])
        self.assertEqual(store.statuses, [("task-1", "running"), ("task-1", "done")])
        self.assertEqual(store.acked, ["task-1-0"])

    def test_reclaims_before_claiming(self) -> None:
        store = _FakeStore(reclaim=[_msg("orphan")], claim=[])
        with patch.object(worker, "get_store", return_value=store), \
             patch.object(worker, "run_analysis", return_value={}), \
             patch.object(worker, "setup_logging"):
            worker.main()

        # 有孤儿时不应再去 claim
        self.assertEqual(store.calls, ["ensure_group", "reclaim_stale"])
        self.assertEqual(store.acked, ["orphan-0"])

    def test_redis_unavailable_backs_off_then_continues(self) -> None:
        store = _FakeStore(claim=[_msg()], fail_first_claim=True, stop_after=2)
        sleeps: list[float] = []
        with patch.object(worker, "get_store", return_value=store), \
             patch.object(worker, "run_analysis", return_value={}), \
             patch.object(worker, "sleep", side_effect=sleeps.append), \
             patch.object(worker, "setup_logging"):
            code = worker.main()

        self.assertEqual(code, 0)                       # 故障不会让 main 退出
        self.assertEqual(len(sleeps), 1)                # 退避一次后重试
        self.assertEqual(store.calls.count("claim"), 2)  # 失败一次、成功一次
        self.assertEqual(store.acked, ["task-1-0"])

    def test_failed_task_marks_failed_and_releases_reservation(self) -> None:
        store = _FakeStore(claim=[_msg()])
        with patch.object(worker, "get_store", return_value=store), \
             patch.object(worker, "run_analysis", side_effect=RuntimeError("model down")), \
             patch.object(worker, "setup_logging"):
            worker.main()

        self.assertEqual(store.statuses, [("task-1", "running"), ("task-1", "failed")])
        self.assertIn("release_reservation", store.calls)
        self.assertEqual(store.acked, ["task-1-0"], "失败的任务也必须 ACK，否则会被反复重做")


if __name__ == "__main__":
    unittest.main()
