"""worker 消息分发的回归测试。

背景：`_handle` 的 finally 里会 `store.ack(...)`，而 ack 失败会抛
RedisUnavailable。原来这个调用写在 main 的 try 之外，异常会冒出 main
把常驻进程杀掉。`_drain` 把它降级成一条 warning：

    - 确认失败 → 消息留在 PEL，120s 后重认领（语义上就是"未完成"）
    - 进程继续跑，不影响后续消息

这组测试锁住三点：不抛、继续处理后续消息、退避一次。
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from tasks import worker
from tasks.store import RedisUnavailable, TaskMessage


def _msg(task_id: str = "task-x") -> TaskMessage:
    return TaskMessage(message_id=f"{task_id}-0", task_id=task_id, jd="jd")


class DrainTests(unittest.TestCase):
    def test_ack_failure_does_not_propagate(self) -> None:
        sleeps: list[float] = []
        with patch.object(worker, "_handle", side_effect=RedisUnavailable("socket timeout")):
            worker._drain([_msg()], sleep_fn=sleeps.append)  # 不抛即通过

        self.assertEqual(sleeps, [worker.RETRY_SLEEP_SECONDS])

    def test_continues_with_remaining_messages(self) -> None:
        calls: list[str] = []

        def fake_handle(message: TaskMessage) -> None:
            calls.append(message.task_id)
            if message.task_id == "bad":
                raise RedisUnavailable("ack failed")

        sleeps: list[float] = []
        with patch.object(worker, "_handle", side_effect=fake_handle):
            worker._drain([_msg("bad"), _msg("good")], sleep_fn=sleeps.append)

        self.assertEqual(calls, ["bad", "good"])      # 第二条仍然被处理
        self.assertEqual(len(sleeps), 1)              # 只退避一次

    def test_other_exceptions_still_propagate(self) -> None:
        # 只兜 RedisUnavailable：编程错误不该被吞掉，否则永远查不出来
        with patch.object(worker, "_handle", side_effect=ValueError("bug")):
            with self.assertRaises(ValueError):
                worker._drain([_msg()], sleep_fn=lambda _: None)


if __name__ == "__main__":
    unittest.main()
