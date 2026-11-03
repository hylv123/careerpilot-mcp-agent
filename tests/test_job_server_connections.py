"""job_server 连接管理回归测试。

sqlite3.Connection 的 `with` 只做事务、**不关连接**。以前五个工具里有四个用
`with _conn() as conn:`，每调用一次泄漏一个连接。这里用"记录 close 的代理"
把行为钉死：每个工具用完必须 close。
"""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mcp_servers import job_server


class _CloseSpy:
    """转发到真连接，但记录 close 是否被调用。"""

    def __init__(self, conn: sqlite3.Connection) -> None:
        object.__setattr__(self, "_conn", conn)
        object.__setattr__(self, "closed", False)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_conn"), name)

    def execute(self, *args, **kwargs):
        return self._conn.execute(*args, **kwargs)

    def __enter__(self):
        self._conn.__enter__()
        return self

    def __exit__(self, *exc_info):
        return self._conn.__exit__(*exc_info)

    def close(self) -> None:
        object.__setattr__(self, "closed", True)
        self._conn.close()


class ConnectionLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "jobs.db"
        self.db_patch = patch.object(job_server, "DB", self.db_path)
        self.db_patch.start()

        self.spies: list[_CloseSpy] = []
        real_connect = sqlite3.connect
        self.connect_patch = patch.object(
            job_server.sqlite3, "connect", side_effect=lambda path, **kw: self._spy(real_connect(path, **kw))
        )
        self.connect_patch.start()

    def _spy(self, conn: sqlite3.Connection) -> _CloseSpy:
        spy = _CloseSpy(conn)
        self.spies.append(spy)
        return spy

    def tearDown(self) -> None:
        self.connect_patch.stop()
        self.db_patch.stop()
        self.temp_dir.cleanup()

    def test_every_tool_closes_its_connection(self) -> None:
        job_server.save_job("AI Intern", "Demo Co", "Python RAG")
        job_server.update_status(1, "面试")
        job_server.list_jobs()
        job_server.get_job(1)
        job_server.list_jobs_from_company("Demo Co")

        self.assertEqual(len(self.spies), 5)
        self.assertTrue(all(spy.closed for spy in self.spies), "有工具没有关闭连接")

    def test_connection_closed_even_when_statement_fails(self) -> None:
        # 传一个 sqlite 绑不出来的对象：异常发生在 _session 内部，
        # 用来自证"异常路径也会关连接"（否则临时库文件会一直被占用删不掉）。
        with self.assertRaises(sqlite3.Error):
            job_server.update_status(object(), "已投")

        self.assertEqual(len(self.spies), 1)
        self.assertTrue(self.spies[0].closed)

    def test_update_status_reads_rowcount_before_close(self) -> None:
        job_server.save_job("AI Intern", "Demo Co", "jd")
        self.assertIn("未找到岗位", job_server.update_status(9999, "已投"))
        self.assertIn("面试", job_server.update_status(1, "面试"))


if __name__ == "__main__":
    unittest.main()
