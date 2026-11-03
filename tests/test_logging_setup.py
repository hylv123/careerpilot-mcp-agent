"""日志初始化的回归测试。

要锁住三件事：
  1. 正常目录：真的写文件，返回 True；
  2. 重复调用：幂等，不会挂第二个 sink；
  3. 目录不可写：**不抛异常**，降级 stderr，返回 False —— 这是本文件的核心。
外加一条进程级断言：日志目录不可用时，`import agent.graph` 依然成功。
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from loguru import logger

import agent.logging_setup as logging_setup

ROOT = Path(__file__).resolve().parent.parent


class LoggingSetupTests(unittest.TestCase):
    def setUp(self) -> None:
        logging_setup.reset_logging()
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        logging_setup.reset_logging()
        self.temp_dir.cleanup()

    def test_writes_file_into_configured_dir(self) -> None:
        log_dir = Path(self.temp_dir.name) / "logs"
        with patch.dict(os.environ, {"CAREERPILOT_LOG_DIR": str(log_dir), "CAREERPILOT_ROLE": ""}):
            self.assertTrue(logging_setup.setup_logging())

        logger.info("probe-message")
        logging_setup.reset_logging()  # 移除 sink 会 flush

        files = list(log_dir.glob("careerpilot_*.log"))
        self.assertEqual(len(files), 1, f"应只有一个日志文件，实际：{files}")
        self.assertIn("probe-message", files[0].read_text(encoding="utf-8"))

    def test_role_becomes_suffix(self) -> None:
        log_dir = Path(self.temp_dir.name) / "logs"
        with patch.dict(os.environ, {"CAREERPILOT_LOG_DIR": str(log_dir)}):
            logging_setup.setup_logging(role="worker")
        logger.info("x")
        logging_setup.reset_logging()

        names = [p.name for p in log_dir.glob("*.log")]
        self.assertTrue(any(name.endswith("_worker.log") for name in names), names)

    def test_second_call_is_idempotent(self) -> None:
        log_dir = Path(self.temp_dir.name) / "logs"
        with patch.dict(os.environ, {"CAREERPILOT_LOG_DIR": str(log_dir)}):
            self.assertTrue(logging_setup.setup_logging())
            handlers = len(logger._core.handlers)
            self.assertTrue(logging_setup.setup_logging())
            self.assertEqual(len(logger._core.handlers), handlers, "重复调用不应新增 sink")

    def test_unwritable_dir_degrades_without_raising(self) -> None:
        blocker = Path(self.temp_dir.name) / "a-file"
        blocker.write_text("x", encoding="utf-8")
        # 把"目录"指向一个普通文件下面：mkdir 会抛 NotADirectoryError
        with patch.dict(os.environ, {"CAREERPILOT_LOG_DIR": str(blocker / "logs")}):
            self.assertFalse(logging_setup.setup_logging())

        logger.info("still-works")  # 降级后日志仍然可用（走 stderr）

    def test_import_graph_survives_bad_log_dir(self) -> None:
        blocker = Path(self.temp_dir.name) / "blocker-file"
        blocker.write_text("x", encoding="utf-8")
        env = dict(os.environ)
        env["CAREERPILOT_LOG_DIR"] = str(blocker / "logs")

        result = subprocess.run(
            [sys.executable, "-c", "import agent.graph; print('import-ok')"],
            cwd=str(ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stderr[-800:])
        self.assertIn("import-ok", result.stdout)


if __name__ == "__main__":
    unittest.main()
