"""环境隔离自检（护栏）。

`tests/__init__.py` 会把 CAREERPILOT_LOG_DIR 指到临时目录，但那只在
"以包的方式导入"时生效。`python -m unittest discover -s tests`（不带 `-t .`）
会把测试模块当**顶层模块**导入，包初始化根本不会执行——日志于是又写回仓库
`logs/`，而且没有任何报错。

所以这里断言"隔离确实生效"：没生效就红，而不是悄悄脏了仓库。
正确命令：`python -m unittest discover -s tests -t .`
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import agent.logging_setup as logging_setup


class EnvironmentIsolationTests(unittest.TestCase):
    def test_log_dir_points_to_temp(self) -> None:
        configured = os.environ.get("CAREERPILOT_LOG_DIR", "")
        self.assertTrue(
            configured,
            "CAREERPILOT_LOG_DIR 未设置：说明 tests 包初始化没执行，"
            "请用 python -m unittest discover -s tests -t . 运行",
        )
        self.assertTrue(
            configured.startswith(tempfile.gettempdir()),
            f"日志目录不在临时目录里：{configured}",
        )
        self.assertNotEqual(
            Path(configured).resolve(),
            logging_setup.DEFAULT_LOG_DIR.resolve(),
            "日志目录仍指向仓库 logs/，测试会污染工作区",
        )

    def test_setup_logging_writes_into_temp_dir(self) -> None:
        logging_setup.reset_logging()
        try:
            self.assertTrue(logging_setup.setup_logging())
            log_dir = Path(os.environ["CAREERPILOT_LOG_DIR"])
            self.assertTrue(list(log_dir.glob("careerpilot_*.log")), f"{log_dir} 下没有日志文件")
        finally:
            logging_setup.reset_logging()


if __name__ == "__main__":
    unittest.main()
