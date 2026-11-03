"""Automated tests for public, offline project checks.

同时负责测试期环境隔离：把日志目录指到临时目录。

为什么必须在这里做：
    `api/main.py` 在模块级调用 `setup_logging()`，默认写到仓库的 `logs/`。
    测试一导入 `api.main` 就往真实日志目录写文件——跑一次测试脏一次仓库，
    还会和正在运行的 api/worker 抢同一个文件（Windows 上轮转会 PermissionError）。

    `unittest discover` 会先导入本包再导入各测试模块，所以这里改环境变量
    早于任何 `setup_logging()` 调用；手工 `python -m unittest tests.xxx` 同样覆盖。
"""

from __future__ import annotations

import atexit
import os
import shutil
import tempfile

_TEST_LOG_DIR = tempfile.mkdtemp(prefix="careerpilot-test-logs-")
os.environ["CAREERPILOT_LOG_DIR"] = _TEST_LOG_DIR

atexit.register(shutil.rmtree, _TEST_LOG_DIR, True)
