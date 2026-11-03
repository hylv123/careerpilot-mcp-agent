"""日志初始化：由进程入口显式调用，不再在 import 时顺手写盘。

为什么要把这件事从 `agent/graph.py` 里搬出来：
    `logger.add()` 会在**导入那一刻**打开日志文件。目录只读、容器只挂只读卷、
    或测试跑在无权限的沙箱里时，`import agent.graph` 直接抛 PermissionError——
    一个和业务毫无关系的路径问题，把整个模块的导入炸掉。
    日志是"观测手段"，观测手段不该有能力让主流程起不来。

约定：
    - `setup_logging()` 可以在任何入口重复调用，只有第一次生效（幂等）；
    - 写不了文件时降级为 stderr，**不抛异常**；
    - 目录可用 `CAREERPILOT_LOG_DIR` 覆盖，角色可用 `CAREERPILOT_ROLE` 区分
      （api 与 worker 各写各的文件，避免轮转时互相抢句柄）。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from loguru import logger

DEFAULT_LOG_DIR = Path(__file__).resolve().parent.parent / "logs"

# "unset" = 还没配过；"file" = 已挂文件 sink；"stderr" = 已降级
_state: str = "unset"
_handler_id: int | None = None


def setup_logging(role: str | None = None) -> bool:
    """挂一个文件 sink（失败则降级为 stderr）。

    Args:
        role: 角色后缀，如 "worker"；默认读环境变量 CAREERPILOT_ROLE。

    Returns:
        True 表示已写入文件；False 表示降级为 stderr（进程仍可正常跑）。
    """
    global _state, _handler_id

    if _state != "unset":
        return _state == "file"

    role = role if role is not None else os.getenv("CAREERPILOT_ROLE", "")
    suffix = f"_{role}" if role else ""
    log_dir = Path(os.getenv("CAREERPILOT_LOG_DIR") or DEFAULT_LOG_DIR)

    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        _handler_id = logger.add(
            str(log_dir / f"careerpilot_{{time:YYYY-MM-DD}}{suffix}.log"),
            encoding="utf-8",
            rotation="10 MB",
            retention="7 days",
        )
        _state = "file"
        return True
    except Exception as exc:  # 只读卷、无权限、路径是文件……都归到这里
        _state = "stderr"
        print(
            f"[logging] 无法写入日志目录 {log_dir}：{type(exc).__name__}: {exc}；"
            "本次只输出到 stderr",
            file=sys.stderr,
        )
        return False


def reset_logging() -> None:
    """移除本模块挂的 sink 并复位状态，仅供测试使用。"""
    global _state, _handler_id
    if _handler_id is not None:
        try:
            logger.remove(_handler_id)
        except ValueError:  # 已经被移除过
            pass
    _handler_id = None
    _state = "unset"
