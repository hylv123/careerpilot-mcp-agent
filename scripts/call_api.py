"""调用本地 CareerPilot HTTP 接口做冒烟测试。

为什么需要这个脚本，而不是直接在 /docs 页面里测：

    Swagger UI 的输入框里直接粘贴多行 JD 会产生**非法 JSON**——
    JSON 字符串不允许出现裸控制字符（换行、制表符必须转义成 \\n、\\t），
    服务端会返回 422 "Invalid control character"。

    用脚本发请求时 json.dumps() 会自动完成转义，既方便又可靠。
    这也是接口测试的一般做法：别手工拼 JSON，交给标准库。

用法（先另开一个终端启动服务）：

    .venv\\Scripts\\python.exe -m uvicorn api.main:app --reload

然后：

    .venv\\Scripts\\python.exe scripts/call_api.py --jd-file examples/sample_jd.txt
    .venv\\Scripts\\python.exe scripts/call_api.py --jd "招聘 AI 应用开发实习生，要求 Python 和 RAG"
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_URL = "http://127.0.0.1:8000/analyze"
# 一次分析要跑完整条 Agent 链路（约 1-3 分钟），超时必须给足
DEFAULT_TIMEOUT = 300.0


def call_analyze(jd: str, url: str = DEFAULT_URL, timeout: float = DEFAULT_TIMEOUT) -> dict[str, Any]:
    """调用分析接口。

    Args:
        jd: 岗位 JD 原文，可以包含换行（由 json.dumps 负责转义）。
        url: 接口地址。
        timeout: 超时秒数。

    Returns:
        解析后的响应字典。

    Raises:
        urllib.error.HTTPError: 服务端返回 4xx / 5xx。
    """
    payload = json.dumps({"jd": jd}, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="调用 CareerPilot 分析接口")
    parser.add_argument("--jd", help="直接传入岗位描述")
    parser.add_argument("--jd-file", type=Path, help="从 UTF-8 文本文件读取岗位描述")
    parser.add_argument("--url", default=DEFAULT_URL, help="接口地址")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="超时秒数")
    args = parser.parse_args()

    jd = args.jd
    if args.jd_file is not None:
        jd = args.jd_file.read_text(encoding="utf-8").strip()
    if not jd:
        print("请通过 --jd 或 --jd-file 提供岗位描述")
        return 1

    try:
        result = call_analyze(jd, args.url, args.timeout)
    except urllib.error.HTTPError as exc:
        # 4xx / 5xx 时把服务端返回的 detail 原样打印，方便直接看错误原因
        body = exc.read().decode("utf-8", errors="replace")
        print(f"HTTP {exc.code}")
        print(body)
        return 1
    except urllib.error.URLError as exc:
        print(f"无法连接服务，请先启动：uvicorn api.main:app --reload  ({exc})")
        return 1

    print(json.dumps(result, ensure_ascii=False, indent=2)[:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
