"""页面脚本冒烟测试：用 Streamlit 官方的 AppTest 无头执行 ui/app.py。

它验证的不是"好不好看"，而是两件更基础的事：
  1. 脚本能完整跑完，没有异常（语法/导入/API 变更都会在这里暴露）；
  2. 关键控件都在（JD 输入框、开始按钮、侧边栏配置）。

没装 streamlit 的环境自动跳过——CI 的 API 依赖里本来就没有它。
"""

from __future__ import annotations

import unittest
from pathlib import Path

try:
    from streamlit.testing.v1 import AppTest

    _HAS_STREAMLIT = True
except Exception:  # pragma: no cover - 没装界面依赖时走这里
    _HAS_STREAMLIT = False

APP_PATH = Path(__file__).resolve().parent.parent / "ui" / "app.py"


@unittest.skipUnless(_HAS_STREAMLIT, "未安装 streamlit，跳过界面冒烟")
class StreamlitAppSmokeTests(unittest.TestCase):
    def test_script_runs_without_exception(self) -> None:
        app = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
        self.assertEqual(list(app.exception), [])

    def test_key_widgets_exist(self) -> None:
        app = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
        self.assertEqual(len(app.text_area), 1)
        self.assertEqual([button.label for button in app.button], ["开始分析"])
        self.assertEqual([box.label for box in app.sidebar.text_input], ["API 地址"])


if __name__ == "__main__":
    unittest.main()
