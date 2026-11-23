"""直读模式的"索引过期"告警测试。

背景：直读模式读的是 FAISS docstore，不是实时文件。改完 knowledge/docs
里的简历/项目描述却不重建索引时，产出里仍是旧事实——排查过一次
"数字明明改了、简历还是旧的"，根因就在这里。
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from knowledge import direct


def _make_tree(root: Path, *, doc_time: float, index_time: float) -> tuple[Path, Path]:
    docs = root / "docs"
    index = root / "index"
    docs.mkdir()
    index.mkdir()
    doc = docs / "resume.md"
    doc.write_text("旧内容", encoding="utf-8")
    os.utime(doc, (doc_time, doc_time))
    for name in ("index.faiss", "index.pkl"):
        path = index / name
        path.write_text("x", encoding="utf-8")
        os.utime(path, (index_time, index_time))
    return docs, index


class StaleIndexWarningTests(unittest.TestCase):
    def test_warns_when_docs_are_newer_than_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            docs, index = _make_tree(Path(tmp), doc_time=2000, index_time=1000)
            with patch.object(direct, "_DOCS_DIR", docs), patch.object(direct, "_INDEX_DIR", index), patch.object(
                direct.logger, "warning"
            ) as warning:
                direct.warn_if_index_stale()
        self.assertTrue(warning.called)
        self.assertIn("build_index", str(warning.call_args))

    def test_silent_when_index_is_fresh(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            docs, index = _make_tree(Path(tmp), doc_time=1000, index_time=2000)
            with patch.object(direct, "_DOCS_DIR", docs), patch.object(direct, "_INDEX_DIR", index), patch.object(
                direct.logger, "warning"
            ) as warning:
                direct.warn_if_index_stale()
        self.assertFalse(warning.called)

    def test_missing_index_does_not_raise(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            docs = Path(tmp) / "docs"
            docs.mkdir()
            (docs / "a.md").write_text("x", encoding="utf-8")
            with patch.object(direct, "_DOCS_DIR", docs), patch.object(
                direct, "_INDEX_DIR", Path(tmp) / "index"
            ):
                direct.warn_if_index_stale()  # 不抛异常即通过


if __name__ == "__main__":
    unittest.main()
