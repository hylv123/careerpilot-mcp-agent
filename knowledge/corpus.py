"""语料范围：哪些文件会进入 FAISS 索引，由这里唯一决定。

为什么单独抽一层：
    索引构建（build_index.py）和经历原子抽取（services/atoms.py）必须看同一批文件。
    之前原子只扫了 private/，公开笔记里的真实经历全部漏在原子库外，
    结果"强制只引用原子"会把可用事实砍掉四分之一。
    两边各写一份 glob 迟早会漂，所以收敛到这里。
"""

from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"
PRIVATE_DIR = DOCS_DIR / "private"

# 早期脱敏示例，内容为虚构经历，不能进入检索结果。
EXCLUDED_SAMPLE_FILES = {
    "简历.md",
    "项目经历.md",
    "后端项目实践.md",
    "竞赛与课外实践.md",
}


def indexed_files() -> list[Path]:
    """返回参与索引的文件（公开笔记 + 本地私有资料），顺序稳定。"""
    files = [
        path for path in sorted(DOCS_DIR.glob("*.md"))
        if path.name not in EXCLUDED_SAMPLE_FILES
    ]
    if PRIVATE_DIR.exists():
        files += sorted(PRIVATE_DIR.glob("*.md"))
    return files
