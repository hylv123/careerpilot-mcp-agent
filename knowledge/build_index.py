"""构建知识库索引：加载 knowledge/docs 与本地私有资料 -> 分割 -> 向量化 -> 存入 FAISS。

目录约定：
- `docs/*.md`：通用学习笔记，随仓库公开；
- `docs/private/*.md`：真实简历等本地私有资料，已被 `.gitignore` 忽略；
- `docs/examples/*.md`：脱敏示例，仅作参考，不参与索引。

`docs/` 根目录下这四份早期示例简历/项目文档按文件名排除，
避免它们被检索出来当成候选人的真实经历（真实内容见 `docs/private/`）。
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"
PRIVATE_DIR = DOCS_DIR / "private"
INDEX_DIR = BASE_DIR / "index"

# 早期脱敏示例，内容为虚构经历，不能进入检索结果。
EXCLUDED_SAMPLE_FILES = {
    "简历.md",
    "项目经历.md",
    "后端项目实践.md",
    "竞赛与课外实践.md",
}

embeddings = DashScopeEmbeddings(
    model=os.getenv("EMBEDDING_MODEL", "text-embedding-v3"),
    dashscope_api_key=(
        os.getenv("DASHSCOPE_API_KEY")
        or os.getenv("aliQwen-api")
    ),
)

def _load_docs() -> list:
    """加载公开笔记与本地私有资料；虚构示例简历不参与索引。"""
    files = [
        path for path in sorted(DOCS_DIR.glob("*.md"))
        if path.name not in EXCLUDED_SAMPLE_FILES
    ]
    if PRIVATE_DIR.exists():
        files += sorted(PRIVATE_DIR.glob("*.md"))

    docs = []
    for path in files:
        # 中文文件显式 UTF-8
        docs.extend(TextLoader(str(path), encoding="utf-8").load())
    return docs


docs = _load_docs()
print(f"加载文档 {len(docs)} 篇：{[Path(d.metadata['source']).name for d in docs]}")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", "；", "，", " ", ""]
)

splits = splitter.split_documents(docs)
print(f"分割后共 {len(splits)} 个片段")

store = FAISS.from_documents(splits, embeddings)
store.save_local(str(INDEX_DIR))
print(f"索引已保存到: {INDEX_DIR}")
