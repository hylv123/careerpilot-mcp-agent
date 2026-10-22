"""构建知识库索引：加载 knowledge/docs 下所有文档 -> 分割 -> 向量化 -> 存入 FAISS。"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"
INDEX_DIR = BASE_DIR / "index"

embeddings = DashScopeEmbeddings(
    model=os.getenv("EMBEDDING_MODEL", "text-embedding-v3"),
    dashscope_api_key=(
        os.getenv("DASHSCOPE_API_KEY")
        or os.getenv("aliQwen-api")
    ),
)

loader = DirectoryLoader(
    str(DOCS_DIR), 
    glob="*.md", 
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"},  # 中文文件显式 UTF-8
    )

docs = loader.load()

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
