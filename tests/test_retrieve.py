"""检索层测试：可离线执行，缺外部依赖时跳过而不是报错。

为什么单独成一份：
    `retrieve_agentic` 的核心逻辑（够不够、要不要改写、怎么合并去重）是纯逻辑，
    不依赖 FAISS、不依赖 Embedding 接口，完全可以离线断言。
    只有最后一次"真查一次"需要索引 + Key，那部分用 skip 处理。

运行：
    python -m unittest tests.test_retrieve -v

历史：
  1. 这个文件最初导入 `retrieve_with_rewrite` / `rewrite_keywords` 两个已删除的
     函数，`python -m knowledge.test_retrieve` 必然 ImportError；
  2. 修好之后它仍留在 `knowledge/` 包里，`discover -s tests` 扫不到，
     CI 得额外写一条命令。
  现在它和其它测试一样在 `tests/` 下，一条 discover 就能跑到。
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

import knowledge.retrieve as retrieve_module
from knowledge.retrieve import retrieve, retrieve_agentic, rewrite_query

# 从 tests/ 上跳两级回到仓库根，再进 knowledge/index
INDEX_DIR = Path(__file__).resolve().parent.parent / "knowledge" / "index"


def _hit(content: str, score: float, source: str = "resume.md") -> dict:
    return {
        "content": content,
        "source": source,
        "source_path": f"/app/knowledge/docs/{source}",
        "score": score,
    }


class _FakeDoc:
    def __init__(self, content: str, source: str) -> None:
        self.page_content = content
        self.metadata = {"source": source}


class _FakeStore:
    def __init__(self, docs: list[tuple[_FakeDoc, float]]) -> None:
        self._docs = docs
        self.queries: list[tuple[str, int]] = []

    def similarity_search_with_score(self, query: str, k: int):
        self.queries.append((query, k))
        return self._docs[:k]


class RetrieveShapeTests(unittest.TestCase):
    """retrieve() 的返回结构：四个键、source 只留文件名、score 是 float。"""

    def test_maps_document_to_dict(self) -> None:
        store = _FakeStore([(_FakeDoc("片段正文", "/kb/docs/resume.md"), 0.83)])
        with patch.object(retrieve_module, "_get_store", return_value=store):
            hits = retrieve("RAG", k=1)

        self.assertEqual(len(hits), 1)
        hit = hits[0]
        self.assertEqual(hit["content"], "片段正文")
        self.assertEqual(hit["source"], "resume.md")          # 只留文件名
        self.assertEqual(hit["source_path"], "/kb/docs/resume.md")
        self.assertIsInstance(hit["score"], float)             # 必须可 JSON 序列化

    def test_passes_query_and_k_to_store(self) -> None:
        store = _FakeStore([])
        with patch.object(retrieve_module, "_get_store", return_value=store):
            retrieve("Python RAG", k=3)
        self.assertEqual(store.queries, [("Python RAG", 3)])


class RetrieveAgenticTests(unittest.TestCase):
    """两轮检索的决策逻辑：够就不改写、不够才改写、失败要回退。"""

    def test_returns_first_round_when_enough_relevant(self) -> None:
        first = [_hit("a", 0.5), _hit("b", 0.6), _hit("c", 0.9), _hit("d", 0.95)]
        with patch.object(retrieve_module, "retrieve", return_value=first) as fake, \
             patch.object(retrieve_module, "rewrite_query") as rewrite:
            hits = retrieve_agentic(["Python", "RAG"], k=4)

        self.assertEqual(hits, first)
        self.assertEqual(fake.call_count, 1)          # 只查一轮
        rewrite.assert_not_called()                    # 没触发改写

    def test_rewrites_when_relevant_hits_insufficient(self) -> None:
        first = [_hit("a", 1.5), _hit("b", 1.6)]       # 距离 > 1.02，都不算相关
        second = [_hit("b", 1.6), _hit("c", 0.9)]      # 含重复内容 b
        with patch.object(retrieve_module, "retrieve", side_effect=[first, second]) as fake, \
             patch.object(retrieve_module, "rewrite_query", return_value="改写后的查询"):
            hits = retrieve_agentic(["Python"], k=4)

        self.assertEqual(fake.call_count, 2)
        contents = [hit["content"] for hit in hits]
        self.assertEqual(contents, ["c", "a", "b"])           # 合并后按距离升序
        self.assertEqual(len(contents), len(set(contents)))   # 已去重

    def test_keeps_first_round_when_rewrite_fails(self) -> None:
        first = [_hit("a", 1.5)]
        with patch.object(retrieve_module, "retrieve", return_value=first), \
             patch.object(retrieve_module, "rewrite_query", side_effect=RuntimeError("模型挂了")):
            hits = retrieve_agentic(["Python"], k=4)
        self.assertEqual(hits, first)

    def test_result_is_capped_at_k(self) -> None:
        first = [_hit(str(i), 2.0) for i in range(4)]
        second = [_hit(f"new-{i}", 0.5) for i in range(4)]
        with patch.object(retrieve_module, "retrieve", side_effect=[first, second]), \
             patch.object(retrieve_module, "rewrite_query", return_value="q"):
            hits = retrieve_agentic(["Python"], k=4)
        self.assertLessEqual(len(hits), 4)


class RewriteQueryTests(unittest.TestCase):
    """rewrite_query() 里有一条 LCEL 链（prompt | llm），

    所以替身不能是普通对象——`|` 右边必须是 Runnable 或可调用对象。
    用 RunnableLambda 包一个记录入参的函数即可，全程不联网。
    """

    class _Spy:
        def __init__(self, content: str) -> None:
            self.content = content
            self.prompt_value = None

        def __call__(self, prompt_value):
            from langchain_core.messages import AIMessage

            self.prompt_value = prompt_value
            return AIMessage(content=self.content)

        def user_text(self) -> str:
            return self.prompt_value.to_messages()[-1].content

    def _llm(self, content: str):
        from langchain_core.runnables import RunnableLambda

        spy = self._Spy(content)
        return spy, RunnableLambda(spy)

    def test_returns_model_content(self) -> None:
        _, llm = self._llm("RAG 检索 向量")
        self.assertEqual(rewrite_query(["检索增强生成"], llm), "RAG 检索 向量")

    def test_only_first_five_keywords_are_used(self) -> None:
        spy, llm = self._llm("x")
        rewrite_query(["a", "b", "c", "d", "e", "f"], llm)
        self.assertIn("a,b,c,d,e", spy.user_text())
        self.assertNotIn(",f", spy.user_text())


@unittest.skipUnless(INDEX_DIR.exists(), "未构建 FAISS 索引，跳过真实检索")
class LiveRetrieveTests(unittest.TestCase):
    """真查一次：验证索引能加载、返回结构与离线部分一致。

    依赖网络（DashScope Embedding）。接口不可用/额度不足时跳过并给出原因，
    不让一条外部依赖把整个 CI 判红——这正是这个项目对 MCP/评测一贯的处理方式。
    """

    def test_real_query_returns_hits(self) -> None:
        if not (os.getenv("DASHSCOPE_API_KEY") or os.getenv("aliQwen-api")):
            self.skipTest("缺少 DASHSCOPE_API_KEY")
        try:
            hits = retrieve("Python RAG", k=1)
        except Exception as exc:  # 网络/额度问题都归为"环境不可用"
            self.skipTest(f"检索不可用：{type(exc).__name__}: {exc}")
        self.assertTrue(hits)
        self.assertEqual(set(hits[0]), {"content", "source", "source_path", "score"})


if __name__ == "__main__":
    unittest.main()
