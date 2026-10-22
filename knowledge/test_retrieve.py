import sys
from pathlib import Path

# 把项目根目录（careerpilot）加入 Python 路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import logging
from retrieve import retrieve_with_rewrite, retrieve, rewrite_keywords
from agent.llm import get_ds_chat_model

logging.basicConfig(level=logging.INFO)

def test_basic_retrieve():
    """测试 1：基础检索（不改写）"""
    print("\n" + "="*50)
    print("测试 1：基础检索")
    print("="*50)
    
    query = "LangChain RAG 项目"
    results = retrieve(query, k=3)
    
    print(f"查询词: {query}")
    print(f"命中 {len(results)} 条结果:")
    for i, doc in enumerate(results, 1):
        print(f"  {i}. {doc['content'][:100]}... (来源: {doc['source']}, 分数: {doc['score']:.4f})")
    
    # 断言：至少返回 1 条
    assert len(results) > 0, "❌ 基础检索返回 0 条，检查向量库是否正常加载"
    print("✅ 基础检索通过")


def test_retrieve_with_rewrite_sufficient():
    """测试 2：检索 + 改写（第一轮就够）"""
    print("\n" + "="*50)
    print("测试 2：检索 + 改写（第一轮够用）")
    print("="*50)
    
    keywords = ["大模型", "RAG", "LangChain"]
    results = retrieve_with_rewrite(keywords, k=3, min_hits=2)
    
    print(f"关键词: {keywords}")
    print(f"命中 {len(results)} 条结果:")
    for i, doc in enumerate(results, 1):
        print(f"  {i}. {doc['content'][:100]}...")
    
    assert len(results) >= 2, f"❌ 预期至少 2 条，实际 {len(results)} 条"
    print("✅ 检索 + 改写（第一轮够用）通过")


def test_retrieve_with_rewrite_insufficient():
    """测试 3：检索 + 改写（第一轮不够，触发改写）"""
    print("\n" + "="*50)
    print("测试 3：检索 + 改写（第一轮不够，触发改写）")
    print("="*50)
    
    # 用一些冷门词，模拟第一轮搜不到的情况
    keywords = ["量子计算", "神经网络", "深度学习"]
    results = retrieve_with_rewrite(keywords, k=3, min_hits=3)
    
    print(f"关键词: {keywords}")
    print(f"最终命中 {len(results)} 条结果:")
    for i, doc in enumerate(results, 1):
        print(f"  {i}. {doc['content'][:100]}...")
    
    # 允许返回 0 条（如果向量库里真的没有相关内容）
    # 但至少应该能正常运行，不报错
    print(f"✅ 检索 + 改写（第一轮不够）完成，返回 {len(results)} 条")


def test_rewrite_keywords():
    """测试 4：单独测试改写功能"""
    print("\n" + "="*50)
    print("测试 4：查询改写功能")
    print("="*50)
    
    keywords = ["大模型应用", "熟悉RAG", "有AI Agent经验"]
    llm = get_ds_chat_model()
    
    print(f"原始关键词: {keywords}")
    rewritten = rewrite_keywords(keywords, llm)
    print(f"改写后查询词: {rewritten}")
    
    assert len(rewritten) > 0, "❌ 改写结果为空"
    print("✅ 查询改写功能通过")


if __name__ == "__main__":
    print("🚀 开始测试 retrieve 模块...")
    
    try:
        # 按顺序执行测试
        test_basic_retrieve()
        test_retrieve_with_rewrite_sufficient()
        test_retrieve_with_rewrite_insufficient()
        test_rewrite_keywords()
        
        print("\n" + "="*50)
        print("🎉 所有测试通过！")
        print("="*50)
        
    except AssertionError as e:
        print(f"\n❌ 测试失败: {e}")
    except Exception as e:
        print(f"\n❌ 运行异常: {e}")
        import traceback
        traceback.print_exc()