import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

# 兜底读超时，只给"短输入短输出"的调用用（例如检索时的查询改写）。
# 真正的节点级超时见 graph.py 里的 TIMEOUT_*。
DEFAULT_TIMEOUT = 60.0


def get_ds_chat_model(temperature: float = 0.3, timeout: float = DEFAULT_TIMEOUT):
    """Create the configured OpenAI-compatible chat model.

    Args:
        temperature: 采样温度。
        timeout: 单次请求的读超时（秒）。不同节点要生成的内容长度差一个数量级，
            所以由调用方按 graph.py 里的 TIMEOUT_* 分节点传入，而不是所有人共用一个值。

    环境变量三件套（见 .env.example）：LLM_API_KEY / LLM_MODEL / LLM_BASE_URL。
    """
    api_key = os.getenv("LLM_API_KEY")
    model = os.getenv("LLM_MODEL")
    base_url = os.getenv("LLM_BASE_URL")

    missing = [
        name
        for name, value in (
            ("LLM_API_KEY", api_key),
            ("LLM_MODEL", model),
            ("LLM_BASE_URL", base_url),
        )
        if not value
    ]
    if missing:
        raise RuntimeError(
            "缺少模型配置：" + "、".join(missing)
            + "。请复制 .env.example 为 .env 并补齐后再运行"
        )

    llm = init_chat_model(
        model=model,
        model_provider="openai",
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        # 读超时的取值依据：2026-10-15 评测 20 例里有 3 例挂在 ReadTimeout，
        #   而正常节点实测 report≈37s、questions≈28s —— 全节点统一 60s 时余量不足一倍，
        #   供应商一端抖动就直接超时，重试三次仍失败导致整条用例判废。
        #   现在改成分节点给不同阈值（见 graph.py 的 TIMEOUT_*）。
        timeout=timeout,
        # 客户端层不再重试，重试统一交给 agent/graph.py 的 invoke_structured / safe_invoke。
        # 为什么必须二选一：两层都重试会**相乘**——
        # 外层 3 次 × 客户端层 3 次 = 同一次节点最坏发 9 次请求，
        # 这正是之前单次分析跑到 560s 的原因之一。
        # 业务层重试的好处是：能区分"输出不合规"和"上游故障"，
        # 能打自己的日志，也能决定什么时候该放弃。
        max_retries=0,
    )
    return llm
