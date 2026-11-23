import json
import random
import threading

from langgraph.graph import END, START, StateGraph
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.tools import StructuredTool
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel
from knowledge.router import resolve_mode, search_for_match, search_evidence
from services.jobs import create_job
from typing import Any, TypeVar, ParamSpec
from collections.abc import Callable
from functools import wraps
from time import perf_counter, sleep
from loguru import logger
from .schemas import JobAnalysisGraph, TailoredResume, InterviewQuestionList
from .llm import get_ds_chat_model
from .state import State
from .mcp_tools import (
    TOOL_REJECT_PREFIX,
    get_mcp_tools,
    is_tool_error,
    validate_tool_calls,
)
from .safety import UNTRUSTED_NOTICE, drop_ungrounded, evidence_block, wrap_untrusted
from .grounding import evidence_corpus, filter_skills, sanitize_text, verify_bullets
from services import atoms as atoms_service
from .style import ANTI_AI_RULES, style_correction, style_issues


MAX_TOOL_ROUNDS = 3

RETRY_BASE_DELAY = 1.0    # 第一次失败后等待的基准秒数
RETRY_MAX_DELAY = 8.0     # 单次退避上限，避免把一次交互请求拖到不可接受
RETRY_JITTER = 0.3        # 抖动幅度 ±30%

# 判定"上游/网络故障"的关键词。命中就按完整退避等待；
# 不命中（典型是模型输出不满足 Pydantic schema）说明不是上游过载，
# 立刻重试才有意义，长等只是白白增加延迟。
UPSTREAM_ERROR_HINTS = (
    "timeout", "timed out", "connection", "rate", "limit",
    "429", "500", "502", "503", "504", "overload", "throttl",
    "temporarily", "unavailable",
)

# agent 决定不再调工具后，同时启动的三个生成节点。
# 它们都只依赖 analysis + evidence，彼此没有数据依赖，所以可以并行：
# 串行耗时是三者之和（早期模型实测约 95s），并行后是三者中最大的那个（约 45s）。
GENERATE_NODES: list[str] = ["report", "rewrite", "questions"]

# 工具结果回流 evidence 时的上限与前缀。
# 为什么要回流：report / rewrite / questions 三个生成节点只读 analysis + evidence，
#   不读 messages。工具执行结果如果只留在消息流里，就永远到不了报告——
#   agent 和 tools 两个节点会变成"只花时间、不产出"的装饰。
# 为什么要前缀：引用校验要求 generated_sources 必须能在 evidence 的 source 里找到；
#   加 `tool:` 前缀既让模型知道这条来自工具而不是知识库，也避免和文件路径混淆。
TOOL_EVIDENCE_PREFIX = "tool:"
TOOL_EVIDENCE_MAX_CHARS = 2000

# 本地检索工具的名字。它不在 MCP 的三个 Server 里，直接跑在主进程：
#   检索和 match 节点共享同一份 FAISS 索引与 Embedding 客户端，
#   走 MCP 只会多一次序列化和一处故障点，没有收益。
SEARCH_TOOL_NAME = "search_evidence"


def _parse_search_hits(text: str) -> list[dict[str, Any]]:
    """把检索工具返回的 JSON 解析成片段列表；解析不了就返回空列表。"""
    try:
        payload = json.loads(text)
    except (TypeError, ValueError):
        return []
    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, dict)]


def _tool_messages_to_evidence(
    tool_calls: list[dict[str, Any]],
    messages: list[Any],
    existing: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """把本轮工具返回的内容转成 evidence 条目。

    Args:
        tool_calls: 本轮 AIMessage.tool_calls，用来把 tool_call_id 映射回工具名。
        messages: ToolNode 返回的 ToolMessage 列表。
        existing: 已有的 evidence，用来去重（同一条工具结果不重复进上下文）。

    Returns:
        新增的 evidence 条目，格式与知识库召回保持一致（content/source/score）。

    为什么要过滤 is_tool_error：
        失败降级成的文本（"MCP 工具调用失败：…"）不是事实依据，
        写进 evidence 会污染引用来源，让报告引用一条错误信息。

    为什么要对 search_evidence 单独处理：
        它返回的是"带 source 的片段数组"，如果整段当成一条 evidence，
        引用来源会退化成 `tool:search_evidence`，模型就分不清结论到底来自
        简历还是某个项目了。拆成逐条，才能保住原来的 source（文件名）。
    """
    name_by_id = {
        call.get("id"): call.get("name")
        for call in tool_calls
        if call.get("id")
    }
    seen = {str(item.get("content", "")) for item in existing}
    added: list[dict[str, Any]] = []

    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue
        raw = msg.content
        if isinstance(raw, list):
            text = "\n".join(
                str(block.get("text", "")) if isinstance(block, dict) else str(block)
                for block in raw
            )
        else:
            text = str(raw)
        text = text.strip()

        if not text or is_tool_error(text):
            continue

        name = name_by_id.get(msg.tool_call_id, "unknown")
        if name == SEARCH_TOOL_NAME:
            for hit in _parse_search_hits(text):
                content = str(hit.get("content", "")).strip()
                if not content or content in seen:
                    continue
                seen.add(content)
                added.append(
                    {
                        "content": content[:TOOL_EVIDENCE_MAX_CHARS],
                        "source": str(hit.get("source") or f"{TOOL_EVIDENCE_PREFIX}{name}"),
                        "source_path": str(
                            hit.get("source_path") or f"{TOOL_EVIDENCE_PREFIX}{name}"
                        ),
                        "score": float(hit.get("score", 0.0) or 0.0),
                    }
                )
            continue

        if text in seen:
            continue

        seen.add(text)
        added.append(
            {
                "content": text[:TOOL_EVIDENCE_MAX_CHARS],
                "source": f"{TOOL_EVIDENCE_PREFIX}{name}",
                "source_path": f"{TOOL_EVIDENCE_PREFIX}{name}",
                "score": 0.0,
            }
        )

    return added

# 日志不在这里初始化：`logger.add` 会在导入期打开文件，
# 目录只读时会让 `import agent.graph` 直接抛 PermissionError。
# 现在由进程入口显式调用 agent.logging_setup.setup_logging()，
# 没配也可以跑——loguru 默认的 stderr sink 仍然是通的。


SchemaType = TypeVar("SchemaType", bound=BaseModel)


P = ParamSpec("P")
R = TypeVar("R")


def trace_node(node_name: str) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """记录节点的开始、完成、失败及耗时"""
    def decorator(func: Callable[P, R]) -> Callable[P, R]:

        @wraps(func)
        def wrapper(
            *args: P.args,
            **kwargs: P.kwargs,
        ) -> R:
            started = perf_counter()

            logger.info("[node:{}] 开始执行", node_name)
            try:
                result = func(*args, **kwargs)
                logger.info(
                    "[node:{}]执行完成，耗时{:.3f}s",
                    node_name,
                    perf_counter() - started,
                )
                return result
            except Exception:
                logger.exception(
                    "[node:{}] 执行失败，耗时{:.3f}s",
                    node_name,
                    perf_counter() - started,
                )
                raise
        return wrapper
    return decorator


def _is_upstream_error(exc: BaseException | None) -> bool:
    """判断一次失败是不是"上游过载/网络故障"。

    为什么要区分：
        上游返回 429/5xx/超时，说明它现在扛不住，越急着重试越失败，
        必须退避一会儿；而模型输出不满足 Pydantic schema 属于内容问题，
        等 8 秒再让它重新生成并没有任何帮助，只会白白拉长响应时间。

    Args:
        exc: 捕获到的异常。

    Returns:
        True 表示应按完整退避等待后再重试。
    """
    if exc is None:
        return False
    text = f"{type(exc).__name__} {exc}".lower()
    return any(hint in text for hint in UPSTREAM_ERROR_HINTS)


def _backoff_delay(attempt: int, exc: BaseException | None) -> float:
    """计算第 attempt 次失败后应该等待多少秒。

    三个设计点：
        1. 指数退避：第 n 次等待 base * 2^(n-1)，给上游留出恢复时间。
        2. 抖动（jitter）：在退避时长上叠加 ±30% 随机量。
           ⚠️ 这一条在并行改造之后是**必需的**：
           report / rewrite / questions 三个节点同时跑、同时失败，
           如果按固定间隔重试，它们会在同一毫秒再次一起打过去，
           把刚缓过来的上游重新打垮（重试风暴）。抖动把重试时刻错开。
        3. 封顶：单次等待不超过 RETRY_MAX_DELAY，
           否则一次交互式请求会被退避时间拖到几分钟。

    Args:
        attempt: 第几次失败（从 1 开始）。
        exc: 本次失败的异常，用于判断是否值得长等。

    Returns:
        等待秒数，非负。
    """
    base = RETRY_BASE_DELAY if _is_upstream_error(exc) else RETRY_BASE_DELAY / 4
    delay = min(base * (2 ** (attempt - 1)), RETRY_MAX_DELAY)
    jitter = delay * RETRY_JITTER
    # 抖动之后再封一次顶，否则上限会被随机量顶穿（8s 的 cap 变成 10.4s）
    return max(0.0, min(delay + random.uniform(-jitter, jitter), RETRY_MAX_DELAY))


def invoke_structured(
    chain: Any,
    inputs: dict[str, Any],
    schema: type[SchemaType],
    retries: int = 2,   #最多调用3次    
) -> SchemaType:
    """调用结构化输出链，生成或检验失败时重试一次"""
    last_error: Exception | None = None

    for i in range(1, retries + 2):
        started = perf_counter()
        try:
            logger.info(
                "[llm:{}] 第{}/{}次调用开始",
                schema.__name__,
                i,
                retries + 1,
            )
            raw_result = chain.invoke(inputs)
            result = schema.model_validate(raw_result)
            logger.info(
                "[llm:{}] 第{}/{}次调用成功，耗时{:.3f}s",
                schema.__name__,
                i,
                retries + 1,
                perf_counter() - started,
            )
            return result
        
        except Exception as e:
            last_error = e

            if i <= retries:
                delay = _backoff_delay(i, e)
                logger.warning(
                    "[llm:{}] 第 {}/{} 次调用失败，{:.2f}s 后重试，"
                    "错误类型={}，错误={}",
                    schema.__name__,
                    i,
                    retries + 1,
                    delay,
                    type(e).__name__,
                    e,
                )
                sleep(delay)
            else:
                logger.exception(
                    "[llm:{}] 第 {}/{} 次调用失败，已无重试次数",
                    schema.__name__,
                    i,
                    retries + 1,
                )

    raise RuntimeError(f"{schema.__name__}结构化输出重试后仍然失败") from last_error
    
def safe_invoke(
    chain: Any,
    inputs: Any,
    operation_name: str,
    retries: int = 2,
) -> Any:
    """调用普通的LLM链，失败后最多重试2次"""

    last_error: Exception | None = None

    for i in range(1, retries + 2):
        started = perf_counter()
        try:
            logger.info(
                "[llm:{}] 第 {}/{} 次调用开始",
                operation_name,
                i,
                retries + 1,
            )

            result = chain.invoke(inputs)

            logger.info(
                "[llm:{}] 第 {}/{} 次调用成功，耗时 {:.3f}s",
                operation_name,
                i,
                retries + 1,
                perf_counter() - started,
            )

            return result
        
        except Exception as e:
            last_error = e

            if i <= retries:
                delay = _backoff_delay(i, e)
                logger.warning(
                    "[llm:{}] 第 {}/{} 次调用失败，{:.2f}s 后重试，"
                    "错误类型={}，错误={}",
                    operation_name,
                    i,
                    retries + 1,
                    delay,
                    type(e).__name__,
                    e,
                )
                sleep(delay)
            else:
                logger.exception(
                    "[llm:{}] 调用失败，重试次数已用完",
                    operation_name,
                )

    raise RuntimeError(f"{operation_name} 调用多次失败") from last_error

# 分节点读超时：一次分析要打 5~7 次模型请求，超时限制的是单次请求；
# 各节点输出长度差一个量级（实测 analysis ≈7s、questions ≈28s、report ≈37s），
# 共用同一阈值会误杀长文本节点。
TIMEOUT_ANALYSIS = 60.0
TIMEOUT_AGENT = 90.0
TIMEOUT_BULLETS = 120.0
TIMEOUT_QUESTIONS = 150.0
TIMEOUT_REPORT = 180.0

# 按超时值缓存模型实例：同一个值只构建一次，不同值各自一份
_llm_cache: dict[float, Any] = {}
_llm_lock = threading.Lock()


def _get_llm(timeout: float = TIMEOUT_AGENT):
    """Initialize the chat model only when a model-backed node is executed.

    为什么要加锁：
        report / rewrite / questions 现在并行跑在三个线程里，
        它们会同时第一次调用 _get_llm()。不加锁就会出现两个线程
        同时看到缓存缺失、各自建一个模型对象，既浪费初始化成本，
        又让它们用上不同的客户端。这里用双重检查加锁，
        只有第一个进来的线程真正初始化，后来的线程直接拿到结果。

    为什么要按 timeout 分桶而不是全局一个实例：
        见上面的 TIMEOUT_* 注释——短输出节点要小阈值，长文本节点要大阈值，
        一个模型对象在创建时就把 timeout 定死了，所以按值缓存成多个实例。

    Args:
        timeout: 单次请求的读超时秒数。
    """
    if timeout not in _llm_cache:
        with _llm_lock:
            if timeout not in _llm_cache:
                _llm_cache[timeout] = get_ds_chat_model(timeout=timeout)
    return _llm_cache[timeout]

@trace_node("analysis")
def analysis_node(state: State) -> dict[str, Any]:
    """解析岗位的要求，结构化输出（json）"""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是一名针对 AI Agent 应用岗位的招聘分析助手。"
                "解析岗位要求并输出 job_title、company、requirements、keywords、level。"
                "尽量提取 job_title 和 company；无法识别时分别返回"
                "未识别岗位和未识别公司。"
                "requirements 是技能清单，keywords 是检索关键词，"
                "level 只能是实习生、初级、中级、高级之一。",
            ),
            ("human", "岗位描述：{user_input}")
        ]
    )
    chain = prompt | _get_llm(TIMEOUT_ANALYSIS).with_structured_output(
        JobAnalysisGraph,
        method="function_calling",
    )
    analysis = invoke_structured(
        chain, 
        {"user_input": state["user_input"]}, 
        JobAnalysisGraph
    )
    return {"analysis": analysis.model_dump()}

@trace_node("match")
def match_evidence(state: State) -> dict[str, Any]:
    """用关键词检索匹配的经历，并把上下文注入消息流（供 agent 阶段使用）"""

    keywords = state["analysis"].get("keywords", [])
    query = " ".join(keywords[:5])

    logger.info(
        "[RAG] 开始检索，query = {}",
        query,
    )

    # 检索模式由规模自动决定：小语料直读（不丢证据），大语料才走向量。
    # 见 knowledge/router.py 与 CAREERPILOT_RETRIEVAL_MODE。
    evidence = search_for_match(keywords, 4)

    logger.info(
        "[RAG] 检索完成，命中{}条",
        len(evidence),
    )

    for idx, item in enumerate(evidence, start=1):
        logger.info(
            "[RAG] 命中 {}， source={}， score={}",
            idx,
            item.get("source", ""),
            item.get("score",""),
        )

        content_preview = item.get("content", "")[:80]
        logger.debug(
            "[rag] 内容摘要={}",
            content_preview,
        )

    context = (
        f"{UNTRUSTED_NOTICE}\n\n"
        "请根据下面的岗位分析和匹配经历，生成一份求职匹配报告，每段必须标注来源。"
        "如果需要简历、项目等补充信息，可以先调用可用工具获取，再输出报告。\n\n"
        f"岗位分析：\n{wrap_untrusted(json.dumps(state['analysis'], ensure_ascii=False), 'job_analysis')}\n\n"
        f"匹配经历：\n{evidence_block(evidence)}"
    )
    return {"evidence": evidence, "messages": [("user", context)]}

@trace_node("report")
def report(state: State) -> dict[str, Any]:
    """生成报告（带引用）"""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是求职顾问。基于岗位要求和匹配经历生成报告，每段必须标注来源。" + UNTRUSTED_NOTICE),
            ("human", "岗位分析：{analysis}\n\n匹配经历：{evidence}"),
        ]
    )
    result = safe_invoke(
        chain = prompt | _get_llm(TIMEOUT_REPORT),
        # 必须 json.dumps：直接把 dict/list 交给占位符，得到的是 Python repr
        #（单引号、True/None），不是合法 JSON，模型容易把 None 当字符串读。
        # rewrite_resume / gen_questions 一直是这么做的，这里以前漏了。
        inputs={
            "analysis": json.dumps(state["analysis"], ensure_ascii=False),
            "evidence": evidence_block(state["evidence"]),
        },
        operation_name="report"
    )
    return {
        "report_text": result.content,
        # 仍然写一份进 messages：eval 里的占位符/幻觉检测要读 messages 全文。
        # 但下游展示一律读 report_text，因为这个字段的顺序是确定的。
        "messages": [("assistant", result.content)],
    }


@trace_node("rewrite")
def rewrite_resume(state: State):
    """按 JD 的硬性要求与偏好，生成一份可投递的定制简历（结构化）。

    和早期版本的区别：
        以前只产出 1~3 条要点，用户还得自己拼成简历；现在直接给出
        headline / 概述 / 技能分组 / 按项目归组的要点，并且每条要点都挂到
        一条真实的 JD 要求或偏好上——这就是"针对 JD 偏好编写"的可核对形态。

    真实性由 `agent/grounding.py` 在模型之后判定，而不是靠提示词自觉：
        来源、项目名、技能词、量化数字、对应要求，五项都要能在本轮证据里落地。
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是一名只做真实改写的简历顾问。任务：把候选人的真实经历改写成"
                "针对这个岗位的简历内容。\n"
                "硬性规则（违反即作废）：\n"
                "1. 只能使用『匹配经历』里已经出现的事实：项目、技术栈、职责、成果。"
                "不得新增任何项目、公司、时间、技术或量化数字；\n"
                "2. 允许调整顺序、突出与岗位相关的部分、把同一件事换成更贴近岗位措辞的表达；\n"
                "3. 每条要点的 jd_requirement 必须逐字取自岗位分析里的 requirements 或 preferences；\n"
                "4. 每条要点的 project 必须逐字取自匹配经历，evidence_source 必须是该经历的 source，"
                "evidence_span 必须是从该经历里**逐字复制**的一段原文（10~80 字），"
                "不得改写、不得用省略号、不得拼接两处；\n"
                "4b. 主张强度不得高于原文：原文写「参与」，就不要写成「主导」「独立负责」「从 0 到 1」；\n"
                "5. preferences 是岗位的加分项/偏好，有对应真实经历时优先写；没有就不要硬凑；\n"
                "6. 技能列表按固定四类组织：编程语言 / AI核心技术 / 框架与工具 / 通用技能，"
                "每类 3~6 个短词，只写匹配经历里真实出现过的技术；不要带括号解释，"
                "不要写 Office、打字、上网这类无信息量的项，也不要把同义词拆成多条；\n"
                "7. 每条项目要点的 summary 是 6~12 字的概括（如“服务化改造”“检索链路评测”），"
                "bullet 要按“遇到什么问题/要做什么 → 怎么做的 → 结果或约束”的顺序叙述，"
                "读起来像人在讲自己做过的事；禁止写成“使用了 A、B、C 技术”这种技能罗列；\n"
                "7. 概述里不要写没有出处的比例、人数、时长等数字。\n"
                + ANTI_AI_RULES
                + "\n必须通过指定的结构化工具返回结果，不要返回普通文本段落。"
                + UNTRUSTED_NOTICE,
            ),
            (
                "human",
                "岗位分析：{analysis}\n\n"
                "匹配经历：{evidence}\n\n"
                "已确认事实原子（优先引用）：\n{atoms_block}\n\n"
                "{style_correction}",
            ),
        ]
    )

    chain = prompt | _get_llm(TIMEOUT_BULLETS).with_structured_output(
        TailoredResume,
        method="function_calling",
    )
    approved = atoms_service.approved_experience_atoms()
    atom_map = {atom["id"]: atom["span"] for atom in approved}
    atoms_block = (
        "\n".join(f"[{atom['id']}] {atom.get('project', '')}：{atom['span']}" for atom in approved)
        or "（当前没有已确认的原子；只能从匹配经历里逐字摘录，atom_id 留空）"
    )
    inputs = {
        "analysis": json.dumps(
            state["analysis"],
            ensure_ascii=False,
        ),
        "evidence": evidence_block(state["evidence"]),
        "atoms_block": atoms_block,
        "style_correction": "",
    }
    result = invoke_structured(chain, inputs, TailoredResume)

    # 去 AI 味：命中模板腔就带着"你犯了这些词"重写一次。
    # 只重写一次——再多就是拿用户的 token 换一点文风，不划算。
    issues = style_issues([result.headline, result.summary, *(b.bullet for b in result.bullets)])
    retried = False
    if issues:
        logger.warning("[style] 首版命中 AI 腔：{}，重写一次", issues)
        try:
            retried_result = invoke_structured(
                chain,
                {**inputs, "style_correction": style_correction(issues)},
                TailoredResume,
            )
        except Exception as exc:  # 重写失败不该让整条任务失败，首版仍可用
            logger.warning("[style] 去 AI 味重写失败，保留首版：{}", exc)
        else:
            result = retried_result
            retried = True
            issues = style_issues(
                [result.headline, result.summary, *(b.bullet for b in result.bullets)]
            )

    corpus = evidence_corpus(state["evidence"])
    kept, _dropped = verify_bullets(
        [bullet.model_dump() for bullet in result.bullets],
        state["evidence"],
        state["analysis"],
        approved_atoms=atom_map,
    )
    analysis = state.get("analysis", {})
    fallback_headline = "针对该岗位的定制简历"

    tailored = {
        "headline": sanitize_text(result.headline, corpus, fallback=fallback_headline),
        "summary": sanitize_text(result.summary, corpus, fallback=""),
        "skill_groups": filter_skills(
            [group.model_dump() for group in result.skill_groups], state["evidence"]
        ),
        "bullets": kept,
        "style": {"issues": issues, "retried": retried},
        "atom_stats": {
            "approved": len(approved),
            "cited": sum(1 for item in kept if str(item.get("atom_id", "")).strip()),
            "total": len(kept),
        },
    }

    # 兼容既有契约：接口与评测仍然读 resume_bullets（1~3 条）。
    bullets = [
        {"bullet": item["bullet"], "evidence_source": item["evidence_source"]}
        for item in kept[:3]
    ]
    bullet_text = "\n".join(
        f"- [{item.get('jd_requirement', '')}] {item['bullet']}\n"
        f"  项目: {item.get('project', '')} / 来源: {item['evidence_source']}"
        for item in kept
    )

    return {
        "tailored_resume": tailored,
        "resume_bullets": bullets,
        "messages":[
            ("assistant", f"## 定制简历要点：\n{bullet_text}")
        ]
    }


@trace_node("questions")
def gen_questions(state: State):
    """生成5道带回答要点和真实经历来源的面试题。"""

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是一名 AI Agent 应用方向的面试辅导老师。"
                "请生成正好5道面试题。"
                "每道题必须包含回答要点，并引用匹配经历中的真实来源。"
                "不得编造项目或经历。"
                "evidence_source 必须来自匹配经历的 source 字段。"
                "必须通过指定的结构化工具返回结果，不要返回普通文本段落。"
                + UNTRUSTED_NOTICE,
            ),
            (
                "human",
                "岗位分析：{analysis}\n\n"
                "匹配经历：{evidence}",
            ),
        ]
    )

    chain = prompt | _get_llm(TIMEOUT_QUESTIONS).with_structured_output(
        InterviewQuestionList,
        method="function_calling",
    )
    result = invoke_structured(
        chain,
        {
            "analysis": json.dumps(
                state["analysis"],
                ensure_ascii=False,
            ),
            "evidence": evidence_block(state["evidence"]),
        },
        InterviewQuestionList
    )

    # 输出侧校验：同简历要点，来源对不上证据的题目直接丢弃（宁缺毋滥）。
    questions_dicts = drop_ungrounded(
        [question.model_dump() for question in result.questions],
        lambda item: item.get("evidence_source", ""),
        state["evidence"],
        kind="面试题",
    )

    questions_sections: list[str] = []
    for idx, ques in enumerate(questions_dicts, start=1):
        ans_text = "\n".join(f"- {point}" for point in ques["answer_points"])
        questions_sections.append(
            f"### Q{idx}: {ques['question']}\n\n"
            f"{ans_text}\n\n"
            f"来源：{ques['evidence_source']}"
        )
    display_text = "\n\n".join(questions_sections)

    return {
        "interview_questions": questions_dicts,
        "messages": [
            ("assistant", f"## 面试准备题\n\n{display_text}")
        ]
    }


# 为什么 agent 节点必须显式带一条 system 指令：
#   原本它是把 state["messages"] 直接丢给模型的，模型只看到一句
#   "岗位分析：xxx / 匹配经历：yyy"，完全不知道自己被期望做什么，
#   于是按"聊天助手"的默认习惯回应 —— 写一段几百字的总结（早期模型实测 21~58 秒）。
#   而这段总结在本项目里【没有任何消费者】：交付的是结构化字段
#   （report_text / resume_bullets / interview_questions），
#   后面三个生成节点各自会去读 analysis + evidence。
# 所以这里明确约束它的输出形态：不需要工具时只回一个词，把这部分开销彻底拿掉。
AGENT_SYSTEM = SystemMessage(
    content=(
        "你是工具调度助手，职责只有一个：判断当前是否还需要调用工具来补充信息。\n"
        "规则：\n"
        "1. 只有确实需要外部数据（候选人简历、项目检索、岗位记录）时才调用工具；\n"
        "2. 信息已经足够时，**只输出 finish 一个词**，"
        "不要输出总结、报告、分析或任何解释性文字；\n"
        "3. 不要向用户提问，不要描述你打算怎么做。"
    )
)


def save_record_impl(state: State) -> dict[str, Any]:
    """把本次岗位写进 SQLite，并把"到底存没存成功"变成结构化字段。

    走原生服务，不走 MCP：岗位是应用自己的数据，写它不需要跨进程。
    以前这一步是 `save_job` 工具，代价是"子进程没起来/适配层吞异常"就会
    整条岗位记录丢失；现在失败会直接抛异常，当场转成 saved=False。

    Args:
        state: 当前图状态。

    Returns:
        含 saved / save_error / messages 三个键的部分状态。

    为什么不再把结果只塞进 messages：
        messages 里的那句"已保存/失败"没有任何下游消费者会去判定，
        于是"岗位没写进库"和"写进去了"在接口层长得一模一样——
        任务状态是 done、报告也正常，用户以为记录上了，其实没有。
        静默丢数据比直接失败更难查，所以必须落成 saved 字段。
    """
    analysis = state.get("analysis", {})
    title = analysis.get("job_title") or "未识别岗位"
    company = analysis.get("company") or "未识别公司"
    try:
        record = create_job(
            title=title,
            company=company,
            jd=state["user_input"],
            requirements=analysis.get("requirements", []),
            preferences=analysis.get("preferences", []),
        )
    except Exception as exc:  # 磁盘满、库被锁、表结构坏掉都要落成可见失败
        message = f"投递记录未保存：{exc}"
        return {"saved": False, "save_error": message, "messages": [("assistant", message)]}

    message = f"已保存岗位 #{record.get('id')}：{title} @ {company}"
    return {"saved": True, "save_error": "", "messages": [("assistant", message)]}


def _build_search_tool() -> StructuredTool:
    """把知识库检索包装成一个模型可调用的工具。

    为什么需要它：
        match 节点只在流程开头检索一次。agent 后面发现证据不足时没有任何
        手段再查——那就不算 agentic。暴露成工具后，模型可以自己判断
        "要不要再查一次"，死循环由 MAX_TOOL_ROUNDS 兜住。

    为什么 description 写得这么具体：
        模型选不选这个工具，几乎完全取决于 description。这里明确"什么时候该调、
        什么时候不该调"，否则它要么永远不调，要么每轮都调。
    """
    return StructuredTool.from_function(
        func=search_evidence,
        name=SEARCH_TOOL_NAME,
        description=(
            "按语义检索候选人的简历与项目经历，返回带来源(source)的片段列表。"
            "当现有证据不足以支撑结论、或需要确认某个技能/项目是否真实存在时调用；"
            "证据已经足够时不要调用。query 用中文描述要查什么，k 为返回条数（默认 4）。"
        ),
    )


def build_graph():
    try:
        logger.info("[mcp] 开始发现 MCP 工具")
        started = perf_counter()

        tools = get_mcp_tools()

        logger.info(
            "[mcp] 工具发现完成，共 {} 个工具，耗时 {:.3f}s，tools={}",
            len(tools),
            perf_counter() - started,
            [tool.name for tool in tools],
        )

    except Exception as e:
        logger.exception(
            "[mcp] 工具发现失败，降级为无 MCP 工具模式。"
            "错误类型={}，错误={}",
            type(e).__name__,
            e,
        )
        tools = []

    # 本地检索工具：和 MCP 工具（简历 2 个 + 文档 2 个）合并后一起 bind_tools。
    # 为什么走本地而不是 MCP：检索和 match 共享同一份 FAISS 索引与 Embedding
    # 客户端，包成 MCP Server 只会多一次序列化和一处故障点，没有收益。
    # 为什么直读模式下不暴露它：所有片段已经在上下文里了，模型再调一次
    # 只会把整库重复塞进消息流，纯浪费 token。
    if resolve_mode() != "direct":
        tools = [*tools, _build_search_tool()]
    else:
        logger.info("[mcp] 直读模式：不暴露检索工具（片段已全量在上下文中）")

    llm_with_tools = _get_llm(TIMEOUT_AGENT).bind_tools(tools)
    tool_map = {tool.name: tool for tool in tools}

    @trace_node("agent")
    def agent_node(state: State) -> dict[str, Any]:
        """把 messages 传给绑定工具的 LLM；裸 Runnable 直接作节点会收到整个 State dict 而报错。

        为什么这个节点必须打点：
            它是整条链路里唯一可能反复执行多次、又唯一会"决定下一步去哪"的节点。
            之前没打 trace，日志里 tools 结束到 dispatch 之间会出现几十秒的空白，
            完全无法归因——那段时间到底是在等模型，还是模型在纠结要不要调工具，
            一概不知。耗时分布里最大的一块找不到主人，优化就只能靠猜。
        """
        # ⚠️ 工具轮次用完之后的这一轮是纯浪费，直接跳过：
        #   经典 ReAct 循环里，模型最后一次被调用时要"收尾"——把工具结果复述成
        #   一段自然语言总结。这个设计只在"用户要的就是 agent 这句话"的聊天场景才成立。
        #   而本项目最终交付的是结构化字段（report_text / resume_bullets /
        #   interview_questions），后面三个生成节点各自会去读 analysis + evidence，
        #   这段总结根本没有任何消费者——早期模型下实测每次白白烧掉 21~24 秒和一份 token。
        # 因此轮次达到上限时不再调用模型，直接交给路由去 dispatch。
        if state.get("tool_rounds", 0) >= MAX_TOOL_ROUNDS:
            logger.info("[agent] 工具轮次已达上限，跳过收尾决策，直接进入生成阶段")
            return {"messages": [AIMessage(content="工具调用结束，进入生成阶段。")]}

        resp = safe_invoke(
            chain=llm_with_tools,
            # 每次调用都把 system 指令放在最前面注入，而不是写进 state：
            #   写进 state 会随着每一轮工具调用不断累积重复指令，白白占用上下文和 token。
            inputs=[AGENT_SYSTEM, *state["messages"]],
            operation_name="agent",  
        )
        tool_calls = getattr(resp, "tool_calls", None) or []
        if tool_calls:
            logger.info(
                "[agent] 本轮决定调用 {} 个工具：{}",
                len(tool_calls),
                [call.get("name") for call in tool_calls],
            )
        else:
            logger.info("[agent] 本轮未请求工具，准备进入生成阶段")
        return {"messages": [resp]}

    @trace_node("save")
    def save_record(state: State) -> dict[str, Any]:
        """通过 job MCP 工具保存本次岗位，成功与否写进 saved / save_error。"""
        return save_record_impl(state)

    @trace_node("dispatch")
    def dispatch_generate(state: State) -> dict[str, Any]:
        """并行扇出的起点，本身不做任何事。

        为什么不让条件边直接返回 ["report", "rewrite", "questions"]：
            那样写运行是对的，但 LangGraph 静态推不出 agent 之后的走向，
            get_graph() 会把 report/rewrite/questions 整段剪掉，
            图既画不出来（draw_ascii / mermaid 全空），也没法做结构校验。
            加一个不带逻辑的 dispatch 节点，图的拓扑才是显式的、
            可被测试和画出来的。代价只是多一次几乎没有开销的 superstep。
        """
        logger.info("[agent] 进入生成阶段，节点并行执行：{}", GENERATE_NODES)
        return {}

    _tool_node = ToolNode(tools)

    @trace_node("tools")
    def tools_with_guard(state: State) -> dict[str, Any]:
        """执行工具并累加轮次计数。

        为什么不直接用 ToolNode 当节点：
            需要在每次工具执行后把 tool_rounds 加一，
            路由函数才能判断是否已经调用太多次。

        为什么执行前要先校验一遍：
            模型给出的 tool_calls 是"自然语言生成的 JSON"，不是可信输入——
            工具名可能是幻觉出来的，args 可能不是对象，tool_call_id 可能缺失。
            缺 id 的那一种最致命：没有它就没法回填 ToolMessage，
            OpenAI 协议会拒绝下一轮请求，整条链路当场断掉。
            校验通过的执行，没通过的把原因回灌给模型，让它自己改。
        """
        last = state["messages"][-1]
        calls = getattr(last, "tool_calls", None) or []
        rounds = state.get("tool_rounds", 0) + 1

        valid, rejected = validate_tool_calls(calls, set(tool_map))
        for call, reason in rejected:
            logger.warning(
                "[tool-guard] 拒绝工具调用 name={} id={}：{}",
                call.get("name"),
                call.get("id"),
                reason,
            )

        result_msgs: list = []
        evidence: list[dict[str, Any]] | None = None
        if valid:
            if rejected:
                # ToolNode 会按名字把所有 tool_calls 都执行一遍，所以这里只把那批
                # 合法的交给它；历史里保留原来的 AIMessage，稍后我们自己把被拒的
                # 那几条也补上 ToolMessage，保证"每个 tool_call 都有且只有一个回应"。
                keep = {call["id"] for call in valid}
                pruned = last.model_copy(
                    update={
                        "tool_calls": [
                            call
                            for call in getattr(last, "tool_calls", [])
                            if call.get("id") in keep
                        ]
                    }
                )
                run_state = {**state, "messages": [*state["messages"][:-1], pruned]}
            else:
                run_state = state

            result = _tool_node.invoke(run_state)
            tool_messages = result.get("messages", [])
            result_msgs.extend(tool_messages)

            # 工具查到的内容必须回流到 evidence，否则三个生成节点看不到它。
            merged = _tool_messages_to_evidence(
                calls, tool_messages, state.get("evidence", [])
            )
            if merged:
                evidence = [*state.get("evidence", []), *merged]
                logger.info(
                    "[tool-guard] 本轮工具结果回流 evidence {} 条：{}",
                    len(merged),
                    [item["source"] for item in merged],
                )

        for call, reason in rejected:
            if not call.get("id"):
                # 没有 id 就无法构造合法的 ToolMessage，只能丢弃并留痕
                continue
            result_msgs.append(
                ToolMessage(
                    content=f"{TOOL_REJECT_PREFIX}{reason}",
                    tool_call_id=call["id"],
                    status="error",
                )
            )

        logger.info("[agent] 工具调用第 {} 轮（执行 {} 个，拒绝 {} 个）",
                    rounds, len(valid), len(rejected))
        updates: dict[str, Any] = {"messages": result_msgs, "tool_rounds": rounds}
        if evidence is not None:
            updates["evidence"] = evidence
        return updates

    def route_after_agent(state: State) -> str | list[str]:
        """决定 agent 节点之后去哪。

        和内置 tools_condition 的区别有两点：
            1. tools_condition 只看"这一轮有没有要调的工具"，不看"已经调了几轮"。
               这里多一层保护：达到上限时不再回 tools，
               让流程**优雅降级继续产出**，而不是抛异常中断。
            2. 不再调工具时不是只去 report，而是去 dispatch，
               由 dispatch 扇出到 report / rewrite / questions 三个节点，
               它们没有数据依赖，会被线程池并发执行，全部结束才继续往下走。

        Returns:
            "tools"：回工具节点继续；"dispatch"：进入生成阶段（三个节点并行）。
        """
        messages = state.get("messages", [])
        last = messages[-1] if messages else None
        has_tool_calls = bool(getattr(last, "tool_calls", None))

        if not has_tool_calls:
            return "dispatch"

        if state.get("tool_rounds", 0) >= MAX_TOOL_ROUNDS:
            logger.warning(
                "[agent] 工具调用已达上限 {} 轮，强制进入生成阶段",
                MAX_TOOL_ROUNDS,
            )
            return "dispatch"

        return "tools"

    def route_after_tools(state: State) -> str:
        """工具执行完之后去哪。

        原来是 `g.add_edge("tools", "agent")` 无条件回 agent，于是即使工具轮次
        已经用完，也还要把模型再叫起来"确认一遍"。而此时它已经不可能再调工具
        （route_after_agent 会拦），唯一能做的事就是写一段没人读的总结，早期模型下实测 21~24 秒。
        现在轮次用完就直接去 dispatch，省掉这次调用。
        """
        if state.get("tool_rounds", 0) >= MAX_TOOL_ROUNDS:
            logger.warning(
                "[agent] 工具调用已达上限 {} 轮，不再回 agent，直接进入生成阶段",
                MAX_TOOL_ROUNDS,
            )
            return "dispatch"
        return "agent"

    g = StateGraph(State)
    g.add_node("analyze", analysis_node)
    g.add_node("match", match_evidence)
    g.add_node("agent", agent_node)

    g.add_node("tools", tools_with_guard)
    g.add_node("dispatch", dispatch_generate)

    g.add_node("report", report)
    g.add_node("rewrite", rewrite_resume)
    g.add_node("questions", gen_questions)
    g.add_node("save", save_record)

    g.add_edge(START, "analyze")
    g.add_edge("analyze", "match")
    g.add_edge("match", "agent")

    # ⚠️ 同一个源节点只能有一条条件边。
    # 之前这里还额外注册了 tools_condition，两条分支会同时生效、互相打架，
    # 现在统一交给 route_after_agent 一个函数裁决。
    # 传 path_map 是为了让 LangGraph 能静态画出这张图（不传它推不出分支走向）。
    g.add_conditional_edges(
        "agent",
        route_after_agent,
        {"dispatch": "dispatch", "tools": "tools"},
    )
    # 工具执行完【不再无条件回 agent】：轮次用完就直接进生成阶段，
    # 否则会多一次只为写总结而存在的模型调用（见 agent_node 与 route_after_tools 的注释）。
    g.add_conditional_edges(
        "tools",
        route_after_tools,
        {"agent": "agent", "dispatch": "dispatch"},
    )

    # 扇出（fan-out）：dispatch 之后三个生成节点并发执行
    g.add_edge("dispatch", "report")
    g.add_edge("dispatch", "rewrite")
    g.add_edge("dispatch", "questions")

    # 扇入（fan-in）：三个生成节点都跑完，才会进入 save，且只跑一次
    g.add_edge("report", "save")
    g.add_edge("rewrite", "save")
    g.add_edge("questions", "save")
    g.add_edge("save", END)

    return g.compile()


if __name__ == "__main__":
    app = build_graph()
    result = app.invoke({
        "user_input": "招聘 AI 应用开发实习生，要求熟悉 RAG、LangChain、大模型调用",
        "analysis": {},
        "evidence": [],
        "report_text": "",
        "resume_bullets": [],
        "tailored_resume": {},
        "interview_questions": [],
        "saved": False,
        "save_error": "",
        "messages": [],
        "tool_rounds": 0,
                        })
    for m in result["messages"]:
        print(m.content)
