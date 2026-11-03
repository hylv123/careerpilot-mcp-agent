import json
import random
import threading

from langgraph.graph import END, START, StateGraph
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel
from knowledge.retrieve import retrieve_agentic
from typing import Any, TypeVar, ParamSpec
from collections.abc import Callable
from functools import wraps
from time import perf_counter, sleep
from loguru import logger
from .schemas import JobAnalysisGraph, ResumeBulletList, InterviewQuestionList
from .llm import get_ds_chat_model
from .state import State
from .mcp_tools import get_mcp_tools, is_tool_error


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
# 串行耗时是三者之和（实测约 95s），并行后是三者中最大的那个（约 45s）。
GENERATE_NODES: list[str] = ["report", "rewrite", "questions"]

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

    evidence = retrieve_agentic(keywords, 4)

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
        "请根据下面的岗位分析和匹配经历，生成一份求职匹配报告，每段必须标注来源。"
        "如果需要简历、项目等补充信息，可以先调用可用工具获取，再输出报告。\n\n"
        f"岗位分析：\n{json.dumps(state['analysis'], ensure_ascii=False)}\n\n"
        f"匹配经历：\n{json.dumps(evidence, ensure_ascii=False)}"
    )
    return {"evidence": evidence, "messages": [("user", context)]}

@trace_node("report")
def report(state: State) -> dict[str, Any]:
    """生成报告（带引用）"""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是求职顾问。基于岗位要求和匹配经历生成报告，每段必须标注来源。"),
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
            "evidence": json.dumps(state["evidence"], ensure_ascii=False),
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
    """根据岗位要求和真实经历生成定制简历要点。"""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是一名简历优化顾问。"
                "请根据岗位分析和匹配经历生成1到3条简历要点。"
                "不得编造不存在的经历。"
                "每条要点的 evidence_source 必须来自匹配经历中的 source 字段。"
                "必须通过指定的结构化工具返回结果，不要返回普通文本段落。",
            ),
            (
                "human",
                "岗位分析：{analysis}\n\n"
                "匹配经历：{evidence}",
            ),
        ]
    )

    chain = prompt | _get_llm(TIMEOUT_BULLETS).with_structured_output(
        ResumeBulletList,
        method="function_calling",
    )
    result = invoke_structured(
        chain,
        {
            "analysis": json.dumps(
                state["analysis"],
                ensure_ascii=False,
            ),
            "evidence": json.dumps(
                state["evidence"],
                ensure_ascii=False,
            ),
        },
        ResumeBulletList
    )

    bullet_dicts = [bullet.model_dump() for bullet in result.bullets]
    bullet_text = "\n".join(
        f"- {bullet.bullet}\n 来源: {bullet.evidence_source}" for bullet in result.bullets
    )

    return {
        "resume_bullets": bullet_dicts,
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
                "必须通过指定的结构化工具返回结果，不要返回普通文本段落。",
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
            "evidence": json.dumps(
                state["evidence"],
                ensure_ascii=False,
            ),
        },
        InterviewQuestionList
    )

    questions_dicts = [question.model_dump() for question in result.questions]

    questions_sections: list[str] = []
    for idx, ques in enumerate(result.questions, start=1):
        ans_text = "\n".join(f"- {point}" for point in ques.answer_points)
        questions_sections.append(
            f"### Q{idx}: {ques.question}\n\n"
            f"{ans_text}\n\n"
            f"来源：{ques.evidence_source}"
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
#   于是按"聊天助手"的默认习惯回应 —— 写一段几百字的总结（实测 21~58 秒）。
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


def save_record_impl(state: State, save_tool: Any) -> dict[str, Any]:
    """把本次岗位写进 SQLite，并把"到底存没存成功"变成结构化字段。

    Args:
        state: 当前图状态。
        save_tool: 已绑定的 save_job 工具；工具发现失败时为 None。

    Returns:
        含 saved / save_error / messages 三个键的部分状态。

    为什么不再把结果只塞进 messages：
        messages 里的那句"已保存/暂时不可用"没有任何下游消费者会去判定，
        于是"岗位没写进库"和"写进去了"在接口层长得一模一样——
        任务状态是 done、报告也正常，用户以为记录上了，其实没有。
        静默丢数据比直接失败更难查，所以必须落成 saved 字段。
    """
    if save_tool is None:
        message = "投递记录未保存：save_job MCP 工具不可用"
        return {"saved": False, "save_error": message, "messages": [("assistant", message)]}

    analysis = state.get("analysis", {})
    payload = {
        "title": analysis.get("job_title") or "未识别岗位",
        "company": analysis.get("company") or "未识别公司",
        "jd": state["user_input"],
    }
    result = save_tool.invoke(payload)
    ok = not is_tool_error(result)
    return {
        "saved": ok,
        "save_error": "" if ok else str(result),
        "messages": [("assistant", str(result))],
    }


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
        #   这段总结根本没有任何消费者——实测每次白白烧掉 21~24 秒和一份 token。
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
        return save_record_impl(state, tool_map.get("save_job"))

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
        """
        result = _tool_node.invoke(state)
        rounds = state.get("tool_rounds", 0) + 1
        logger.info("[agent] 工具调用第 {} 轮", rounds)
        return {**result, "tool_rounds": rounds}

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
        （route_after_agent 会拦），唯一能做的事就是写一段没人读的总结，实测 21~24 秒。
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
        "interview_questions": [],
        "saved": False,
        "save_error": "",
        "messages": [],
        "tool_rounds": 0,
                        })
    for m in result["messages"]:
        print(m.content)
