from typing import TypedDict, Annotated, Any
from langgraph.graph.message import add_messages

class State(TypedDict):
    user_input: str
    analysis: dict[str, Any]
    evidence: list[dict[str, Any]]
    # 报告正文单独存一个字段，而不是只塞进 messages：
    # report / rewrite / questions 三个节点并行执行后，
    # messages 里这三条的先后次序是不确定的，
    # 下游（CLI、HTTP 接口）要用固定顺序展示就必须读确定的字段。
    report_text: str
    resume_bullets: list[dict[str, Any]]
    interview_questions: list[dict[str, Any]]
    messages: Annotated[list[Any], add_messages]
    tool_rounds: int