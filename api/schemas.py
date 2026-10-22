"""接口层的数据契约（Request / Response）。

为什么单独拆一个 schemas.py：

    把"进来什么样、出去什么样"集中定义，路由函数里就不会散落校验逻辑。
    FastAPI 会读这些类自动生成 /docs 交互文档，别人不用问你怎么调。

    你在 agent/schemas.py 里已经用 Pydantic 约束过大模型的结构化输出，
    这里是同一套东西，只是用在 HTTP 入参上——这是知识复用，不是新概念。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    """岗位分析请求体。

    Attributes:
        jd: 岗位 JD 原文。设上下限是为了在入口就挡掉非法输入，
            避免把一次昂贵的大模型调用浪费在一段空文本上。
    """

    jd: str = Field(
        ...,
        min_length=10,
        max_length=10000,
        description="岗位 JD 原文",
    )


class AnalyzeResponse(BaseModel):
    """岗位分析响应体。

    这里返回的是**结构化字段**而不是一整段报告文本。
    因为你的 graph 本来就已经产出了结构化的 analysis / resume_bullets /
    interview_questions，把它们原样透出比拼一段字符串有用得多——
    前端可以分块渲染，别的服务可以按字段消费。

    Attributes:
        status: done 或 failed。
        analysis: 岗位结构化分析结果。
        evidence_count: 命中并经判定为相关的经历条数。
        report_text: 报告正文。单独透出是因为三个生成节点并行后，
                     messages 里的顺序不确定，接口必须给调用方确定的东西。
        resume_bullets: 定制简历要点。
        interview_questions: 生成的面试题。
    """

    status: str = Field(description="done | failed")
    analysis: dict[str, Any] = Field(default_factory=dict)
    evidence_count: int = Field(default=0)
    report_text: str = Field(default="")
    resume_bullets: list[dict[str, Any]] = Field(default_factory=list)
    interview_questions: list[dict[str, Any]] = Field(default_factory=list)


class TaskAccepted(BaseModel):
    """异步任务受理回执（HTTP 202）。

    为什么要立刻回一个 task_id 而不是让客户端干等：
        一次分析要 50~120 秒，远超网关、浏览器、负载均衡常见的 30~60 秒超时。
        同步等待的结果往往是"连接被断开，但服务端还在跑"——
        用户既拿不到结果，也不知道该不该重试，最坏的情况是重复提交烧两份钱。

    Attributes:
        task_id: 任务编号，用 GET /tasks/{task_id} 查询结果。
        status: 受理时的状态，恒为 pending。
        reused: True 表示命中幂等 —— 这份 JD 已有任务在跑或已完成，
                task_id 指向的是那个已存在的任务，不是新开的。
    """

    task_id: str = Field(description="任务编号")
    status: str = Field(default="pending", description="pending")
    reused: bool = Field(
        default=False, description="是否复用了同一份 JD 的已有任务"
    )


class TaskStatusResponse(BaseModel):
    """任务状态查询结果。

    Attributes:
        task_id: 任务编号。
        status: pending | running | done | failed。
        progress: 人类可读的进度描述。
        error: 失败时的错误信息。
        updated_at: 最后一次更新状态的 Unix 秒级时间戳。
        result: 仅在 status 为 done 时存在。
    """

    task_id: str
    status: str = Field(description="pending | running | done | failed")
    progress: str = Field(default="")
    error: str = Field(default="")
    updated_at: int = Field(default=0)
    result: AnalyzeResponse | None = Field(default=None)
