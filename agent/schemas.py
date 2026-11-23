from typing import Literal
from pydantic import BaseModel, Field

class JobAnalysisGraph(BaseModel):
    job_title: str = Field(default="未识别岗位", description="岗位名称，无法识别时使用未识别岗位")
    company: str = Field(default="未识别公司", description="公司名称，无法识别时使用未识别公司")
    requirements: list[str] = Field(min_length = 1, description="岗位要求的技能清单（硬性要求）")
    preferences: list[str] = Field(default_factory=list, description="岗位的加分项/偏好，例如“有 MCP 项目经验优先”；没有则返回空列表")
    keywords: list[str] = Field(min_length = 1, description="检索关键词")
    level: Literal["实习生", "初级", "中级", "高级"] = Field(description="岗位级别")

class ResumeBullet(BaseModel):
    bullet: str = Field(min_length = 1, description="根据岗位要求改写后的简历要点")
    evidence_source: str = Field(min_length = 1, description="该简历要点引用的真实经历来源")

class ResumeBulletList(BaseModel):
    bullets: list[ResumeBullet] = Field(min_length = 1, max_length = 3,description="1到3条定制简历要点")

SKILL_CATEGORIES = ("编程语言", "AI核心技术", "框架与工具", "通用技能")


class SkillGroup(BaseModel):
    category: Literal["编程语言", "AI核心技术", "框架与工具", "通用技能"] = Field(
        description="固定四类之一，不要自造分类"
    )
    items: list[str] = Field(
        default_factory=list,
        description="该分组下的技能，3~6 个短词，必须来自真实经历；不要带括号解释，不要写“Office/打字”这类无信息量的项",
    )

class TailoredBullet(BaseModel):
    project: str = Field(min_length = 1, description="这条要点所属的项目/经历名称，必须逐字来自匹配经历")
    summary: str = Field(
        min_length = 1,
        description="这条要点的概括，6~12 个字，例如“服务化改造”“检索链路评测”，用来替代“做了什么”的空泛开头",
    )
    bullet: str = Field(min_length = 1, description="针对该岗位改写后的简历要点，只能换表达，不能新增事实")
    jd_requirement: str = Field(min_length = 1, description="对应岗位分析里的哪条要求或偏好，必须逐字取自 requirements/preferences")
    evidence_source: str = Field(min_length = 1, description="事实来源，必须是匹配经历里的 source")
    evidence_span: str = Field(
        min_length = 4,
        max_length = 200,
        description="支撑这条要点的原文片段，必须从匹配经历里逐字复制（不要改写、不要省略号）",
    )
    atom_id: str = Field(
        default="",
        description="如果这条要点由某个已确认的事实原子支撑，填该原子的 id；没有对应原子时留空",
    )

class TailoredResume(BaseModel):
    headline: str = Field(min_length = 1, description="一句话定位，说明候选人与该岗位最相关的身份")
    summary: str = Field(min_length = 1, description="2到3句概述，只允许使用匹配经历里已有的事实")
    skill_groups: list[SkillGroup] = Field(default_factory=list, description="按类别归组的技能，全部来自真实经历")
    bullets: list[TailoredBullet] = Field(min_length = 1, max_length = 8, description="3到8条针对该岗位的简历要点")

class InterviewQuestion(BaseModel):
    question: str = Field(min_length = 1, description="面试问题（定制）")
    answer_points: list[str] = Field(min_length = 1, description="回答该问题时需要覆盖的要点")
    evidence_source: str = Field(min_length = 1, description="该面试问题引用的真实经历来源")

class InterviewQuestionList(BaseModel):
    questions: list[InterviewQuestion] = Field(min_length = 5, max_length = 5, description="5条定制面试问题")
