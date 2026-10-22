from typing import Literal
from pydantic import BaseModel, Field

class JobAnalysisGraph(BaseModel):
    job_title: str = Field(default="未识别岗位", description="岗位名称，无法识别时使用未识别岗位")
    company: str = Field(default="未识别公司", description="公司名称，无法识别时使用未识别公司")
    requirements: list[str] = Field(min_length = 1, description="岗位要求的技能清单")
    keywords: list[str] = Field(min_length = 1, description="检索关键词")
    level: Literal["实习生", "初级", "中级", "高级"] = Field(description="岗位级别")

class ResumeBullet(BaseModel):
    bullet: str = Field(min_length = 1, description="根据岗位要求改写后的简历要点")
    evidence_source: str = Field(min_length = 1, description="该简历要点引用的真实经历来源")

class ResumeBulletList(BaseModel):
    bullets: list[ResumeBullet] = Field(min_length = 1, max_length = 3,description="1到3条定制简历要点")

class InterviewQuestion(BaseModel):
    question: str = Field(min_length = 1, description="面试问题（定制）")
    answer_points: list[str] = Field(min_length = 1, description="回答该问题时需要覆盖的要点")
    evidence_source: str = Field(min_length = 1, description="该面试问题引用的真实经历来源")

class InterviewQuestionList(BaseModel):
    questions: list[InterviewQuestion] = Field(min_length = 5, max_length = 5, description="5条定制面试问题")
