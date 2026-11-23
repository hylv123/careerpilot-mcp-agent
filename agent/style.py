"""去 AI 味的写作约束与检测。

为什么单独一层：
    "不要写得太像 AI" 如果只写进 prompt，模型照样会写"通过…实现…，不仅…而且…"。
    这里做两件事：
      1. 把约束固化成一份规则文本（prompt 里直接引用，避免各处措辞不一致）；
      2. 生成后做一次确定性检测，命中就带着"你犯了这些词"重写一次。
    检测只看词表和句法特征，不做语义判断——它拦不住"意思空洞"，
    但能拦住绝大多数一眼假的腔调。

复查口径：宁可少拦，不要误伤技术词。
    比如"沉淀"在"数据沉淀"里是术语，但简历语境下几乎都是 AI 腔，所以按命中处理；
    而"架构""链路"这类真实工程词不在词表里。
"""

from __future__ import annotations

import re
from typing import Iterable

# 这些词一出现，基本就是"模板腔"：不是技术事实，只是语气填充。
BANNED_PHRASES: tuple[str, ...] = (
    "赋能",
    "闭环",
    "抓手",
    "生态",
    "全链路",
    "助力",
    "致力于",
    "打造",
    "沉淀",
    "全方位",
    "多维度",
    "一站式",
    "无缝",
    "至关重要",
    "不可或缺",
    "值得一提的是",
    "综上所述",
    "总而言之",
    "极大地",
    "深耕",
    "深度融合",
    "强力支撑",
)

# 结构性套话：出现两次以上就算腔调重复。
_REPEATED_FRAMES: tuple[str, ...] = (
    "基于",
    "通过",
    "负责",
    "参与",
)

ANTI_AI_RULES = """写作风格（必须遵守，读者是招聘方，不是评审专家）：
1. 用「动词 + 做了什么 + 结果/约束」写句子，不要用形容词堆砌价值；
2. 单句不超过 45 个汉字，长短句交替，避免每条要点的句式完全一样；
3. 禁止使用这些词：赋能、闭环、抓手、生态、全链路、助力、致力于、打造、沉淀、
   全方位、多维度、一站式、无缝、至关重要、不可或缺、值得一提的是、综上所述、
   总而言之、极大地、深耕、深度融合、强力支撑；
4. 不要连续用「基于…」「通过…」「负责…」开头的句式；同一段里最多出现一次；
5. 不要为了凑气势写三连排比，也不要给每句话都加"显著提升/有效解决"这类评价；
6. 只做表达层改写：不新增事实、不新增数字、不改变技术栈和职责范围。"""


def style_issues(texts: Iterable[str]) -> list[str]:
    """返回命中的 AI 腔词汇（去重、保持首次出现顺序）。

    Args:
        texts: 待检测文本（标题、概述、各条要点）。

    Returns:
        命中的词列表；空列表表示没检测到明显腔调。
    """
    joined = "\n".join(str(text or "") for text in texts)
    hits: list[str] = []
    for phrase in BANNED_PHRASES:
        if phrase in joined and phrase not in hits:
            hits.append(phrase)
    for frame in _REPEATED_FRAMES:
        if len(re.findall(rf"(?:^|[。；;\n])\s*{frame}", joined)) >= 3 and frame not in hits:
            hits.append(f"{frame}…（连续 3 句同头）")
    return hits


def style_correction(hits: list[str]) -> str:
    """把命中项变成一句可直接塞进 prompt 的纠正要求。"""
    if not hits:
        return ""
    listed = "、".join(hits)
    return (
        f"上一版出现了这些 AI 腔表达：{listed}。"
        "请重写受影响的句子：去掉这些词，改成具体动作和结果的陈述，"
        "保持所有事实、数字、技术栈与来源不变。"
    )
