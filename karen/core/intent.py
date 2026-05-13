"""意图识别系统 - 双层架构（规则引擎 + 上下文感知）。

识别策略：
1. 规则引擎（0ms）：正则/关键词匹配常见模式，覆盖 80% 场景
2. 上下文感知（0ms）：结合历史消息判断澄清/追问

使用方式：
    result = classify_intent_sync("你好", history=[])
    if result.skip_search:
        # 直接回复，跳过搜索
"""

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from langchain_core.messages import BaseMessage

logger = logging.getLogger(__name__)

# ========== 意图类型定义 ==========


class IntentType(str, Enum):
    """用户消息意图类型"""

    GREETING = "greeting"          # 问候（你好、hi）
    FAREWELL = "farewell"          # 告别（再见、bye）
    THANKS = "thanks"              # 感谢（谢谢、thx）
    CHITCHAT = "chitchat"          # 闲聊（在吗、忙吗）
    MATH = "math"                  # 数学计算（2+2、x²+3x）
    CODING = "coding"              # 代码请求（写个函数）
    CREATIVE = "creative"          # 创意写作（写诗、编故事）
    FACTUAL = "factual"            # 事实查询（什么是、为什么）→ 需搜索
    OPINION = "opinion"            # 观点询问（你怎么看）
    CLARIFY = "clarify"            # 澄清/追问（详细说说、举个例子）
    TRANSLATION = "translation"    # 翻译请求
    COMPARISON = "comparison"      # 对比请求（A和B的区别）
    UNKNOWN = "unknown"            # 不确定


# ========== 意图结果 ==========


@dataclass
class IntentResult:
    """意图识别结果"""

    intent: str
    confidence: float = 1.0          # 0.0-1.0
    skip_search: bool = False        # 是否跳过联网搜索
    skip_memory: bool = False        # 是否跳过记忆搜索
    skip_knowledge: bool = False     # 是否跳过知识库
    use_coding_prompt: bool = False  # 是否使用代码专用提示词
    # 上下文相关：如果是 clarify，这里记录上一轮意图
    parent_intent: Optional[str] = None
    # 分类器来源："rule" | "context" | "llm"
    source: str = "rule"


# ========== 第一层：规则引擎 ==========

# 问候语（中英文）
_GREETING_PATTERNS = [
    r"^\s*(你?好|您好|嗨|hello|hi|hey)\s*[!.！。]?\s*$",
    r"^\s*(早上好|下午好|晚上好|早安|晚安)\s*[!.！。]?\s*$",
    r"^\s*(在吗|在嘛|在?|有人吗)\s*[?？]?\s*$",
]

# 告别语
_FAREWELL_PATTERNS = [
    r"^\s*(再见|拜拜|bye|goodbye|see you|下次见)\s*[!.！。]?\s*$",
]

# 感谢
_THANKS_PATTERNS = [
    r"^\s*(谢谢|感谢|thx|thanks|thank you|谢了|多谢)\s*[!.！。]?\s*$",
]

# 数学表达式（简单判断）
_MATH_PATTERNS = [
    r"[\d\s]+[+\-*/^=]+[\d\s+xX]+",           # 2+2, x^2+3
    r"^\s*计算\s*[:：]?\s*",                     # 计算：...
    r"^\s*等于多少\s*[?？]?\s*$",               # ...等于多少
    r"[\d\s]+的\s*[\d\s]+次方",                 # 2的10次方
    r"[\d\s]+[倍%％分之和差积商]",               # 百分比、倍数
]

# 代码请求
_CODING_KEYWORDS = [
    "写个", "写一个", "写一段", "给个", "给一段",
    "代码", "函数", "程序", "脚本", "实现",
    "python", "javascript", "js", "java", "c++", "go", "rust",
    "algorithm", "算法", "排序", "递归",
]

# 创意写作
_CREATIVE_KEYWORDS = [
    "写一首", "写篇", "编个", "编一个", "创作",
    "诗", "故事", "小说", "剧本", "歌词",
    " poem", " story", " novel", " song",
]

# 翻译
_TRANSLATION_KEYWORDS = [
    "翻译成", "翻译为", "用中文", "用英文", "用日语",
    "translate", "translation",
]

# 对比
_COMPARISON_KEYWORDS = [
    "区别", "对比", "比较", "versus", "vs", "和.*不同",
    "哪个更好", "哪个更快", "哪个更强",
]

# 事实查询（需要搜索）
_FACTUAL_KEYWORDS = [
    "什么是", "为什么", "怎么", "如何",
    "介绍", "解释", "说明", "原理", "历史",
    "who is", "what is", "how to", "why",
    "search", "find", "look up", "查询", "搜索",
    "最新", "news", "趋势", "trend",
    # 时效/实时类（覆盖"今天沈阳天气""现在股价"这类隐式时效查询）
    "今天", "现在", "目前", "近期", "最近",
    "天气", "股价", "汇率", "新闻", "热点",
    "today", "weather", "stock", "price", "current",
]

# 闲聊
_CHITCHAT_PATTERNS = [
    r"^\s*(嗯|哦|啊|哈哈|呵呵|嘿嘿)\s*$",
    r"^\s*(好的|ok|可以|行|没问题)\s*[!.！。]?\s*$",
    r"^\s*(真的吗|是吗|不会吧|太棒了|厉害了)\s*[?？!.！。]?\s*$",
]

# 澄清/追问
_CLARIFY_PATTERNS = [
    r"^\s*(详细说说|详细点|再详细|展开讲讲|具体说说)\s*$",
    r"^\s*(举个例子|举例说明|比如呢|例如)\s*[?？]?\s*$",
    r"^\s*(然后呢|还有呢|接着说|继续说)\s*[?？]?\s*$",
    r"^\s*(为什么|怎么回事|什么意思|怎么理解)\s*[?？]?\s*$",
    r"^\s*(那.*呢|还有.*吗|.*怎么样)\s*[?？]?\s*$",
]


def _match_patterns(text: str, patterns: list[str]) -> bool:
    """正则匹配"""
    t = text.strip().lower()
    for p in patterns:
        if re.search(p, t):
            return True
    return False


def _match_keywords(text: str, keywords: list[str], min_len: int = 4) -> bool:
    """关键词匹配（至少命中一个）"""
    if len(text.strip()) < min_len:
        return False
    t = text.lower()
    return any(kw.lower() in t for kw in keywords)


def _rule_classify(query: str) -> Optional[IntentResult]:
    """规则引擎分类。命中则返回结果，未命中返回 None（交给下一层）。"""
    q = query.strip()

    # 1. 问候（最高优先级，短句）
    if len(q) <= 15 and _match_patterns(q, _GREETING_PATTERNS):
        return IntentResult(IntentType.GREETING, confidence=0.95,
                            skip_search=True, skip_memory=True, skip_knowledge=True)

    # 2. 告别
    if len(q) <= 15 and _match_patterns(q, _FAREWELL_PATTERNS):
        return IntentResult(IntentType.FAREWELL, confidence=0.95,
                            skip_search=True, skip_memory=True, skip_knowledge=True)

    # 3. 感谢
    if len(q) <= 15 and _match_patterns(q, _THANKS_PATTERNS):
        return IntentResult(IntentType.THANKS, confidence=0.95,
                            skip_search=True, skip_memory=True, skip_knowledge=True)

    # 4. 数学计算
    if _match_patterns(q, _MATH_PATTERNS):
        return IntentResult(IntentType.MATH, confidence=0.85,
                            skip_search=True, skip_memory=True, skip_knowledge=True)

    # 5. 代码请求
    if _match_keywords(q, _CODING_KEYWORDS):
        return IntentResult(IntentType.CODING, confidence=0.80,
                            skip_search=True, skip_memory=True, skip_knowledge=True,
                            use_coding_prompt=True)

    # 6. 翻译
    if _match_keywords(q, _TRANSLATION_KEYWORDS):
        return IntentResult(IntentType.TRANSLATION, confidence=0.80,
                            skip_search=True, skip_memory=False, skip_knowledge=True)

    # 7. 创意写作
    if _match_keywords(q, _CREATIVE_KEYWORDS):
        return IntentResult(IntentType.CREATIVE, confidence=0.80,
                            skip_search=True, skip_memory=True, skip_knowledge=True)

    # 8. 对比 - 一般知识类比较模型自己就会，不浪费 4-5s 等 web search 超时
    if _match_keywords(q, _COMPARISON_KEYWORDS):
        return IntentResult(IntentType.COMPARISON, confidence=0.75,
                            skip_search=True, skip_memory=True, skip_knowledge=True)

    # 9. 事实查询（明确需要搜索）
    if _match_keywords(q, _FACTUAL_KEYWORDS):
        return IntentResult(IntentType.FACTUAL, confidence=0.75,
                            skip_search=False, skip_memory=False, skip_knowledge=False)

    # 10. 闲聊
    if len(q) <= 20 and _match_patterns(q, _CHITCHAT_PATTERNS):
        return IntentResult(IntentType.CHITCHAT, confidence=0.80,
                            skip_search=True, skip_memory=True, skip_knowledge=True)

    # 未命中规则
    return None


# ========== 第二层：上下文感知 ==========


def _context_classify(query: str, history: list[BaseMessage]) -> Optional[IntentResult]:
    """结合历史消息判断意图。

    主要用于识别：
    - 澄清/追问（"详细说说"、"举个例子"）
    - 多轮对话中的上下文延续
    """
    if not history or len(history) < 2:
        return None

    q = query.strip()

    # 1. 澄清/追问检测
    if _match_patterns(q, _CLARIFY_PATTERNS):
        # 找上一轮用户的意图（从最近的人类消息推断）
        parent = _infer_parent_intent(history)
        return IntentResult(
            IntentType.CLARIFY,
            confidence=0.80,
            skip_search=parent.skip_search if parent else False,
            skip_memory=True,  # 澄清通常不需要查记忆
            skip_knowledge=parent.skip_knowledge if parent else False,
            parent_intent=parent.intent if parent else None,
            source="context",
        )

    return None


def _infer_parent_intent(history: list[BaseMessage]) -> Optional[IntentResult]:
    """从历史消息推断上一轮的主要意图。"""
    # 找最近的用户消息（排除当前这条）
    for msg in reversed(history[:-1]):
        content = getattr(msg, "content", "")
        if content:
            # 尝试用规则识别上一轮意图
            result = _rule_classify(content)
            if result:
                return result
    return None


# ========== 主入口 ==========


def classify_intent_sync(
    query: str,
    history: Optional[list[BaseMessage]] = None,
) -> IntentResult:
    """同步版本的意图识别（仅使用规则和上下文，无 LLM）。

    用于无法使用 async 的场景（如图编译期的路由判断）。
    """
    # 第一层：规则引擎
    result = _rule_classify(query)
    if result:
        return result

    # 第二层：上下文感知
    if history:
        result = _context_classify(query, history)
        if result:
            return result

    return IntentResult(
        IntentType.UNKNOWN,
        confidence=0.3,
        skip_search=False,
        skip_memory=False,
        skip_knowledge=False,
        source="sync_fallback",
    )
