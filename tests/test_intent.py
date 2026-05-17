"""意图分类边界 case 测试"""

import pytest
from core.intent import classify_intent_sync, IntentType


@pytest.mark.parametrize("query,expected_intent,skip_search", [
    ("你好", IntentType.GREETING, True),
    ("早上好", IntentType.GREETING, True),
    ("再见", IntentType.FAREWELL, True),
    ("谢谢", IntentType.THANKS, True),
    ("3+5等于多少", IntentType.MATH, True),
    ("写个Python排序函数", IntentType.CODING, True),
    ("写一首诗", IntentType.CREATIVE, True),
    ("什么是量子计算", IntentType.FACTUAL, False),
    ("为什么天空是蓝色的", IntentType.FACTUAL, False),
    ("最新新闻", IntentType.FACTUAL, False),
    ("翻译成英文", IntentType.TRANSLATION, True),
])
def test_intent_classification(query, expected_intent, skip_search):
    result = classify_intent_sync(query)
    assert result.intent == expected_intent
    assert result.skip_search == skip_search


def test_coding_intent_with_code_blocks():
    """包含代码块标记应识别为编码意图"""
    result = classify_intent_sync("帮我写一个 ```python 函数")
    assert result.intent == IntentType.CODING
    assert result.use_coding_prompt is True


def test_non_coding_just_language_name():
    """纯语言名称不含编码动作词 → 非编码意图（由 model_router 的 _is_coding_intent 处理）"""
    result = classify_intent_sync("今天天气怎么样")
    assert result.intent != IntentType.CODING


def test_intent_unknown_fallback():
    """无法识别的输入应 fallback 到 unknown"""
    result = classify_intent_sync("xyzabc123 未定义")
    assert result.intent == IntentType.UNKNOWN
    assert result.confidence == 0.3
    assert result.source == "sync_fallback"
