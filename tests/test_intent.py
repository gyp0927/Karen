"""意图分类边界 case 测试"""

import pytest

from core.intent import IntentType, classify_intent_sync


@pytest.mark.parametrize(
    "query,expected_intent,skip_search",
    [
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
    ],
)
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


# ========== Skill 触发测试 ==========


def test_skill_trigger_by_keyword():
    """关键词应触发对应的 skill"""
    result = classify_intent_sync("帮我看看这段代码")
    assert result.intent == IntentType.SKILL
    assert result.skill_name == "code-review"
    assert result.source == "skill"


def test_skill_trigger_exact():
    """精确 trigger 匹配"""
    result = classify_intent_sync("代码审查")
    assert result.intent == IntentType.SKILL
    assert result.skill_name == "code-review"


def test_skill_trigger_code_review_english():
    """英文 trigger 也应匹配"""
    result = classify_intent_sync("review this code")
    assert result.intent == IntentType.SKILL
    assert result.skill_name == "code-review"


def test_skill_trigger_debug():
    """debug 关键词触发 debug skill"""
    result = classify_intent_sync("调试")
    assert result.intent == IntentType.SKILL
    assert result.skill_name == "debug"


def test_skill_trigger_refactor():
    """重构关键词触发 refactor skill"""
    result = classify_intent_sync("这段代码太乱了")
    assert result.intent == IntentType.SKILL
    assert result.skill_name == "refactor"


def test_skill_exit_detection():
    """退出关键词应清空 active_skill"""
    result = classify_intent_sync("退出审查", active_skill="code-review")
    assert result.source == "skill_exit"
    assert result.intent == IntentType.UNKNOWN


def test_skill_exit_done():
    """done 也应视为退出"""
    result = classify_intent_sync("我完成了", active_skill="debug")
    assert result.source == "skill_exit"


def test_skill_preserved_when_no_trigger():
    """非退出、非触发消息应保持原有 skill"""
    result = classify_intent_sync("继续分析")
    # 没有 active_skill 时不应触发 skill
    assert result.intent != IntentType.SKILL
    assert result.skill_name is None
