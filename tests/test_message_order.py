"""消息归一化测试"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


def test_reorder_system_first():
    """SystemMessage 应全部移到非 SystemMessage 之前"""
    from agents.nodes import _reorder_system_first

    messages = [
        SystemMessage(content="sys1"),
        HumanMessage(content="human1"),
        SystemMessage(content="sys2"),
        AIMessage(content="ai1"),
        SystemMessage(content="sys3"),
    ]
    result = _reorder_system_first(messages)

    assert len(result) == 5
    assert all(isinstance(m, SystemMessage) for m in result[:3])
    assert isinstance(result[3], HumanMessage)
    assert isinstance(result[4], AIMessage)
    # 顺序保持
    assert result[0].content == "sys1"
    assert result[1].content == "sys2"
    assert result[2].content == "sys3"


def test_normalize_message_order_named_system():
    """连续具名 SystemMessage 按 name 排序"""
    from agents.nodes import _normalize_message_order

    sys_b = SystemMessage(content="b", name="web_search")
    sys_a = SystemMessage(content="a", name="memory_search")
    sys_c = SystemMessage(content="c", name="knowledge_search")
    human = HumanMessage(content="hello")

    messages = [sys_b, sys_c, sys_a, human]
    result = _normalize_message_order(messages)

    assert len(result) == 4
    # 连续的具名 SystemMessage 按 name 字母序
    assert result[0].name == "knowledge_search"
    assert result[1].name == "memory_search"
    assert result[2].name == "web_search"
    assert result[3].content == "hello"


def test_normalize_and_reorder_combined():
    """组合使用：先 normalize 再 reorder_system_first"""
    from agents.nodes import _normalize_message_order, _reorder_system_first

    sys_named = SystemMessage(content="named", name="zzz")
    human = HumanMessage(content="human")
    sys_unnamed = SystemMessage(content="unnamed")

    messages = [sys_named, human, sys_unnamed]
    result = _reorder_system_first(_normalize_message_order(messages))

    assert len(result) == 3
    assert all(isinstance(m, SystemMessage) for m in result[:2])
    assert isinstance(result[2], HumanMessage)
