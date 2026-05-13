"""
全局工作空间 - 意识的"舞台"

基于Baars的全局工作空间理论：
- 多个无意识处理模块竞争进入有限的工作空间
- 进入的内容被广播到全系统，成为意识到的内容
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
import heapq


@dataclass
class WorkspaceContent:
    """工作空间中的内容"""
    content: str
    content_type: str           # thought/perception/memory/goal
    source_module: str          # 来源模块
    salience: float = 0.5       # 显著性（竞争优先级）
    emotion_valence: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    broadcasted: bool = False


class GlobalWorkspace:
    """
    全局工作空间 - 意识的舞台

    核心机制：
    1. 多个候选内容竞争进入
    2. 容量有限（一次1-2个主要内容）
    3. 进入后被广播到所有模块
    4. 注意力焦点决定哪个内容被深度加工
    """

    CAPACITY = 2    # 一次能容纳的主要内容数

    def __init__(self):
        self.contents: List[WorkspaceContent] = []
        self.attention_focus: Optional[str] = None
        self.broadcast_history: List[Dict] = []

    def compete(self, candidates: List[WorkspaceContent]) -> List[WorkspaceContent]:
        """
        竞争进入工作空间

        显著性计算考虑：
        - 情感显著性
        - 新奇性
        - 与当前目标的关联
        - 强度
        """
        # 按显著性排序
        candidates.sort(key=lambda c: c.salience, reverse=True)

        # 选择优胜者
        winners = candidates[:self.CAPACITY]

        # 更新工作空间内容
        self.contents = winners

        # 设置注意力焦点为最高显著性的内容
        if winners:
            self.attention_focus = winners[0].content[:50]

        # 广播到全系统
        for winner in winners:
            self._broadcast(winner)

        return winners

    def _broadcast(self, content: WorkspaceContent):
        """广播内容到所有模块"""
        content.broadcasted = True
        self.broadcast_history.append({
            "content": content.content[:100],
            "type": content.content_type,
            "source": content.source_module,
            "salience": content.salience,
            "timestamp": datetime.now()
        })

        # 限制历史长度
        if len(self.broadcast_history) > 100:
            self.broadcast_history = self.broadcast_history[-100:]

    def add_content(self, content: str, content_type: str,
                   source_module: str, salience: float = 0.5,
                   emotion_valence: float = 0.0):
        """直接添加内容到工作空间（绕过竞争）"""
        item = WorkspaceContent(
            content=content,
            content_type=content_type,
            source_module=source_module,
            salience=salience,
            emotion_valence=emotion_valence
        )

        self.contents.append(item)

        # 如果超过容量，移除最低显著性的
        if len(self.contents) > self.CAPACITY:
            self.contents.sort(key=lambda c: c.salience)
            removed = self.contents.pop(0)
            # 被移除的内容"淡出"意识

        # 广播
        self._broadcast(item)

    def get_current_contents(self) -> List[str]:
        """获取当前意识内容"""
        return [c.content for c in self.contents]

    def get_focus(self) -> Optional[str]:
        """获取当前注意力焦点"""
        return self.attention_focus

    def clear(self):
        """清空工作空间"""
        self.contents = []
        self.attention_focus = None

    def to_dict(self) -> Dict:
        return {
            "contents": self.get_current_contents(),
            "attention_focus": self.attention_focus,
            "capacity": self.CAPACITY
        }
