"""
记忆系统 - 人类思维的时间维度

基于Tulving的多重记忆系统模型：
- 感觉记忆、工作记忆、情景记忆、语义记忆、程序记忆
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import uuid
import math
import random


@dataclass
class Memory:
    """记忆的基本单元"""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    content: str = ""                           # 记忆内容
    memory_type: str = "episodic"               # episodic/semantic/procedural/emotional
    timestamp: datetime = field(default_factory=datetime.now)
    emotion_valence: float = 0.0                # 编码时的情感标记
    emotion_intensity: float = 0.5              # 情感强度
    importance: float = 0.5                     # 重要性（0-1）
    retrieval_count: int = 0                    # 被提取次数
    last_retrieved: Optional[datetime] = None   # 上次提取时间
    retrieval_cues: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0                     # 记忆可靠性（重构后可能降低）

    def get_age_hours(self) -> float:
        """获取记忆的年龄（小时）"""
        delta = datetime.now() - self.timestamp
        return delta.total_seconds() / 3600

    def get_retrievability(self, current_emotion_valence: float = 0.0) -> float:
        """
        计算记忆的可提取性

        基于：
        1. 艾宾浩斯遗忘曲线
        2. 提取练习效应
        3. 情感一致性
        4. 重要性
        """
        age_hours = self.get_age_hours()

        # 艾宾浩斯遗忘曲线: R = e^(-t/S)
        # S 取决于重要性（重要记忆遗忘更慢）
        stability = 24 * (1 + self.importance * 4)  # 基础稳定时间
        forgetting = math.exp(-age_hours / stability)

        # 提取练习效应：提取过的记忆更稳定
        retrieval_boost = min(0.3, self.retrieval_count * 0.05)

        # 情感一致性：与当前情绪一致的记忆更容易提取
        emotion_match = 1.0 - abs(current_emotion_valence - self.emotion_valence) * 0.5
        emotion_match = max(0.1, emotion_match)

        # 闪光灯记忆效应：高情绪强度记忆更容易提取
        flashbulb_boost = min(0.2, self.emotion_intensity * 0.1)

        # 综合
        retrievability = (forgetting + retrieval_boost + flashbulb_boost) * emotion_match
        return min(1.0, retrievability)


class WorkingMemory:
    """
    工作记忆 - 意识的临时工作台

    基于Baddeley模型：
    - 中央执行系统（注意力控制）
    - 语音环路
    - 视觉空间画板
    """

    CAPACITY = 4        # Cowan的估计：4±1个组块
    DECAY_SECONDS = 30  # 20-30秒自动衰减

    def __init__(self):
        self.slots: List[Dict] = []         # 当前内容
        self.focus: Optional[str] = None    # 当前注意力焦点
        self.cognitive_load: float = 0.0    # 认知负荷

    def add(self, content: str, content_type: str = "semantic",
            priority: float = 0.5) -> bool:
        """
        添加内容到工作记忆

        如果满，则根据优先级替换
        """
        item = {
            "content": content,
            "type": content_type,
            "priority": priority,
            "entry_time": datetime.now()
        }

        if len(self.slots) < self.CAPACITY:
            self.slots.append(item)
        else:
            # 替换优先级最低的
            min_idx = min(range(len(self.slots)),
                         key=lambda i: self.slots[i]["priority"])
            if self.slots[min_idx]["priority"] < priority:
                self.slots[min_idx] = item
            else:
                return False  # 无法插入

        self._update_load()
        return True

    def clear_expired(self):
        """清除过期的内容"""
        now = datetime.now()
        self.slots = [
            slot for slot in self.slots
            if (now - slot["entry_time"]).total_seconds() < self.DECAY_SECONDS
        ]
        self._update_load()

    def get_contents(self) -> List[str]:
        """获取当前工作记忆中的内容"""
        self.clear_expired()
        return [slot["content"] for slot in self.slots]

    def set_focus(self, focus: str):
        """设置注意力焦点"""
        self.focus = focus

    def _update_load(self):
        """更新认知负荷"""
        self.cognitive_load = len(self.slots) / self.CAPACITY

    def to_dict(self) -> Dict:
        return {
            "slots": self.get_contents(),
            "focus": self.focus,
            "load": self.cognitive_load,
            "capacity": self.CAPACITY
        }


class MemorySystem:
    """
    记忆系统 - 所有记忆的统一管理

    包含：
    - 工作记忆（短期）
    - 情景记忆（个人经历）
    - 语义记忆（一般知识）
    - 程序记忆（技能）
    """

    def __init__(self):
        self.working_memory = WorkingMemory()

        # 长期记忆存储
        self.episodic_memories: List[Memory] = []   # 情景记忆
        self.semantic_memories: List[Memory] = []   # 语义记忆
        self.procedural_memories: List[Memory] = [] # 程序记忆

        # 自传体记忆索引（重要记忆）
        self.autobiographical_memories: List[str] = []

        # 关联网络（概念连接）
        self.association_network: Dict[str, List[Tuple[str, float]]] = {}

    def encode(self, content: str, memory_type: str = "episodic",
               emotion_valence: float = 0.0, emotion_intensity: float = 0.5,
               importance: float = 0.5, context: Dict = None,
               cues: List[str] = None) -> str:
        """
        编码新记忆

        模拟记忆的编码过程：
        1. 首先进入工作记忆
        2. 根据重要性决定是否进入长期记忆
        3. 高情绪事件优先编码
        """
        memory = Memory(
            content=content,
            memory_type=memory_type,
            emotion_valence=emotion_valence,
            emotion_intensity=emotion_intensity,
            importance=importance,
            context=context or {},
            retrieval_cues=cues or []
        )

        # 存入工作记忆
        self.working_memory.add(content, memory_type, importance)

        # 根据类型存入长期记忆
        if memory_type == "episodic":
            self.episodic_memories.append(memory)
            # 高重要性或高情绪的成为自传体记忆
            if importance > 0.7 or emotion_intensity > 0.8:
                self.autobiographical_memories.append(memory.id)
        elif memory_type == "semantic":
            self.semantic_memories.append(memory)
        elif memory_type == "procedural":
            self.procedural_memories.append(memory)

        # 更新关联网络
        self._update_associations(memory)

        return memory.id

    def _update_associations(self, memory: Memory):
        """更新概念关联网络"""
        # 提取关键词作为节点
        words = memory.content.lower().split()
        for word in words:
            if word not in self.association_network:
                self.association_network[word] = []
            # 与其他词建立连接
            for other_word in words:
                if other_word != word:
                    existing = next((item for item in self.association_network[word]
                                    if item[0] == other_word), None)
                    if existing:
                        # 强化已有连接
                        idx = self.association_network[word].index(existing)
                        self.association_network[word][idx] = (
                            other_word, min(1.0, existing[1] + 0.05)
                        )
                    else:
                        self.association_network[word].append((other_word, 0.1))

    def retrieve(self, cue: str, memory_type: Optional[str] = None,
                 current_emotion_valence: float = 0.0,
                 top_k: int = 3) -> List[Tuple[Memory, float]]:
        """
        基于线索检索记忆

        模拟人类的记忆检索：
        1. 线索匹配
        2. 可提取性计算（遗忘曲线 + 情感一致性）
        3. 返回最相关的记忆
        """
        all_memories = []
        if memory_type == "episodic" or memory_type is None:
            all_memories.extend(self.episodic_memories)
        if memory_type == "semantic" or memory_type is None:
            all_memories.extend(self.semantic_memories)
        if memory_type == "procedural" or memory_type is None:
            all_memories.extend(self.procedural_memories)

        scored_memories = []
        for memory in all_memories:
            # 线索匹配度
            cue_match = self._calculate_cue_match(cue, memory)

            # 可提取性
            retrievability = memory.get_retrievability(current_emotion_valence)

            # 综合得分
            score = cue_match * retrievability * memory.importance

            if score > 0.1:  # 阈值
                scored_memories.append((memory, score))

        # 按得分排序
        scored_memories.sort(key=lambda x: x[1], reverse=True)

        # 更新检索统计
        for memory, _ in scored_memories[:top_k]:
            memory.retrieval_count += 1
            memory.last_retrieved = datetime.now()

        return scored_memories[:top_k]

    def _calculate_cue_match(self, cue: str, memory: Memory) -> float:
        """计算线索与记忆的匹配度"""
        cue_lower = cue.lower()
        content_lower = memory.content.lower()

        # 直接内容匹配
        if cue_lower in content_lower:
            return 0.8

        # 检索线索匹配
        for mem_cue in memory.retrieval_cues:
            if cue_lower in mem_cue.lower() or mem_cue.lower() in cue_lower:
                return 0.7

        # 关键词重叠
        cue_words = set(cue_lower.split())
        content_words = set(content_lower.split())
        if cue_words and content_words:
            overlap = len(cue_words & content_words) / len(cue_words)
            return overlap * 0.6

        # 关联网络匹配（远距离联想）
        for cue_word in cue_lower.split():
            if cue_word in self.association_network:
                for associated_word, strength in self.association_network[cue_word]:
                    if associated_word in content_lower:
                        return strength * 0.4

        return 0.0

    def retrieve_similar(self, content: str, current_emotion_valence: float = 0.0,
                        top_k: int = 3) -> List[Tuple[Memory, float]]:
        """基于内容相似性检索"""
        return self.retrieve(content, None, current_emotion_valence, top_k)

    def get_recent_memories(self, hours: float = 24, memory_type: Optional[str] = None) -> List[Memory]:
        """获取最近的记忆"""
        cutoff = datetime.now() - timedelta(hours=hours)

        all_memories = []
        if memory_type == "episodic" or memory_type is None:
            all_memories.extend(self.episodic_memories)
        if memory_type == "semantic" or memory_type is None:
            all_memories.extend(self.semantic_memories)

        return [m for m in all_memories if m.timestamp > cutoff]

    def reconstruct_memory(self, memory_id: str, current_context: Dict = None) -> Optional[Memory]:
        """
        记忆重构 - 模拟人类记忆的重建过程

        每次回忆都是一次重构，可能融入当前知识和信念
        """
        # 查找记忆
        all_memories = (self.episodic_memories + self.semantic_memories +
                       self.procedural_memories)
        memory = next((m for m in all_memories if m.id == memory_id), None)

        if not memory:
            return None

        # 模拟重构：置信度略有下降（记忆被"污染"）
        memory.confidence *= 0.98

        # 如果多次重构，可能添加"虚假"细节
        if memory.retrieval_count > 5 and memory.confidence < 0.7:
            # 模拟记忆重构：融入当前语境
            if current_context:
                memory.context.update(current_context)

        memory.retrieval_count += 1
        memory.last_retrieved = datetime.now()

        return memory

    def get_working_memory(self) -> Dict:
        """获取工作记忆状态"""
        return self.working_memory.to_dict()

    def get_memory_stats(self) -> Dict:
        """获取记忆统计"""
        return {
            "episodic": len(self.episodic_memories),
            "semantic": len(self.semantic_memories),
            "procedural": len(self.procedural_memories),
            "autobiographical": len(self.autobiographical_memories),
            "working_memory": self.working_memory.to_dict(),
            "association_nodes": len(self.association_network)
        }
