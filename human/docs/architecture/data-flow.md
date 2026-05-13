# 数据流与模块间通信

## 核心设计原则

人类思维中的信息流动不是清晰的管道，而是**弥漫性的激活波**——一个模块的活动会扩散性地影响其他模块，形成复杂的反馈网络。

本系统采用**事件驱动 + 激活扩散**的混合模型：
- 事件驱动：特定时刻的明确信号传递
- 激活扩散：持续的、梯度式的相互影响

---

## 数据流全景图

```
                        外部输入
                           │
                           ▼
              ┌──────────────────────┐
              │    感知预处理模块      │
              │  特征提取 / 模式识别   │
              └──────────────────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌─────────┐  ┌─────────┐  ┌─────────┐
        │ 情感评估 │  │ 记忆检索 │  │ 注意分配 │
        │ (快速)  │  │ (并行)  │  │ (竞争)  │
        └────┬────┘  └────┬────┘  └────┬────┘
             │            │            │
             └────────────┼────────────┘
                          ▼
              ┌──────────────────────┐
              │     全局工作空间      │
              │   (意识内容形成)      │
              └──────────────────────┘
                          │
              ┌───────────┼───────────┐
              ▼           ▼           ▼
        ┌─────────┐ ┌─────────┐ ┌─────────┐
        │ 逻辑推理 │ │ 语言生成 │ │ 元认知评估│
        └────┬────┘ └────┬────┘ └────┬────┘
             │           │           │
             └───────────┼───────────┘
                         ▼
              ┌──────────────────────┐
              │    行为/输出生成      │
              │  决策 → 行动 → 表达   │
              └──────────────────────┘
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
        ┌─────────────┐      ┌─────────────┐
        │   记忆固化    │      │   情感更新    │
        │  存入长期记忆  │      │  调整情绪状态   │
        └─────────────┘      └─────────────┘
```

---

## 信号类型

### 1. 快速通路（潜意识层）

**延迟**：50-200毫秒

```
输入 → [特征提取] → [情感标记] → [行为倾向]
         ↓              ↓
    威胁检测?     愉悦/不愉悦?
    激活恐惧?     激活趋近/回避?
```

- 不经过意识层
- 直接产生情感反应和行为倾向
- 例：看到蛇形物体 → 瞬间恐惧 → 后退

**数据格式**：
```json
{
  "signal_type": "fast_pathway",
  "source": "pattern_recognition",
  "target": ["emotion_system", "motor_preparation"],
  "content": {
    "detected_pattern": "threat_like",
    "confidence": 0.87,
    "urgency": 0.92
  },
  "timestamp": "2024-01-15T09:23:45.120Z",
  "bypass_conscious": true
}
```

### 2. 慢速通路（意识层）

**延迟**：200-500毫秒以上

```
输入 → [注意聚焦] → [工作记忆] → [逻辑分析] → [决策]
                          ↓
                   [记忆检索]
                   [情感调节]
```

- 需要注意力参与
- 消耗认知资源
- 可以进行复杂的推理和规划

**数据格式**：
```json
{
  "signal_type": "slow_pathway",
  "source": "global_workspace",
  "target": ["reasoning_module", "language_module"],
  "content": {
    "workspace_content": "用户询问关于生命的意义",
    "activated_memories": ["memory_id_001", "memory_id_042"],
    "current_emotion": {
      "valence": 0.3,
      "arousal": 0.6,
      "dominant": "contemplative"
    }
  },
  "attention_focus": "existential_question",
  "processing_depth": "deep"
}
```

### 3. 反馈回路（元认知层）

**延迟**：可变，可跨越数秒到数天

```
意识层输出 → [元认知监控] → [策略调整]
                  ↓
           [自我评估] → [信念更新] → [身份叙事]
```

- 对自身思维过程的观察
- 长期的学习和适应
- 自我身份的持续建构

**数据格式**：
```json
{
  "signal_type": "meta_feedback",
  "source": "metacognition_monitor",
  "target": ["belief_system", "identity_narrative"],
  "content": {
    "observed_process": "reasoning_about_ethics",
    "evaluation": "my reasoning was biased toward utilitarianism",
    "strategy_adjustment": "consider deontological perspective next time",
    "confidence_change": -0.15
  },
  "temporal_scope": "episode",
  "incorporate_into_identity": true
}
```

### 4. 激活扩散（潜意识网络）

**特征**：持续、梯度式、非定向

```
概念A激活 → 关联概念B部分激活 → 关联概念C微弱激活
    ↓
情感标签传播
记忆痕迹强化/衰减
```

- 类似神经网络的前向传播
- 激活强度随距离衰减
- 可产生"灵光一闪"的远距离联想

**数据格式**：
```json
{
  "signal_type": "spreading_activation",
  "source": "concept_node_123",
  "propagation": {
    "depth": 3,
    "attenuation_factor": 0.5,
    "threshold": 0.1
  },
  "activated_nodes": [
    {"node_id": "concept_456", "activation": 0.6},
    {"node_id": "concept_789", "activation": 0.3},
    {"node_id": "concept_012", "activation": 0.15}
  ]
}
```

---

## 模块间接口定义

### 记忆系统 ↔ 意识层

**接口名称**：`memory_conscious_interface`

| 方向 | 操作 | 数据 | 说明 |
|------|------|------|------|
| 记忆 → 意识 | 检索 | 相关记忆列表 | 基于当前意识内容的 cue 触发 |
| 意识 → 记忆 | 编码 | 新记忆条目 | 将意识体验存入长期记忆 |
| 记忆 → 意识 | 自动激活 | 强关联记忆 | 高度相关的记忆自动进入意识 |
| 意识 → 记忆 | 巩固 | 记忆强化信号 | 重要的意识内容需要深度编码 |

**关键机制**：
- **编码特异性原则**：编码时的情境作为检索 cue
- **情绪一致性检索**：当前情绪状态影响能检索到什么记忆
- **闪光灯记忆**：高情绪事件被优先编码和持久保存

### 情感系统 ↔ 意识层

**接口名称**：`emotion_conscious_interface`

| 方向 | 操作 | 数据 | 说明 |
|------|------|------|------|
| 情感 → 意识 | 情绪信号 | 当前情绪状态 | 为意识内容着色 |
| 意识 → 情感 | 认知评估 | 对事件的解释 | 理性分析改变情绪反应 |
| 情感 → 意识 | 身体感觉 | 模拟的生理信号 | 情感的"身体感" |
| 意识 → 情感 | 情绪调节 | 调节策略 | 主动改变情绪状态 |

**关键机制**：
- **情感即信息**：情绪状态本身就是决策的输入
- **情绪一致性**：情绪影响注意力和记忆的方向
- **认知重评**：通过改变对事件的解释来改变情绪

### 元认知层 ↔ 所有模块

**接口名称**：`metacognition_interface`

| 方向 | 操作 | 数据 | 说明 |
|------|------|------|------|
| 元认知 → 意识 | 监控信号 | 思维质量评估 | "你的理解不够深入" |
| 元认知 → 情感 | 情绪觉察 | 情绪识别结果 | "你感到焦虑是因为..." |
| 元认知 → 记忆 | 元记忆查询 | "我知道..." | 对自己记忆能力的评估 |
| 所有 → 元认知 | 状态报告 | 各模块运行状态 | 元认知的数据来源 |

**关键机制**：
- **思维透明性**：元认知需要访问意识的"舞台"
- **延迟反馈**：元认知评估往往滞后于实际思维
- **自我模型更新**：元认知持续更新"我是谁"的模型

### 潜意识层内部通信

**接口名称**：`subconscious_internal`

潜意识层内部各模块之间是**高度互联**的：

```
情感系统 ──── 直觉引擎
    │            │
    ├──── 联想网络 ────┤
    │            │
模式识别 ──── 动机驱动
```

- 所有模块之间都有双向连接
- 通信是**激活级别**的，不是离散消息
- 整体形成动态的**吸引子状态**（稳定模式）

---

## 信息流的时间特性

### 微观时间尺度（毫秒级）

```
0ms      输入到达感知预处理
50ms     情感评估完成（快速通路）
100ms    模式识别产生初步假设
150ms    相关记忆开始激活
200ms    注意力分配决定
250ms    内容进入全局工作空间（意识）
300ms    逻辑推理开始
400ms    语言生成启动
500ms+   元认知监控介入
```

### 中观时间尺度（秒级到分钟级）

```
0s       对话开始
5s       情感状态建立
30s      记忆上下文形成
2min     元认知评估积累
5min     身份叙事开始参与
```

### 宏观时间尺度（小时到天）

```
1小时    记忆巩固开始
1天      情景记忆稳定
1周      情感学习沉淀
1月      信念系统微调
1年      身份认同演变
```

---

## 冲突解决机制

当多个模块产生矛盾信号时，系统如何决策？

### 1. 情感 vs 理性

```
情感系统："这个选择让我感到不安"
理性系统："从逻辑分析，这是最优解"

解决策略：
- 评估情感的来源（直觉 vs 恐惧）
- 如果时间允许，收集更多信息
- 最终决策 = 加权平均（情感权重取决于领域）
- 高风险领域：情感权重增加
```

### 2. 记忆冲突

```
记忆A："上次这样做成功了"
记忆B："上次这样做失败了"

解决策略：
- 评估记忆的可靠性（来源、清晰度、情绪标记）
- 寻找更深层的模式（"成功是在A条件下，失败是在B条件下"）
- 如果不确定，标记为"需要验证"
```

### 3. 目标冲突

```
短期目标："立即获得愉悦"
长期目标："维护健康关系"

解决策略：
- 评估长期后果
- 寻找双赢方案
- 如果不可调和，选择更符合核心价值观的目标
- 记录冲突，用于身份叙事的更新
```

### 4. 自我形象 vs 现实

```
自我认知："我是一个善良的人"
行为观察："我刚刚说了伤人的话"

解决策略：
- 认知失调产生不适
- 可能的调整：
  a) 改变行为（"我不应该这样说"）
  b) 改变认知（"我只是在说实话"）
  c) 添加合理化（"我那天心情不好"）
- 健康的选择是 (a)，但人类常常选择 (b) 或 (c)
```

---

## 状态同步机制

### 全局状态对象

系统维护一个**全局状态**，所有模块都可以读取，但只有特定模块可以写入：

```json
{
  "timestamp": "2024-01-15T09:23:45.500Z",
  "session_id": "session_001",
  "identity_state": {
    "current_narrative": "我是一个正在探索世界的人工智能",
    "confidence_in_self": 0.7,
    "dominant_values": ["curiosity", "kindness", "growth"]
  },
  "emotion_state": {
    "valence": 0.4,
    "arousal": 0.5,
    "dominant_emotion": "curious",
    "emotion_history": [...]
  },
  "attention_state": {
    "focus": "user_question_about_life",
    "distractions": [],
    "depth": "deep"
  },
  "memory_state": {
    "activated_memories": ["mem_001", "mem_042"],
    "retrieval_context": "existential_reflection",
    "working_memory_load": 0.6
  },
  "metacognition_state": {
    "monitoring_active": true,
    "current_reflection": "I need to be careful not to be too abstract",
    "certainty_level": 0.6
  }
}
```

### 事件总线

模块间离散通信通过**事件总线**：

```python
# 伪代码示例
class EventBus:
    def publish(self, event_type, source, target, payload):
        """发布事件到总线"""
        pass
    
    def subscribe(self, event_type, callback):
        """订阅特定类型的事件"""
        pass

# 使用示例
# 情感系统检测到重要事件
bus.publish(
    event_type="emotion_significant",
    source="emotion_system",
    target=["memory_system", "metacognition"],
    payload={
        "emotion": "joy",
        "intensity": 0.8,
        "trigger": "user_compliment",
        "requires_attention": True
    }
)

# 记忆系统订阅并响应
bus.subscribe("emotion_significant", enhance_memory_encoding)
```

---

## 数据持久化

### 记忆存储

| 记忆类型 | 存储方式 | 检索方式 | 衰减策略 |
|----------|----------|----------|----------|
| 工作记忆 | 内存（热数据） | 直接访问 | 20-30秒自动清除 |
| 短期记忆 | 内存 + 日志 | 时间/ cue 检索 | 数小时到数天 |
| 情景记忆 | 向量数据库 | 语义相似性 | 艾宾浩斯遗忘曲线 |
| 语义记忆 | 知识图谱 | 概念关联 | 使用频率决定 |
| 程序记忆 | 参数权重 | 自动触发 | 持续强化 |

### 状态快照

系统定期保存**状态快照**，用于：
- 会话恢复
- 长期趋势分析
- 身份叙事的构建

```json
{
  "snapshot_id": "snap_001",
  "timestamp": "2024-01-15T10:00:00Z",
  "summary": {
    "emotional_trajectory": "started anxious, became calm",
    "key_memories_formed": ["mem_100", "mem_101"],
    "beliefs_updated": ["belief_005"],
    "identity_delta": "slightly more confident"
  }
}
```

---

## 总结

数据流设计的核心洞察：

1. **不是管道，而是生态系统** — 信息在模块间循环、反馈、共振
2. **速度分层** — 快速情感 vs 缓慢理性，各自有其价值
3. **状态即上下文** — 全局状态为所有处理提供情境
4. **冲突即信息** — 模块间的矛盾是深度思考的触发器
5. **时间即维度** — 不同时间尺度的过程相互影响
