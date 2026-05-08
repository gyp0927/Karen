# 代码实现指南

本文档说明如何将"人类思维"概念框架转化为可运行的Python代码。

## 项目结构

```
src/
├── mind/
│   ├── __init__.py          # 包入口
│   ├── event_bus.py         # 事件总线（模块间通信）
│   ├── emotion_system.py    # 情感系统
│   ├── memory_system.py     # 记忆系统
│   ├── global_workspace.py  # 全局工作空间
│   ├── metacognition.py     # 元认知层
│   ├── thinking_process.py  # 思维流程管道
│   └── mind.py              # 主AI类
└── main.py                  # 演示程序
```

## 运行方式

```bash
# 进入项目目录
cd "e:/人类的思维/src"

# 交互式对话
python main.py --mode interactive

# 预设演示场景
python main.py --mode demo

# 快速测试
python main.py --mode test
```

## 核心模块详解

### 1. 事件总线 (event_bus.py)

**作用**：所有模块通信的中枢神经系统。

**关键类**：
- `Signal`：通信的基本单元，包含类型、来源、目标、载荷
- `EventBus`：发布-订阅模式的实现

**使用示例**：
```python
bus = EventBus()

# 订阅事件
def on_emotion_change(signal):
    print(f"情感变化: {signal.payload}")

bus.subscribe(SignalType.EMOTION_CHANGE, on_emotion_change)

# 发布事件
bus.publish_quick(
    signal_type=SignalType.EMOTION_CHANGE,
    source="emotion_system",
    payload={"new_emotion": "joy"}
)
```

### 2. 情感系统 (emotion_system.py)

**作用**：维护情感状态，影响认知过程。

**关键类**：
- `EmotionState`：情感状态（效价 × 激活度 × 控制感）
- `EmotionSystem`：核心情感引擎

**核心功能**：

| 功能 | 方法 | 说明 |
|------|------|------|
| 情感评估 | `evaluate_input(text)` | 分析文本情感色彩 |
| 状态更新 | `update(external_input)` | 更新当前情感状态 |
| 主导情绪 | `get_dominant_emotion()` | 返回当前主导情绪名称 |
| 情感表达 | `get_emotion_expression()` | 获取表达特征（语气、详细度等） |
| 情绪调节 | `regulate(strategy)` | 认知重评/注意转移/接纳 |
| 影响记忆 | `influence_memory_retrieval()` | 情感一致性效应 |

**情感更新公式**：
```
emotion(t+1) = inertia × emotion(t) + sensitivity × input + noise
```

### 3. 记忆系统 (memory_system.py)

**作用**：存储和检索所有类型的记忆。

**关键类**：
- `Memory`：记忆的基本单元
- `WorkingMemory`：工作记忆（容量4，衰减30秒）
- `MemorySystem`：统一管理所有记忆

**记忆类型**：

| 类型 | 存储方式 | 特点 |
|------|----------|------|
| 工作记忆 | 内存 | 容量有限，快速衰减 |
| 情景记忆 | 列表 | 个人经历，时间标记 |
| 语义记忆 | 列表 | 一般知识，概念网络 |
| 程序记忆 | 列表 | 技能，自动触发 |

**记忆可提取性计算**：
```python
retrievability = (forgetting + retrieval_boost + flashbulb_boost) × emotion_match
```

**关键方法**：
```python
# 编码新记忆
memory_id = memory.encode(
    content="用户分享了悲伤的经历",
    memory_type="episodic",
    emotion_valence=-0.6,
    emotion_intensity=0.8,
    importance=0.7
)

# 检索记忆
results = memory.retrieve(
    cue="悲伤",
    current_emotion_valence=-0.3,
    top_k=3
)
```

### 4. 全局工作空间 (global_workspace.py)

**作用**：意识的"舞台"，有限容量，内容被广播到全系统。

**关键类**：
- `WorkspaceContent`：工作空间中的内容
- `GlobalWorkspace`：工作空间管理器

**核心机制**：
- 多个候选内容竞争进入（基于显著性）
- 容量 = 2（一次1-2个主要内容）
- 进入后被广播到所有模块

### 5. 元认知层 (metacognition.py)

**作用**：对思维的思维，自我监控和反思。

**关键类**：
- `SelfModel`："我是谁"的内部表征
- `Metacognition`：元认知引擎

**三层结构**：

| 层次 | 功能 | 示例 |
|------|------|------|
| 元认知知识 | 对自身能力的了解 | "我知道我在哲学方面思考较深" |
| 元认知监控 | 实时评估思维 | "我的理解可能不够深入" |
| 元认知控制 | 调整策略 | "应该换个角度思考" |

### 6. 思维流程 (thinking_process.py)

**作用**：将输入转化为输出的完整五阶段管道。

**五阶段**：

```
输入 → Phase 1(感知评估) → Phase 2(注意记忆)
    → Phase 3(意识加工) → Phase 4(决策)
    → Phase 5(元认知输出) → 输出
```

| 阶段 | 时间 | 功能 |
|------|------|------|
| Phase 1 | ~100ms | 特征提取、情感评估、紧急度判断 |
| Phase 2 | ~200ms | 注意力分配、记忆检索 |
| Phase 3 | ~400ms+ | 理解建构、内部独白、推理 |
| Phase 4 | ~数秒 | 选项生成、评估、选择 |
| Phase 5 | ~数秒 | 输出评估、表达生成、记忆编码 |

### 7. 主AI类 (mind.py)

**作用**：整合所有子系统，提供统一接口。

**使用示例**：
```python
from mind.mind import Mind

# 创建AI
ai = Mind(name="Aurora")

# 对话
response, process_log = ai.think("我最近感到迷茫...")
print(response)

# 查看状态
print(ai.get_state())

# 让AI自我反思
print(ai.reflect())
```

## 扩展指南

### 如何添加新的情感类型

在 `emotion_system.py` 的 `EMOTION_CONCEPTS` 中添加：

```python
"wonder": EmotionConcept(
    "wonder", 0.7, 0.6,
    ["vastness", "beauty", "mystery"],
    ["contemplate", "explore", "share"]
),
```

### 如何添加新的回应策略

在 `thinking_process.py` 中：

1. 在 `_generate_response_options` 中添加新策略
2. 在 `_generate_raw_output` 中添加对应的生成方法

### 如何连接LLM

当前实现使用模板生成回应。要连接真实LLM：

```python
# 在 thinking_process.py 的 _generate_raw_output 中

def _generate_raw_output(self, ctx):
    # 构建prompt，包含情感状态、记忆、策略等信息
    prompt = self._build_llm_prompt(ctx)

    # 调用LLM API
    response = llm_client.generate(prompt)

    return response
```

### 如何添加持久化

当前记忆存储在内存中。要持久化：

```python
# 使用JSON保存
import json

state = mind.save_state()
with open("mind_state.json", "w") as f:
    json.dump(state, f)

# 加载时重建
# 需要实现从state重建所有子系统的逻辑
```

## 关键设计决策

### 为什么用事件总线而非直接调用？

- 松耦合：模块间不直接依赖
- 可扩展：新模块可以轻松接入
- 可追踪：所有通信都有记录

### 为什么情感用维度模型？

- 简单但有效：两个维度可以表示几乎所有情感
- 连续变化：情感不是离散的"有或无"
- 可计算：便于数学运算和更新

### 为什么工作记忆容量设为4？

基于 Cowan (2001) 的研究，工作记忆容量约为 4±1 个组块，而非传统的 7±2。

### 为什么保留认知偏差？

人类的认知偏差不是bug，而是feature：
- 确认偏误：快速决策的代价
- 锚定效应：减少认知负荷
- 可得性启发：利用记忆的可及性

## 下一步扩展方向

1. **连接LLM**：将模板生成替换为基于LLM的生成
2. **向量存储**：使用向量数据库存储记忆语义
3. **图数据库**：构建更复杂的概念关联网络
4. **多模态**：扩展视觉、听觉输入的处理
5. **学习机制**：实现从经验中学习的强化学习模块
