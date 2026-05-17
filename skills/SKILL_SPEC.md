# 技能文件格式规范

## 文件位置

`skills/<category>/<skill-name>/SKILL.md`

例子：`skills/software-development/code-review/SKILL.md`

## 文件结构

```markdown
---
name: code-review
description: 代码审查，检查安全、性能、风格问题
triggers:
  - 代码审查
  - 审查代码
  - code review
  - review this code
category: software-development
model: powerful  # 使用哪档模型：light/default/powerful
---

# 技能内容

## 角色定义
你是一个经验丰富的代码审查员...

## 工作流程
1. 先看整体架构
2. 再查具体实现
3. 最后给出修复建议

## 示例
### 输入
```python
def add(a, b):
    return a + b
```

### 输出
- 建议添加类型注解
- 建议添加文档字符串
```

## Skill 字段规范

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| name | str | 是 | 技能 ID，唯一，小写，用下划线连接 |
| description | str | 是 | 人类可读的说明 |
| triggers | list[str] | 是 | 触发关键词列表，匹配用户输入 |
| category | str | 是 | 分类目录名，小写，用下划线连接 |
| model | str | 否 | 默认使用的模型档位：light/default/powerful |
| temperature | float | 否 | 覆盖默认 temperature |
| max_tokens | int | 否 | 覆盖默认 max_tokens |
| tools | list[str] | 否 | 该技能可调用的工具名列表 |
| requires_mcp | list[str] | 否 | 需要的 MCP 服务器名 |

## 设计原则

1. **纯文本优先**：技能文件是 YAML + Markdown，无需编译或打包
2. **自描述**：每个技能包含触发条件、角色定义、工作流程，无需外部文档
3. **可扩展**：新增技能只需新建目录和 SKILL.md，重启后自动加载
4. **与 Hermes 兼容**：YAML frontmatter 格式与 Hermes skill 系统一致
