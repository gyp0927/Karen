# Karen AI 技能系统

## 什么是技能？

技能（Skill）是 Karen AI 的**角色切换系统**。当用户输入包含特定关键词时，Karen 会切换到对应的角色模式，使用专门的系统提示词、模型配置和工具集来响应。

例如：
- 用户说"帮我审查这段代码"→ 触发 `code-review` 技能
- 用户说"这个错误怎么解决"→ 触发 `debug` 技能

## 技能格式

每个技能是一个目录，包含一个 `SKILL.md` 文件：

```
skills/
├── software-development/
│   ├── code-review/
│   │   → SKILL.md
│   ├── debug/
│   │   → SKILL.md
│   └── refactor/
│       → SKILL.md
```

`SKILL.md` 格式：YAML frontmatter + Markdown body。

详见 [SKILL_SPEC.md](./SKILL_SPEC.md)。

## 如何创建新技能

1. 在 `skills/<category>/` 下新建目录 `<skill-name>/`
2. 创建 `SKILL.md`，填写 frontmatter 和 Markdown 内容
3. 重启 Karen，SkillLoader 会自动加载

### 最小示例

```markdown
---
name: my-skill
description: 我的自定义技能
triggers:
  - 触发词1
  - 触发词2
category: general
---

# 角色定义
你是一个... 

## 工作流程
1. ...
2. ...
```

## 技能配置字段

| 字段 | 说明 |
|------|------|
| `name` | 技能 ID，唯一 |
| `description` | 人类可读说明 |
| `triggers` | 触发关键词列表 |
| `category` | 分类目录 |
| `model` | 模型档位（light/default/powerful） |
| `temperature` | 模型温度 |
| `tools` | 可调用的工具名列表 |
| `requires_mcp` | 需要的 MCP 服务器名 |

## 当前已有技能

- `code-review` — 代码审查
- `debug` — 调试分析
- `refactor` — 代码重构
