---
name: claude-dashboard
description: >
  配置 Claude Code 底部状态栏为丰富的实时 dashboard，包含时间、项目信息、模型状态、上下文占比、
  token 消耗量和可爱宠物动画。当用户提到 statusline、状态栏、底部 dashboard、宠物、实时监控、
  token 用量、上下文占比，或想要美化/增强 Claude Code 底部显示时触发。也适用于用户说"我想在
  Claude Code 底部看到..."或"给我个 dashboard"等场景。
metadata:
  version: 1.0.0
  author: Claude
---

# Claude Dashboard — 增强状态栏 Skill

将 Claude Code 底部状态栏升级为信息丰富的实时 dashboard，带可爱宠物动画。

## 功能特性

- **时间日期** — 实时显示当前时间
- **项目信息** — 当前目录、git 分支
- **模型状态** — 当前使用的 Claude 模型名称
- **上下文占比** — 上下文窗口使用百分比
- **Token 消耗** — 从 rate_limits 估算 token 使用
- **可爱宠物** — ASCII/Emoji 宠物动画，定时切换帧
- **系统信息** — 用户名、主机名
- **CPU / 内存** — 实时 CPU 与内存占用率（需 `psutil`）
- **天气** — 当前位置（或指定城市）的天气与温度，10 分钟缓存（来源 wttr.in）
- **会话名称** — 自定义会话名（如有）

## 快速开始

### 第一步：安装 dashboard 脚本

将 `scripts/statusline_dashboard.py` 复制到用户的 Claude 配置目录：

```bash
# Windows
mkdir -p "%USERPROFILE%\.claude\scripts"
cp scripts/statusline_dashboard.py "%USERPROFILE%\.claude\scripts\"

# macOS/Linux
mkdir -p ~/.claude/scripts
cp scripts/statusline_dashboard.py ~/.claude/scripts/
```

### 第二步：配置 settings.json

编辑 `~/.claude/settings.json`（或 `%USERPROFILE%\.claude\settings.json`），添加 statusLine 配置：

```json
{
  "statusLine": {
    "type": "command",
    "command": "python \"%USERPROFILE%\\.claude\\scripts\\statusline_dashboard.py\"",
    "refreshInterval": 2
  }
}
```

**macOS/Linux** 请使用对应路径：
```json
{
  "statusLine": {
    "type": "command",
    "command": "python ~/.claude/scripts/statusline_dashboard.py",
    "refreshInterval": 2
  }
}
```

`refreshInterval: 2` 表示每 2 秒刷新一次，宠物动画会随之切换。

### 第三步：验证

保存配置后，Claude Code 底部状态栏会立即更新。如果未生效，重启 Claude Code 或执行 `/statusline` 命令刷新。

## 自定义配置

编辑 `~/.claude/scripts/statusline_dashboard.py` 顶部的 `CONFIG` 字典来自定义显示：

```python
CONFIG = {
    "show_pet": True,           # 是否显示宠物
    "show_time": True,          # 是否显示时间
    "show_cwd": True,           # 是否显示当前目录
    "show_git": True,           # 是否显示 git 分支
    "show_model": True,         # 是否显示模型名
    "show_context": True,       # 是否显示上下文占比
    "show_rate_limit": True,    # 是否显示 rate limit
    "show_user_host": True,     # 是否显示用户名@主机名
    "show_session": False,      # 是否显示会话名
    "show_cpu": True,           # 是否显示 CPU 使用率（需 psutil）
    "show_memory": True,        # 是否显示内存使用率（需 psutil）
    "show_weather": True,       # 是否显示天气（来源 wttr.in）
    "weather_city": "",         # 天气城市，留空=自动IP定位，如 "Beijing"
    "weather_cache_minutes": 10,# 天气缓存分钟数
    "time_format": "%H:%M:%S",  # 时间格式
    "pet_style": "emoji",       # 宠物样式: "emoji" | "ascii"
    "separator": " | ",         # 分隔符
    "compact_mode": False,      # 紧凑模式（减少信息密度）
}
```

### 宠物样式

- `emoji` — 使用 Unicode emoji 表情（🐱😺😸😻🙀😽），切换流畅
- `ascii` — 使用 ASCII 艺术猫，更有复古感

### 紧凑模式

开启 `compact_mode: True` 后，只显示最关键信息：
```
🐱 project [main] 14:32:15 | op4.7 | ctx:45%
```

## 工作原理

Claude Code 的 statusline 机制：
1. 每次刷新时，Claude Code 向配置的命令发送 JSON 数据（stdin）
2. 命令解析 JSON，提取 workspace、model、context_window 等信息
3. 命令输出一行文本，显示在 Claude Code 底部
4. `refreshInterval` 控制刷新频率（秒）

### 输入 JSON 结构

```json
{
  "workspace": {
    "current_dir": "/path/to/project",
    "git_worktree": { "branch": "main", ... },
    "added_dirs": ["/path/to/added"]
  },
  "model": {
    "id": "claude-opus-4-7",
    "display_name": "Claude Opus 4.7"
  },
  "context_window": {
    "used_percentage": 45.2
  },
  "rate_limits": {
    "five_hour": { "used": 1000, "limit": 10000 },
    "seven_day": { "used": 50000, "limit": 100000 }
  },
  "session_name": "my-session",
  "worktree": { ... }
}
```

## 故障排除

| 问题 | 解决方法 |
|------|----------|
| 状态栏无变化 | 检查 Python 路径是否正确，尝试使用 `python3` 替代 `python` |
| 宠物不动 | 确认 `refreshInterval` 已设置（建议 2-5 秒） |
| 显示乱码 | 确保终端支持 Unicode emoji |
| 信息不完整 | 某些字段（如 git 分支）仅在对应目录下才显示 |
| Windows 路径问题 | 使用双反斜杠 `\\` 或正斜杠 `/` |
| 天气不显示 | 检查网络连接，`weather_cache_minutes` 可改短；也可填 `weather_city` 指定城市 |
| CPU/内存不显示 | 确保已安装 `psutil`：`pip install psutil` |

## 扩展建议

用户可以根据需要修改脚本，添加更多自定义信息：
- 显示当前虚拟环境（venv/conda）
- 显示 Node.js/Python 版本
- 显示未读消息数（配合其他工具）
- 自定义宠物（换成 🐶🦊🐰 等）
