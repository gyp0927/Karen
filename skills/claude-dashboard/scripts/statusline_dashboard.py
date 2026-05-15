#!/usr/bin/env python3
"""
Claude Dashboard — Enhanced Statusline Script
显示时间、项目信息、模型状态、上下文占比、宠物动画等。

用法：由 Claude Code statusline 机制调用，从 stdin 接收 JSON 数据。
"""

import json
import os
import subprocess
import sys
from datetime import datetime

# ==================== 用户配置 ====================
CONFIG = {
    "show_pet": True,
    "show_time": True,
    "show_cwd": True,
    "show_git": True,
    "show_model": True,
    "show_context": True,
    "show_rate_limit": False,
    "show_user_host": True,
    "show_session": False,
    "show_cpu": True,
    "show_memory": True,
    "show_weather": True,
    "weather_city": "",  # 留空 = 自动按 IP 定位；或填城市名如 "Beijing"
    "weather_cache_minutes": 10,  # 天气缓存分钟数
    "time_format": "%H:%M:%S",
    "pet_style": "emoji",  # "emoji" or "ascii"
    "separator": " | ",
    "compact_mode": False,
}

# ==================== 宠物动画帧 ====================
PET_FRAMES_EMOJI = ["🐱", "😺", "😸", "😻", "🙀", "😽"]

PET_FRAMES_ASCII = [
    " /\\_/\\  ",
    "( o.o ) ",
    " > ^ <  ",
]

# 另一种 ASCII 猫（多帧）
PET_FRAMES_ASCII_ALT = [
    " /\\_/\\ ",
    "( o.o )",
    " > ^ < ",
]


def get_pet_frame():
    """根据当前秒数选择宠物帧，实现动画效果。"""
    if not CONFIG["show_pet"]:
        return ""

    second = datetime.now().second

    if CONFIG["pet_style"] == "ascii":
        frames = PET_FRAMES_ASCII
        # ASCII 模式下显示静态猫（空间有限）
        return " /\\_/\\ ( o.o ) > ^ < "
    else:
        frames = PET_FRAMES_EMOJI
        frame_idx = second % len(frames)
        return frames[frame_idx]


def get_cpu_memory():
    """获取 CPU 和内存使用率，psutil 不可用时返回 (None, None)。"""
    try:
        import psutil
        cpu = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory().percent
        return cpu, mem
    except Exception:
        return None, None


def get_weather():
    """获取天气，带文件缓存。缓存过期或失败时尝试刷新。"""
    import tempfile
    import time
    import urllib.request
    import urllib.parse

    cache_dir = tempfile.gettempdir()
    cache_file = os.path.join(cache_dir, "claude_dashboard_weather.txt")
    cache_age_max = CONFIG["weather_cache_minutes"] * 60

    if os.path.exists(cache_file):
        age = time.time() - os.path.getmtime(cache_file)
        if age < cache_age_max:
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    return f.read().strip()
            except Exception:
                pass

    city = CONFIG.get("weather_city", "")
    path = urllib.parse.quote(city) if city else ""
    url = f"https://wttr.in/{path}?format=%c+%t&lang=zh"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "curl/7.0"})
        with urllib.request.urlopen(req, timeout=2) as resp:
            data = resp.read().decode("utf-8").strip()
        if data and not data.lower().startswith("unknown"):
            try:
                with open(cache_file, "w", encoding="utf-8") as f:
                    f.write(data)
            except Exception:
                pass
            return data
    except Exception:
        pass

    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception:
            pass
    return ""


def get_git_branch(cwd):
    """获取当前目录的 git 分支。"""
    if not cwd:
        return ""
    try:
        result = subprocess.run(
            ["git", "symbolic-ref", "--short", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass

    # 尝试获取 detached HEAD 的短 hash
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass

    return ""


def format_model_name(model_id, display_name):
    """将模型名格式化为简短版本。"""
    if display_name:
        name = display_name
    elif model_id:
        name = model_id
    else:
        return ""

    # 常见模型名缩写
    abbreviations = {
        "claude-opus-4-7": "op4.7",
        "claude-opus-4-6": "op4.6",
        "claude-opus-4-5": "op4.5",
        "claude-sonnet-4-6": "sn4.6",
        "claude-sonnet-4-5": "sn4.5",
        "claude-haiku-4-5": "hk4.5",
        "claude-opus": "opus",
        "claude-sonnet": "sonnet",
        "claude-haiku": "haiku",
    }

    name_normalized = name.lower().replace(" ", "-")
    for full, short in abbreviations.items():
        if full in name_normalized:
            return short

    # 如果名字太长，截断
    if len(name) > 15:
        return name[:12] + "..."
    return name


def format_context_percentage(pct):
    """格式化上下文占比，带颜色提示（通过 emoji）。"""
    if pct is None:
        return ""

    pct_int = int(pct)

    # 根据使用量选择提示
    if pct_int >= 80:
        indicator = "🔴"
    elif pct_int >= 50:
        indicator = "🟡"
    else:
        indicator = "🟢"

    return f"{indicator}ctx:{pct_int}%"


def format_rate_limits(rate_limits):
    """格式化 rate limit 信息。"""
    if not rate_limits:
        return ""

    parts = []
    fh = rate_limits.get("five_hour", {})
    if fh:
        used = fh.get("used", 0)
        limit = fh.get("limit", 1)
        pct = int(used / limit * 100) if limit else 0
        parts.append(f"5h:{pct}%")

    sd = rate_limits.get("seven_day", {})
    if sd:
        used = sd.get("used", 0)
        limit = sd.get("limit", 1)
        pct = int(used / limit * 100) if limit else 0
        parts.append(f"7d:{pct}%")

    return " ".join(parts) if parts else ""


def build_statusline(data):
    """根据配置和输入数据构建状态栏文本。"""
    parts = []

    # 宠物
    pet = get_pet_frame()
    if pet:
        parts.append(pet)

    # 当前目录
    workspace = data.get("workspace", {})
    cwd = workspace.get("current_dir", "")
    git_worktree = workspace.get("git_worktree", {})

    if CONFIG["show_cwd"] and cwd:
        cwd_name = os.path.basename(cwd) or cwd
        if CONFIG["compact_mode"]:
            parts.append(cwd_name)
        else:
            parts.append(f"📁 {cwd_name}")

    # Git 分支
    if CONFIG["show_git"]:
        # 优先使用 git_worktree 中的分支信息
        branch = git_worktree.get("branch", "") if git_worktree else ""
        if not branch and cwd:
            branch = get_git_branch(cwd)
        if branch:
            if CONFIG["compact_mode"]:
                parts.append(f"[{branch}]")
            else:
                parts.append(f"🌿 {branch}")

    # 时间
    if CONFIG["show_time"]:
        now = datetime.now().strftime(CONFIG["time_format"])
        if CONFIG["compact_mode"]:
            parts.append(now)
        else:
            parts.append(f"⏰ {now}")

    # 用户名@主机名
    if CONFIG["show_user_host"]:
        user = os.environ.get("USERNAME") or os.environ.get("USER", "")
        host = os.environ.get("COMPUTERNAME") or os.environ.get("HOSTNAME", "")
        if user and host:
            if CONFIG["compact_mode"]:
                parts.append(f"{user}@{host}")
            else:
                parts.append(f"💻 {user}@{host}")

    # 模型
    if CONFIG["show_model"]:
        model = data.get("model", {})
        model_id = model.get("id", "")
        display_name = model.get("display_name", "")
        model_short = format_model_name(model_id, display_name)
        if model_short:
            if CONFIG["compact_mode"]:
                parts.append(model_short)
            else:
                parts.append(f"🤖 {model_short}")

    # 上下文占比
    if CONFIG["show_context"]:
        ctx = data.get("context_window", {})
        pct = ctx.get("used_percentage")
        if pct is not None:
            parts.append(format_context_percentage(pct))

    # Rate limits
    if CONFIG["show_rate_limit"]:
        rate_limits = data.get("rate_limits", {})
        rl_text = format_rate_limits(rate_limits)
        if rl_text:
            if CONFIG["compact_mode"]:
                parts.append(rl_text)
            else:
                parts.append(f"📊 {rl_text}")

    # 会话名
    if CONFIG["show_session"]:
        session = data.get("session_name", "")
        if session:
            parts.append(f"📌 {session}")

    # CPU / 内存
    if CONFIG["show_cpu"] or CONFIG["show_memory"]:
        cpu, mem = get_cpu_memory()
        if CONFIG["show_cpu"] and cpu is not None:
            cpu_icon = "🔴" if cpu >= 80 else "🟡" if cpu >= 50 else "🟢"
            if CONFIG["compact_mode"]:
                parts.append(f"cpu:{int(cpu)}%")
            else:
                parts.append(f"{cpu_icon}cpu:{int(cpu)}%")
        if CONFIG["show_memory"] and mem is not None:
            mem_icon = "🔴" if mem >= 80 else "🟡" if mem >= 50 else "🟢"
            if CONFIG["compact_mode"]:
                parts.append(f"mem:{int(mem)}%")
            else:
                parts.append(f"{mem_icon}mem:{int(mem)}%")

    # 天气
    if CONFIG["show_weather"]:
        weather = get_weather()
        if weather:
            parts.append(weather)

    # 组合输出
    return CONFIG["separator"].join(parts)


def main():
    try:
        data = json.load(sys.stdin)
    except json.JSONDecodeError:
        # 如果 stdin 没有有效 JSON，显示基础信息
        data = {}

    statusline = build_statusline(data)
    print(statusline, end="")


if __name__ == "__main__":
    main()
