#!/usr/bin/env python3
"""
Claude Dashboard — Enhanced Statusline Script
宠物固定第一位置，其他信息按秒轮播。

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
    "weather_city": "",
    "weather_cache_minutes": 10,
    "time_format": "%H:%M",
    "pet_style": "emoji",
    "separator": " | ",
    "compact_mode": True,
}

# ==================== 宠物动画帧 ====================
PET_FRAMES_EMOJI = ["🐱", "😺", "😸", "😻", "🙀", "😽"]

PET_FRAMES_ASCII = [
    " /\\_/\\  ",
    "( o.o ) ",
    " > ^ <  ",
]


def get_pet_frame():
    """根据当前秒数选择宠物帧，实现动画效果。"""
    if not CONFIG["show_pet"]:
        return ""

    second = datetime.now().second

    if CONFIG["pet_style"] == "ascii":
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
    """获取天气，带文件缓存。"""
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

    import re
    m = re.search(r'(opus|sonnet|haiku)[^\d]*(\d+\.\d+)', name.lower())
    if m:
        family = m.group(1)[:2]
        return f"{family}{m.group(2)}"

    if len(name) > 12:
        return name[:10] + ".."
    return name


def format_context_percentage(pct):
    """格式化上下文占比，带颜色提示。"""
    if pct is None:
        return ""

    pct_int = int(pct)
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
    """构建状态栏：宠物固定第一，其他信息按秒轮播。"""
    parts = []

    # === 宠物：始终固定在第一位置 ===
    pet = get_pet_frame()
    if pet:
        parts.append(pet)

    # === 收集其他信息项 ===
    info_parts = []

    workspace = data.get("workspace", {})
    cwd = workspace.get("current_dir", "")
    git_worktree = workspace.get("git_worktree", {})

    if CONFIG["show_cwd"] and cwd:
        info_parts.append(os.path.basename(cwd) or cwd)

    if CONFIG["show_git"]:
        branch = git_worktree.get("branch", "") if git_worktree else ""
        if not branch and cwd:
            branch = get_git_branch(cwd)
        if branch:
            info_parts.append(f"[{branch}]")

    if CONFIG["show_time"]:
        info_parts.append(datetime.now().strftime(CONFIG["time_format"]))

    if CONFIG["show_model"]:
        model = data.get("model", {})
        model_short = format_model_name(model.get("id", ""), model.get("display_name", ""))
        if model_short:
            info_parts.append(model_short)

    if CONFIG["show_context"]:
        ctx = data.get("context_window", {})
        pct = ctx.get("used_percentage")
        if pct is not None:
            info_parts.append(format_context_percentage(pct))

    if CONFIG["show_rate_limit"]:
        rl_text = format_rate_limits(data.get("rate_limits", {}))
        if rl_text:
            info_parts.append(rl_text)

    if CONFIG["show_session"]:
        session = data.get("session_name", "")
        if session:
            info_parts.append(f"📌 {session}")

    if CONFIG["show_user_host"]:
        user = os.environ.get("USERNAME") or os.environ.get("USER", "")
        host = os.environ.get("COMPUTERNAME") or os.environ.get("HOSTNAME", "")
        if user and host:
            info_parts.append(f"{user}@{host}")

    if CONFIG["show_cpu"] or CONFIG["show_memory"]:
        cpu, mem = get_cpu_memory()
        if CONFIG["show_cpu"] and cpu is not None:
            info_parts.append(f"C:{int(cpu)}%")
        if CONFIG["show_memory"] and mem is not None:
            info_parts.append(f"M:{int(mem)}%")

    if CONFIG["show_weather"]:
        weather = get_weather()
        if weather:
            info_parts.append(weather)

    # === 轮播：每秒只显示一条信息 ===
    if info_parts:
        second = datetime.now().second
        idx = second % len(info_parts)
        parts.append(info_parts[idx])

    return CONFIG["separator"].join(parts)


def main():
    try:
        data = json.load(sys.stdin)
    except json.JSONDecodeError:
        data = {}

    statusline = build_statusline(data)
    print(statusline, end="")


if __name__ == "__main__":
    main()
