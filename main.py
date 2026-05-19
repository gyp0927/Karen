import asyncio
import logging
import os
import re
import time
import warnings

# 在 import langgraph 之前过滤其弃用警告
# 必须用具体类名，标准 PendingDeprecationWarning 子类匹配不生效
from langchain_core._api.deprecation import LangChainPendingDeprecationWarning
warnings.filterwarnings("ignore", category=LangChainPendingDeprecationWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)

from agents.nodes import create_agents
from interface.human_interface import HumanInterface
from state.manager import SessionManager

# ── 日志配置：文件记录全量，控制台仅 WARNING ───────────────
_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(_LOG_DIR, exist_ok=True)

# 清除已有 handlers，防止重复
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# 文件 handler：记录完整日志
_file_level = getattr(logging, _LOG_LEVEL, logging.INFO)
file_handler = logging.FileHandler(
    os.path.join(_LOG_DIR, "karen.log"),
    encoding="utf-8",
)
file_handler.setLevel(_file_level)
file_handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
))
logging.root.addHandler(file_handler)

# 控制台 handler：默认只显示 WARNING+，可通过 LOG_LEVEL 环境变量调整
_console_level = logging.WARNING
if "LOG_LEVEL" in os.environ:
    _console_level = _file_level
console_handler = logging.StreamHandler()
console_handler.setLevel(_console_level)
console_handler.setFormatter(logging.Formatter(
    "%(levelname)s: %(message)s",
))
logging.root.addHandler(console_handler)

logging.root.setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)

# ── ANSI 颜色 ──────────────────────────────────────────────
C = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "italic": "\033[3m",
    "underline": "\033[4m",
    # 前景色
    "black": "\033[30m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "white": "\033[37m",
    # 亮前景色
    "lred": "\033[91m",
    "lgreen": "\033[92m",
    "lyellow": "\033[93m",
    "lblue": "\033[94m",
    "lmagenta": "\033[95m",
    "lcyan": "\033[96m",
    "lwhite": "\033[97m",
}

_ANSI_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


def _c(style: str, text: str) -> str:
    """用指定样式（支持 + 连接多种样式）包裹文本。"""
    codes = ""
    for key in style.split("+"):
        key = key.strip()
        if key in C:
            codes += C[key]
    return f"{codes}{text}{C['reset']}"


def _visual_len(text: str) -> int:
    """计算文本在终端中的视觉宽度（去除 ANSI 转义序列 + CJK 双宽）。"""
    text = _ANSI_RE.sub("", text)
    width = 0
    for ch in text:
        o = ord(ch)
        # CJK Unified Ideographs + Extension A
        if 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF:
            width += 2
        # Fullwidth ASCII + CJK symbols
        elif 0xFF01 <= o <= 0xFF60 or 0xFFE0 <= o <= 0xFFE6 or 0x3000 <= o <= 0x303F:
            width += 2
        # Hangul
        elif 0xAC00 <= o <= 0xD7AF or 0x1100 <= o <= 0x11FF or 0x3130 <= o <= 0x318F:
            width += 2
        else:
            width += 1
    return width


def _split_visual(text: str, max_width: int) -> tuple[str, str]:
    """按视觉宽度分割文本为 (chunk, remaining)。"""
    width = 0
    for i, ch in enumerate(text):
        o = ord(ch)
        w = 2 if (
            0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF
            or 0xFF01 <= o <= 0xFF60 or 0xFFE0 <= o <= 0xFFE6
            or 0x3000 <= o <= 0x303F or 0xAC00 <= o <= 0xD7AF
            or 0x1100 <= o <= 0x11FF or 0x3130 <= o <= 0x318F
        ) else 1
        if width + w > max_width:
            return text[:i], text[i:]
        width += w
    return text, ""


# ── 终端宽度 ────────────────────────────────────────────────
try:
    _TERM_W = os.get_terminal_size().columns
except OSError:
    _TERM_W = 100
_MSG_W = max(min(_TERM_W - 2, 120), 64)  # 留边距，最小 64 最大 120

# ── 像素头像（ANSI 256 色背景）─────────────────────────────
_AVATAR_BG = {
    "H": "\033[48;5;225m",  # 最亮粉-头发高光
    "h": "\033[48;5;218m",  # 亮粉-头发主体
    "n": "\033[48;5;211m",  # 中粉-头发过渡
    "d": "\033[48;5;168m",  # 深粉-头发阴影
    "D": "\033[48;5;132m",  # 暗紫-轮廓/最深阴影
    "r": "\033[48;5;160m",  # 暗红-发带
    "s": "\033[48;5;224m",  # 肤色
    "e": "\033[48;5;235m",  # 近黑-瞳孔
    "w": "\033[48;5;15m",   # 白-眼睛高光
    "c": "\033[48;5;117m",  # 柔和蓝-衣服
    "b": "\033[48;5;217m",  # 腮红粉
}

_AVATAR_PIXELS = [
    [" ", " ", " ", "D", "d", "h", "H", "H", "H", "H", "H", "h", "d", "D", "D", " ", " ", " "],
    [" ", " ", "D", "h", "H", "H", "H", "H", "H", "H", "H", "H", "H", "h", "d", "D", " ", " "],
    [" ", "D", "h", "H", "H", "n", "n", "n", "n", "n", "n", "H", "H", "H", "h", "d", "D", " "],
    ["D", "h", "H", "H", "n", "n", "s", "s", "s", "s", "n", "n", "H", "H", "H", "h", "d", "D"],
    ["D", "h", "H", "n", "n", "s", "s", "e", "w", "s", "n", "n", "H", "H", "H", "h", "d", "D"],
    [" ", "d", "h", "n", "s", "s", "s", "s", "s", "s", "s", "n", "n", "H", "H", "h", "d", " "],
    [" ", " ", "d", "n", "s", "s", "b", "s", "s", "b", "s", "n", "n", "h", "h", "d", " ", " "],
    [" ", " ", " ", "d", "h", "n", "s", "s", "s", "s", "n", "h", "h", "d", "d", " ", " ", " "],
    [" ", " ", " ", " ", "d", "h", "n", "r", "r", "r", "n", "h", "d", " ", " ", " ", " ", " "],
    [" ", " ", " ", " ", " ", "d", "h", "h", "s", "h", "h", "d", " ", " ", " ", " ", " ", " "],
    [" ", " ", " ", " ", " ", " ", "c", "c", "c", "c", "c", " ", " ", " ", " ", " ", " ", " "],
    [" ", " ", " ", " ", " ", "c", "c", "c", "c", "c", "c", "c", " ", " ", " ", " ", " ", " "],
    [" ", " ", " ", " ", "c", "c", "c", "c", "c", "c", "c", "c", " ", " ", " ", " ", " ", " "],
    [" ", " ", " ", "c", "c", "c", "c", "c", "c", "c", "c", "c", "c", " ", " ", " ", " ", " "],
]


def _render_avatar_row(pixels: list[str]) -> str:
    """将像素行渲染为带 ANSI 背景色的字符串（每个像素 = 2 空格）。"""
    out = ""
    current = None
    for p in pixels:
        if p == " ":
            if current != "reset":
                out += C["reset"]
                current = "reset"
            out += "  "
        else:
            color = _AVATAR_BG.get(p)
            if color is None:
                color = C["reset"]
            if color != current:
                out += color
                current = color
            out += "  "
    if current != "reset":
        out += C["reset"]
    return out


async def _thinking_spinner(stop_event: asyncio.Event, interval: float = 0.3):
    """AI 处理期间显示动态思考动画，用 \\r 覆盖同一行。"""
    frames = ["(⊙_⊙)", "(◉_◉)", "(⊙ω⊙)", "(◕_◕)"]
    texts = ["思考中...", "搜索中...", "整理中...", "构思中..."]
    i = 0
    while not stop_event.is_set():
        frame = frames[i % len(frames)]
        text = texts[i % len(texts)]
        line = f"\r{_c('yellow', frame)} {_c('dim', text)}"
        print(line, end="", flush=True)
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval)
            break
        except TimeoutError:
            pass
        i += 1
    # 清除 spinner 行
    print(f"\r{' ' * 40}\r", end="", flush=True)


def _render_status_bar(model: str, used_tokens: int, elapsed_sec: int) -> str:
    """渲染终端状态栏：$ 模型名 | 用量/上限 [进度条] 百分比 | 时间"""
    # token 格式化
    if used_tokens >= 1000:
        token_str = f"{used_tokens / 1000:.1f}K"
    else:
        token_str = str(used_tokens)

    # 根据模型确定上限
    model_lower = model.lower()
    if "kimi" in model_lower:
        max_tokens = 260000
    else:
        max_tokens = 128000
    max_str = f"{max_tokens / 1000:.0f}K"

    # 进度条
    pct = min(used_tokens / max_tokens, 1.0) if max_tokens > 0 else 0.0
    bar_len = 20
    filled = int(pct * bar_len)
    bar = "█" * filled + "░" * (bar_len - filled)

    # 百分比
    pct_str = f"{int(pct * 100)}%"

    # 时间
    hours = elapsed_sec // 3600
    mins = (elapsed_sec % 3600) // 60
    if hours > 0:
        time_str = f"{hours}h {mins:02d}m"
    else:
        time_str = f"{mins}m"

    # 组装（带颜色）
    model_colored = _c("yellow", model or "default")
    bar_colored = _c("green", bar)
    return f"$ {model_colored} | {token_str}/{max_str} [{bar_colored}] {pct_str} | {time_str}"


async def main():
    # ── 启动画面（头像 + 标题并排）──────────────────────────
    avatar_lines = [_render_avatar_row(row) for row in _AVATAR_PIXELS]
    _T = " " * 24
    titles = [
        _T,
        _c("bold+lcyan", "    ✦  凯 伦  ✦        "),
        _T,
        _c("dim", "  输入消息开始对话      "),
        _T,
        _c("dim", "  exit    退出对话      "),
        _c("dim", "  /review 审查开关      "),
        _c("dim", "  /fast   模式切换      "),
        _c("dim", "  /clear  清空历史      "),
        _T,
        _T,
        _T,
    ]
    # 启动框固定 64 宽（与终端无关）
    _BOOT_W = 62
    print()
    print(_c("bold+lmagenta", "╭" + "─" * _BOOT_W + "╮"))
    for i in range(12):
        line = avatar_lines[i] + "  " + titles[i]
        pad = _BOOT_W - _visual_len(line)
        print(_c("bold+lmagenta", "│") + line + " " * max(pad, 0) + _c("bold+lmagenta", "│"))
    print(_c("bold+lmagenta", "╰" + "─" * _BOOT_W + "╯"))
    print()

    # 静默初始化
    coordinator, researcher, responder, reviewer = create_agents()
    msg_manager = SessionManager()
    interface = HumanInterface(
        message_manager=msg_manager,
        coordinator=coordinator,
        researcher=researcher,
        responder=responder,
        reviewer=reviewer,
        fast_mode=True,
        review=False,
        review_language="zh",
    )

    session_start_time = time.time()

    # 对话循环
    while True:
        try:
            # ── 用户输入 ────────────────────────────────────
            print()
            user_input = input(_c("yellow", "● ")).strip()
            if not user_input:
                continue

            # 处理命令
            if user_input.lower() == "exit":
                print()
                print(_c("lcyan", "  再见！"))
                print()
                break
            elif user_input.lower() == "/review":
                interface.review = not interface.review
                status = _c("lgreen", "已开启") if interface.review else _c("lred", "已关闭")
                print(_c("dim", f"\n  审查模式: {status}\n"))
                continue
            elif user_input.lower() == "/fast":
                interface.fast_mode = not interface.fast_mode
                mode = _c("lgreen", "快速") if interface.fast_mode else _c("lyellow", "协调")
                print(_c("dim", f"\n  当前模式: {mode}\n"))
                continue
            elif user_input.lower() == "/clear":
                msg_manager.clear()
                print(_c("dim", "\n  对话已清空。\n"))
                continue
            elif user_input.lower() == "/history":
                history = msg_manager.get_messages()
                print()
                header = f" 历史记录 ({len(history)} 条) "
                header_v = _visual_len(header)
                left = (_MSG_W - 2 - header_v) // 2
                right = _MSG_W - 2 - header_v - left
                print(_c("lcyan", "┌" + "─" * left) + _c("bold+lcyan", header) + _c("lcyan", "─" * right + "┐"))
                for msg in history:
                    sender = getattr(msg, "name", "unknown")
                    content = msg.content[:60].replace("\n", " ")
                    line = f"  [{sender}]: {content}"
                    chunk, _ = _split_visual(line, _MSG_W - 2)
                    pad = _MSG_W - 2 - _visual_len(chunk)
                    print(_c("lcyan", "│") + chunk + " " * max(pad, 0) + _c("lcyan", "│"))
                print(_c("lcyan", "└" + "─" * (_MSG_W - 2) + "┘"))
                print()
                continue

            # ── AI 回复 ─────────────────────────────────────
            # 获取路由信息用于状态栏
            from core.model_router import get_router
            router = get_router()
            history = msg_manager.get_messages()
            history_turns = len(history) // 2
            route_result = router.route(user_input, history_turns)
            model_name = route_result["config"].get("model", "default")

            # 启动思考动画
            stop_event = asyncio.Event()
            spinner_task = asyncio.create_task(_thinking_spinner(stop_event))
            try:
                response = await interface.send_message(user_input)
            finally:
                stop_event.set()
                try:
                    await asyncio.wait_for(spinner_task, timeout=0.5)
                except TimeoutError:
                    pass

            # 分隔线
            print(_c("dim", "─" * _MSG_W))

            # 全宽回复框
            label = _c("bold+lcyan", "🧠 Karen")
            label_plain = "🧠 Karen"
            label_v = _visual_len(label_plain)
            top_dash = _MSG_W - 6 - label_v  # ╭─  + 标签 + 空格 + ╮
            print(_c("lcyan", "╭─ ") + label + _c("lcyan", " " + "─" * top_dash + "╮"))

            content_w = _MSG_W - 4  # │ 内容 │
            for line in response.split("\n"):
                while line:
                    chunk, line = _split_visual(line, content_w)
                    pad = content_w - _visual_len(chunk)
                    print(_c("lcyan", "│ ") + chunk + " " * max(pad, 0) + _c("lcyan", " │"))

            print(_c("lcyan", "╰" + "─" * (_MSG_W - 2) + "╯"))

            # 状态栏
            sid = msg_manager.get_current_session_id()
            from state.stats import get_session_stats
            session_stats = get_session_stats(sid)
            elapsed = int(time.time() - session_start_time)
            status_line = _render_status_bar(
                session_stats.get("last_model") or model_name,
                session_stats.get("total_tokens", 0),
                elapsed,
            )
            print(status_line)

        except KeyboardInterrupt:
            print()
            print(_c("lcyan", "  再见！"))
            print()
            break
        except Exception as e:
            logger.exception("Error in conversation loop")
            print(_c("lred", f"\n  ✗ 错误: {e}\n"))


def run_cli():
    """同步入口，供 console_scripts 调用。"""
    asyncio.run(main())


if __name__ == "__main__":
    run_cli()
