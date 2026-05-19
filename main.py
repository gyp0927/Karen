import asyncio
import logging
import os
import re

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
    """计算文本在终端中的视觉宽度（去除 ANSI 转义序列）。"""
    return len(_ANSI_RE.sub("", text))


def _pad_visual(text: str, width: int) -> str:
    """按视觉宽度左对齐补齐。"""
    pad = width - _visual_len(text)
    return text + " " * max(pad, 0)


def _box(lines: list[str], width: int = 80, color: str = "lcyan") -> str:
    """用 Unicode 单线框包裹多行文本。"""
    out = []
    out.append(_c(color, "┌" + "─" * (width - 2) + "┐"))
    for line in lines:
        pad = width - 2 - _visual_len(line)
        out.append(_c(color, "│") + line + " " * max(pad, 0) + _c(color, "│"))
    out.append(_c(color, "└" + "─" * (width - 2) + "┘"))
    return "\n".join(out)


def _separator(char: str = "─", width: int = 80, color: str = "dim") -> str:
    return _c(color, char * width)


# ── 像素头像（ANSI 256 色背景）─────────────────────────────
# 基于粉色短发少女形象：亮粉头发 + 红色发带 + 紫红瞳孔 + 淡蓝衣服
_AVATAR_BG = {
    "H": "\033[48;5;219m",  # 亮粉-头发高光
    "h": "\033[48;5;212m",  # 中粉-头发主体
    "d": "\033[48;5;168m",  # 深粉-头发阴影/轮廓
    "r": "\033[48;5;196m",  # 红-发带
    "s": "\033[48;5;224m",  # 肤色
    "p": "\033[48;5;217m",  # 粉-腮红
    "e": "\033[48;5;201m",  # 紫红-瞳孔
    "w": "\033[48;5;15m",   # 白-高光
    "m": "\033[48;5;204m",  # 玫红-嘴巴
    "c": "\033[48;5;195m",  # 淡蓝-衣服
}

_AVATAR_PIXELS = [
    [" ", " ", " ", " ", " ", " ", "h", "h", "h", "h", "h", "h", " ", " ", " ", " ", " ", " "],
    [" ", " ", " ", " ", "h", "h", "H", "H", "H", "H", "H", "H", "h", "h", " ", " ", " ", " "],
    [" ", " ", " ", "d", "H", "h", "h", "s", "s", "s", "s", "h", "h", "H", "d", " ", " ", " "],
    [" ", "d", "H", "h", "s", "s", "e", "w", "s", "s", "e", "w", "s", "s", "h", "H", "d", " "],
    [" ", "d", "H", "s", "p", "s", "s", "s", "s", "s", "s", "s", "p", "s", "s", "H", "d", " "],
    ["d", "h", "s", "s", "s", "s", "s", "s", "m", "m", "s", "s", "s", "s", "s", "s", "h", "d"],
    [" ", "d", "h", "r", "s", "s", "s", "s", "s", "s", "s", "s", "s", "s", "r", "h", "d", " "],
    [" ", " ", "d", "h", "r", "s", "s", "s", "s", "s", "s", "s", "s", "r", "h", "d", " ", " "],
    [" ", " ", " ", "d", "h", "h", "s", "s", "s", "s", "h", "h", "d", " ", " ", " ", " ", " "],
    [" ", " ", " ", " ", " ", "r", "r", "r", "r", "r", "r", " ", " ", " ", " ", " ", " ", " "],
    [" ", " ", " ", " ", " ", "c", "c", "c", "c", "c", "c", "c", "c", " ", " ", " ", " ", " "],
    [" ", " ", " ", " ", "c", "c", "c", "c", "c", "c", "c", "c", "c", "c", " ", " ", " ", " "],
    [" ", " ", " ", "c", "c", "c", "c", "c", "c", "c", "c", "c", "c", "c", "c", " ", " ", " "],
    [" ", " ", "c", "c", "c", "c", "c", "c", "c", "c", "c", "c", "c", "c", "c", "c", " ", " "],
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


async def main():
    # ── 启动画面（头像 + 标题并排）──────────────────────────
    avatar_lines = [_render_avatar_row(row) for row in _AVATAR_PIXELS]
    # 标题区域 24 字符宽度，12 行与头像对齐
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
    # 补齐标题到统一视觉宽度
    titles = [_pad_visual(t, 24) for t in titles]

    # 单线框，总宽 64（头像36 + 间距2 + 标题24 + 边框2）
    _W = 62
    print()
    print(_c("bold+lmagenta", "╭" + "─" * _W + "╮"))
    for i in range(12):
        line = avatar_lines[i] + "  " + titles[i]
        pad = _W - _visual_len(line)
        print(_c("bold+lmagenta", "│") + line + " " * max(pad, 0) + _c("bold+lmagenta", "│"))
    print(_c("bold+lmagenta", "╰" + "─" * _W + "╯"))
    print()

    # 静默初始化（不再打印初始化信息，日志写入文件）
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

    # 对话循环
    while True:
        try:
            # ── 用户输入提示 ─────────────────────────────────
            user_input = input(_c("bold+lgreen", "💬 ") + _c("bold+lwhite", "You") + _c("dim", " ───> ")).strip()
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
                header_width = _visual_len(header)
                left_dash = (78 - header_width) // 2
                right_dash = 78 - header_width - left_dash
                print(_c("lcyan", "┌" + "─" * left_dash) + _c("bold+lcyan", header) + _c("lcyan", "─" * right_dash + "┐"))
                for msg in history:
                    sender = getattr(msg, "name", "unknown")
                    content = msg.content[:60].replace("\n", " ")
                    line = f"  [{sender}]: {content}"
                    if len(line) > 76:
                        line = line[:73] + "..."
                    padded = _pad_visual(line, 78)
                    print(_c("lcyan", "│") + padded + _c("lcyan", "│"))
                print(_c("lcyan", "└" + "─" * 78 + "┘"))
                print()
                continue

            # ── AI 回复 ─────────────────────────────────────
            response = await interface.send_message(user_input)
            print()
            print(_c("bold+lcyan", "╭─ Assistant ") + _c("dim", "─" * 66 + "╮"))
            # 多行回复逐行打印，保持框线对齐
            for line in response.split("\n"):
                # 逐行处理，每行最多 76 个字符（留边距）
                while line:
                    chunk = line[:76] if len(line) > 76 else line
                    line = line[76:] if len(line) > 76 else ""
                    pad = 76 - len(chunk)
                    print(_c("lcyan", "│ ") + chunk + " " * pad + _c("lcyan", " │"))
            print(_c("lcyan", "╰" + "─" * 78 + "╯"))
            print()

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
