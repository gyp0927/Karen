import asyncio
import logging
import os

from agents.nodes import create_agents
from interface.human_interface import HumanInterface
from state.manager import SessionManager

# 配置控制台日志（可通过环境变量 LOG_LEVEL 调整，默认 INFO）
_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, _LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
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


def _c(style: str, text: str) -> str:
    """用指定样式（支持 + 连接多种样式）包裹文本。"""
    codes = ""
    for key in style.split("+"):
        key = key.strip()
        if key in C:
            codes += C[key]
    return f"{codes}{text}{C['reset']}"


def _box(lines: list[str], width: int = 50, color: str = "lcyan") -> str:
    """用 Unicode 单线框包裹多行文本。"""
    out = []
    out.append(_c(color, "┌" + "─" * (width - 2) + "┐"))
    for line in lines:
        pad = width - 2 - len(line)
        out.append(_c(color, "│") + line + " " * max(pad, 0) + _c(color, "│"))
    out.append(_c(color, "└" + "─" * (width - 2) + "┘"))
    return "\n".join(out)


def _separator(char: str = "─", width: int = 50, color: str = "dim") -> str:
    return _c(color, char * width)


async def main():
    # ── 启动画面 ───────────────────────────────────────────
    print()
    print(_c("bold+lmagenta", "╔" + "═" * 48 + "╗"))
    print(_c("bold+lmagenta", "║") + " " * 48 + _c("bold+lmagenta", "║"))
    print(_c("bold+lmagenta", "║") + _c("bold+lcyan", "           ✦    凯 伦    ✦             ".center(48)) + _c("bold+lmagenta", "║"))
    print(_c("bold+lmagenta", "║") + " " * 48 + _c("bold+lmagenta", "║"))
    print(_c("bold+lmagenta", "║") + _c("dim", "      Karen AI · Multi-Agent System    ".center(48)) + _c("bold+lmagenta", "║"))
    print(_c("bold+lmagenta", "║") + " " * 48 + _c("bold+lmagenta", "║"))
    print(_c("bold+lmagenta", "╚" + "═" * 48 + "╝"))
    print()

    print(_c("dim", "  正在初始化智能体..."))

    # 创建 Agents
    coordinator, researcher, responder, reviewer = create_agents()
    print(_c("lgreen", "  ✓ 智能体初始化完成"))

    # 初始化消息管理器
    msg_manager = SessionManager()

    # 创建用户接口（启用协调模式 + 审查）
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

    # ── 就绪提示 ───────────────────────────────────────────
    print()
    print(_c("lcyan", "┌─ 系统就绪 ") + _c("dim", "─" * 37 + "┐"))
    print(_c("lcyan", "│") + _c("lwhite", "  输入消息开始对话") + " " * 31 + _c("lcyan", "│"))
    print(_c("lcyan", "│") + _c("dim", "  命令:") + _c("lyellow", " /review") + _c("dim", " 开关审查") + " " * 21 + _c("lcyan", "│"))
    print(_c("lcyan", "│") + "        " + _c("lyellow", "/fast  ") + _c("dim", "切换模式") + " " * 21 + _c("lcyan", "│"))
    print(_c("lcyan", "│") + "        " + _c("lyellow", "/clear ") + _c("dim", "清空对话") + " " * 21 + _c("lcyan", "│"))
    print(_c("lcyan", "│") + "        " + _c("lyellow", "/history ") + _c("dim", "查看历史") + " " * 18 + _c("lcyan", "│"))
    print(_c("lcyan", "│") + "        " + _c("lyellow", "exit   ") + _c("dim", "退出程序") + " " * 21 + _c("lcyan", "│"))
    print(_c("lcyan", "└" + "─" * 48 + "┘"))
    print()

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
                print(_c("lcyan", "┌─ 历史记录 ") + _c("dim", f"({len(history)} 条) ") + _c("dim", "─" * 28 + "┐"))
                for msg in history:
                    sender = getattr(msg, "name", "unknown")
                    content = msg.content[:60].replace("\n", " ")
                    pad = 40 - len(sender) - len(content)
                    line = f"  [{sender}]: {content}..."
                    print(_c("lcyan", "│") + line + " " * max(48 - len(line), 0) + _c("lcyan", "│"))
                print(_c("lcyan", "└" + "─" * 48 + "┘"))
                print()
                continue

            # ── AI 回复 ─────────────────────────────────────
            response = await interface.send_message(user_input)
            print()
            print(_c("bold+lcyan", "╭─ Assistant ") + _c("dim", "─" * 37 + "╮"))
            # 多行回复逐行打印，保持框线对齐
            for line in response.split("\n"):
                # 逐行处理，每行最多 46 个字符（留边距）
                while line:
                    chunk = line[:46] if len(line) > 46 else line
                    line = line[46:] if len(line) > 46 else ""
                    pad = 46 - len(chunk)
                    print(_c("lcyan", "│ ") + chunk + " " * pad + _c("lcyan", " │"))
            print(_c("lcyan", "╰" + "─" * 48 + "╯"))
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
