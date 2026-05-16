#!/usr/bin/env python3
"""
Windows 桌面通知脚本 — 使用原生 Toast Notification，美观现代，来源显示 Claude。

用法:
    python notify.py --title "Claude" --message "任务已完成" --duration 5
    python notify.py -t "Claude" -m "代码修复完成" -d 5
"""

import argparse
import sys
from pathlib import Path

def get_icon_path() -> str | None:
    """获取 Claude 图标路径。"""
    script_dir = Path(__file__).parent
    icon_file = script_dir.parent / "assets" / "claude_icon.png"
    if icon_file.exists():
        return f"file:///{icon_file.absolute().as_posix()}"
    return None


def show_notification(title: str, message: str, duration: int = 5) -> None:
    """使用 Windows Toast Notification 显示通知。

    通知来源显示为 "Claude"，并使用 Claude 品牌图标。
    """
    try:
        from win11toast import toast
    except ImportError:
        print("错误：未安装 win11toast。请运行: pip install win11toast", file=sys.stderr)
        sys.exit(1)

    duration_str = "long" if duration >= 7 else "short"
    icon = get_icon_path()

    try:
        toast(
            title=title,
            body=message,
            app_id="Claude",
            icon=icon,
            duration=duration_str,
        )
    except Exception as e:
        print(f"通知发送失败: {e}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Windows 桌面通知工具")
    parser.add_argument("-t", "--title", default="Claude", help="通知标题（默认: Claude）")
    parser.add_argument("-m", "--message", default="任务已完成", help="通知正文内容")
    parser.add_argument("-d", "--duration", type=int, default=5, help="显示时长，秒（默认: 5）")
    args = parser.parse_args()

    show_notification(args.title, args.message, args.duration)


if __name__ == "__main__":
    main()
