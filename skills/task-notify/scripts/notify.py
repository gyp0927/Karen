#!/usr/bin/env python3
"""
Windows 桌面通知脚本 — 在桌面右下角弹出通知气泡。

用法:
    python notify.py --title "Claude" --message "任务已完成" --duration 5
    python notify.py -t "Claude" -m "代码修复完成" -d 5
"""

import argparse
import subprocess
import sys


def show_notification(title: str, message: str, duration: int = 5) -> None:
    """在 Windows 桌面右下角显示通知气泡。

    Args:
        title: 通知标题
        message: 通知正文内容
        duration: 显示时长（秒），默认 5 秒
    """
    ps_script = f'''
    Add-Type -AssemblyName System.Windows.Forms
    $balloon = New-Object System.Windows.Forms.NotifyIcon
    $balloon.Icon = [System.Drawing.SystemIcons]::Information
    $balloon.BalloonTipIcon = [System.Windows.Forms.ToolTipIcon]::Info
    $balloon.BalloonTipTitle = "{title.replace('"', '`"')}"
    $balloon.BalloonTipText = "{message.replace('"', '`"').replace(chr(10), ' ')}"
    $balloon.Visible = $true
    $balloon.ShowBalloonTip({duration * 1000})
    Start-Sleep -Milliseconds ({duration * 1000 + 500})
    $balloon.Dispose()
    '''

    try:
        subprocess.run(
            ["powershell", "-WindowStyle", "Hidden", "-Command", ps_script],
            check=True,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
    except subprocess.CalledProcessError as e:
        print(f"通知发送失败: {e}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print("错误：未找到 PowerShell。此脚本仅支持 Windows 系统。", file=sys.stderr)
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
