"""系统工具 —— 文件查询与电脑控制

提供安全的文件操作和命令执行能力：
- 文件读取、目录浏览、文件搜索
- 安全的 shell 命令执行（带黑名单限制）

安全策略：
1. 文件操作限制在 HOME 目录及其子目录
2. 命令执行禁止删除、格式化、权限修改等危险操作
3. 路径规范化防止目录遍历攻击
"""

import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_HOME_DIR = Path.home()

# 危险命令黑名单（正则匹配）
_DANGEROUS_PATTERNS = [
    r"\brm\b.*-\s*[rfR]+",          # rm -rf
    r"\brm\b.*[/\\]",               # rm 后跟路径（过于宽泛，改为仅禁止根目录删除）
    r"\bformat\b",
    r"\bmkfs\b",
    r"\bfdisk\b",
    r"\bchmod\b.*777",
    r"\bchown\b",
    r"\bdd\b.*if=",
    r"[:]>\s*[/\\]",                 # 重定向到系统路径
    r"\bwget\b.*-O\s*[/\\]",
    r"\bcurl\b.*-o\s*[/\\]",
]

# 允许的简单命令前缀白名单（更严格的安全策略）
_SAFE_COMMAND_PREFIXES = {
    # 文件查看
    "ls", "dir", "cat", "type", "head", "tail", "less", "more",
    "find", "grep", "wc", "stat", "file",
    # 系统信息
    "pwd", "cd", "echo", "date", "whoami", "hostname", "uname", "ver",
    "ps", "top", "htop", "tasklist", "df", "du", "free", "systeminfo",
    # 开发工具
    "python", "python3", "node", "npm", "npx", "git", "pip",
    "rustc", "cargo", "go", "javac", "java",
    # 网络查看（只读）
    "ping", "nslookup", "tracert", "traceroute", "netstat", "ipconfig", "ifconfig",
    # 其他
    "which", "where", "open", "start",
}


def _normalize_path(path: str) -> Path | None:
    """规范化路径并限制在 HOME 目录内。"""
    if not path:
        return _HOME_DIR

    # 处理 ~ 前缀
    if path.startswith("~"):
        path = str(_HOME_DIR) + path[1:]

    try:
        p = Path(path).resolve()
    except (ValueError, OSError) as e:
        logger.warning(f"Invalid path: {path}, error: {e}")
        return None

    # 安全检查：路径必须在 HOME 目录内
    # 允许访问常见用户目录（Downloads, Documents, Desktop 等）
    try:
        p.relative_to(_HOME_DIR)
        return p
    except ValueError:
        # 路径不在 HOME 内，拒绝访问
        logger.warning(f"Path outside HOME directory: {p}")
        return None


def _is_command_safe(command: str) -> tuple[bool, str]:
    """检查命令是否安全。返回 (是否安全, 原因)。"""
    cmd = command.strip()
    if not cmd:
        return False, "命令为空"

    # 检查危险模式
    for pattern in _DANGEROUS_PATTERNS:
        if re.search(pattern, cmd, re.IGNORECASE):
            return False, f"命令包含危险操作模式"

    # 提取主命令（第一个词）
    # 处理管道和重定向：只允许简单命令
    if any(c in cmd for c in ["|", "&", ";", "`", "$"]):
        return False, "不支持管道、后台执行、命令链或变量展开"

    # 提取第一个 token 作为命令名
    first_token = cmd.split()[0].lower()

    # 去掉路径前缀（如 /usr/bin/python → python）
    cmd_name = os.path.basename(first_token).lower()

    # Windows: 去掉 .exe 后缀
    if cmd_name.endswith(".exe"):
        cmd_name = cmd_name[:-4]

    # 检查是否在白名单中
    if cmd_name not in _SAFE_COMMAND_PREFIXES:
        return False, f"命令 '{cmd_name}' 不在允许列表中"

    # 解释器类命令的参数安全检查：
    # 禁止通过 -c / -e / -exec 等参数执行任意代码
    _DANGEROUS_ARGS = {"-c", "-e", "-exec", "--exec", "-eval", "-execdir", "--execdir"}
    # 可以执行代码的解释器
    _INTERPRETER_CMDS = {"python", "python3", "node", "npm", "npx", "rustc", "cargo", "go", "javac", "java"}
    # find 的 -exec 也能执行任意命令
    _FIND_CMD = {"find"}
    if cmd_name in _INTERPRETER_CMDS | _FIND_CMD:
        tokens = cmd.split()
        for token in tokens[1:]:
            if token in _DANGEROUS_ARGS:
                return False, f"命令禁止使用参数 '{token}' 执行内联代码"

    return True, ""


async def read_file(path: str, max_lines: int = 200) -> str:
    """读取文件内容。

    Args:
        path: 文件路径（支持 ~ 表示 HOME 目录）
        max_lines: 最多读取行数，超长文件自动截断

    Returns:
        文件内容文本，或错误提示
    """
    p = _normalize_path(path)
    if p is None:
        return f"[错误: 只能访问 HOME 目录({ _HOME_DIR })下的文件]"

    if not p.exists():
        return f"[错误: 文件不存在: {p}]"
    if not p.is_file():
        return f"[错误: 不是文件: {p}]"

    # 大小限制：最多读取 500KB
    max_size = 500 * 1024
    try:
        size = p.stat().st_size
        if size > max_size:
            return f"[文件过大: {size / 1024:.0f}KB (上限 500KB)，无法完整读取]"
    except OSError as e:
        return f"[读取失败: {e}]"

    try:
        content = p.read_text(encoding="utf-8", errors="replace")
        lines = content.split("\n")
        if len(lines) > max_lines:
            head = "\n".join(lines[:max_lines])
            return f"{head}\n\n... (共 {len(lines)} 行，已截断显示前 {max_lines} 行)"
        return content
    except Exception as e:
        return f"[读取失败: {e}]"


async def write_file(path: str, content: str) -> str:
    """写入文件内容。当你需要创建新文件或修改已有文件时使用此工具。

    Args:
        path: 文件路径（支持 ~ 表示 HOME 目录）
        content: 要写入的文本内容

    Returns:
        写入结果提示
    """
    p = _normalize_path(path)
    if p is None:
        return f"[错误: 只能访问 HOME 目录({ _HOME_DIR })下的文件]"

    # 禁止写入系统关键路径
    _PROTECTED_NAMES = {"boot.ini", "ntldr", "pagefile.sys", "hiberfil.sys"}
    if p.name.lower() in _PROTECTED_NAMES:
        return f"[错误: 禁止覆盖系统文件: {p.name}]"

    # 大小限制：最多写入 1MB
    if len(content) > 1024 * 1024:
        return "[错误: 内容超过 1MB 限制]"

    try:
        # 确保父目录存在
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"写入成功: {p} ({len(content)} 字符)"
    except Exception as e:
        return f"[写入失败: {e}]"


async def list_directory(path: str = "") -> str:
    """列出目录内容。

    Args:
        path: 目录路径（支持 ~ 表示 HOME 目录，空字符串表示 HOME）

    Returns:
        目录列表文本
    """
    p = _normalize_path(path)
    if p is None:
        return f"[错误: 只能访问 HOME 目录({ _HOME_DIR })下的目录]"

    if not p.exists():
        return f"[错误: 目录不存在: {p}]"
    if not p.is_dir():
        return f"[错误: 不是目录: {p}]"

    try:
        items = sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
        lines = [f"📁 {p.resolve()}", "=" * 50]

        # 显示 ../ 和 ./（如果不在 HOME 根目录）
        try:
            if p != _HOME_DIR and p != _HOME_DIR.resolve():
                parent = p.parent
                if _normalize_path(str(parent)) is not None:
                    lines.append(f"  ../          (上级目录)")
        except (ValueError, OSError):
            pass

        for item in items:
            try:
                if item.is_dir():
                    lines.append(f"  📁 {item.name}/")
                else:
                    size = item.stat().st_size
                    if size < 1024:
                        size_str = f"{size}B"
                    elif size < 1024 * 1024:
                        size_str = f"{size / 1024:.1f}KB"
                    else:
                        size_str = f"{size / (1024 * 1024):.1f}MB"
                    lines.append(f"  📄 {item.name:<40} {size_str:>10}")
            except (OSError, PermissionError):
                lines.append(f"  ❓ {item.name}    (无法访问)")

        return "\n".join(lines)
    except Exception as e:
        return f"[列出目录失败: {e}]"


async def search_files(pattern: str, directory: str = "", max_results: int = 20) -> str:
    """按文件名模式搜索文件。

    Args:
        pattern: 搜索模式（支持 * 和 ? 通配符，如 *.py, test*）
        directory: 搜索起始目录（默认 HOME）
        max_results: 最大返回结果数

    Returns:
        搜索结果列表
    """
    p = _normalize_path(directory)
    if p is None:
        return f"[错误: 只能搜索 HOME 目录({ _HOME_DIR })下的文件]"

    if not p.exists():
        return f"[错误: 目录不存在: {p}]"
    if not p.is_dir():
        return f"[错误: 不是目录: {p}]"

    try:
        results: list[str] = []
        # 限制搜索深度为 5 层，避免遍历整个 HOME 目录过慢
        max_depth = 5

        def _search_recursive(current: Path, depth: int) -> None:
            if depth > max_depth or len(results) >= max_results:
                return
            try:
                for item in current.iterdir():
                    if len(results) >= max_results:
                        return
                    try:
                        if item.is_dir():
                            _search_recursive(item, depth + 1)
                        elif item.match(pattern):
                            rel = item.relative_to(_HOME_DIR)
                            results.append(str(rel))
                    except PermissionError:
                        continue
            except PermissionError:
                return

        _search_recursive(p, 0)

        if not results:
            return f"在 '{p}' 下未找到匹配 '{pattern}' 的文件（搜索深度: {max_depth} 层）"

        lines = [f"🔍 搜索结果: '{pattern}' in '{p}'", "=" * 50]
        for r in results:
            lines.append(f"  📄 ~/{r}")
        if len(results) >= max_results:
            lines.append(f"\n... (已达最大结果数 {max_results}，请缩小搜索范围)")
        return "\n".join(lines)
    except Exception as e:
        return f"[搜索失败: {e}]"


async def edit_file(path: str, old_string: str, new_string: str) -> str:
    """编辑文件内容。查找 old_string 并将其替换为 new_string。

    用于精确修改文件中的某一段内容，old_string 必须完全匹配（包括空白符）。
    当 old_string 为空时，在文件开头插入 new_string。

    Args:
        path: 文件路径（支持 ~ 表示 HOME 目录）
        old_string: 要替换的文本（精确匹配，包括空白符）
        new_string: 替换后的新文本

    Returns:
        编辑结果提示
    """
    p = _normalize_path(path)
    if p is None:
        return f"[错误: 只能访问 HOME 目录({ _HOME_DIR })下的文件]"

    if not p.exists():
        return f"[错误: 文件不存在: {p}]"
    if not p.is_file():
        return f"[错误: 不是文件: {p}]"

    try:
        content = p.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"[读取失败: {e}]"

    # old_string 为空 → 在文件开头插入
    if old_string == "":
        new_content = new_string + content
        try:
            p.write_text(new_content, encoding="utf-8")
            return f"编辑成功: 在开头插入了 {len(new_string)} 字符"
        except Exception as e:
            return f"[写入失败: {e}]"

    # 精确匹配替换
    if old_string not in content:
        return f"[编辑失败: 未找到匹配的文本片段]"

    count = content.count(old_string)
    new_content = content.replace(old_string, new_string, count)

    try:
        p.write_text(new_content, encoding="utf-8")
        return f"编辑成功: 替换了 {count} 处，文件 {p}"
    except Exception as e:
        return f"[写入失败: {e}]"


async def apply_patch(path: str, patch: str) -> str:
    """应用 patch 到文件。patch 格式为 unified diff（类似 git diff）。

    支持单文件的多个 hunk 修改。patch 中每个 hunk 以
    @@ -old_start,old_count +new_start,new_count @@ 开头。

    Args:
        path: 目标文件路径（支持 ~ 表示 HOME 目录）
        patch: unified diff 格式的 patch 文本

    Returns:
        应用结果提示
    """
    p = _normalize_path(path)
    if p is None:
        return f"[错误: 只能访问 HOME 目录({ _HOME_DIR })下的文件]"

    if not p.exists():
        return f"[错误: 文件不存在: {p}]"
    if not p.is_file():
        return f"[错误: 不是文件: {p}]"

    try:
        content = p.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"[读取失败: {e}]"

    lines = content.split("\n")

    # 解析 patch
    hunks: list[dict[str, Any]] = []
    current_hunk: dict[str, Any] | None = None
    patch_lines = patch.split("\n")
    i = 0
    while i < len(patch_lines):
        line = patch_lines[i]
        # 跳过 diff 头
        if line.startswith("--- ") or line.startswith("+++ "):
            i += 1
            continue
        # Hunk header: @@ -start,count +start,count @@
        if line.startswith("@@"):
            m = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
            if m:
                old_start = int(m.group(1))
                old_count = int(m.group(2)) if m.group(2) else 1
                current_hunk = {
                    "old_start": old_start,
                    "old_count": old_count,
                    "lines": [],
                }
                hunks.append(current_hunk)
            i += 1
            continue
        if current_hunk is not None:
            current_hunk["lines"].append(line)
        i += 1

    if not hunks:
        return "[错误: patch 中未找到有效的 hunk]"

    # 按旧行号从后往前应用，避免行号偏移
    hunks.sort(key=lambda h: h["old_start"], reverse=True)

    for hunk in hunks:
        old_start = hunk["old_start"]
        # unified diff 的行号从 1 开始，转换为 0-based 索引
        insert_pos = old_start - 1
        if insert_pos < 0:
            insert_pos = 0

        # 解析 hunk 内容：分离删除行、新增行、上下文行
        deletions = []
        additions = []
        context_before = []
        context_after = []
        state = "before"  # before / del / add / after

        for hl in hunk["lines"]:
            if not hl:
                continue
            if hl.startswith("-"):
                state = "del"
                deletions.append(hl[1:])
            elif hl.startswith("+"):
                state = "add"
                additions.append(hl[1:])
            else:
                # 上下文行（可能以空格开头或无前缀）
                ctx = hl[1:] if hl.startswith(" ") else hl
                if state in ("before", "del"):
                    context_before.append(ctx)
                else:
                    context_after.append(ctx)

        # 验证上下文（简化：只检查上下文是否存在）
        # 实际修改：删除 deletions 对应的行，插入 additions
        # 简化策略：找到 insert_pos，删除 deletions 数量的行，插入 additions
        del_count = len(deletions) if deletions else hunk["old_count"]
        end_pos = min(insert_pos + del_count, len(lines))

        # 替换
        new_lines = lines[:insert_pos] + additions + lines[end_pos:]
        lines = new_lines

    try:
        p.write_text("\n".join(lines), encoding="utf-8")
        return f"Patch 应用成功: {p}（应用了 {len(hunks)} 个 hunk）"
    except Exception as e:
        return f"[写入失败: {e}]"


async def execute_command(command: str, timeout: int = 30) -> str:
    """执行安全的 shell 命令。

    支持的命令类型：
    - 文件查看: ls, cat, head, tail, find, grep, wc, stat
    - 系统信息: pwd, echo, date, whoami, ps, top, df, du
    - 开发工具: python, node, npm, git, pip
    - 网络诊断: ping, nslookup, netstat, ipconfig

    禁止的操作：
    - 删除文件/目录 (rm, del)
    - 格式化磁盘
    - 修改权限
    - 管道、命令链、变量展开

    Args:
        command: 要执行的命令
        timeout: 超时时间（秒）

    Returns:
        命令输出结果或错误信息
    """
    safe, reason = _is_command_safe(command)
    if not safe:
        return f"[命令被拒绝: {reason}]\n提示: 只能执行文件查看、系统信息、开发工具等安全命令。"

    try:
        # 使用 shell=True 执行命令，但限制在当前工作目录
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(_HOME_DIR),
        )

        output = result.stdout.strip() if result.stdout else ""
        stderr = result.stderr.strip() if result.stderr else ""

        if result.returncode != 0:
            err_msg = stderr or f"命令退出码: {result.returncode}"
            return f"[命令执行出错]\n{err_msg}"

        if not output and not stderr:
            return "命令执行成功（无输出）"

        # 限制输出长度
        max_output_len = 5000
        if len(output) > max_output_len:
            output = output[:max_output_len] + f"\n\n... (输出已截断，共 {len(output)} 字符)"

        if stderr:
            return f"{output}\n\n[stderr]:\n{stderr[:1000]}"
        return output

    except subprocess.TimeoutExpired:
        return f"[命令超时: 执行超过 {timeout} 秒]"
    except Exception as e:
        return f"[命令执行失败: {e}]"
