"""Python 代码执行沙箱 - 安全运行 Agent 生成的代码。"""

import ast
import json
import logging
import os
import subprocess
import sys
import time
import traceback
from typing import Optional

logger = logging.getLogger(__name__)

# 禁止导入的危险模块
_FORBIDDEN_MODULES = {
    "os", "sys", "subprocess", "importlib", "ctypes", "socket",
    "urllib", "http", "ftplib", "smtplib", "pickle", "marshal",
    "compileall", "py_compile", "bdb", "pdb", "trace",
    "shutil", "pathlib", "tempfile", "multiprocessing",
    "builtins",  # 防 import builtins 拿 __import__
    "io",        # 可通过 io.open 获取真实 builtins
    "operator",  # 可通过 operator.attrgetter 逃逸
    "inspect",   # 可内省 frame、获取 globals
    "types",     # 可构造新类型、获取 FrameType
    "warnings",  # 可触发 warnings.showwarning 写文件
    "code",      # 可动态编译代码
    "codeop",    # 编译辅助
}

# 禁止在代码中使用的危险函数/方法名（全局禁止 + 模块级禁止）
_FORBIDDEN_CALLS_GLOBAL = {
    "eval", "exec", "compile", "open", "input", "__import__",
    "system", "popen", "call", "run", "exec_",
    "getattr", "setattr", "delattr",  # 动态属性
    "globals", "locals", "vars",       # 命名空间内省
    "attrgetter", "itemgetter", "methodcaller",  # operator 逃逸链
}
# 以下函数名被导入到当前作用域时也禁止（如 from importlib import import_module）
_FORBIDDEN_CALLS_ALIASED = {
    "import_module", "find_loader", "spec_from_file_location",  # importlib
}

# 禁止访问的 dunder 属性 — 经典逃逸链 (().__class__.__base__.__subclasses__())
_FORBIDDEN_ATTRS = {
    "__class__", "__bases__", "__base__", "__subclasses__",
    "__mro__", "__globals__", "__builtins__", "__import__",
    "__getattribute__", "__dict__", "__init_subclass__",
    "__loader__", "__spec__", "__code__", "__closure__",
    "f_globals", "f_locals", "f_back",  # frame inspection
    "func_globals", "gi_frame",
}

# 允许使用的安全模块白名单（如果启用白名单模式）
_ALLOWED_MODULES = {
    "math", "random", "statistics", "datetime", "decimal", "fractions",
    "itertools", "collections", "functools", "heapq", "bisect",
    "json", "re", "string", "textwrap", "hashlib", "base64",
    "typing", "copy", "pprint", "numbers", "abc",
    "numpy", "pandas",
}


class SecurityError(Exception):
    """代码安全检查失败"""
    pass


def _check_ast(code: str) -> bool:
    """通过 AST 检查代码安全性。"""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise SecurityError(f"语法错误: {e}")

    for node in ast.walk(tree):
        # 禁止导入
        if isinstance(node, ast.Import):
            for alias in node.names:
                mod_name = alias.name.split(".")[0]
                if mod_name in _FORBIDDEN_MODULES:
                    raise SecurityError(f"禁止导入模块: {mod_name}")
        elif isinstance(node, ast.ImportFrom):
            mod_name = node.module.split(".")[0] if node.module else ""
            if mod_name in _FORBIDDEN_MODULES:
                raise SecurityError(f"禁止导入模块: {mod_name}")

        # 禁止危险的函数调用
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                fname = node.func.id
                if fname in _FORBIDDEN_CALLS_GLOBAL or fname in _FORBIDDEN_CALLS_ALIASED:
                    raise SecurityError(f"禁止调用函数: {fname}")
            elif isinstance(node.func, ast.Attribute):
                if node.func.attr in _FORBIDDEN_CALLS_GLOBAL:
                    raise SecurityError(f"禁止调用方法: {node.func.attr}")
                # 检查模块级危险调用（如 importlib.import_module）
                if isinstance(node.func.value, ast.Name):
                    if node.func.value.id == "importlib" and node.func.attr in _FORBIDDEN_CALLS_ALIASED:
                        raise SecurityError(f"禁止调用: importlib.{node.func.attr}")

        # 禁止经典逃逸链的 dunder 属性访问 (().__class__.__base__.__subclasses__())
        if isinstance(node, ast.Attribute):
            if node.attr in _FORBIDDEN_ATTRS:
                raise SecurityError(f"禁止访问 dunder 属性: .{node.attr}")

        # 禁止 lambda（可绕过函数名检查构造逃逸链）
        if isinstance(node, ast.Lambda):
            raise SecurityError("禁止 lambda 表达式")

        # 禁止推导式（内部表达式可构造逃逸链，且难以逐层审查）
        if isinstance(node, (ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp)):
            raise SecurityError("禁止列表/字典/集合推导式与生成器表达式")

        # 禁止通过 Subscript 访问危险属性（如 __builtins__['__import__']）
        if isinstance(node, ast.Subscript):
            if isinstance(node.value, ast.Name):
                if node.value.id in ("__builtins__", "builtins", "__import__"):
                    raise SecurityError(f"禁止通过下标访问: {node.value.id}[...]")
            if isinstance(node.value, ast.Attribute):
                if node.value.attr in _FORBIDDEN_ATTRS:
                    raise SecurityError(f"禁止通过下标访问 dunder 属性: .{node.value.attr}[...]")

    return True


def execute_python(code: str, timeout: int = 30) -> dict:
    """安全执行 Python 代码。

    参数:
        code: Python 代码字符串
        timeout: 超时时间（秒）

    返回:
        {
            "success": bool,
            "stdout": str,
            "stderr": str,
            "error": str | None,
            "traceback": str | None,
            "duration_ms": int,
        }
    """
    # 步骤1: AST 安全检查
    try:
        _check_ast(code)
    except SecurityError as e:
        return {
            "success": False,
            "stdout": "",
            "stderr": str(e),
            "error": str(e),
            "traceback": "",
            "duration_ms": 0,
        }

    # 步骤2: 在子进程中执行（超时保护）
    runner_path = os.path.join(os.path.dirname(__file__), "_code_runner.py")
    start_time = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, runner_path],
            input=code,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        duration_ms = int((time.time() - start_time) * 1000)

        stdout_text = proc.stdout
        lines = stdout_text.splitlines()
        result: dict
        if lines:
            try:
                result = json.loads(lines[-1])
            except json.JSONDecodeError:
                result = {
                    "success": proc.returncode == 0,
                    "stdout": stdout_text,
                    "stderr": proc.stderr,
                    "error": None if proc.returncode == 0 else f"退出码: {proc.returncode}",
                    "traceback": None,
                }
        else:
            result = {
                "success": proc.returncode == 0,
                "stdout": "",
                "stderr": proc.stderr,
                "error": None if proc.returncode == 0 else f"退出码: {proc.returncode}",
                "traceback": None,
            }
        result["duration_ms"] = duration_ms
        return result

    except subprocess.TimeoutExpired:
        duration_ms = int((time.time() - start_time) * 1000)
        return {
            "success": False,
            "stdout": "",
            "stderr": "",
            "error": f"代码执行超时（>{timeout}秒）",
            "traceback": "",
            "duration_ms": duration_ms,
        }
    except Exception as e:
        logger.exception("Code execution failed")
        return {
            "success": False,
            "stdout": "",
            "stderr": str(e),
            "error": str(e),
            "traceback": traceback.format_exc(),
            "duration_ms": 0,
        }


def format_result(result: dict) -> str:
    """将执行结果格式化为易读的文本。"""
    lines = []
    if result["success"]:
        lines.append("执行成功 ✅")
    else:
        lines.append("执行失败 ❌")
    if result["stdout"]:
        lines.append("\n[标准输出]")
        lines.append(result["stdout"])
    if result["stderr"]:
        lines.append("\n[标准错误]")
        lines.append(result["stderr"])
    if result["error"]:
        lines.append(f"\n[错误] {result['error']}")
    if result.get("traceback"):
        lines.append("\n[堆栈跟踪]")
        lines.append(result["traceback"])
    lines.append(f"\n耗时: {result.get('duration_ms', 0)}ms")
    return "\n".join(lines)
