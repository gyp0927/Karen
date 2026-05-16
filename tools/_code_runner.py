"""Code execution runner — invoked by code_executor.py via subprocess.

Reads Python code from stdin, executes in a restricted environment,
and prints a JSON result dict as the last line of stdout.
"""

import builtins
import io
import json
import sys
import traceback
import types

# Restricted builtins — 移除危险函数及所有下划线开头属性（防遍历绕过）
_SAFE_BUILTINS = dict(builtins.__dict__)
for _fname in (
    "open",
    "input",
    "exec",
    "eval",
    "compile",
    "__import__",
    "__build_class__",
    "globals",
    "locals",
    "vars",
    "memoryview",
    "getattr",
    "setattr",
    "delattr",
    "breakpoint",
    "help",
    "exit",
    "quit",
    "attrgetter",
    "itemgetter",
    "methodcaller",
):
    _SAFE_BUILTINS.pop(_fname, None)

# 移除所有下划线开头的属性，防止通过 __builtins__ 遍历拿到危险对象
for _fname in list(_SAFE_BUILTINS.keys()):
    if _fname.startswith("_"):
        _SAFE_BUILTINS.pop(_fname, None)

_FROZEN_BUILTINS = types.MappingProxyType(_SAFE_BUILTINS)
_SAFE_GLOBALS = {
    "__builtins__": _FROZEN_BUILTINS,
    "__name__": "__main__",
}


def main() -> None:
    code = sys.stdin.read()

    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()

    try:
        import contextlib

        with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
            exec(code, _SAFE_GLOBALS)
        result = {
            "success": True,
            "stdout": stdout_buf.getvalue(),
            "stderr": stderr_buf.getvalue(),
            "error": None,
            "traceback": None,
        }
    except Exception as e:
        result = {
            "success": False,
            "stdout": stdout_buf.getvalue(),
            "stderr": stderr_buf.getvalue(),
            "error": str(e),
            "traceback": traceback.format_exc(),
        }

    # JSON result is the final line of stdout so the parent can easily
    # locate it even if user code printed unrelated text.
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
