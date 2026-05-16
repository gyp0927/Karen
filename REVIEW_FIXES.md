# karen-ai 代码审查修复指令

## 合并来源
- Hermes电脑版: 15 个问题
- Hermes微信版(Karen): 10 个问题
- 去重后合并为以下 18 个修复项

---

## 【P0 严重—立即修复】

### P0-1: /api/execute 未受 LOCAL_ONLY 保护 (电脑版#1)
**文件**: `web/app.py` 第256-275行 (`handle_execute_code` 函数)
**问题**: 该端点仅受 `@auth_required` 保护，不在 `LOCAL_ONLY_PREFIXES` 列表中。任何拥有有效 API Key 的远程用户可 POST 任意 Python 代码并执行。
**修复**:
1. 将 `/api/execute` 加入 `LOCAL_ONLY_PREFIXES` 列表（第83行）
2. 或在 `handle_execute_code` 函数内增加 IP 白名单校验

### P0-2: LLM 自动触发代码执行 (电脑版#2)
**文件**: `cognition/tool_engine.py:86-111`, `agents/nodes.py:543`, `agents/tools.py:77`
**问题**: 当 `ENABLE_CODE_EXECUTION=true` 时，LLM 可通过 function calling 自动触发 `execute_python`，无需用户确认。
**修复**:
1. 在 `tool_engine.py` 中增加用户确认步骤：代码执行前需要用户点击"确认执行"
2. 或在 `agents/tools.py` 的 `_need_tool_call()` 中，对 `execute_python` 工具增加额外限制

### P0-3: `_code_runner.py` 移除 `type` 和 `object` 导致库初始化失败 (电脑版#3)
**文件**: `tools/_code_runner.py` 第27-28行
**问题**: `_SAFE_BUILTINS.pop("type")` 和 `pop("object")` 会导致 numpy/pandas 等被允许的库初始化失败。
**修复**:
```python
# 不要移除 type 和 object，它们是 Python 运行时基础
for _fname in (
    "open", "input", "exec", "eval", "compile", "__import__",
    "__build_class__", "globals", "locals", "vars",
    # "type",      # 保留
    # "object",    # 保留
    "memoryview", "getattr", "setattr", "delattr",
    "breakpoint", "help", "exit", "quit",
    "attrgetter", "itemgetter", "methodcaller",
):
```

### P0-4: Semaphore 跨 loop 死锁 (微信版#1)
**文件**: `agents/nodes.py` 第36-53行 (`_get_llm_sem`)
**问题**: `id(loop)` 在 loop 被 GC 后可能被重用，新 loop 拿到旧 Semaphore（可能已被占满），导致协程永远卡住。
**修复**:
```python
import weakref
# 用 WeakKeyDictionary 替代 dict，loop 被 GC 时自动清除
_llm_semaphores: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()

def _get_llm_sem() -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    sem = _llm_semaphores.get(loop)
    if sem is None:
        sem = asyncio.Semaphore(_LLM_SEM_LIMIT)
        _llm_semaphores[loop] = sem
    return sem
```

### P0-5: `web/app.py` 模块级代码位置错误 (微信版#2)
**文件**: `web/app.py` 第1241-1246行
**问题**: `from web.api import api_bp`、`app.register_blueprint(api_bp)`、`init_agents()` 在 `if __name__ == "__main__":` 之后。导致循环导入风险和 `init_agents()` 重复调用。
**修复**:
1. 将 `from web.api import api_bp` 和 `app.register_blueprint(api_bp)` 移到第67行 `app = Flask(__name__)` 之后
2. 删除第1246行的 `init_agents()`（保留 `__main__` 里的那个）

### P0-6: SocketIO debug + threading 冲突 (微信版#4)
**文件**: `web/app.py` 第1235-1238行
**问题**: `socketio.run(app, ..., debug=debug_mode)` 当 `FLASK_DEBUG=true` 时，Flask reloader 创建双进程，SocketIO 连接异常。
**修复**:
```python
# 强制禁用 reloader，避免双进程问题
socketio.run(app, host=bind_host, port=bind_port, debug=debug_mode, use_reloader=False)
```

---

## 【P1 高危】

### P1-7: CSRF 风险 (电脑版#6)
**文件**: `core/auth.py` 第278-298行 (`auth_required`)
**问题**: Cookie 认证时无 CSRF 防护。
**修复**:
1. 对修改性操作（POST/PUT/DELETE）强制使用 Header 认证
2. 或添加 CSRF Token 机制

### P1-8: API Key 存储使用 SHA256 无盐哈希 (电脑版#7)
**文件**: `core/auth.py` 第143-146行
**问题**: SHA256 无盐哈希安全性不足，泄露后可被暴力破解。
**修复**: 使用 `bcrypt` 或 `argon2-cffi` 替代 SHA256。如果不想引入新依赖，可使用 `hashlib.pbkdf2_hmac`。

### P1-9: 前端 XSS 风险 (电脑版#5)
**文件**: `web/static/script.js` (多处 innerHTML)
**问题**: `escapeHtml()` 只转义 `< > & "`，未转义 `'` 和 `/`。
**修复**:
```javascript
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
// 或者优先使用 textContent 替代 innerHTML
```

### P1-10: 全局状态竞争/内存泄漏 (微信版#5 + 电脑版#8)
**文件**: `state/stop_flag.py`、`tools/search.py`、`web/state.py`
**问题**: 多个模块级全局 dict 无 TTL。
**修复**:
1. `tools/search.py` 的 `_search_cache` 添加 TTL 清理
2. `web/state.py` 的 `socket_states` 和 `socket_configs` 确保 `cleanup_socket()` 在 disconnect 时完全清除

### P1-11: Skeleton/Stub 代码 (微信版#6)
**文件**: `core/mcp_manager.py` 第57、62、267行；`core/vector_store.py` 第135、138行
**问题**: 多处 `return {}` / `return []`，可能是 AI 生成的占位实现。
**修复**: 检查这些函数是否应该抛 `NotImplementedError`，或补全实现。

### P1-12: 安全 HTTP 响应头缺失 (电脑版#10)
**文件**: `web/app.py`
**问题**: 缺少 Content-Security-Policy、X-Frame-Options、X-Content-Type-Options 等。
**修复**:
```python
@app.after_request
def add_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['Content-Security-Policy'] = "default-src 'self'"
    return response
```

### P1-13: `graph/orchestrator.py` 无限循环风险 (电脑版#9)
**文件**: `graph/orchestrator.py` 第40-103行 (`create_multi_agent_graph`)
**问题**: reviewer->responder->reviewer 无限循环，虽标记 DEPRECATED 但仍可被调用。
**修复**: 添加硬编码的最大迭代次数限制（如 10 次），超出后强制退出循环。

### P1-14: `web/app.py` 结构臃肿 (微信版#8)
**文件**: `web/app.py` 1246行
**问题**: SocketIO 事件处理器全在主文件中。
**修复**: 将 `@socketio.on` 处理器拆分到 `web/socket_handlers.py`。

---

## 【P2 中危】

### P2-15: SECRET_KEY 非持久化 (电脑版#15)
**文件**: `web/app.py` 第68行
**问题**: 未设置 `FLASK_SECRET_KEY` 时每次重启生成新随机密钥，session 失效。
**修复**:
```python
import os, secrets
secret_path = os.path.join(os.path.dirname(__file__), '..', 'data', '.secret_key')
os.makedirs(os.path.dirname(secret_path), exist_ok=True)
if os.path.exists(secret_path):
    with open(secret_path, 'r') as f:
        secret_key = f.read().strip()
else:
    secret_key = secrets.token_hex(32)
    with open(secret_path, 'w') as f:
        f.write(secret_key)
app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", secret_key)
```

### P2-16: 依赖版本未锁定 (电脑版#14)
**文件**: `requirements.txt`
**问题**: 使用 `>=` 范围，自动升级可能引入漏洞。
**修复**: `pip freeze > requirements.txt` 锁定精确版本。

### P2-17: CDN 资源缺少 SRI 校验 (电脑版#12)
**文件**: `web/templates/index.html` 第230-234行
**问题**: 从 CDN 引入 socket.io、marked、DOMPurify、highlight.js，无 integrity 属性。
**修复**: 为每个 CDN 资源添加 `integrity` 和 `crossorigin="anonymous"`。

### P2-18: 缺少测试覆盖 (微信版#7 + 电脑版#13)
**文件**: 整个项目
**问题**: 0 个测试文件。
**修复**: 至少添加核心模块的单元测试（`test_code_executor.py`、`test_stop_flag.py`、`test_auth.py`）。

---

## 执行方式

1. 按上述顺序修复，每修完一个运行测试确认
2. 所有修复完成后执行 `python test_all.py`
3. 测试通过后 `git add -A && git commit -m "代码审查修复批次3: 安全+结构+质量问题修复" && git push karen fix-root-code:main`
4. 向用户汇报修复结果
