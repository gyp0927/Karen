# Phase 1 执行清单（交给 Claude 实现）

> 来源：Karen 的代码审查 + 开源路线图
> 优先级：P0 → P1 → P2 → 工程规范
> 验收：每页完成后运行 `python test_all.py`，全部通过后按 CLAUDE.md 流程 git push

---

## 页 1：P0 修复（必须先做）

### Task 1.1：默认管理员密码随机化
**[影响程度] Critical — 安全漏洞**

- **位置：** `core/auth.py`，第 58 行
- **现状：** `_DEFAULT_ADMIN_PASSWORD = os.getenv("DEFAULT_ADMIN_PASSWORD", "admin123")`
- **修改要求：**
  1. 如果环境变量 `DEFAULT_ADMIN_PASSWORD` 未设置，启动时生成随机密码（比如 `secrets.token_urlsafe(16)`）
  2. 把随机密码打印到控制台，格式：`[INIT] 管理员密码已随机生成: xxxx`
  3. 同时把这个密码写入 `data/.admin_password`文件（如果文件已存在则直接读取，不重新生成）
  4. 下次启动时检测该文件，避免每次重启都变密码
- **验收标准：**
  ```bash
  rm -f data/.admin_password
  python -c "from core.auth import init_db; init_db()"
  # 控制台应打印随机密码
  cat data/.admin_password
  # 再次运行，密码应该与上次相同
  ```

---

### Task 1.2：FLASK_SECRET_KEY 持久化
**[影响程度] Critical — 服务重启后 session 全失**

- **位置：** `web/app.py`，第 78 行
- **现状：** `app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", secrets.token_hex(32))`
- **修改要求：**
  1. 检查环境变量 `FLASK_SECRET_KEY`
  2. 如果未设置，检查 `data/.secret_key` 文件
  3. 如果文件不存在，生成随机 key 并写入文件（64 位十六进制）
  4. 下次启动直接从文件读取
- **验收标准：**
  ```bash
  rm -f data/.secret_key
  python web/app.py &
  sleep 2 && kill %1
  cat data/.secret_key  # 应该有 64 位字符
  # 再次启动，密钥应不变
  ```

---

### Task 1.3：代码执行沙箱加固
**[影响程度] Critical — 静态 AST 无法防御所有运行时逃逸**

- **位置：** `tools/_code_runner.py` + `tools/code_executor.py`
- **现状：** 子进程用 `exec(code, _SAFE_GLOBALS)`，但仍运行在同一容器/操作系统中
- **修改要求：**
  1. 在 `code_executor.py` 的 `subprocess.run()` 中加入 resource limit：
     - `resource.setrlimit(resource.RLIMIT_CPU, (timeout, timeout))` 限制 CPU
     - `resource.setrlimit(resource.RLIMIT_AS, (512*1024*1024, 512*1024*1024))` 限制内存 512MB
     - `resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))` 限制文件句柄
  2. 将 `_code_runner.py` 作为独立进程启动，并传入上述 resource limit
  3. 如果不能用 resource 模块（Windows 兼容），至少加一个检测：非 Linux 平台时打 warning 日志 `"[WARN] 代码执行在非 Linux 平台上运行，沙箱隔离有限"`
- **验收标准：**
  ```bash
  ENABLE_CODE_EXECUTION=true python -c "
  from tools.code_executor import execute_python
  # 测试超内存代码
  r = execute_python('a=[1]*10**9')
  print(r)
  # 应该因超内存而失败或被杀
  "
  ```

---

## 页 2：P1 架构修复

### Task 2.1：Socket 状态字典加硬上限
**[影响程度] High — 内存泄漏风险**

- **位置：** `web/state.py`，第 62-67 行
- **修改要求：**
  1. `socket_states` 和 `socket_configs` 添加 `max_size = 5000`
  2. 超出时使用 LRU 淘汰：删除最久未活跃的 sid
  3. 在 `get_socket_state()` 中检查当前大小，超限时触发一次扫描清理
  4. 添加日志：`logger.warning(f"Socket states 达到上限 {max_size}，淘汰最旧的 sid={oldest_sid}")`
- **验收标准：**
  ```python
  # 单元测试：模拟创建 5001 个 socket 状态
  for i in range(5001):
      s = get_socket_state(f"test-{i}")
      s.touch()
  assert len(socket_states) <= 5000
  ```

---

### Task 2.2：httpx client 缓存安全校验
**[影响程度] Medium — 线程 ID 重用可能导致错误 client**

- **位置：** `agents/llm.py`，第 20-58 行
- **修改要求：**
  1. 将 key 从 `(thread_id, loop_id)` 改为 `(thread_id, loop_id, id(client))`
  2. 或更简单：获取 client 时检查 `client._loop` 是否是当前运行中的 loop，不是就是重建
  3. 在 `_get_http_async_client()` 中，如果从缓存拿到 client，验证 `client._loop == asyncio.get_running_loop()`
- **验收标准：**
  ```python
  # 模拟跨线程重用
  import asyncio
  from agents.llm import _get_http_async_client
  
  async def check():
      c1 = _get_http_async_client()
      # 在另一个线程的新 loop 中获取
      def other_thread():
          new_loop = asyncio.new_event_loop()
          return new_loop.run_until_complete(_get_http_async_client())
      import threading
      t = threading.Thread(target=lambda: other_loop_client := other_thread())
      t.start(); t.join()
      # c1 和 other_loop_client 应该是不同对象
  ```

---

### Task 2.3：删除废弃的多 Agent 图
**[影响程度] Medium — reviewer 无限循环风险**

- **位置：** `graph/orchestrator.py`，第 40-83 行
- **修改要求：**
  1. 直接删除 `create_multi_agent_graph` 函数
  2. 检查项目中所有导入该函数的地方（`test_all.py`、`main.py` 等）
  3. `test_all.py` 中对应测试改为测试 `create_coordination_graph`
  4. `main.py` 如果还在用，改为调用 `create_coordination_graph`
- **验收标准：**
  ```bash
  grep -rn "create_multi_agent_graph" agents/ graph/ web/ core/ interface/ main.py test_all.py
  # 应该没有任何匹配
  ```

---

### Task 2.4：SQLite 连接添加 TTL
**[影响程度] High — 线程池场景下连接句柄泄漏**

- **位置：** `core/db_utils.py`
- **现状：** `get_sqlite_conn()` 使用 `use_thread_local=True` 缓存连接，但不会关闭
- **修改要求：**
  1. 在 `get_sqlite_conn()` 中，记录每个线程本地连接的创建时间
  2. 如果连接已存在且超过 5 分钟未使用，先关闭旧连接再创建新连接
  3. 或者在 `put_sqlite_conn()` 方法中添加 TTL 检查
  4. 如果 `db_utils.py` 没有 `put_sqlite_conn`，可以在 `get_sqlite_conn` 入口处检查并重置
- **验收标准：**
  ```python
  import threading, time
  from core.db_utils import get_sqlite_conn
  
  conn1 = get_sqlite_conn("data/test_ttl.db", use_thread_local=True)
  # 模拟一个新线程获取同一个路径
  def get_again():
      conn2 = get_sqlite_conn("data/test_ttl.db", use_thread_local=True)
      print(conn1 is conn2)  # 同线程应该是 True
  get_again()
  # 清理
  import os; os.remove("data/test_ttl.db")
  ```

---

## 页 3：P2 质量修复

### Task 3.1：搜索超时调整
**[影响程度] Low — 快速模式下网络搜索几乎总是超时**

- **位置：** `agents/search.py`，第 12 行
- **修改要求：**
  1. `WEB_SEARCH_TIMEOUT_FAST_S` 从 1.5 秒改为 3 秒
  2. 在 `_safe_search` 超时时，如果是快速模式且超时，记录 `logger.info(f"{label} 快速模式超时跳过（{timeout}s）")`
- **验收标准：**
  ```python
  from agents.search import WEB_SEARCH_TIMEOUT_FAST_S
  assert WEB_SEARCH_TIMEOUT_FAST_S >= 3.0
  ```

---

### Task 3.2：废弃自定义测试框架，迁移到 pytest
**[影响程度] Medium — 行业标准工具**

- **位置：** `test_all.py`
- **修改要求：**
  1. 保留 `test_all.py` 作为简单的自检脚本（向后兼容）
  2. 新建 `tests/` 目录：`tests/test_graph.py`、`tests/test_agents.py`、`tests/test_auth.py`
  3. 把 `test_all.py` 中的测试例逐一迁移，转成 `def test_xxx():` 形式
  4. 使用 `pytest-asyncio` 处理异步测试
  5. 添加 `pytest.ini`：
     ```ini
     [pytest]
     asyncio_mode = auto
     testpaths = tests
     ```
- **验收标准：**
  ```bash
  pip install pytest pytest-asyncio
  pytest tests/ -v
  # 全部通过
  ```

---

### Task 3.3：补充测试（至少 3 个新测试文件）
**[影响程度] High — 安全相关逻辑必须覆盖**

- **要求：**
  1. `tests/test_local_only.py` — 测试 LOCAL_ONLY 中间件
     - 从 `127.0.0.1` 访问 `/api/config` 应返回 200
     - 从 `192.168.1.1` 访问 `/api/config` 应返回 403
  2. `tests/test_intent.py` — 测试意图分类边界 case
     - "你好" → skip_search=True
     - "什么是量子计算" → skip_search=False
     - "Python" 不含编码动作 → 非编码意图
  3. `tests/test_message_order.py` — 测试消息归一化
     - 传入 [SystemA, SystemB, Human, SystemC] → 输出 [SystemA, SystemB, SystemC, Human]
- **验收标准：**
  ```bash
  pytest tests/test_local_only.py tests/test_intent.py tests/test_message_order.py -v
  # 全部通过
  ```

---

## 页 4：工程规范

### Task 4.1：引入 ruff + mypy
**[影响程度] Medium — 面试官看代码的第一眼**

- **要求：**
  1. 创建 `pyproject.toml`：
     ```toml
     [tool.ruff]
     line-length = 120
     target-version = "py311"
     
     [tool.ruff.lint]
     select = ["E", "F", "I", "W", "UP"]
     ignore = ["E501"]
     
     [tool.mypy]
     python_version = "3.11"
     warn_return_any = true
     warn_unused_ignores = true
     ignore_missing_imports = true
     ```
  2. 安装：`pip install ruff mypy`
  3. 对 `agents/`、`graph/`、`core/`、`web/` 运行 ruff 格式化：`ruff format agents/ graph/ core/ web/`
  4. 对同上运行 ruff lint：`ruff check agents/ graph/ core/ web/`
  5. 对同上运行 mypy：`mypy agents/ graph/ core/ web/`
  6. 修复所有报错，无法修的加 `# type: ignore` 注释
- **验收标准：**
  ```bash
  ruff check agents/ graph/ core/ web/ tests/  # 零报错
  ruff format --check agents/ graph/ core/ web/ tests/  # 格式正确
  mypy agents/ graph/ core/ web/  # 零报错
  ```

---

### Task 4.2：GitHub Actions CI
**[影响程度] Medium — 开源项目的标准配置**

- **要求：**
  1. 创建 `.github/workflows/ci.yml`：
     ```yaml
     name: CI
     on: [push, pull_request]
     jobs:
       test:
         runs-on: ubuntu-latest
         steps:
           - uses: actions/checkout@v4
           - uses: actions/setup-python@v5
             with:
               python-version: '3.11'
           - run: pip install -r requirements.txt
           - run: pip install ruff mypy pytest pytest-asyncio
           - run: ruff check agents/ graph/ core/ web/ tests/
           - run: mypy agents/ graph/ core/ web/
           - run: pytest tests/ -v
     ```
  2. 推送到 GitHub，确保第一次 CI run 是绿的
- **验收标准：**
  - GitHub 仓库的 Actions 页面显示绿勾勾

---

### Task 4.3：清理 git 历史
**[影响程度] Low — 但影响第一印象**

- **要求：**
  1. 将 `.db`、`.pyc`、`__pycache__/` 从 git 历史中彻底删除
  2. 完善 `.gitignore`：
     ```
     data/*.db
     data/*.db-journal
     data/.secret_key
     data/.admin_password
     __pycache__/
     *.pyc
     *.pyo
     *.egg-info/
     .app.lock
     *.spec
     .claude/
     ```
  3. 如果 git 历史已污染，用 BFG Repo-Cleaner 或 git filter-repo 清理
- **验收标准：**
  ```bash
  git log --all --full-history -- data/cache.db
  # 应该没有输出（已彻底删除）
  ```

---

### Task 4.4：依赖版本上限
**[影响程度] Low**

- **要求：**
  1. 给 `requirements.txt` 每个包加上限：
     ```
     flask>=3.0.0,<4.0.0
     flask-socketio>=5.3.0,<6.0.0
     langchain-openai>=0.2.0,<0.3.0
     ...
     ```
  2. 或者更好：创建 `pyproject.toml`，用 `[project.dependencies]` 管理
- **验收标准：**
  ```bash
  pip install -r requirements.txt
  # 无报错无警告
  ```

---

## 页 5：最终验收

全部完成后，运行：

```bash
# 1. 清理并重新安装
rm -rf __pycache__ data/*.db data/.secret_key data/.admin_password
pip install -r requirements.txt
pip install ruff mypy pytest pytest-asyncio

# 2. 启动服务验证
python web/app.py &
APP_PID=$!
sleep 3
curl -s http://127.0.0.1:5000/api/health || echo "未提供 health endpoint，请确保服务启动成功"
kill $APP_PID

# 3. 质量检查
ruff check agents/ graph/ core/ web/ tests/  # 必须零报错
mypy agents/ graph/ core/ web/               # 必须零报错
pytest tests/ -v                              # 必须全部通过

# 4. 按 CLAUDE.md 流程推送
python test_all.py   # 保持向后兼容
git add -A
git commit -m "Phase 1: 代码质量与安全固化
details:
- 随机化默认管理员密码
- FLASK_SECRET_KEY 持久化
- 代码执行沙箱加固 resource limit
- socket 状态字典加硬上限
- 删除废弃的 create_multi_agent_graph
- SQLite 连接加 TTL
- 搜索超时调整
- 迁移到 pytest + 补充测试
- 引入 ruff + mypy
- 添加 GitHub Actions CI
- 清理 git 历史
- 依赖版本上限"
git push karen fix-root-code:main
```

---

## 附录：修改检查清单

每个 Task 完成后在这里打勾：

- [ ] Task 1.1 管理员密码随机化
- [ ] Task 1.2 FLASK_SECRET_KEY 持久化
- [ ] Task 1.3 代码沙箱加固
- [ ] Task 2.1 socket 状态上限
- [ ] Task 2.2 httpx client 安全校验
- [ ] Task 2.3 删除废弃图
- [ ] Task 2.4 SQLite TTL
- [ ] Task 3.1 搜索超时调整
- [ ] Task 3.2 迁移到 pytest
- [ ] Task 3.3 补充测试
- [ ] Task 4.1 ruff + mypy
- [ ] Task 4.2 GitHub Actions
- [ ] Task 4.3 清理 git
- [ ] Task 4.4 版本上限
- [ ] 最终验收全通过
