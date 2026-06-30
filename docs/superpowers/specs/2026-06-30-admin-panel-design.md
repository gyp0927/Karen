# Karen 管理后台设计文档

## 背景

当前 Karen Web UI 只有一个聊天页面，大量后端能力没有前端入口：

- 模型配置管理（增删改查、测试、激活）
- 插件管理（列表、启用/禁用、执行、上传）
- RAG / 知识库后端切换
- MCP 服务器管理
- 缓存管理
- 模型路由配置
- 用户认证（登录/注册/用户列表）

本设计补齐这些能力，新增一个独立的管理后台页面 `/settings`。

## 目标

1. 用独立页面统一管理后端所有配置类能力。
2. 视觉风格与现有聊天页一致（深色/浅色主题、青绿色强调色、侧边栏布局）。
3. 不引入新依赖，使用 React + Tailwind + Lucide。
4. 保持移动端可用，后台以桌面端为主、小屏下可滚动。

## 设计方向

后台采用**双栏布局**：

- 左侧：固定 220px 导航栏，图标 + 文字标签。
- 右侧：内容区，顶部有面包屑/标题，下方是模块卡片/表单/表格。

导航项（按使用频率和逻辑分组）：

1. **模型配置** — 增删改查、测试、激活保存的配置
2. **模型路由** — 启用/禁用路由、配置 light/default/powerful 三档
3. **插件** — 列表、启用/禁用、执行、上传
4. **MCP 服务器** — 列表、添加、删除、启用/禁用、查看工具
5. **RAG / 知识库** — 当前后端和可用后端列表，切换后端
6. **缓存** — 统计、清空、配置 TTL
7. **用户认证** — 登录/注册/登出、用户列表（仅当认证开启时显示）
8. **返回聊天** — 返回 `/`

## 视觉与文案

- 主背景：深色 `#0a0a0f`，浅色 `#f5f7fa`
- 面板：深色 `#111118`，浅色 `#ffffff`
- 强调色：`#00d4aa`（深色模式） / `#059669`（浅色模式）
- 危险色：`#ef4444`
- 边框：`rgba(255,255,255,0.06)`（深色） / `rgba(0,0,0,0.06)`（浅色）
- 文案：句子大小写、主动语态、操作与结果同名（“保存” → “已保存”）。

## 技术决策

- 路由：React Router，新增 `/settings` 和 `/settings/:section`。
- 数据获取：复用 `web-ui/src/lib/api.ts` 的 REST wrapper。
- Socket 事件不动，管理后台纯 HTTP API 驱动。
- 表单状态用本地 React state；复杂表格/列表用简单状态机（loading/error/data）。

## 后端 API 映射

| 功能 | 方法 | 端点 |
|------|------|------|
| 列出配置 | GET | `/api/configs` |
| 保存配置 | POST | `/api/configs` |
| 删除配置 | DELETE | `/api/configs/:id` |
| 激活配置 | POST | `/api/configs/:id/activate` |
| 测试配置 | POST | `/api/configs/test` |
| 插件列表 | GET | `/api/plugins` |
| 启用插件 | POST | `/api/plugins/:name/enable` |
| 禁用插件 | POST | `/api/plugins/:name/disable` |
| 执行插件 | POST | `/api/plugins/:name/execute` |
| 上传插件 | POST | `/api/plugins/upload` |
| RAG 后端列表 | GET | `/api/rag/backends` |
| 切换 RAG 后端 | POST | `/api/rag/backend` |
| MCP 服务器列表 | GET | `/api/mcp/servers` |
| 添加 MCP 服务器 | POST | `/api/mcp/servers` |
| 删除 MCP 服务器 | DELETE | `/api/mcp/servers/:name` |
| 启用/禁用 MCP | POST | `/api/mcp/servers/:name/toggle` |
| MCP 工具列表 | GET | `/api/mcp/tools` |
| 调用 MCP 工具 | POST | `/api/mcp/tools/:server/:tool` |
| 缓存统计 | GET | `/api/cache/stats` |
| 清空缓存 | POST | `/api/cache/clear` |
| 缓存配置 | POST | `/api/cache/config` |
| 路由状态 | GET | `/api/router/status` |
| 路由启用/禁用 | POST | `/api/router/config` |
| 设置档位 | POST | `/api/router/tiers/:tier` |
| 认证状态 | GET | `/api/auth/status` |
| 注册 | POST | `/api/auth/register` |
| API Key 登录 | POST | `/api/auth/login` |
| 密码登录 | POST | `/api/auth/login-password` |
| 登出 | POST | `/api/auth/logout` |
| 用户列表 | GET | `/api/auth/users` |
| 删除用户 | DELETE | `/api/auth/users/:id` |

## 成功标准

1. 从聊天页点击「管理」能进入 `/settings`。
2. 八个模块都能正常读取/写入对应后端 API。
3. 操作后有 Toast 反馈。
4. 未启用认证时，用户认证模块显示提示并隐藏表单。
5. 所有修改后运行 `python test_all.py` 通过。
