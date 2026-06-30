# karen-web-ui

React + TypeScript + Vite 前端，分阶段替换 `web/templates/` + `web/static/script.js`。
第一阶段只覆盖聊天主页（`/`）；其它页面继续走 Flask 模板，由根路由懒迁移。

## 开发

```bash
# 1) 启动 Flask 后端（项目根）
python main.py            # 或 python desktop_app.py

# 2) 启动 Vite 开发服务器
cd web-ui
npm install               # 首次
npm run dev               # http://localhost:5173
```

Vite 通过 proxy 把 `/api`、`/static`、`/socket.io` 全部转发到 Flask :5000。

## 构建

```bash
npm run build
# 产物输出到 ../web/static/dist/
```

构建产物可由 Flask 直接 serve。集成方式有两种：

1. 把 `web/templates/index.html` 改成加载 `static/dist/index.html`（生产推荐）。
2. 或单独配一个 `/app` 路由，开发期不影响旧 UI。

## 类型一致性

所有 SocketIO 事件 payload 在 `src/lib/types.ts` 集中定义。新增事件时
同步更新 `ServerToClientEvents` / `ClientToServerEvents`，TS 会强制检查。
