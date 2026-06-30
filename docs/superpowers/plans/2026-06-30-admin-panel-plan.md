# Karen 管理后台实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 Karen Web UI 中新增 `/settings` 独立管理后台页面，补齐后端已有的配置、插件、RAG、MCP、缓存、模型路由、用户认证等能力的前端入口。

**Architecture:** 新增 `SettingsPage` 作为双栏布局容器，左侧导航，右侧按 section 渲染子面板。所有子面板通过复用 `lib/api.ts` 调用 Flask REST API。路由使用 React Router，聊天页「管理」按钮改为 `navigate('/settings')`。

**Tech Stack:** React 18, TypeScript, Tailwind CSS, lucide-react, react-router-dom, 既有 REST wrapper (`lib/api.ts`)。

## Global Constraints

- 不引入新的 npm 依赖。
- 颜色、圆角、字体必须复用现有 CSS 变量和 Tailwind 自定义类（如 `bg-bg-soft`, `border-border`, `text-text-soft`, `bg-accent` 等）。
- 所有 API 调用使用 `lib/api.ts` 中已有的 `api.get/post/del/upload`。
- 模块文案使用句子大小写、主动语态，操作按钮与结果提示同名。
- 后端认证相关 API 带 `@auth_required`，前端需在失败时显示 `message`。
- 每次任务完成后运行 `python test_all.py`。
- 每次任务完成后按 CLAUDE.md 流程提交并推送（`git push origin main:main`），除非用户明确说不要推送。

---

### Task 1: 更新路由与入口

**Files:**
- Modify: `web-ui/src/main.tsx`
- Modify: `web-ui/src/pages/ChatPage.tsx`

**Interfaces:**
- Consumes: 现有 `BrowserRouter`, `Routes`, `Route`。
- Produces: `/settings` 路由指向新页面；聊天页侧边栏「管理」按钮导航到 `/settings`。

- [ ] **Step 1: 新增 SettingsPage 导入与路由**

在 `web-ui/src/main.tsx` 中：

```tsx
import SettingsPage from "./pages/SettingsPage";
```

并在 `Routes` 中添加：

```tsx
<Route path="/settings/*" element={<SettingsPage />} />
```

- [ ] **Step 2: 修改 ChatPage 管理按钮**

在 `web-ui/src/pages/ChatPage.tsx` 中，把 `onManage` 从空函数改为：

```tsx
onManage={() => navigate("/settings")}
```

并移除 TODO 注释。

- [ ] **Step 3: 编译检查**

Run: `cd web-ui && npm run lint`
Expected: 无 TypeScript 错误。

- [ ] **Step 4: 提交**

```bash
git add web-ui/src/main.tsx web-ui/src/pages/ChatPage.tsx
git commit -m "feat(settings): add /settings route and link from chat sidebar"
```

---

### Task 2: 创建 SettingsPage 容器与导航

**Files:**
- Create: `web-ui/src/pages/SettingsPage.tsx`
- Create: `web-ui/src/components/settings/SettingsNav.tsx`
- Create: `web-ui/src/components/settings/index.ts`

**Interfaces:**
- Consumes: React Router `useParams`, `NavLink`, `useNavigate`；Lucide icons。
- Produces: 双栏布局容器，左侧 `SettingsNav`，右侧根据 `:section` 渲染占位符或默认选择模型配置模块。

- [ ] **Step 1: 创建 SettingsNav 组件**

文件 `web-ui/src/components/settings/SettingsNav.tsx`：

```tsx
import { NavLink } from "react-router-dom";
import {
  Cpu,
  Route,
  Puzzle,
  Server,
  Database,
  HardDrive,
  Users,
  MessageSquare,
  type LucideIcon,
} from "lucide-react";

interface NavItem {
  to: string;
  label: string;
  icon: LucideIcon;
}

const items: NavItem[] = [
  { to: "/settings/models", label: "模型配置", icon: Cpu },
  { to: "/settings/router", label: "模型路由", icon: Route },
  { to: "/settings/plugins", label: "插件", icon: Puzzle },
  { to: "/settings/mcp", label: "MCP 服务器", icon: Server },
  { to: "/settings/rag", label: "RAG / 知识库", icon: Database },
  { to: "/settings/cache", label: "缓存", icon: HardDrive },
  { to: "/settings/auth", label: "用户认证", icon: Users },
];

export default function SettingsNav() {
  return (
    <aside className="flex h-full w-[220px] shrink-0 flex-col border-r border-border bg-bg-soft">
      <div className="flex items-center gap-2 border-b border-border px-4 py-3">
        <span className="text-base font-semibold tracking-wide">管理后台</span>
      </div>
      <nav className="flex-1 overflow-y-auto p-2">
        {items.map((item) => {
          const Icon = item.icon;
          return (
            <NavLink
              key={item.to}
              to={item.to}
              end
              className={({ isActive }) =>
                `flex items-center gap-2 rounded-md px-3 py-2 text-sm transition ${
                  isActive
                    ? "bg-accent/10 text-accent"
                    : "text-text-soft hover:bg-surface-hover hover:text-text"
                }`
              }
            >
              <Icon size={16} />
              <span>{item.label}</span>
            </NavLink>
          );
        })}
      </nav>
      <div className="border-t border-border p-2">
        <NavLink
          to="/"
          className="flex items-center gap-2 rounded-md px-3 py-2 text-sm text-text-soft transition hover:bg-surface-hover hover:text-text"
        >
          <MessageSquare size={16} />
          <span>返回聊天</span>
        </NavLink>
      </div>
    </aside>
  );
}
```

- [ ] **Step 2: 创建 SettingsPage 容器**

文件 `web-ui/src/pages/SettingsPage.tsx`：

```tsx
import { useParams, useNavigate, Routes, Route, Navigate } from "react-router-dom";
import SettingsNav from "@/components/settings/SettingsNav";

function Placeholder({ title }: { title: string }) {
  return (
    <div className="flex h-full items-center justify-center text-text-soft">
      <span>{title} 模块开发中</span>
    </div>
  );
}

export default function SettingsPage() {
  const { section } = useParams();

  return (
    <div className="flex h-screen w-screen bg-bg text-text">
      <SettingsNav />
      <main className="flex flex-1 flex-col overflow-hidden">
        <header className="flex h-12 items-center border-b border-border bg-bg-soft px-4 text-sm font-medium">
          <span className="text-text-soft">{section ? section.toUpperCase() : "模型配置"}</span>
        </header>
        <div className="flex-1 overflow-y-auto p-6">
          <Routes>
            <Route path="models" element={<Placeholder title="模型配置" />} />
            <Route path="router" element={<Placeholder title="模型路由" />} />
            <Route path="plugins" element={<Placeholder title="插件" />} />
            <Route path="mcp" element={<Placeholder title="MCP 服务器" />} />
            <Route path="rag" element={<Placeholder title="RAG / 知识库" />} />
            <Route path="cache" element={<Placeholder title="缓存" />} />
            <Route path="auth" element={<Placeholder title="用户认证" />} />
            <Route path="*" element={<Navigate to="/settings/models" replace />} />
          </Routes>
        </div>
      </main>
    </div>
  );
}
```

- [ ] **Step 3: 创建组件索引（可选）**

文件 `web-ui/src/components/settings/index.ts`：

```tsx
export { default as SettingsNav } from "./SettingsNav";
```

- [ ] **Step 4: 编译检查**

Run: `cd web-ui && npm run lint`
Expected: 无 TypeScript 错误。

- [ ] **Step 5: 提交**

```bash
git add web-ui/src/pages/SettingsPage.tsx web-ui/src/components/settings/SettingsNav.tsx web-ui/src/components/settings/index.ts
git commit -m "feat(settings): add settings container and navigation"
```

---

### Task 3: 模型配置模块

**Files:**
- Create: `web-ui/src/components/settings/ModelConfigPanel.tsx`

**Interfaces:**
- Consumes: `api` from `lib/api.ts`；后端返回 `{ configs: ConfigItem[], activeConfigId?: string }`。
- Produces: 可新增、编辑、删除、测试、激活配置的表单与列表。

- [ ] **Step 1: 创建 ModelConfigPanel**

文件 `web-ui/src/components/settings/ModelConfigPanel.tsx`：

```tsx
import { useEffect, useState } from "react";
import { Plus, Trash2, Check, RefreshCw, Activity } from "lucide-react";
import { api } from "@/lib/api";
import type { ConfigItem } from "@/lib/types";

const PROVIDERS = ["deepseek", "qwen", "openai", "kimi", "siliconflow", "doubao", "minimax", "ollama"];

export default function ModelConfigPanel() {
  const [configs, setConfigs] = useState<ConfigItem[]>([]);
  const [activeId, setActiveId] = useState<string | undefined>();
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [editing, setEditing] = useState<Partial<ConfigItem> & { apiKey?: string; baseUrl?: string } | null>(null);

  const fetchConfigs = async () => {
    try {
      const res = await api.get<{ configs: ConfigItem[]; activeConfigId?: string }>("/api/configs");
      setConfigs(res.configs || []);
      setActiveId(res.activeConfigId);
    } catch (e) {
      setMessage(`加载失败: ${(e as Error).message}`);
    }
  };

  useEffect(() => {
    fetchConfigs();
  }, []);

  const save = async () => {
    if (!editing?.name || !editing.provider || !editing.model) return;
    setLoading(true);
    try {
      await api.post("/api/configs", {
        id: editing.id,
        name: editing.name,
        provider: editing.provider,
        model: editing.model,
        apiKey: editing.apiKey || "",
        baseUrl: editing.baseUrl || "",
      });
      setEditing(null);
      setMessage("已保存");
      await fetchConfigs();
    } catch (e) {
      setMessage(`保存失败: ${(e as Error).message}`);
    } finally {
      setLoading(false);
    }
  };

  const remove = async (id: string) => {
    if (!confirm("删除这个配置？")) return;
    try {
      await api.del(`/api/configs/${id}`);
      setMessage("已删除");
      await fetchConfigs();
    } catch (e) {
      setMessage(`删除失败: ${(e as Error).message}`);
    }
  };

  const activate = async (id: string) => {
    try {
      await api.post(`/api/configs/${id}/activate`);
      setMessage("已激活");
      await fetchConfigs();
    } catch (e) {
      setMessage(`激活失败: ${(e as Error).message}`);
    }
  };

  const test = async (cfg: ConfigItem & { apiKey?: string; baseUrl?: string }) => {
    setLoading(true);
    try {
      const res = await api.post<{ success: boolean; message: string }>("/api/configs/test", {
        provider: cfg.provider,
        model: cfg.model,
        apiKey: cfg.apiKey || "",
      });
      setMessage(res.message);
    } catch (e) {
      setMessage(`测试失败: ${(e as Error).message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="mx-auto max-w-3xl space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold">模型配置</h2>
        <button
          onClick={() => setEditing({ provider: "deepseek", model: "", name: "" })}
          className="flex items-center gap-1 rounded-md bg-accent px-3 py-1.5 text-sm font-medium text-bg hover:bg-accent-hover"
        >
          <Plus size={14} /> 新增配置
        </button>
      </div>

      {message && (
        <div className="rounded-md border border-border bg-surface px-3 py-2 text-sm text-text-soft">
          {message}
        </div>
      )}

      {editing && (
        <div className="space-y-3 rounded-xl border border-border bg-bg-soft p-4">
          <div className="grid grid-cols-2 gap-3">
            <input
              value={editing.name || ""}
              onChange={(e) => setEditing({ ...editing, name: e.target.value })}
              placeholder="配置名称"
              className="rounded-md border border-border bg-bg px-3 py-2 text-sm outline-none focus:border-accent"
            />
            <select
              value={editing.provider || "deepseek"}
              onChange={(e) => setEditing({ ...editing, provider: e.target.value })}
              className="rounded-md border border-border bg-bg px-3 py-2 text-sm outline-none focus:border-accent"
            >
              {PROVIDERS.map((p) => (
                <option key={p} value={p}>
                  {p}
                </option>
              ))}
            </select>
            <input
              value={editing.model || ""}
              onChange={(e) => setEditing({ ...editing, model: e.target.value })}
              placeholder="模型 ID"
              className="rounded-md border border-border bg-bg px-3 py-2 text-sm outline-none focus:border-accent"
            />
            <input
              value={editing.baseUrl || ""}
              onChange={(e) => setEditing({ ...editing, baseUrl: e.target.value })}
              placeholder="Base URL（可选）"
              className="rounded-md border border-border bg-bg px-3 py-2 text-sm outline-none focus:border-accent"
            />
          </div>
          <input
            type="password"
            value={editing.apiKey || ""}
            onChange={(e) => setEditing({ ...editing, apiKey: e.target.value })}
            placeholder="API Key"
            className="w-full rounded-md border border-border bg-bg px-3 py-2 text-sm outline-none focus:border-accent"
          />
          <div className="flex justify-end gap-2">
            <button
              onClick={() => setEditing(null)}
              className="rounded-md border border-border px-3 py-1.5 text-sm text-text-soft hover:bg-surface-hover"
            >
              取消
            </button>
            <button
              onClick={save}
              disabled={loading}
              className="rounded-md bg-accent px-3 py-1.5 text-sm font-medium text-bg hover:bg-accent-hover disabled:opacity-50"
            >
              保存
            </button>
          </div>
        </div>
      )}

      <div className="space-y-2">
        {configs.map((c) => {
          const isActive = c.id === activeId;
          return (
            <div
              key={c.id}
              className="flex items-center justify-between rounded-xl border border-border bg-bg-soft p-4"
            >
              <div>
                <div className="font-medium">{c.name}</div>
                <div className="text-xs text-text-muted">
                  {c.provider_name || c.provider} · {c.model}
                </div>
              </div>
              <div className="flex items-center gap-1">
                {!isActive && (
                  <button
                    onClick={() => activate(c.id)}
                    title="激活"
                    className="rounded p-1.5 text-text-soft hover:bg-surface-hover hover:text-accent"
                  >
                    <Check size={15} />
                  </button>
                )}
                {isActive && <span className="px-2 text-xs text-accent">当前激活</span>}
                <button
                  onClick={() => test({ ...c })}
                  title="测试连接"
                  className="rounded p-1.5 text-text-soft hover:bg-surface-hover hover:text-accent"
                >
                  <Activity size={15} />
                </button>
                <button
                  onClick={() => setEditing({ ...c, apiKey: "" })}
                  title="编辑"
                  className="rounded p-1.5 text-text-soft hover:bg-surface-hover hover:text-accent"
                >
                  <RefreshCw size={15} />
                </button>
                <button
                  onClick={() => remove(c.id)}
                  title="删除"
                  className="rounded p-1.5 text-text-soft hover:bg-danger/10 hover:text-danger"
                >
                  <Trash2 size={15} />
                </button>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
```

- [ ] **Step 2: 接入 SettingsPage**

修改 `web-ui/src/pages/SettingsPage.tsx`：

```tsx
import ModelConfigPanel from "@/components/settings/ModelConfigPanel";
```

并把 `<Route path="models" element={<Placeholder title="模型配置" />} />` 替换为：

```tsx
<Route path="models" element={<ModelConfigPanel />} />
```

- [ ] **Step 3: 编译与测试**

Run: `cd web-ui && npm run lint`
Expected: 无 TypeScript 错误。

Run: `python test_all.py`
Expected: 22 项通过。

- [ ] **Step 4: 提交**

```bash
git add web-ui/src/components/settings/ModelConfigPanel.tsx web-ui/src/pages/SettingsPage.tsx
git commit -m "feat(settings): add model config management panel"
```

---

### Task 4: 模型路由模块

**Files:**
- Create: `web-ui/src/components/settings/RouterConfigPanel.tsx`
- Modify: `web-ui/src/pages/SettingsPage.tsx`

**Interfaces:**
- Consumes: `/api/router/status`, `/api/router/config`, `/api/router/tiers/:tier`。
- Produces: 开关 + 三档（light/default/powerful）配置表单。

- [ ] **Step 1: 创建 RouterConfigPanel**

文件 `web-ui/src/components/settings/RouterConfigPanel.tsx`：

```tsx
import { useEffect, useState } from "react";
import { Save, Power } from "lucide-react";
import { api } from "@/lib/api";

interface TierConfig {
  provider: string;
  model: string;
  apiKey: string;
  baseUrl: string;
  description: string;
}

interface RouterStatus {
  enabled: boolean;
  tiers: Record<string, TierConfig>;
}

const TIERS = [
  { key: "light", label: "轻量档", desc: "简单问答、问候" },
  { key: "default", label: "默认档", desc: "一般问题" },
  { key: "powerful", label: "强力档", desc: "复杂编码、推理" },
];

export default function RouterConfigPanel() {
  const [enabled, setEnabled] = useState(false);
  const [tiers, setTiers] = useState<Record<string, TierConfig>>({});
  const [message, setMessage] = useState<string | null>(null);

  const fetchStatus = async () => {
    try {
      const res = await api.get<{ success: boolean; enabled: boolean; tiers: Record<string, TierConfig> }>("/api/router/status");
      setEnabled(res.enabled);
      setTiers(res.tiers || {});
    } catch (e) {
      setMessage(`加载失败: ${(e as Error).message}`);
    }
  };

  useEffect(() => {
    fetchStatus();
  }, []);

  const toggle = async () => {
    try {
      await api.post("/api/router/config", { enabled: !enabled });
      setEnabled(!enabled);
      setMessage("已更新");
    } catch (e) {
      setMessage(`更新失败: ${(e as Error).message}`);
    }
  };

  const saveTier = async (key: string) => {
    const cfg = tiers[key];
    if (!cfg) return;
    try {
      await api.post(`/api/router/tiers/${key}`, {
        provider: cfg.provider,
        model: cfg.model,
        apiKey: cfg.apiKey,
        baseUrl: cfg.baseUrl,
      });
      setMessage("档位已保存");
    } catch (e) {
      setMessage(`保存失败: ${(e as Error).message}`);
    }
  };

  const updateTier = (key: string, patch: Partial<TierConfig>) => {
    setTiers((prev) => ({ ...prev, [key]: { ...prev[key], ...patch } }));
  };

  return (
    <div className="mx-auto max-w-3xl space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold">模型路由</h2>
        <button
          onClick={toggle}
          className={`flex items-center gap-1 rounded-md px-3 py-1.5 text-sm font-medium ${
            enabled
              ? "bg-accent text-bg hover:bg-accent-hover"
              : "border border-border text-text-soft hover:bg-surface-hover"
          }`}
        >
          <Power size={14} />
          {enabled ? "已启用" : "已禁用"}
        </button>
      </div>

      {message && (
        <div className="rounded-md border border-border bg-surface px-3 py-2 text-sm text-text-soft">{message}</div>
      )}

      {!enabled && (
        <p className="text-sm text-text-muted">模型路由禁用时会使用默认配置响应所有请求。</p>
      )}

      <div className="space-y-4">
        {TIERS.map(({ key, label, desc }) => {
          const cfg = tiers[key] || { provider: "", model: "", apiKey: "", baseUrl: "" };
          return (
            <div key={key} className="rounded-xl border border-border bg-bg-soft p-4">
              <div className="mb-3 flex items-center justify-between">
                <div>
                  <div className="font-medium">{label}</div>
                  <div className="text-xs text-text-muted">{desc}</div>
                </div>
                <button
                  onClick={() => saveTier(key)}
                  className="flex items-center gap-1 rounded-md bg-accent px-3 py-1.5 text-sm font-medium text-bg hover:bg-accent-hover"
                >
                  <Save size={14} /> 保存
                </button>
              </div>
              <div className="grid grid-cols-2 gap-3">
                <input
                  value={cfg.provider}
                  onChange={(e) => updateTier(key, { provider: e.target.value })}
                  placeholder="provider"
                  className="rounded-md border border-border bg-bg px-3 py-2 text-sm outline-none focus:border-accent"
                />
                <input
                  value={cfg.model}
                  onChange={(e) => updateTier(key, { model: e.target.value })}
                  placeholder="model"
                  className="rounded-md border border-border bg-bg px-3 py-2 text-sm outline-none focus:border-accent"
                />
                <input
                  type="password"
                  value={cfg.apiKey}
                  onChange={(e) => updateTier(key, { apiKey: e.target.value })}
                  placeholder="API Key"
                  className="rounded-md border border-border bg-bg px-3 py-2 text-sm outline-none focus:border-accent"
                />
                <input
                  value={cfg.baseUrl}
                  onChange={(e) => updateTier(key, { baseUrl: e.target.value })}
                  placeholder="Base URL"
                  className="rounded-md border border-border bg-bg px-3 py-2 text-sm outline-none focus:border-accent"
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
```

- [ ] **Step 2: 接入 SettingsPage**

在 `SettingsPage.tsx` 中导入并替换 `router` 路由：

```tsx
import RouterConfigPanel from "@/components/settings/RouterConfigPanel";
```

```tsx
<Route path="router" element={<RouterConfigPanel />} />
```

- [ ] **Step 3: 编译与测试**

Run: `cd web-ui && npm run lint`
Run: `python test_all.py`
Expected: 通过。

- [ ] **Step 4: 提交**

```bash
git add web-ui/src/components/settings/RouterConfigPanel.tsx web-ui/src/pages/SettingsPage.tsx
git commit -m "feat(settings): add model router configuration panel"
```

---

### Task 5: 插件管理模块

**Files:**
- Create: `web-ui/src/components/settings/PluginsPanel.tsx`
- Modify: `web-ui/src/pages/SettingsPage.tsx`

**Interfaces:**
- Consumes: `/api/plugins` (GET), `/:name/enable`, `/:name/disable`, `/:name/execute`, `/upload`。
- Produces: 插件卡片列表，支持启用/禁用、执行（弹出参数输入）、上传 `.py` 文件。

- [ ] **Step 1: 创建 PluginsPanel**

文件 `web-ui/src/components/settings/PluginsPanel.tsx`：

```tsx
import { useEffect, useRef, useState } from "react";
import { Play, Power, PowerOff, Upload } from "lucide-react";
import { api } from "@/lib/api";

interface Plugin {
  name: string;
  enabled: boolean;
  description?: string;
  version?: string;
  triggers?: string[];
}

export default function PluginsPanel() {
  const [plugins, setPlugins] = useState<Plugin[]>([]);
  const [message, setMessage] = useState<string | null>(null);
  const [running, setRunning] = useState<string | null>(null);
  const [argsJson, setArgsJson] = useState<string>("{}");
  const [selectedPlugin, setSelectedPlugin] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const fetchPlugins = async () => {
    try {
      const res = await api.get<{ plugins: Plugin[] }>("/api/plugins");
      setPlugins(res.plugins || []);
    } catch (e) {
      setMessage(`加载失败: ${(e as Error).message}`);
    }
  };

  useEffect(() => {
    fetchPlugins();
  }, []);

  const toggle = async (name: string, enabled: boolean) => {
    try {
      await api.post(`/api/plugins/${name}/${enabled ? "disable" : "enable"}`);
      setMessage(enabled ? "已禁用" : "已启用");
      await fetchPlugins();
    } catch (e) {
      setMessage(`操作失败: ${(e as Error).message}`);
    }
  };

  const execute = async () => {
    if (!selectedPlugin) return;
    let args = {};
    try {
      args = JSON.parse(argsJson || "{}");
    } catch {
      setMessage("参数必须是合法 JSON");
      return;
    }
    setRunning(selectedPlugin);
    try {
      const res = await api.post<{ success: boolean; result?: unknown; message?: string }>(
        `/api/plugins/${selectedPlugin}/execute`,
        { args }
      );
      setMessage(res.success ? JSON.stringify(res.result, null, 2) : res.message || "执行失败");
    } catch (e) {
      setMessage(`执行失败: ${(e as Error).message}`);
    } finally {
      setRunning(null);
      setSelectedPlugin(null);
    }
  };

  const upload = async (file: File) => {
    try {
      const res = await api.upload<{ message: string }>("/api/plugins/upload", file);
      setMessage(res.message);
      await fetchPlugins();
    } catch (e) {
      setMessage(`上传失败: ${(e as Error).message}`);
    }
  };

  return (
    <div className="mx-auto max-w-3xl space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold">插件</h2>
        <label className="flex cursor-pointer items-center gap-1 rounded-md bg-accent px-3 py-1.5 text-sm font-medium text-bg hover:bg-accent-hover">
          <Upload size={14} /> 上传插件
          <input
            type="file"
            accept=".py"
            className="hidden"
            ref={fileRef}
            onChange={(e) => {
              const f = e.target.files?.[0];
              e.target.value = "";
              if (f) upload(f);
            }}
          />
        </label>
      </div>

      {message && (
        <div className="whitespace-pre-wrap rounded-md border border-border bg-surface px-3 py-2 text-sm text-text-soft">
          {message}
        </div>
      )}

      <div className="space-y-2">
        {plugins.map((p) => (
          <div key={p.name} className="rounded-xl border border-border bg-bg-soft p-4">
            <div className="flex items-start justify-between">
              <div>
                <div className="flex items-center gap-2 font-medium">
                  {p.name}
                  {p.enabled ? (
                    <span className="rounded-full bg-success/15 px-1.5 py-0.5 text-[10px] text-success">启用</span>
                  ) : (
                    <span className="rounded-full bg-text-muted/15 px-1.5 py-0.5 text-[10px] text-text-muted">禁用</span>
                  )}
                </div>
                <div className="text-xs text-text-muted">{p.description || "无描述"}</div>
              </div>
              <div className="flex gap-1">
                <button
                  onClick={() => setSelectedPlugin(p.name)}
                  className="rounded p-1.5 text-text-soft hover:bg-surface-hover hover:text-accent"
                  title="执行"
                >
                  <Play size={15} />
                </button>
                <button
                  onClick={() => toggle(p.name, p.enabled)}
                  className="rounded p-1.5 text-text-soft hover:bg-surface-hover hover:text-accent"
                  title={p.enabled ? "禁用" : "启用"}
                >
                  {p.enabled ? <PowerOff size={15} /> : <Power size={15} />}
                </button>
              </div>
            </div>
          </div>
        ))}
      </div>

      {selectedPlugin && (
        <div className="rounded-xl border border-border bg-bg-soft p-4">
          <div className="mb-2 text-sm font-medium">执行 {selectedPlugin}</div>
          <textarea
            value={argsJson}
            onChange={(e) => setArgsJson(e.target.value)}
            rows={4}
            className="w-full rounded-md border border-border bg-bg px-3 py-2 text-sm font-mono outline-none focus:border-accent"
          />
          <div className="mt-2 flex justify-end gap-2">
            <button
              onClick={() => setSelectedPlugin(null)}
              className="rounded-md border border-border px-3 py-1.5 text-sm text-text-soft hover:bg-surface-hover"
            >
              取消
            </button>
            <button
              onClick={execute}
              disabled={running === selectedPlugin}
              className="rounded-md bg-accent px-3 py-1.5 text-sm font-medium text-bg hover:bg-accent-hover disabled:opacity-50"
            >
              执行
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 2: 接入 SettingsPage**

导入并替换 `plugins` 路由。

- [ ] **Step 3: 编译与测试**

Run: `cd web-ui && npm run lint`
Run: `python test_all.py`
Expected: 通过。

- [ ] **Step 4: 提交**

```bash
git add web-ui/src/components/settings/PluginsPanel.tsx web-ui/src/pages/SettingsPage.tsx
git commit -m "feat(settings): add plugin management panel"
```

---

### Task 6: MCP 服务器模块

**Files:**
- Create: `web-ui/src/components/settings/McpPanel.tsx`
- Modify: `web-ui/src/pages/SettingsPage.tsx`

**Interfaces:**
- Consumes: `/api/mcp/servers`, `/api/mcp/tools`, `/api/mcp/servers/:name`, `/api/mcp/servers/:name/toggle`, `/api/mcp/tools/:server/:tool`。
- Produces: 服务器列表（添加/删除/启用/禁用）和工具列表（测试调用）。

- [ ] **Step 1: 创建 McpPanel**

文件 `web-ui/src/components/settings/McpPanel.tsx`：

```tsx
import { useEffect, useState } from "react";
import { Plus, Trash2, Power, PowerOff, Play } from "lucide-react";
import { api } from "@/lib/api";

interface McpServer {
  name: string;
  command?: string;
  args?: string[];
  env?: Record<string, string>;
  url?: string;
  transport?: string;
  enabled: boolean;
}

interface McpTool {
  name: string;
  server: string;
  description?: string;
}

export default function McpPanel() {
  const [servers, setServers] = useState<McpServer[]>([]);
  const [tools, setTools] = useState<McpTool[]>([]);
  const [message, setMessage] = useState<string | null>(null);
  const [form, setForm] = useState<Partial<McpServer>>({ transport: "stdio" });
  const [showForm, setShowForm] = useState(false);
  const [selectedTool, setSelectedTool] = useState<McpTool | null>(null);
  const [argsJson, setArgsJson] = useState("{}");

  const fetchServers = async () => {
    try {
      const res = await api.get<{ servers: McpServer[] }>("/api/mcp/servers");
      setServers(res.servers || []);
    } catch (e) {
      setMessage(`加载服务器失败: ${(e as Error).message}`);
    }
  };

  const fetchTools = async () => {
    try {
      const res = await api.get<{ tools: McpTool[] }>("/api/mcp/tools");
      setTools(res.tools || []);
    } catch (e) {
      setMessage(`加载工具失败: ${(e as Error).message}`);
    }
  };

  useEffect(() => {
    fetchServers();
    fetchTools();
  }, []);

  const addServer = async () => {
    if (!form.name) return;
    try {
      await api.post("/api/mcp/servers", {
        name: form.name,
        command: form.command,
        args: form.args?.filter(Boolean),
        env: form.env,
        url: form.url,
        transport: form.transport,
      });
      setMessage("服务器已添加");
      setShowForm(false);
      setForm({ transport: "stdio" });
      await fetchServers();
      await fetchTools();
    } catch (e) {
      setMessage(`添加失败: ${(e as Error).message}`);
    }
  };

  const removeServer = async (name: string) => {
    if (!confirm(`删除服务器 ${name}？`)) return;
    try {
      await api.del(`/api/mcp/servers/${name}`);
      setMessage("已删除");
      await fetchServers();
      await fetchTools();
    } catch (e) {
      setMessage(`删除失败: ${(e as Error).message}`);
    }
  };

  const toggleServer = async (name: string, enabled: boolean) => {
    try {
      await api.post(`/api/mcp/servers/${name}/toggle`, { enabled: !enabled });
      setMessage("已更新");
      await fetchServers();
      await fetchTools();
    } catch (e) {
      setMessage(`更新失败: ${(e as Error).message}`);
    }
  };

  const callTool = async () => {
    if (!selectedTool) return;
    let args = {};
    try {
      args = JSON.parse(argsJson || "{}");
    } catch {
      setMessage("参数必须是合法 JSON");
      return;
    }
    try {
      const res = await api.post<{ success: boolean; result?: unknown; message?: string }>(
        `/api/mcp/tools/${selectedTool.server}/${selectedTool.name}`,
        { args }
      );
      setMessage(res.success ? JSON.stringify(res.result, null, 2) : res.message || "调用失败");
    } catch (e) {
      setMessage(`调用失败: ${(e as Error).message}`);
    }
  };

  return (
    <div className="mx-auto max-w-3xl space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold">MCP 服务器</h2>
        <button
          onClick={() => setShowForm(true)}
          className="flex items-center gap-1 rounded-md bg-accent px-3 py-1.5 text-sm font-medium text-bg hover:bg-accent-hover"
        >
          <Plus size={14} /> 添加服务器
        </button>
      </div>

      {message && (
        <div className="whitespace-pre-wrap rounded-md border border-border bg-surface px-3 py-2 text-sm text-text-soft">
          {message}
        </div>
      )}

      {showForm && (
        <div className="space-y-2 rounded-xl border border-border bg-bg-soft p-4">
          <input
            value={form.name || ""}
            onChange={(e) => setForm({ ...form, name: e.target.value })}
            placeholder="名称"
            className="w-full rounded-md border border-border bg-bg px-3 py-2 text-sm outline-none focus:border-accent"
          />
          <select
            value={form.transport || "stdio"}
            onChange={(e) => setForm({ ...form, transport: e.target.value })}
            className="w-full rounded-md border border-border bg-bg px-3 py-2 text-sm outline-none focus:border-accent"
          >
            <option value="stdio">stdio</option>
            <option value="sse">sse</option>
          </select>
          <input
            value={form.command || ""}
            onChange={(e) => setForm({ ...form, command: e.target.value })}
            placeholder="命令（stdio）"
            className="w-full rounded-md border border-border bg-bg px-3 py-2 text-sm outline-none focus:border-accent"
          />
          <input
            value={form.args?.join(",") || ""}
            onChange={(e) => setForm({ ...form, args: e.target.value.split(",") })}
            placeholder="参数，逗号分隔"
            className="w-full rounded-md border border-border bg-bg px-3 py-2 text-sm outline-none focus:border-accent"
          />
          <input
            value={form.url || ""}
            onChange={(e) => setForm({ ...form, url: e.target.value })}
            placeholder="URL（sse）"
            className="w-full rounded-md border border-border bg-bg px-3 py-2 text-sm outline-none focus:border-accent"
          />
          <div className="flex justify-end gap-2">
            <button
              onClick={() => setShowForm(false)}
              className="rounded-md border border-border px-3 py-1.5 text-sm text-text-soft hover:bg-surface-hover"
            >
              取消
            </button>
            <button
              onClick={addServer}
              className="rounded-md bg-accent px-3 py-1.5 text-sm font-medium text-bg hover:bg-accent-hover"
            >
              添加
            </button>
          </div>
        </div>
      )}

      <div className="space-y-2">
        {servers.map((s) => (
          <div key={s.name} className="rounded-xl border border-border bg-bg-soft p-4">
            <div className="flex items-center justify-between">
              <div>
                <div className="font-medium">{s.name}</div>
                <div className="text-xs text-text-muted">
                  {s.transport} · {s.enabled ? "启用" : "禁用"}
                </div>
              </div>
              <div className="flex gap-1">
                <button
                  onClick={() => toggleServer(s.name, s.enabled)}
                  className="rounded p-1.5 text-text-soft hover:bg-surface-hover hover:text-accent"
                >
                  {s.enabled ? <PowerOff size={15} /> : <Power size={15} />}
                </button>
                <button
                  onClick={() => removeServer(s.name)}
                  className="rounded p-1.5 text-text-soft hover:bg-danger/10 hover:text-danger"
                >
                  <Trash2 size={15} />
                </button>
              </div>
            </div>
          </div>
        ))}
      </div>

      <div>
        <h3 className="mb-2 text-sm font-medium text-text-soft">可用工具</h3>
        <div className="flex flex-wrap gap-2">
          {tools.map((t) => (
            <button
              key={`${t.server}/${t.name}`}
              onClick={() => setSelectedTool(t)}
              className="flex items-center gap-1 rounded-md border border-border bg-surface px-2 py-1 text-xs hover:bg-surface-hover"
            >
              {t.server}/{t.name}
              <Play size={12} />
            </button>
          ))}
        </div>
      </div>

      {selectedTool && (
        <div className="rounded-xl border border-border bg-bg-soft p-4">
          <div className="mb-2 text-sm font-medium">
            调用 {selectedTool.server}/{selectedTool.name}
          </div>
          <textarea
            value={argsJson}
            onChange={(e) => setArgsJson(e.target.value)}
            rows={4}
            className="w-full rounded-md border border-border bg-bg px-3 py-2 text-sm font-mono outline-none focus:border-accent"
          />
          <div className="mt-2 flex justify-end gap-2">
            <button
              onClick={() => setSelectedTool(null)}
              className="rounded-md border border-border px-3 py-1.5 text-sm text-text-soft hover:bg-surface-hover"
            >
              取消
            </button>
            <button
              onClick={callTool}
              className="rounded-md bg-accent px-3 py-1.5 text-sm font-medium text-bg hover:bg-accent-hover"
            >
              调用
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 2: 接入 SettingsPage**

导入并替换 `mcp` 路由。

- [ ] **Step 3: 编译与测试**

Run: `cd web-ui && npm run lint`
Run: `python test_all.py`
Expected: 通过。

- [ ] **Step 4: 提交**

```bash
git add web-ui/src/components/settings/McpPanel.tsx web-ui/src/pages/SettingsPage.tsx
git commit -m "feat(settings): add MCP server management panel"
```

---

### Task 7: RAG / 知识库模块

**Files:**
- Create: `web-ui/src/components/settings/RagPanel.tsx`
- Modify: `web-ui/src/pages/SettingsPage.tsx`

**Interfaces:**
- Consumes: `/api/rag/backends`, `/api/rag/backend`。
- Produces: 显示当前后端、可用后端列表、切换后端。

- [ ] **Step 1: 创建 RagPanel**

文件 `web-ui/src/components/settings/RagPanel.tsx`：

```tsx
import { useEffect, useState } from "react";
import { Save } from "lucide-react";
import { api } from "@/lib/api";

export default function RagPanel() {
  const [backends, setBackends] = useState<string[]>([]);
  const [current, setCurrent] = useState<string>("");
  const [selected, setSelected] = useState<string>("");
  const [persistPath, setPersistPath] = useState<string>("");
  const [message, setMessage] = useState<string | null>(null);

  const fetchBackends = async () => {
    try {
      const res = await api.get<{ backends: string[]; current: string }>("/api/rag/backends");
      setBackends(res.backends || []);
      setCurrent(res.current || "");
      setSelected(res.current || "");
    } catch (e) {
      setMessage(`加载失败: ${(e as Error).message}`);
    }
  };

  useEffect(() => {
    fetchBackends();
  }, []);

  const save = async () => {
    try {
      const res = await api.post<{ message: string }>("/api/rag/backend", {
        backend: selected,
        persist_path: persistPath || undefined,
      });
      setMessage(res.message);
      await fetchBackends();
    } catch (e) {
      setMessage(`切换失败: ${(e as Error).message}`);
    }
  };

  return (
    <div className="mx-auto max-w-3xl space-y-6">
      <h2 className="text-lg font-semibold">RAG / 知识库</h2>

      {message && (
        <div className="rounded-md border border-border bg-surface px-3 py-2 text-sm text-text-soft">{message}</div>
      )}

      <div className="rounded-xl border border-border bg-bg-soft p-4">
        <div className="mb-4 text-sm text-text-soft">
          当前后端：<span className="font-medium text-text">{current || "未配置"}</span>
        </div>
        <div className="space-y-3">
          <select
            value={selected}
            onChange={(e) => setSelected(e.target.value)}
            className="w-full rounded-md border border-border bg-bg px-3 py-2 text-sm outline-none focus:border-accent"
          >
            {backends.map((b) => (
              <option key={b} value={b}>
                {b}
              </option>
            ))}
          </select>
          <input
            value={persistPath}
            onChange={(e) => setPersistPath(e.target.value)}
            placeholder="持久化路径（可选）"
            className="w-full rounded-md border border-border bg-bg px-3 py-2 text-sm outline-none focus:border-accent"
          />
          <div className="flex justify-end">
            <button
              onClick={save}
              className="flex items-center gap-1 rounded-md bg-accent px-3 py-1.5 text-sm font-medium text-bg hover:bg-accent-hover"
            >
              <Save size={14} /> 切换后端
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: 接入 SettingsPage**

导入并替换 `rag` 路由。

- [ ] **Step 3: 编译与测试**

Run: `cd web-ui && npm run lint`
Run: `python test_all.py`
Expected: 通过。

- [ ] **Step 4: 提交**

```bash
git add web-ui/src/components/settings/RagPanel.tsx web-ui/src/pages/SettingsPage.tsx
git commit -m "feat(settings): add RAG backend management panel"
```

---

### Task 8: 缓存管理模块

**Files:**
- Create: `web-ui/src/components/settings/CachePanel.tsx`
- Modify: `web-ui/src/pages/SettingsPage.tsx`

**Interfaces:**
- Consumes: `/api/cache/stats`, `/api/cache/clear`, `/api/cache/config`。
- Produces: 统计展示、清空按钮、启用/TTL 配置。

- [ ] **Step 1: 创建 CachePanel**

文件 `web-ui/src/components/settings/CachePanel.tsx`：

```tsx
import { useEffect, useState } from "react";
import { Trash2, Save } from "lucide-react";
import { api } from "@/lib/api";

export default function CachePanel() {
  const [stats, setStats] = useState<Record<string, unknown> | null>(null);
  const [enabled, setEnabled] = useState(true);
  const [ttl, setTtl] = useState(24);
  const [message, setMessage] = useState<string | null>(null);

  const fetchStats = async () => {
    try {
      const res = await api.get<{ stats: Record<string, unknown> }>("/api/cache/stats");
      setStats(res.stats || {});
    } catch (e) {
      setMessage(`加载失败: ${(e as Error).message}`);
    }
  };

  useEffect(() => {
    fetchStats();
  }, []);

  const clear = async () => {
    if (!confirm("确定清空缓存？")) return;
    try {
      const res = await api.post<{ message: string }>("/api/cache/clear");
      setMessage(res.message);
      await fetchStats();
    } catch (e) {
      setMessage(`清空失败: ${(e as Error).message}`);
    }
  };

  const saveConfig = async () => {
    try {
      const res = await api.post<{ message: string }>("/api/cache/config", {
        enabled,
        ttl_hours: ttl,
      });
      setMessage(res.message);
    } catch (e) {
      setMessage(`配置失败: ${(e as Error).message}`);
    }
  };

  return (
    <div className="mx-auto max-w-3xl space-y-6">
      <h2 className="text-lg font-semibold">缓存</h2>

      {message && (
        <div className="rounded-md border border-border bg-surface px-3 py-2 text-sm text-text-soft">{message}</div>
      )}

      <div className="rounded-xl border border-border bg-bg-soft p-4">
        <h3 className="mb-2 text-sm font-medium">统计</h3>
        <pre className="max-h-48 overflow-auto rounded-md bg-bg p-3 text-xs text-text-soft">
          {stats ? JSON.stringify(stats, null, 2) : "加载中..."}
        </pre>
        <div className="mt-3 flex justify-end">
          <button
            onClick={clear}
            className="flex items-center gap-1 rounded-md bg-danger/15 px-3 py-1.5 text-sm font-medium text-danger hover:bg-danger/25"
          >
            <Trash2 size={14} /> 清空缓存
          </button>
        </div>
      </div>

      <div className="rounded-xl border border-border bg-bg-soft p-4">
        <h3 className="mb-3 text-sm font-medium">配置</h3>
        <div className="flex items-center gap-3">
          <label className="flex items-center gap-2 text-sm text-text-soft">
            <input
              type="checkbox"
              checked={enabled}
              onChange={(e) => setEnabled(e.target.checked)}
              className="accent-accent"
            />
            启用缓存
          </label>
          <input
            type="number"
            value={ttl}
            onChange={(e) => setTtl(Number(e.target.value))}
            placeholder="TTL（小时）"
            className="w-24 rounded-md border border-border bg-bg px-3 py-2 text-sm outline-none focus:border-accent"
          />
          <button
            onClick={saveConfig}
            className="ml-auto flex items-center gap-1 rounded-md bg-accent px-3 py-1.5 text-sm font-medium text-bg hover:bg-accent-hover"
          >
            <Save size={14} /> 保存
          </button>
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: 接入 SettingsPage**

导入并替换 `cache` 路由。

- [ ] **Step 3: 编译与测试**

Run: `cd web-ui && npm run lint`
Run: `python test_all.py`
Expected: 通过。

- [ ] **Step 4: 提交**

```bash
git add web-ui/src/components/settings/CachePanel.tsx web-ui/src/pages/SettingsPage.tsx
git commit -m "feat(settings): add cache management panel"
```

---

### Task 9: 用户认证模块

**Files:**
- Create: `web-ui/src/components/settings/AuthPanel.tsx`
- Modify: `web-ui/src/pages/SettingsPage.tsx`

**Interfaces:**
- Consumes: `/api/auth/status`, `/api/auth/login-password`, `/api/auth/logout`, `/api/auth/users`, `/api/auth/users/:id`。
- Produces: 认证状态提示、登录表单、用户列表、登出。

- [ ] **Step 1: 创建 AuthPanel**

文件 `web-ui/src/components/settings/AuthPanel.tsx`：

```tsx
import { useEffect, useState } from "react";
import { LogIn, LogOut, Trash2 } from "lucide-react";
import { api } from "@/lib/api";

interface User {
  id: string;
  name: string;
  role: string;
  api_key_mask: string;
}

export default function AuthPanel() {
  const [enabled, setEnabled] = useState(false);
  const [loggedIn, setLoggedIn] = useState(false);
  const [users, setUsers] = useState<User[]>([]);
  const [name, setName] = useState("");
  const [password, setPassword] = useState("");
  const [message, setMessage] = useState<string | null>(null);

  const fetchStatus = async () => {
    try {
      const res = await api.get<{ enabled: boolean }>("/api/auth/status");
      setEnabled(res.enabled);
    } catch (e) {
      setMessage(`加载失败: ${(e as Error).message}`);
    }
  };

  const fetchUsers = async () => {
    try {
      const res = await api.get<{ users: User[] }>("/api/auth/users");
      setUsers(res.users || []);
    } catch (e) {
      // 未登录或不是管理员时忽略
    }
  };

  useEffect(() => {
    fetchStatus();
    fetchUsers();
  }, []);

  const login = async () => {
    try {
      const res = await api.post<{ success: boolean; message?: string }>("/api/auth/login-password", {
        name,
        password,
      });
      if (res.success) {
        setLoggedIn(true);
        setMessage("登录成功");
        await fetchUsers();
      } else {
        setMessage(res.message || "登录失败");
      }
    } catch (e) {
      setMessage(`登录失败: ${(e as Error).message}`);
    }
  };

  const logout = async () => {
    try {
      await api.post("/api/auth/logout");
      setLoggedIn(false);
      setUsers([]);
      setMessage("已退出");
    } catch (e) {
      setMessage(`退出失败: ${(e as Error).message}`);
    }
  };

  const deleteUser = async (id: string) => {
    if (!confirm("删除该用户？")) return;
    try {
      await api.del(`/api/auth/users/${id}`);
      setMessage("已删除");
      await fetchUsers();
    } catch (e) {
      setMessage(`删除失败: ${(e as Error).message}`);
    }
  };

  if (!enabled) {
    return (
      <div className="mx-auto max-w-3xl space-y-6">
        <h2 className="text-lg font-semibold">用户认证</h2>
        <p className="text-sm text-text-muted">认证系统未启用。可在服务端配置开启。</p>
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-3xl space-y-6">
      <h2 className="text-lg font-semibold">用户认证</h2>

      {message && (
        <div className="rounded-md border border-border bg-surface px-3 py-2 text-sm text-text-soft">{message}</div>
      )}

      <div className="rounded-xl border border-border bg-bg-soft p-4">
        {loggedIn ? (
          <div className="flex items-center justify-between">
            <span className="text-sm text-text-soft">已登录</span>
            <button
              onClick={logout}
              className="flex items-center gap-1 rounded-md border border-border px-3 py-1.5 text-sm text-text-soft hover:bg-surface-hover"
            >
              <LogOut size={14} /> 退出
            </button>
          </div>
        ) : (
          <div className="space-y-3">
            <input
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="用户名"
              className="w-full rounded-md border border-border bg-bg px-3 py-2 text-sm outline-none focus:border-accent"
            />
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="密码"
              className="w-full rounded-md border border-border bg-bg px-3 py-2 text-sm outline-none focus:border-accent"
            />
            <div className="flex justify-end">
              <button
                onClick={login}
                className="flex items-center gap-1 rounded-md bg-accent px-3 py-1.5 text-sm font-medium text-bg hover:bg-accent-hover"
              >
                <LogIn size={14} /> 登录
              </button>
            </div>
          </div>
        )}
      </div>

      {users.length > 0 && (
        <div className="rounded-xl border border-border bg-bg-soft p-4">
          <h3 className="mb-3 text-sm font-medium">用户列表</h3>
          <div className="space-y-2">
            {users.map((u) => (
              <div key={u.id} className="flex items-center justify-between rounded-md bg-bg px-3 py-2 text-sm">
                <div>
                  <div className="font-medium">{u.name}</div>
                  <div className="text-xs text-text-muted">
                    {u.role} · {u.api_key_mask}
                  </div>
                </div>
                <button
                  onClick={() => deleteUser(u.id)}
                  className="rounded p-1.5 text-text-soft hover:bg-danger/10 hover:text-danger"
                >
                  <Trash2 size={15} />
                </button>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 2: 接入 SettingsPage**

导入并替换 `auth` 路由。

- [ ] **Step 3: 编译与测试**

Run: `cd web-ui && npm run lint`
Run: `python test_all.py`
Expected: 通过。

- [ ] **Step 4: 提交**

```bash
git add web-ui/src/components/settings/AuthPanel.tsx web-ui/src/pages/SettingsPage.tsx
git commit -m "feat(settings): add auth management panel"
```

---

### Task 10: 集成测试与最终提交

**Files:**
- Modify: `web-ui/src/pages/SettingsPage.tsx`（清理占位符和未使用导入）

**Interfaces:**
- Consumes: 所有面板组件。
- Produces: 完整的 `/settings/*` 路由表。

- [ ] **Step 1: 清理 SettingsPage**

确保 `SettingsPage.tsx` 只导入真实面板，删除 `Placeholder` 组件和相关路由。

- [ ] **Step 2: 运行全部测试**

Run: `python test_all.py`
Expected: 22 项通过。

- [ ] **Step 3: 构建前端**

Run: `cd web-ui && npm run build`
Expected: 构建成功。

- [ ] **Step 4: 提交并推送**

```bash
git add -A
git commit -m "feat(settings): complete admin panel with all backend modules"
git push origin main:main
```

---

## Self-Review

- **Spec coverage:** 八个模块全部映射到具体任务。
- **Placeholder scan:** 无 TBD/TODO；代码完整。
- **Type consistency:** 所有面板返回 JSX，接口与后端一致。
- **Dependencies:** 未引入新依赖。
- **测试：** 每个任务末尾运行 `python test_all.py`。
