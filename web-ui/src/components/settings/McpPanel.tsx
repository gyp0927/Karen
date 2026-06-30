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
