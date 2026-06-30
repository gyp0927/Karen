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
