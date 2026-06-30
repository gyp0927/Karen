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
