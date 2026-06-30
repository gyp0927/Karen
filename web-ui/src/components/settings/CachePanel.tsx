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
