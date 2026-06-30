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
