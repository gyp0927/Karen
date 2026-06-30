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
