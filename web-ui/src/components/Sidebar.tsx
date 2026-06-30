import { Plus, MessageSquare, Trash2, Settings, Moon, Sun, Laptop, Cpu } from "lucide-react";
import { useEffect, useState } from "react";
import type { ModelInfo, SessionMeta } from "@/lib/types";

interface Props {
  sessions: SessionMeta[];
  currentSessionId: string | null;
  modelInfo: ModelInfo | null;
  onNew: () => void;
  onSwitch: (id: string) => void;
  onDelete: (id: string) => void;
  onManage: () => void;
  onPickModel: () => void;
}

type Theme = "system" | "dark" | "light";

function applyTheme(t: Theme) {
  const prefersLight = window.matchMedia("(prefers-color-scheme: light)").matches;
  const resolved = t === "system" ? (prefersLight ? "light" : "dark") : t;
  document.documentElement.setAttribute("data-theme", resolved);
}

export default function Sidebar({
  sessions,
  currentSessionId,
  modelInfo,
  onNew,
  onSwitch,
  onDelete,
  onManage,
  onPickModel,
}: Props) {
  const [theme, setTheme] = useState<Theme>(
    () => (localStorage.getItem("theme") as Theme) || "system",
  );

  useEffect(() => {
    localStorage.setItem("theme", theme);
    applyTheme(theme);
  }, [theme]);

  const cycleTheme = () => setTheme(theme === "system" ? "dark" : theme === "dark" ? "light" : "system");
  const ThemeIcon = theme === "system" ? Laptop : theme === "dark" ? Moon : Sun;

  return (
    <aside className="flex h-full w-[260px] shrink-0 flex-col border-r border-border bg-bg-soft">
      <div className="flex items-center justify-between p-4">
        <span className="text-lg font-semibold tracking-wide">凯伦</span>
      </div>
      <div className="px-3">
        <button
          onClick={onNew}
          className="flex w-full items-center justify-center gap-2 rounded-lg border border-border-strong bg-surface px-3 py-2 text-sm font-medium transition hover:bg-surface-hover"
        >
          <Plus size={16} />
          新建对话
        </button>
      </div>
      <div className="mt-4 flex-1 overflow-y-auto px-2">
        {sessions.length === 0 ? (
          <div className="px-3 py-6 text-center text-xs text-text-muted">暂无历史对话</div>
        ) : (
          sessions.map((s) => {
            const active = s.id === currentSessionId;
            return (
              <div
                key={s.id}
                onClick={() => !active && onSwitch(s.id)}
                className={`group mb-1 flex cursor-pointer items-center gap-2 rounded-md px-3 py-2 text-sm transition ${
                  active
                    ? "bg-accent/10 text-accent"
                    : "text-text-soft hover:bg-surface-hover hover:text-text"
                }`}
              >
                <MessageSquare size={14} className="shrink-0" />
                <span className="flex-1 truncate">{s.title || "新对话"}</span>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    if (confirm("删除这个对话？")) onDelete(s.id);
                  }}
                  className="rounded p-1 opacity-0 transition hover:bg-danger/10 hover:text-danger group-hover:opacity-100"
                  title="删除对话"
                >
                  <Trash2 size={13} />
                </button>
              </div>
            );
          })
        )}
      </div>
      <div className="flex flex-col gap-1 border-t border-border p-3">
        <button
          onClick={onPickModel}
          className="flex items-center gap-2 rounded-md px-3 py-2 text-left text-sm text-text-soft transition hover:bg-surface-hover hover:text-text"
          title="切换模型"
        >
          <Cpu size={15} />
          <span className="flex-1 truncate">{modelInfo?.name ?? "加载中..."}</span>
        </button>
        <button
          onClick={onManage}
          className="flex items-center gap-2 rounded-md px-3 py-2 text-left text-sm text-text-soft transition hover:bg-surface-hover hover:text-text"
        >
          <Settings size={15} />
          管理
        </button>
        <button
          onClick={cycleTheme}
          className="flex items-center gap-2 rounded-md px-3 py-2 text-left text-sm text-text-soft transition hover:bg-surface-hover hover:text-text"
          title={`主题：${theme}`}
        >
          <ThemeIcon size={15} />
          主题
        </button>
      </div>
    </aside>
  );
}
