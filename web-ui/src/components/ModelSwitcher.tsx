import { X, Check } from "lucide-react";
import { useEffect, useState } from "react";
import { getSocket } from "@/lib/socket";
import type { ConfigItem } from "@/lib/types";

interface Props {
  onClose: () => void;
}

export default function ModelSwitcher({ onClose }: Props) {
  const [configs, setConfigs] = useState<ConfigItem[]>([]);
  const [activeId, setActiveId] = useState<string | undefined>();

  useEffect(() => {
    const s = getSocket();
    const handler = (d: { configs: ConfigItem[]; activeConfigId?: string }) => {
      setConfigs(d.configs || []);
      setActiveId(d.activeConfigId);
    };
    s.on("configs_list", handler);
    s.emit("get_configs");
    const closeOnEsc = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", closeOnEsc);
    return () => {
      s.off("configs_list", handler);
      window.removeEventListener("keydown", closeOnEsc);
    };
  }, [onClose]);

  const activate = (id: string) => {
    getSocket().emit("activate_config", { configId: id });
  };

  return (
    <div className="fixed inset-0 z-40 flex items-center justify-center bg-black/60 p-4">
      <div className="w-full max-w-lg rounded-xl border border-border-strong bg-bg-soft shadow-2xl">
        <div className="flex items-center justify-between border-b border-border p-4">
          <h3 className="text-base font-semibold">选择模型</h3>
          <button onClick={onClose} className="rounded p-1 text-text-soft hover:bg-surface-hover hover:text-text">
            <X size={16} />
          </button>
        </div>
        <div className="max-h-[60vh] overflow-y-auto p-2">
          {configs.length === 0 ? (
            <p className="px-3 py-6 text-center text-sm text-text-soft">
              暂无配置。请到「管理 → 模型配置」添加。
            </p>
          ) : (
            configs.map((c) => {
              const active = (c.is_active ?? c.id === activeId) === true;
              return (
                <button
                  key={c.id}
                  onClick={() => activate(c.id)}
                  className={`flex w-full items-center justify-between rounded-md px-3 py-2 text-left text-sm transition hover:bg-surface-hover ${
                    active ? "bg-accent/10 text-accent" : ""
                  }`}
                >
                  <span className="flex flex-col">
                    <span className="font-medium">{c.name}</span>
                    <span className="text-xs text-text-muted">
                      {(c.provider_name || c.provider) + " · " + c.model}
                    </span>
                  </span>
                  {active && <Check size={16} />}
                </button>
              );
            })
          )}
        </div>
      </div>
    </div>
  );
}
