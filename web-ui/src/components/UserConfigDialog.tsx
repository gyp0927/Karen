import { X } from "lucide-react";
import { useEffect, useState } from "react";
import { saveUserConfig, loadUserConfig } from "@/lib/socket";
import type { UserLLMConfig } from "@/lib/types";

interface Props {
  onSubmit: (cfg: UserLLMConfig) => void;
  onClose: () => void;
}

const PROVIDERS = [
  "deepseek", "qwen", "minimax", "doubao", "glm", "ernie", "hunyuan", "spark",
  "kimi", "siliconflow", "yi", "baichuan", "openai", "anthropic", "gemini",
  "grok", "mistral", "cohere", "perplexity", "groq", "together", "azure", "ollama",
];

export default function UserConfigDialog({ onSubmit, onClose }: Props) {
  const existing = loadUserConfig();
  const [provider, setProvider] = useState(existing?.provider ?? "deepseek");
  const [model, setModel] = useState(existing?.model ?? "");
  const [apiKey, setApiKey] = useState(existing?.apiKey ?? "");
  const [baseUrl, setBaseUrl] = useState(existing?.baseUrl ?? "");
  const [name, setName] = useState(existing?.name ?? "");

  useEffect(() => {
    const close = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", close);
    return () => window.removeEventListener("keydown", close);
  }, [onClose]);

  const needsApiKey = provider !== "ollama";
  const canSubmit = !!provider && !!model && (!needsApiKey || !!apiKey);

  const submit = () => {
    if (!canSubmit) return;
    const cfg: UserLLMConfig = { provider, model, apiKey, baseUrl: baseUrl || undefined, name: name || undefined };
    saveUserConfig(cfg);
    onSubmit(cfg);
  };

  return (
    <div className="fixed inset-0 z-40 flex items-center justify-center bg-black/60 p-4">
      <div className="w-full max-w-md rounded-xl border border-border-strong bg-bg-soft shadow-2xl">
        <div className="flex items-center justify-between border-b border-border p-4">
          <h3 className="text-base font-semibold">配置你的模型</h3>
          <button onClick={onClose} className="rounded p-1 text-text-soft hover:bg-surface-hover hover:text-text">
            <X size={16} />
          </button>
        </div>
        <div className="space-y-3 p-4">
          <Field label="提供商">
            <select
              value={provider}
              onChange={(e) => setProvider(e.target.value)}
              className="w-full rounded-md border border-border-strong bg-surface px-3 py-2 text-sm outline-none focus:border-accent"
            >
              {PROVIDERS.map((p) => (
                <option key={p} value={p}>{p}</option>
              ))}
            </select>
          </Field>
          <Field label="模型名称">
            <input
              value={model}
              onChange={(e) => setModel(e.target.value)}
              placeholder="如 deepseek-chat / gpt-4o-mini"
              className="w-full rounded-md border border-border-strong bg-surface px-3 py-2 text-sm outline-none focus:border-accent"
            />
          </Field>
          <Field label="API Key">
            <input
              type="password"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder={needsApiKey ? "sk-..." : "Ollama 可留空"}
              className="w-full rounded-md border border-border-strong bg-surface px-3 py-2 text-sm outline-none focus:border-accent"
            />
          </Field>
          <Field label="Base URL（可选）">
            <input
              value={baseUrl}
              onChange={(e) => setBaseUrl(e.target.value)}
              placeholder="留空使用默认"
              className="w-full rounded-md border border-border-strong bg-surface px-3 py-2 text-sm outline-none focus:border-accent"
            />
          </Field>
          <Field label="备注名（可选）">
            <input
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="w-full rounded-md border border-border-strong bg-surface px-3 py-2 text-sm outline-none focus:border-accent"
            />
          </Field>
        </div>
        <div className="flex justify-end gap-2 border-t border-border p-3">
          <button onClick={onClose} className="rounded-md border border-border-strong px-3 py-1.5 text-sm hover:bg-surface-hover">
            取消
          </button>
          <button
            onClick={submit}
            disabled={!canSubmit}
            className="rounded-md bg-accent px-3 py-1.5 text-sm text-bg transition hover:bg-accent-hover disabled:opacity-40"
          >
            保存并使用
          </button>
        </div>
      </div>
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="block">
      <span className="mb-1 block text-xs text-text-soft">{label}</span>
      {children}
    </label>
  );
}
