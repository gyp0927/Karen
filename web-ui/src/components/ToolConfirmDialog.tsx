import { Play, X } from "lucide-react";
import type { ToolConfirmPayload } from "@/lib/types";

interface Props {
  payload: ToolConfirmPayload;
  onDecide: (id: string, approve: boolean) => void;
}

export default function ToolConfirmDialog({ payload, onDecide }: Props) {
  return (
    <div className="fixed inset-0 z-40 flex items-center justify-center bg-black/60 p-4">
      <div className="w-full max-w-2xl rounded-xl border border-border-strong bg-bg-soft shadow-2xl">
        <div className="flex items-center justify-between border-b border-border p-4">
          <h3 className="text-base font-semibold">是否执行工具调用？</h3>
          <button
            onClick={() => onDecide(payload.confirm_id, false)}
            className="rounded p-1 text-text-soft hover:bg-surface-hover hover:text-text"
          >
            <X size={16} />
          </button>
        </div>
        <div className="p-4">
          {payload.tool_name && (
            <p className="mb-2 text-sm text-text-soft">
              工具: <span className="font-mono text-text">{payload.tool_name}</span>
            </p>
          )}
          <pre className="max-h-72 overflow-auto rounded-md border border-border bg-[var(--code-bg)] p-3 font-mono text-xs leading-6">
            {payload.code_preview || "（无代码预览）"}
          </pre>
        </div>
        <div className="flex justify-end gap-2 border-t border-border p-3">
          <button
            onClick={() => onDecide(payload.confirm_id, false)}
            className="rounded-md border border-border-strong px-3 py-1.5 text-sm hover:bg-surface-hover"
          >
            拒绝
          </button>
          <button
            onClick={() => onDecide(payload.confirm_id, true)}
            className="flex items-center gap-1 rounded-md bg-accent px-3 py-1.5 text-sm text-bg transition hover:bg-accent-hover"
          >
            <Play size={13} />
            允许执行
          </button>
        </div>
      </div>
    </div>
  );
}
