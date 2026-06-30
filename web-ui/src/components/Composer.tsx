import { Send, Square, FileText, X, ShieldCheck } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { api } from "@/lib/api";
import type { AttachedFile } from "@/lib/types";

interface Props {
  isProcessing: boolean;
  awaitingReview: boolean;
  onSend: (msg: string, ctx?: string) => void;
  onStop: () => void;
  onTriggerReview: () => void;
}

export default function Composer({ isProcessing, awaitingReview, onSend, onStop, onTriggerReview }: Props) {
  const [text, setText] = useState("");
  const [attached, setAttached] = useState<AttachedFile | null>(null);
  const [uploading, setUploading] = useState(false);
  const ref = useRef<HTMLTextAreaElement | null>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 220)}px`;
  }, [text]);

  const canSend = !!text.trim() && !isProcessing;

  const handleSend = () => {
    if (!canSend) return;
    onSend(text.trim(), attached?.content);
    setText("");
    setAttached(null);
  };

  const onKey = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleFile = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    e.target.value = "";
    if (!f) return;
    setUploading(true);
    try {
      const res = await api.upload<{ filename: string; content: string }>("/api/upload", f);
      setAttached({ filename: res.filename, content: res.content });
    } catch (err) {
      alert(`上传失败：${(err as Error).message}`);
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="border-t border-border bg-bg-soft px-4 py-3">
      <div className="mx-auto max-w-content">
        {attached && (
          <div className="mb-2 flex items-center gap-2 rounded-md border border-border bg-surface px-3 py-2 text-sm text-text-soft">
            <FileText size={14} />
            <span className="flex-1 truncate">{attached.filename}</span>
            <button onClick={() => setAttached(null)} className="rounded p-1 hover:bg-surface-hover">
              <X size={14} />
            </button>
          </div>
        )}
        <div className="flex items-end gap-2 rounded-xl border border-border bg-surface px-3 py-2 focus-within:border-accent/60">
          <label className="cursor-pointer rounded-md p-2 text-text-soft transition hover:bg-surface-hover hover:text-text" title="上传文档">
            <input type="file" className="hidden" onChange={handleFile} accept=".pdf,.docx,.txt,.md" />
            {uploading ? <div className="karen-spinner" /> : <FileText size={16} />}
          </label>
          <textarea
            ref={ref}
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={onKey}
            placeholder={awaitingReview ? "继续追问，或点右侧三角按钮触发审查" : "和凯伦聊点什么..."}
            rows={1}
            className="flex-1 resize-none bg-transparent py-2 text-[15px] outline-none placeholder:text-text-muted"
          />
          {awaitingReview && !isProcessing && (
            <button
              onClick={onTriggerReview}
              title="触发审查"
              className="flex h-9 w-9 items-center justify-center rounded-md text-violet transition hover:bg-violet/10"
            >
              <ShieldCheck size={18} />
            </button>
          )}
          {isProcessing ? (
            <button
              onClick={onStop}
              title="停止生成"
              className="flex h-9 w-9 items-center justify-center rounded-md bg-danger/15 text-danger transition hover:bg-danger/25"
            >
              <Square size={16} />
            </button>
          ) : (
            <button
              onClick={handleSend}
              disabled={!canSend}
              title="发送"
              className="flex h-9 w-9 items-center justify-center rounded-md bg-accent text-bg transition hover:bg-accent-hover disabled:cursor-not-allowed disabled:opacity-40"
            >
              <Send size={16} />
            </button>
          )}
        </div>
        <p className="mt-2 text-center text-xs text-text-muted">
          Enter 发送，Shift+Enter 换行
        </p>
      </div>
    </div>
  );
}
