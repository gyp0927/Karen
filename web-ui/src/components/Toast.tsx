import { useEffect, useState } from "react";

type Kind = "info" | "error" | "success";

interface Props {
  toast: { text: string; kind: Kind } | null;
}

const styles: Record<Kind, string> = {
  info: "bg-surface text-text border-border-strong",
  error: "bg-danger/15 text-danger border-danger/40",
  success: "bg-accent/15 text-accent border-accent/40",
};

export default function Toast({ toast }: Props) {
  const [show, setShow] = useState(false);
  useEffect(() => {
    if (!toast) {
      setShow(false);
      return;
    }
    setShow(true);
  }, [toast]);
  if (!toast) return null;
  return (
    <div
      className={`pointer-events-none fixed left-1/2 top-6 z-50 -translate-x-1/2 rounded-lg border px-4 py-2 text-sm shadow-lg transition-opacity ${
        styles[toast.kind]
      } ${show ? "opacity-100" : "opacity-0"}`}
    >
      {toast.text}
    </div>
  );
}
