import { useEffect, useRef } from "react";
import { highlightWithin, renderMarkdown } from "@/lib/markdown";
import type { ChatMessage } from "@/lib/types";

interface Props {
  msg: ChatMessage;
}

export default function MessageView({ msg }: Props) {
  const bodyRef = useRef<HTMLDivElement | null>(null);

  // Highlight after the streaming finishes; while streaming we keep plain text
  // so token append stays cheap.
  useEffect(() => {
    if (msg.streaming) return;
    if (msg.sender === "user") return;
    if (bodyRef.current) highlightWithin(bodyRef.current);
  }, [msg.streaming, msg.content, msg.sender]);

  const isUser = msg.sender === "user";
  const isReview = msg.sender === "review";

  return (
    <div
      className={`flex w-full gap-3 px-4 py-3 ${isUser ? "justify-end" : "justify-start"}`}
    >
      {!isUser && (
        <div
          className={`mt-1 h-8 w-8 shrink-0 overflow-hidden rounded-full border ${
            isReview ? "border-violet bg-violet/10" : "border-border-strong"
          }`}
        >
          {isReview ? (
            <div className="flex h-full w-full items-center justify-center text-violet">⚖</div>
          ) : (
            <img
              src="/static/img/avatar.png"
              alt="凯伦"
              className="h-full w-full object-cover"
              onError={(e) => ((e.currentTarget as HTMLImageElement).style.display = "none")}
            />
          )}
        </div>
      )}
      <div className={`max-w-[min(720px,80%)] ${isUser ? "items-end" : "items-start"}`}>
        {!isUser && (
          <div className="mb-1 text-xs text-text-muted">{msg.senderName ?? "凯伦"}</div>
        )}
        {msg.streaming ? (
          <div
            className="whitespace-pre-wrap rounded-lg border border-border bg-surface px-4 py-3 text-[15px] leading-7"
          >
            {msg.content}
            <span className="ml-0.5 inline-block h-4 w-[2px] -translate-y-0.5 animate-pulse bg-accent" />
          </div>
        ) : (
          <div
            ref={bodyRef}
            className={`markdown rounded-lg border px-4 py-3 ${
              isUser
                ? "border-accent/40 bg-accent/10 text-text"
                : isReview
                  ? "border-violet/40 bg-violet/5"
                  : "border-border bg-surface"
            }`}
            // eslint-disable-next-line react/no-danger
            dangerouslySetInnerHTML={{
              __html: isUser
                ? renderMarkdown(msg.content)
                : renderMarkdown(msg.content),
            }}
          />
        )}
      </div>
    </div>
  );
}
