import { useCallback, useEffect, useRef, useState } from "react";
import { getSocket, loadUserConfig } from "./socket";
import type {
  ChatMessage,
  ModelInfo,
  SessionMeta,
  ToolConfirmPayload,
  UserLLMConfig,
} from "./types";

function uid(): string {
  return Math.random().toString(36).slice(2, 10);
}

export interface ChatUiState {
  messages: ChatMessage[];
  sessions: SessionMeta[];
  currentSessionId: string | null;
  thinking: string | null;
  isProcessing: boolean;
  awaitingReview: boolean;
  modelInfo: ModelInfo | null;
  pendingTool: ToolConfirmPayload | null;
  toast: { text: string; kind: "info" | "error" | "success" } | null;
  needsUserConfig: boolean;
}

const initial: ChatUiState = {
  messages: [],
  sessions: [],
  currentSessionId: null,
  thinking: null,
  isProcessing: false,
  awaitingReview: false,
  modelInfo: null,
  pendingTool: null,
  toast: null,
  needsUserConfig: false,
};

type RawSocket = {
  on: (e: string, h: (d: unknown) => void) => void;
  off: (e: string, h: (d: unknown) => void) => void;
};

export function useChat() {
  const socketRef = useRef(getSocket());
  const [state, setState] = useState<ChatUiState>(initial);

  // Stream accumulator lives outside React state so token append is O(1).
  const streamIdRef = useRef<string | null>(null);
  const streamBufRef = useRef<string>("");

  const showToast = useCallback(
    (text: string, kind: "info" | "error" | "success" = "info") => {
      setState((s) => ({ ...s, toast: { text, kind } }));
      window.setTimeout(
        () => setState((s) => (s.toast?.text === text ? { ...s, toast: null } : s)),
        2400,
      );
    },
    [],
  );

  useEffect(() => {
    const s = socketRef.current;
    const push = (m: Omit<ChatMessage, "id" | "createdAt">) =>
      setState((st) => ({
        ...st,
        messages: [...st.messages, { id: uid(), createdAt: Date.now(), ...m }],
      }));

    const handlers: Record<string, (data: never) => void> = {
      thinking: (d: { message: string }) =>
        setState((st) => ({ ...st, thinking: d.message })),
      user_message: (d: { message: string }) => {
        push({ sender: "user", senderName: "你", content: d.message });
        s.emit("get_sessions");
      },
      bot_history: (d: { message: string; sender: string }) =>
        push({ sender: "assistant", senderName: d.sender, content: d.message }),
      review_history: (d: { review_result: string }) =>
        push({ sender: "review", senderName: "审查者", content: d.review_result }),
      history_restored: (d: { awaiting_review?: boolean }) =>
        setState((st) => ({ ...st, awaitingReview: !!d.awaiting_review })),
      stream_start: () => {
        streamBufRef.current = "";
        const id = uid();
        streamIdRef.current = id;
        setState((st) => ({
          ...st,
          thinking: null,
          messages: [
            ...st.messages,
            {
              id,
              sender: "assistant",
              senderName: "凯伦",
              content: "",
              streaming: true,
              createdAt: Date.now(),
            },
          ],
        }));
      },
      token_chunk: (d: { token: string }) => {
        const id = streamIdRef.current;
        if (!id) return;
        streamBufRef.current += d.token;
        const buf = streamBufRef.current;
        setState((st) => ({
          ...st,
          messages: st.messages.map((m) => (m.id === id ? { ...m, content: buf } : m)),
        }));
      },
      stream_end: (d: { message: string }) => {
        const id = streamIdRef.current;
        const final = d.message || streamBufRef.current;
        streamIdRef.current = null;
        streamBufRef.current = "";
        setState((st) => ({
          ...st,
          thinking: null,
          isProcessing: false,
          messages: st.messages.map((m) =>
            m.id === id ? { ...m, content: final, streaming: false } : m,
          ),
        }));
      },
      review_complete: (d: { review_result: string }) => {
        push({ sender: "review", senderName: "审查者", content: d.review_result });
        setState((st) => ({
          ...st,
          thinking: null,
          awaitingReview: false,
          isProcessing: false,
        }));
      },
      message_failed: (d: { error?: string }) => {
        showToast(d.error || "发送失败", "error");
        setState((st) => {
          const msgs = [...st.messages];
          for (let i = msgs.length - 1; i >= 0; i--) {
            if (msgs[i].sender === "user") {
              msgs.splice(i, 1);
              break;
            }
          }
          return { ...st, isProcessing: false, thinking: null, messages: msgs };
        });
        streamIdRef.current = null;
        streamBufRef.current = "";
      },
      error: (d: { message: string }) => {
        showToast(d.message, "error");
        setState((st) => ({ ...st, isProcessing: false, thinking: null }));
        streamIdRef.current = null;
        streamBufRef.current = "";
      },
      sessions_list: (d: { sessions: SessionMeta[] }) =>
        setState((st) => {
          const cur = d.sessions.find((x) => x.is_current);
          return {
            ...st,
            sessions: d.sessions,
            currentSessionId: cur?.id ?? st.currentSessionId,
          };
        }),
      session_created: (d: { session_id: string; sessions: SessionMeta[] }) => {
        streamIdRef.current = null;
        streamBufRef.current = "";
        setState((st) => ({
          ...st,
          messages: [],
          thinking: null,
          awaitingReview: false,
          sessions: d.sessions,
          currentSessionId: d.session_id,
        }));
        showToast("已开启新对话");
      },
      session_switched: (d: { session_id: string; sessions: SessionMeta[] }) => {
        streamIdRef.current = null;
        streamBufRef.current = "";
        setState((st) => ({
          ...st,
          messages: [],
          thinking: null,
          awaitingReview: false,
          sessions: d.sessions,
          currentSessionId: d.session_id,
        }));
      },
      session_deleted: (d: { sessions: SessionMeta[] }) => {
        const cur = d.sessions.find((x) => x.is_current);
        streamIdRef.current = null;
        streamBufRef.current = "";
        setState((st) => ({
          ...st,
          messages: [],
          thinking: null,
          awaitingReview: false,
          sessions: d.sessions,
          currentSessionId: cur?.id ?? null,
        }));
      },
      model_info: (d: ModelInfo) =>
        setState((st) => ({
          ...st,
          modelInfo: d,
          needsUserConfig:
            st.needsUserConfig || (!d.server_has_config && st.modelInfo == null),
        })),
      config_activated: (d: {
        name: string;
        provider_name: string;
        model: string;
        message: string;
      }) => {
        setState((st) => ({
          ...st,
          modelInfo: st.modelInfo
            ? { ...st.modelInfo, name: d.name, provider_name: d.provider_name, model: d.model }
            : st.modelInfo,
        }));
        showToast(d.message, "success");
      },
      config_error: (d: { message: string }) => {
        showToast(d.message, "error");
        setState((st) => ({ ...st, needsUserConfig: true }));
      },
      config_required: () =>
        setState((st) => ({
          ...st,
          isProcessing: false,
          thinking: null,
          needsUserConfig: true,
        })),
      generation_stopped: () => {
        streamIdRef.current = null;
        streamBufRef.current = "";
        setState((st) => ({ ...st, isProcessing: false, thinking: null }));
        showToast("生成已停止");
      },
      tool_confirmation_required: (d: ToolConfirmPayload) =>
        setState((st) => ({ ...st, pendingTool: d })),
      tool_confirmation_ack: () => setState((st) => ({ ...st, pendingTool: null })),
    };

    const raw = s as unknown as RawSocket;
    for (const [ev, fn] of Object.entries(handlers)) raw.on(ev, fn as (d: unknown) => void);

    const onConnect = () => {
      const cfg = loadUserConfig();
      if (cfg) s.emit("set_user_config", cfg);
      s.emit("get_sessions");
      s.emit("get_model_info");
    };
    s.on("connect", onConnect);
    if (s.connected) onConnect();

    return () => {
      for (const [ev, fn] of Object.entries(handlers)) raw.off(ev, fn as (d: unknown) => void);
      s.off("connect", onConnect);
    };
  }, [showToast]);

  // ---------- actions ----------
  const send = useCallback((message: string, documentContext?: string) => {
    const s = socketRef.current;
    if (!message.trim()) return;
    setState((st) => ({ ...st, isProcessing: true }));
    s.emit(
      "send_message",
      documentContext ? { message, document_context: documentContext } : { message },
    );
  }, []);

  const stop = useCallback(() => socketRef.current.emit("stop_generation"), []);
  const triggerReview = useCallback(() => {
    setState((st) => ({ ...st, isProcessing: true }));
    socketRef.current.emit("trigger_review");
  }, []);
  const newSession = useCallback(() => socketRef.current.emit("new_session"), []);
  const switchSession = useCallback(
    (id: string) => socketRef.current.emit("switch_session", { session_id: id }),
    [],
  );
  const deleteSession = useCallback(
    (id: string) => socketRef.current.emit("delete_session", { session_id: id }),
    [],
  );
  const confirmTool = useCallback(
    (id: string, approve: boolean) =>
      socketRef.current.emit("confirm_tool_execution", { confirm_id: id, approve }),
    [],
  );
  const setUserConfig = useCallback((cfg: UserLLMConfig) => {
    socketRef.current.emit("set_user_config", cfg);
    setState((st) => ({ ...st, needsUserConfig: false }));
  }, []);
  const dismissUserConfig = useCallback(
    () => setState((st) => ({ ...st, needsUserConfig: false })),
    [],
  );

  return {
    state,
    actions: {
      send,
      stop,
      triggerReview,
      newSession,
      switchSession,
      deleteSession,
      confirmTool,
      setUserConfig,
      dismissUserConfig,
      showToast,
    },
  };
}
