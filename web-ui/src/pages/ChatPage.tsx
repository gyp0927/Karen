import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import Sidebar from "@/components/Sidebar";
import Composer from "@/components/Composer";
import MessageView from "@/components/MessageView";
import Toast from "@/components/Toast";
import ToolConfirmDialog from "@/components/ToolConfirmDialog";
import UserConfigDialog from "@/components/UserConfigDialog";
import ModelSwitcher from "@/components/ModelSwitcher";
import { useChat } from "@/lib/useChat";

export default function ChatPage() {
  const navigate = useNavigate();
  const { state, actions } = useChat();
  const [showModelSwitcher, setShowModelSwitcher] = useState(false);

  const scrollRef = useRef<HTMLDivElement | null>(null);
  const autoFollowRef = useRef(true);

  // Track whether the user has scrolled up; if so we stop auto-following.
  const onScroll = () => {
    const el = scrollRef.current;
    if (!el) return;
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 40;
    autoFollowRef.current = atBottom;
  };
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    if (autoFollowRef.current) el.scrollTop = el.scrollHeight;
  }, [state.messages, state.thinking]);

  const empty = state.messages.length === 0 && !state.thinking;

  return (
    <div className="flex h-screen w-screen bg-bg text-text">
      <Sidebar
        sessions={state.sessions}
        currentSessionId={state.currentSessionId}
        modelInfo={state.modelInfo}
        onNew={actions.newSession}
        onSwitch={actions.switchSession}
        onDelete={actions.deleteSession}
        onManage={() => navigate("/settings")}
        onPickModel={() => setShowModelSwitcher(true)}
      />
      <main className="flex flex-1 flex-col">
        <header className="flex h-12 items-center justify-between border-b border-border bg-bg-soft px-4 text-sm">
          <span className="text-text-soft">
            {state.modelInfo
              ? `${state.modelInfo.provider_name || state.modelInfo.provider} · ${state.modelInfo.model}`
              : "未连接模型"}
          </span>
          {state.awaitingReview && (
            <span className="rounded-full bg-violet/15 px-2 py-0.5 text-xs text-violet">等待审查</span>
          )}
        </header>

        <div
          ref={scrollRef}
          onScroll={onScroll}
          className="flex-1 overflow-y-auto"
        >
          {empty ? (
            <div className="flex h-full flex-col items-center justify-center px-6 text-center">
              <h1 className="bg-gradient-to-r from-accent to-violet bg-clip-text text-3xl font-semibold text-transparent">
                我是凯伦，你想问什么？
              </h1>
              <p className="mt-3 text-sm text-text-soft"></p>
            </div>
          ) : (
            <div className="mx-auto max-w-content py-4">
              {state.messages.map((m) => (
                <MessageView key={m.id} msg={m} />
              ))}
              {state.thinking && (
                <div className="flex items-center gap-2 px-4 py-2 text-sm text-text-soft">
                  <div className="karen-spinner" />
                  <span>{state.thinking}</span>
                </div>
              )}
            </div>
          )}
        </div>

        <Composer
          isProcessing={state.isProcessing}
          awaitingReview={state.awaitingReview}
          onSend={actions.send}
          onStop={actions.stop}
          onTriggerReview={actions.triggerReview}
        />
      </main>

      <Toast toast={state.toast} />
      {state.pendingTool && (
        <ToolConfirmDialog payload={state.pendingTool} onDecide={actions.confirmTool} />
      )}
      {state.needsUserConfig && (
        <UserConfigDialog
          onSubmit={actions.setUserConfig}
          onClose={actions.dismissUserConfig}
        />
      )}
      {showModelSwitcher && <ModelSwitcher onClose={() => setShowModelSwitcher(false)} />}
    </div>
  );
}
