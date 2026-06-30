// All server payload shapes derived from web/app.py SocketIO emits + script.js usages.

export type Sender = "user" | "assistant" | "review" | "system";

export interface SessionMeta {
  id: string;
  title?: string;
  is_current?: boolean;
  updated_at?: number;
  message_count?: number;
}

export interface ModelInfo {
  name: string;
  provider: string;
  provider_name: string;
  model: string;
  server_has_config: boolean;
}

export interface ConfigItem {
  id: string;
  name: string;
  provider: string;
  provider_name?: string;
  model: string;
  is_active?: boolean;
}

export interface UserLLMConfig {
  provider: string;
  model: string;
  apiKey: string;
  baseUrl?: string;
  name?: string;
}

export interface ToolConfirmPayload {
  confirm_id: string;
  code_preview?: string;
  tool_name?: string;
}

export interface AttachedFile {
  filename: string;
  content: string;
}

// --------- Server → Client events ---------
export interface ServerToClientEvents {
  thinking: (data: { message: string }) => void;
  stream_start: (data: { sender?: string }) => void;
  token_chunk: (data: { token: string }) => void;
  stream_end: (data: { message: string; sender?: string }) => void;
  token_usage: (data: {
    prompt_tokens?: number;
    completion_tokens?: number;
    total_tokens?: number;
    cost?: number;
  }) => void;
  user_message: (data: { message: string }) => void;
  bot_history: (data: { message: string; sender: string }) => void;
  review_history: (data: { review_result: string }) => void;
  history_restored: (data: { awaiting_review?: boolean }) => void;
  review_complete: (data: { review_result: string }) => void;
  message_failed: (data: { message?: string; error?: string }) => void;
  error: (data: { message: string }) => void;
  sessions_list: (data: { sessions: SessionMeta[] }) => void;
  session_created: (data: { session_id: string; sessions: SessionMeta[] }) => void;
  session_switched: (data: { session_id: string; sessions: SessionMeta[] }) => void;
  session_deleted: (data: { sessions: SessionMeta[] }) => void;
  model_info: (data: ModelInfo) => void;
  config_activated: (data: { name: string; provider_name: string; model: string; message: string }) => void;
  configs_list: (data: { configs: ConfigItem[]; activeConfigId?: string }) => void;
  user_config_set: (data: {
    success: boolean;
    name?: string;
    provider_name?: string;
    model?: string;
    message?: string;
  }) => void;
  config_error: (data: { message: string }) => void;
  config_required: (data: { reason?: string }) => void;
  generation_stopped: (data: Record<string, never>) => void;
  generation_stopping: (data: Record<string, never>) => void;
  tool_confirmation_required: (data: ToolConfirmPayload) => void;
  tool_confirmation_ack: (data: { confirm_id: string }) => void;
}

// --------- Client → Server events ---------
export interface ClientToServerEvents {
  send_message: (payload: { message: string; document_context?: string }) => void;
  stop_generation: () => void;
  trigger_review: () => void;
  confirm_tool_execution: (payload: { confirm_id: string; approve: boolean }) => void;
  new_session: () => void;
  switch_session: (payload: { session_id: string }) => void;
  delete_session: (payload: { session_id: string }) => void;
  clear_history: () => void;
  get_sessions: () => void;
  get_model_info: () => void;
  get_mode: () => void;
  set_mode: (payload: { mode: string }) => void;
  set_model: (payload: { provider: string; model: string }) => void;
  get_configs: () => void;
  activate_config: (payload: { configId: string }) => void;
  set_user_config: (payload: UserLLMConfig) => void;
  set_review_language: (payload: { language: string }) => void;
}

// --------- Chat message UI model ---------
export interface ChatMessage {
  id: string;
  sender: Sender;
  senderName?: string;
  content: string;
  streaming?: boolean;
  createdAt: number;
}
