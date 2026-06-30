import { io, type Socket } from "socket.io-client";
import type { ClientToServerEvents, ServerToClientEvents, UserLLMConfig } from "./types";

export type KarenSocket = Socket<ServerToClientEvents, ClientToServerEvents>;

// Dev: connect directly to Flask so socket.io WebSocket never goes through
// the Vite proxy (Vite's ws proxy has sticky-session problems with
// Flask-SocketIO's threading mode).  Production: same-origin relative path.
const _SERVER = import.meta.env.DEV ? "http://127.0.0.1:5000" : undefined;

const STORAGE_KEY = "karen_user_llm_config";

export function loadUserConfig(): UserLLMConfig | null {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? (JSON.parse(raw) as UserLLMConfig) : null;
  } catch {
    return null;
  }
}

export function saveUserConfig(cfg: UserLLMConfig | null): void {
  if (cfg) localStorage.setItem(STORAGE_KEY, JSON.stringify(cfg));
  else localStorage.removeItem(STORAGE_KEY);
}

let _socket: KarenSocket | null = null;

export function getSocket(): KarenSocket {
  if (_socket) return _socket;
  _socket = io(_SERVER, {
    reconnectionAttempts: 1,
    reconnectionDelay: 1000,
  }) as KarenSocket;
  return _socket;
}

export function resetSocket(): void {
  _socket?.disconnect();
  _socket = null;
}
