import { useEffect, useState } from "react";
import { LogIn, LogOut, Trash2 } from "lucide-react";
import { api } from "@/lib/api";

interface User {
  id: string;
  name: string;
  role: string;
  api_key_mask: string;
}

export default function AuthPanel() {
  const [enabled, setEnabled] = useState(false);
  const [loggedIn, setLoggedIn] = useState(false);
  const [users, setUsers] = useState<User[]>([]);
  const [name, setName] = useState("");
  const [password, setPassword] = useState("");
  const [message, setMessage] = useState<string | null>(null);

  const fetchStatus = async () => {
    try {
      const res = await api.get<{ enabled: boolean }>("/api/auth/status");
      setEnabled(res.enabled);
    } catch (e) {
      setMessage(`加载失败: ${(e as Error).message}`);
    }
  };

  const fetchUsers = async () => {
    try {
      const res = await api.get<{ users: User[] }>("/api/auth/users");
      setUsers(res.users || []);
    } catch (e) {
      // 未登录或不是管理员时忽略
    }
  };

  useEffect(() => {
    fetchStatus();
    fetchUsers();
  }, []);

  const login = async () => {
    try {
      const res = await api.post<{ success: boolean; message?: string }>("/api/auth/login-password", {
        name,
        password,
      });
      if (res.success) {
        setLoggedIn(true);
        setMessage("登录成功");
        await fetchUsers();
      } else {
        setMessage(res.message || "登录失败");
      }
    } catch (e) {
      setMessage(`登录失败: ${(e as Error).message}`);
    }
  };

  const logout = async () => {
    try {
      await api.post("/api/auth/logout");
      setLoggedIn(false);
      setUsers([]);
      setMessage("已退出");
    } catch (e) {
      setMessage(`退出失败: ${(e as Error).message}`);
    }
  };

  const deleteUser = async (id: string) => {
    if (!confirm("删除该用户？")) return;
    try {
      await api.del(`/api/auth/users/${id}`);
      setMessage("已删除");
      await fetchUsers();
    } catch (e) {
      setMessage(`删除失败: ${(e as Error).message}`);
    }
  };

  if (!enabled) {
    return (
      <div className="mx-auto max-w-3xl space-y-6">
        <h2 className="text-lg font-semibold">用户认证</h2>
        <p className="text-sm text-text-muted">认证系统未启用。可在服务端配置开启。</p>
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-3xl space-y-6">
      <h2 className="text-lg font-semibold">用户认证</h2>

      {message && (
        <div className="rounded-md border border-border bg-surface px-3 py-2 text-sm text-text-soft">{message}</div>
      )}

      <div className="rounded-xl border border-border bg-bg-soft p-4">
        {loggedIn ? (
          <div className="flex items-center justify-between">
            <span className="text-sm text-text-soft">已登录</span>
            <button
              onClick={logout}
              className="flex items-center gap-1 rounded-md border border-border px-3 py-1.5 text-sm text-text-soft hover:bg-surface-hover"
            >
              <LogOut size={14} /> 退出
            </button>
          </div>
        ) : (
          <div className="space-y-3">
            <input
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="用户名"
              className="w-full rounded-md border border-border bg-bg px-3 py-2 text-sm outline-none focus:border-accent"
            />
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="密码"
              className="w-full rounded-md border border-border bg-bg px-3 py-2 text-sm outline-none focus:border-accent"
            />
            <div className="flex justify-end">
              <button
                onClick={login}
                className="flex items-center gap-1 rounded-md bg-accent px-3 py-1.5 text-sm font-medium text-bg hover:bg-accent-hover"
              >
                <LogIn size={14} /> 登录
              </button>
            </div>
          </div>
        )}
      </div>

      {users.length > 0 && (
        <div className="rounded-xl border border-border bg-bg-soft p-4">
          <h3 className="mb-3 text-sm font-medium">用户列表</h3>
          <div className="space-y-2">
            {users.map((u) => (
              <div key={u.id} className="flex items-center justify-between rounded-md bg-bg px-3 py-2 text-sm">
                <div>
                  <div className="font-medium">{u.name}</div>
                  <div className="text-xs text-text-muted">
                    {u.role} · {u.api_key_mask}
                  </div>
                </div>
                <button
                  onClick={() => deleteUser(u.id)}
                  className="rounded p-1.5 text-text-soft hover:bg-danger/10 hover:text-danger"
                >
                  <Trash2 size={15} />
                </button>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
