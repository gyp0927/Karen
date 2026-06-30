import { NavLink } from "react-router-dom";
import {
  Cpu,
  Route,
  Puzzle,
  Server,
  Database,
  HardDrive,
  Users,
  MessageSquare,
  type LucideIcon,
} from "lucide-react";

interface NavItem {
  to: string;
  label: string;
  icon: LucideIcon;
}

const items: NavItem[] = [
  { to: "/settings/models", label: "模型配置", icon: Cpu },
  { to: "/settings/router", label: "模型路由", icon: Route },
  { to: "/settings/plugins", label: "插件", icon: Puzzle },
  { to: "/settings/mcp", label: "MCP 服务器", icon: Server },
  { to: "/settings/rag", label: "RAG / 知识库", icon: Database },
  { to: "/settings/cache", label: "缓存", icon: HardDrive },
  { to: "/settings/auth", label: "用户认证", icon: Users },
];

export default function SettingsNav() {
  return (
    <aside className="flex h-full w-[220px] shrink-0 flex-col border-r border-border bg-bg-soft">
      <div className="flex items-center gap-2 border-b border-border px-4 py-3">
        <span className="text-base font-semibold tracking-wide">管理后台</span>
      </div>
      <nav className="flex-1 overflow-y-auto p-2">
        {items.map((item) => {
          const Icon = item.icon;
          return (
            <NavLink
              key={item.to}
              to={item.to}
              end
              className={({ isActive }) =>
                `flex items-center gap-2 rounded-md px-3 py-2 text-sm transition ${
                  isActive
                    ? "bg-accent/10 text-accent"
                    : "text-text-soft hover:bg-surface-hover hover:text-text"
                }`
              }
            >
              <Icon size={16} />
              <span>{item.label}</span>
            </NavLink>
          );
        })}
      </nav>
      <div className="border-t border-border p-2">
        <NavLink
          to="/"
          className="flex items-center gap-2 rounded-md px-3 py-2 text-sm text-text-soft transition hover:bg-surface-hover hover:text-text"
        >
          <MessageSquare size={16} />
          <span>返回聊天</span>
        </NavLink>
      </div>
    </aside>
  );
}
