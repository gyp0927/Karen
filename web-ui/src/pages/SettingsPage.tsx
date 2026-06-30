import { useParams, Routes, Route, Navigate } from "react-router-dom";
import SettingsNav from "@/components/settings/SettingsNav";

function Placeholder({ title }: { title: string }) {
  return (
    <div className="flex h-full items-center justify-center text-text-soft">
      <span>{title} 模块开发中</span>
    </div>
  );
}

export default function SettingsPage() {
  const { section } = useParams();

  return (
    <div className="flex h-screen w-screen bg-bg text-text">
      <SettingsNav />
      <main className="flex flex-1 flex-col overflow-hidden">
        <header className="flex h-12 items-center border-b border-border bg-bg-soft px-4 text-sm font-medium">
          <span className="text-text-soft">{section ? section.toUpperCase() : "模型配置"}</span>
        </header>
        <div className="flex-1 overflow-y-auto p-6">
          <Routes>
            <Route path="models" element={<Placeholder title="模型配置" />} />
            <Route path="router" element={<Placeholder title="模型路由" />} />
            <Route path="plugins" element={<Placeholder title="插件" />} />
            <Route path="mcp" element={<Placeholder title="MCP 服务器" />} />
            <Route path="rag" element={<Placeholder title="RAG / 知识库" />} />
            <Route path="cache" element={<Placeholder title="缓存" />} />
            <Route path="auth" element={<Placeholder title="用户认证" />} />
            <Route path="*" element={<Navigate to="/settings/models" replace />} />
          </Routes>
        </div>
      </main>
    </div>
  );
}
