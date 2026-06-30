import { useParams, Routes, Route, Navigate } from "react-router-dom";
import SettingsNav from "@/components/settings/SettingsNav";

import ModelConfigPanel from "@/components/settings/ModelConfigPanel";
import RouterConfigPanel from "@/components/settings/RouterConfigPanel";
import PluginsPanel from "@/components/settings/PluginsPanel";
import McpPanel from "@/components/settings/McpPanel";
import RagPanel from "@/components/settings/RagPanel";

import CachePanel from "@/components/settings/CachePanel";
import AuthPanel from "@/components/settings/AuthPanel";

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
            <Route path="models" element={<ModelConfigPanel />} />
            <Route path="router" element={<RouterConfigPanel />} />
            <Route path="plugins" element={<PluginsPanel />} />
            <Route path="mcp" element={<McpPanel />} />
            <Route path="rag" element={<RagPanel />} />
            <Route path="cache" element={<CachePanel />} />
            <Route path="auth" element={<AuthPanel />} />
            <Route path="*" element={<Navigate to="/settings/models" replace />} />
          </Routes>
        </div>
      </main>
    </div>
  );
}
