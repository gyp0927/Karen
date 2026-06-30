import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import ChatPage from "./pages/ChatPage";
import SettingsPage from "./pages/SettingsPage";
import "./index.css";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<ChatPage />} />
        <Route path="/settings/*" element={<SettingsPage />} />
        {
          /* Other pages land in later batches; for now render chat page for any route. */
        }
        <Route path="*" element={<ChatPage />} />
      </Routes>
    </BrowserRouter>
  </React.StrictMode>
);
