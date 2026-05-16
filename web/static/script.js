// ============ 凯伦 Client ============

// HTML escape helper to prevent XSS
function escapeHtml(text) {
  if (text == null) return "";
  return String(text)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

// AI 头像。图片放在 web/static/img/avatar.png(由用户提供)。
// object-fit: cover 让图按容器中心裁切,32x32 圆角框里只露脸部。
const ASSISTANT_AVATAR = `<img src="/static/img/avatar.png" alt="凯伦" style="width:100%;height:100%;object-fit:cover;border-radius:inherit;display:block">`;

// 在 Socket.IO 连接前先加载用户配置，使 API Key 通过 auth handshake 传递（不入日志）
let userConfig = null;
try {
  const raw = sessionStorage.getItem("user_llm_config");
  if (raw) userConfig = JSON.parse(raw);
} catch (e) {
  userConfig = null;
}

const socket = io({
  auth: userConfig ? { api_key: userConfig.apiKey } : {},
  reconnectionAttempts: 1,
  reconnectionDelay: 1000,
});

// DOM Elements
const chatMessages = document.getElementById("chatMessages");
const chatInput = document.getElementById("chatInput");
const sendBtn = document.getElementById("sendBtn");
const emptyState = document.getElementById("emptyState");
const newChatBtn = document.getElementById("newChatBtn");
const sidebar = document.getElementById("sidebar");
const sidebarToggle = document.getElementById("sidebarToggle");
const sidebarOverlay = document.getElementById("sidebarOverlay");
const sidebarHistory = document.getElementById("sidebarHistory");
const themeToggleBtn = document.getElementById("themeToggleBtn");
const themeIcon = document.getElementById("themeIcon");
const currentModelText = document.getElementById("currentModelText");
const modelHint = document.getElementById("modelHint");
const modelSwitcherBtn = document.getElementById("modelSwitcherBtn");
const modelSwitcherOverlay = document.getElementById("modelSwitcherOverlay");
const modelSwitcherClose = document.getElementById("modelSwitcherClose");
const modelSwitcherList = document.getElementById("modelSwitcherList");
const reviewBar = document.getElementById("reviewBar");
const reviewBarBtn = document.getElementById("reviewBarBtn");
const stopBtn = document.getElementById("stopBtn");
const uploadBtn = document.getElementById("uploadBtn");
const fileInput = document.getElementById("fileInput");
const fileAttachment = document.getElementById("fileAttachment");
const fileAttachmentName = document.getElementById("fileAttachmentName");
const fileAttachmentRemove = document.getElementById("fileAttachmentRemove");

// State
let isProcessing = false;
let awaitingReview = false;
let messageCount = 0;
let currentTheme = localStorage.getItem("theme") || "system";
let sessions = [];
let currentSessionId = "";
let currentMode = "fast";
let fastMode = true;
let attachedFile = null; // { filename, content }

// Preview Panel DOM (initialized later)
const previewFrame = document.getElementById("previewFrame");
let previewContent = "";
let previewFilename = "preview";

// User Config Dialog DOM
const userConfigOverlay = document.getElementById("userConfigOverlay");
const userCfgProvider = document.getElementById("userCfgProvider");
const userCfgModel = document.getElementById("userCfgModel");
const userCfgApiKey = document.getElementById("userCfgApiKey");
const userCfgSave = document.getElementById("userCfgSave");
const userCfgCancel = document.getElementById("userCfgCancel");
const userCfgToggleKey = document.getElementById("userCfgToggleKey");

// ============ Theme ============

function applyTheme(theme) {
  currentTheme = theme;
  const hljsTheme = document.getElementById("hljs-theme");
  if (theme === "system") {
    const prefersLight = window.matchMedia("(prefers-color-scheme: light)").matches;
    document.documentElement.setAttribute("data-theme", prefersLight ? "light" : "dark");
    if (hljsTheme) hljsTheme.href = prefersLight
      ? "https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github.min.css"
      : "https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css";
  } else {
    document.documentElement.setAttribute("data-theme", theme);
    if (hljsTheme) hljsTheme.href = theme === "light"
      ? "https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github.min.css"
      : "https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css";
  }
}

applyTheme(currentTheme);

// ============ Markdown Renderer ============

// 安全渲染：CDN 加载失败时回退到纯文本
function safeRenderMarkdown(text) {
  if (typeof marked === "undefined" || typeof DOMPurify === "undefined") {
    return escapeHtml(text);
  }
  try {
    return DOMPurify.sanitize(marked.parse(text), {
      ALLOWED_TAGS: ['p','div','span','h1','h2','h3','h4','h5','h6','br','hr',
        'strong','b','em','i','u','strike','del','a','img',
        'ul','ol','li','table','thead','tbody','tr','th','td',
        'pre','code','blockquote','sup','sub'],
      ALLOWED_ATTR: ['href','title','alt','src','width','height','class','id',
        'style','target','rel','lang'],
      ALLOW_DATA_ATTR: false,
    });
  } catch (e) {
    return escapeHtml(text);
  }
}

marked.setOptions({
  breaks: true,
  gfm: true,
  headerIds: false,
});

const renderer = new marked.Renderer();

renderer.code = function(code, language) {
  const lang = escapeHtml(language || "");
  const escaped = code.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  const isHtml = lang.toLowerCase() === "html" || lang.toLowerCase() === "htm";
  const codeId = "cb-" + Math.random().toString(36).substr(2, 9);

  let actions = `<button class="copy-code-btn" onclick="copyCode(this)">复制</button>`;

  if (isHtml) {
    actions += `<button class="copy-code-btn preview-btn" onclick="previewHtml(this)">预览</button>`;
  }

  // 下载下拉菜单
  const formats = [];
  if (isHtml) formats.push({ fmt: "html", label: "HTML" });
  formats.push({ fmt: "doc", label: "Word" });
  formats.push({ fmt: "txt", label: "TXT" });
  if (lang) formats.push({ fmt: lang, label: lang.toUpperCase() });

  const uniqueId = "dl-" + Math.random().toString(36).substr(2, 9);
  const options = formats.map(f =>
    `<div class="dl-option" onclick="downloadCodeBlock(this, '${f.fmt}')">${f.label}</div>`
  ).join("");

  actions += `
    <div class="dl-dropdown" id="${uniqueId}">
      <button class="copy-code-btn dl-btn" onclick="toggleDlMenu('${uniqueId}')">下载 ▼</button>
      <div class="dl-menu">${options}</div>
    </div>
  `;

  return `
    <div class="code-block-wrapper" data-lang="${lang}" data-code-id="${codeId}">
      <div class="code-block-header">
        <span>${lang}</span>
        <div class="code-actions">${actions}</div>
      </div>
      <pre><code class="hljs ${lang}">${escaped}</code></pre>
    </div>
  `;
};

marked.use({ renderer });

function copyCode(btn) {
  const code = btn.closest(".code-block-wrapper").querySelector("code").textContent;
  navigator.clipboard.writeText(code).then(() => {
    btn.textContent = "已复制";
    setTimeout(() => btn.textContent = "复制", 2000);
  });
}

window.copyCode = copyCode;

function toggleDlMenu(menuId) {
  const dropdown = document.getElementById(menuId);
  if (!dropdown) return;
  const isOpen = dropdown.classList.contains("open");
  // 关闭所有其他下拉菜单
  document.querySelectorAll(".dl-dropdown.open").forEach(d => d.classList.remove("open"));
  if (!isOpen) {
    dropdown.classList.add("open");
  }
}

window.toggleDlMenu = toggleDlMenu;

// 点击页面其他地方关闭下拉菜单
document.addEventListener("click", (e) => {
  if (!e.target.closest(".dl-dropdown")) {
    document.querySelectorAll(".dl-dropdown.open").forEach(d => d.classList.remove("open"));
  }
});

function getCodeFromBtn(btn) {
  const wrapper = btn.closest(".code-block-wrapper");
  return wrapper.querySelector("code").textContent;
}

function getLangFromBtn(btn) {
  const wrapper = btn.closest(".code-block-wrapper");
  return wrapper.dataset.lang || "";
}

async function downloadCodeBlock(btnOrEl, format) {
  const code = getCodeFromBtn(btnOrEl);
  const lang = getLangFromBtn(btnOrEl);
  const fmt = format.toLowerCase();

  // 关闭下拉菜单
  document.querySelectorAll(".dl-dropdown.open").forEach(d => d.classList.remove("open"));

  try {
    const resp = await fetch("/api/generate-file", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        content: code,
        filename: `code-export-${Date.now()}`,
        format: fmt
      })
    });
    const data = await resp.json();
    if (data.success) {
      // 触发下载
      const a = document.createElement("a");
      a.href = data.download_url;
      a.download = data.filename;
      document.body.appendChild(a);
      a.click();
      a.remove();
      showToast(`已下载: ${data.filename}`, "success");
    } else {
      showToast(data.message || "下载失败", "error");
    }
  } catch (e) {
    showToast("下载失败: " + e.message, "error");
  }
}

window.downloadCodeBlock = downloadCodeBlock;

function previewHtml(btn) {
  try {
    let code = getCodeFromBtn(btn);
    if (!code) {
      showToast("无法获取代码内容", "error");
      return;
    }

    // 消毒：移除危险标签和事件处理器，防止 XSS
    if (typeof DOMPurify !== "undefined") {
      code = DOMPurify.sanitize(code, {
        ALLOWED_TAGS: ['p','div','span','h1','h2','h3','h4','h5','h6','br','hr',
          'strong','b','em','i','u','strike','del','a','img',
          'ul','ol','li','table','thead','tbody','tr','th','td',
          'pre','code','blockquote','sup','sub','style'],
        ALLOWED_ATTR: ['href','title','alt','src','width','height','class','id',
          'style','target','rel','charset','name','content','lang'],
        ALLOW_DATA_ATTR: false,
      });
    }

    previewContent = code;
    previewFilename = `preview-${Date.now()}.html`;

    // 如果不是完整的 HTML 文档，自动包装
    const hasDocType = code.trim().toLowerCase().startsWith("<!doctype") || code.trim().toLowerCase().startsWith("<html");
    if (!hasDocType) {
      code = `<!DOCTYPE html>\n<html lang="zh-CN">\n<head>\n  <meta charset="UTF-8">\n  <meta name="viewport" content="width=device-width, initial-scale=1.0">\n  <title>预览</title>\n  <style>\n    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; padding: 20px; line-height: 1.6; }\n  </style>\n</head>\n<body>\n${code}\n</body>\n</html>`;
    }

    if (!previewPanel) {
      showToast("预览面板未初始化", "error");
      return;
    }
    if (!previewFrame) {
      showToast("预览框架未初始化", "error");
      return;
    }

    // 写入 iframe
    const doc = previewFrame.contentDocument || previewFrame.contentWindow.document;
    if (!doc) {
      showToast("无法访问预览框架", "error");
      return;
    }
    doc.open();
    doc.write(code);
    doc.close();

    // 展开右侧面板
    previewPanel.classList.add("open");
    showToast("预览已打开", "success");
  } catch (err) {
    showToast("预览出错: " + err.message, "error");
    console.error("previewHtml error:", err);
  }
}

window.previewHtml = previewHtml;

// Preview Panel DOM
const previewPanel = document.getElementById("previewPanel");
const previewPanelClose = document.getElementById("previewPanelClose");
const previewPanelDl = document.getElementById("previewPanelDl");

if (previewPanelClose) {
  previewPanelClose.addEventListener("click", () => {
    previewPanel.classList.remove("open");
    previewFrame.src = "about:blank";
    previewContent = "";
  });
}

if (previewPanelDl) {
  previewPanelDl.addEventListener("click", async () => {
    if (!previewContent) return;
    try {
      const resp = await fetch("/api/generate-file", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ content: previewContent, filename: "preview-page", format: "html" })
      });
      const data = await resp.json();
      if (data.success) {
        const a = document.createElement("a");
        a.href = data.download_url;
        a.download = data.filename;
        document.body.appendChild(a);
        a.click();
        a.remove();
        showToast(`已下载: ${data.filename}`, "success");
      }
    } catch (e) {
      showToast("下载失败", "error");
    }
  });
}

// ============ UI Functions ============

function autoResize(textarea) {
  textarea.style.height = "auto";
  textarea.style.height = Math.min(textarea.scrollHeight, 200) + "px";
  sendBtn.disabled = !textarea.value.trim();
}

window.autoResize = autoResize;

function setInput(text) {
  chatInput.value = text;
  autoResize(chatInput);
  chatInput.focus();
}

window.setInput = setInput;

function showEmptyState(show) {
  emptyState.style.display = show ? "block" : "none";
  chatMessages.style.display = show ? "none" : "block";
}

function appendMessage(content, type, sender, stream = false) {
  showEmptyState(false);
  messageCount++;

  const div = document.createElement("div");
  div.className = `message ${type}`;

  let avatar = ASSISTANT_AVATAR;
  if (type === "user") {
    avatar = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>`;
  }
  else if (type === "assistant") { avatar = ASSISTANT_AVATAR; }
  else if (type === "review") { avatar = "👀"; }

  if (type === "user") {
    const htmlContent = escapeHtml(content).replace(/\n/g, "<br>");
    // User message: avatar on the right (via CSS flex-direction: row-reverse)
    div.innerHTML = `
      <div class="message-inner">
        <div class="message-avatar" title="你">${avatar}</div>
        <div class="message-content">${htmlContent}</div>
      </div>
    `;
    chatMessages.appendChild(div);
    chatMessages.scrollTop = chatMessages.scrollHeight;
  } else {
    // AI 消息：创建容器，流式显示 (avatar on the left by default)
    div.innerHTML = `
      <div class="message-inner">
        <div class="message-avatar" title="${sender || '助手'}">${avatar}</div>
        <div class="message-content"></div>
      </div>
      <div class="token-footer" id="token-footer-${messageCount}">
        <span class="token-item">输入: <span class="token-value prompt-tokens">-</span></span>
        <span class="token-item">输出: <span class="token-value completion-tokens">-</span></span>
        <span class="token-item">总计: <span class="token-value total-tokens">-</span></span>
        <span class="token-item">费用: $<span class="token-value cost-usd">-</span></span>
        <span class="token-item">耗时: <span class="token-value duration-ms">-</span>ms</span>
      </div>
    `;
    chatMessages.appendChild(div);
    chatMessages.scrollTop = chatMessages.scrollHeight;

    div.querySelector(".message-content").innerHTML = safeRenderMarkdown(content);
    div.querySelectorAll("pre code").forEach(block => {
      hljs.highlightElement(block);
    });
  }

  return div;
}

function clearMessages() {
  chatMessages.innerHTML = "";
  messageCount = 0;
}

function showThinking(text) {
  showEmptyState(false);
  // 如果已有 thinking 指示器，只更新文本，避免重复创建
  let div = document.getElementById("thinkingIndicator");
  if (div) {
    const span = div.querySelector(".thinking-msg span");
    if (span) span.textContent = text;
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return;
  }
  div = document.createElement("div");
  div.className = "message assistant";
  div.id = "thinkingIndicator";
  div.innerHTML = `
    <div class="message-inner">
      <div class="message-avatar">${ASSISTANT_AVATAR}</div>
      <div class="message-content">
        <div class="thinking-msg">
          <div class="thinking-dots"><span></span><span></span><span></span></div>
          <span>${escapeHtml(text)}</span>
        </div>
      </div>
    </div>
  `;
  chatMessages.appendChild(div);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function clearThinking() {
  const el = document.getElementById("thinkingIndicator");
  if (el) el.remove();
}

function updateInputHint() {
  if (awaitingReview) {
    chatInput.placeholder = "发送消息继续对话，或点击上方按钮审查...";
    if (reviewBar) reviewBar.style.display = "flex";
  } else {
    chatInput.placeholder = "发送消息...";
    if (reviewBar) reviewBar.style.display = "none";
  }
}

function showToast(message, type = "info") {
  const toast = document.createElement("div");
  toast.className = `toast ${type}`;
  toast.textContent = message;
  document.body.appendChild(toast);
  setTimeout(() => toast.remove(), 3000);
}

// ============ Sidebar ============

function toggleSidebar() {
  sidebar.classList.toggle("open");
  sidebarOverlay.classList.toggle("show");
}

sidebarToggle.addEventListener("click", toggleSidebar);
sidebarOverlay.addEventListener("click", toggleSidebar);

newChatBtn.addEventListener("click", () => {
  socket.emit("new_session");
});

if (themeToggleBtn) {
  themeToggleBtn.addEventListener("click", () => {
    const order = ["light", "dark", "system"];
    const next = order[(order.indexOf(currentTheme) + 1) % order.length];
    currentTheme = next;
    localStorage.setItem("theme", next);
    applyTheme(next);
    const icons = { light: "☀️", dark: "🌙", system: "💻" };
    if (themeIcon) themeIcon.textContent = icons[next] || "💻";
  });
}

// File upload
if (uploadBtn && fileInput) {
  uploadBtn.addEventListener("click", () => {
    fileInput.click();
  });
}

if (fileInput) {
  fileInput.addEventListener("change", async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    showToast("正在解析文件...", "info");

    try {
      const resp = await fetch("/api/upload", { method: "POST", body: formData });
      const data = await resp.json();
      if (data.success) {
        attachedFile = { filename: data.filename, content: data.content };
        showFileAttachment(data.filename);
        showToast(`文件 "${data.filename}" 已上传`, "success");
      } else {
        showToast(data.message || "上传失败", "error");
      }
    } catch (err) {
      showToast("上传失败: " + err.message, "error");
    }

    fileInput.value = "";
  });
}

function showFileAttachment(filename) {
  if (fileAttachmentName) fileAttachmentName.textContent = filename;
  if (fileAttachment) fileAttachment.style.display = "flex";
}

function hideFileAttachment() {
  attachedFile = null;
  if (fileAttachment) fileAttachment.style.display = "none";
}

if (fileAttachmentRemove) {
  fileAttachmentRemove.addEventListener("click", () => {
    hideFileAttachment();
  });
}

// ============ Session Management ============

function renderSessionList() {
  if (!sidebarHistory) return;

  if (sessions.length === 0) {
    sidebarHistory.innerHTML = "";
    return;
  }

  sidebarHistory.innerHTML = sessions.map(s => `
    <div class="history-item ${s.is_current ? 'active' : ''}" data-session-id="${escapeHtml(s.id)}">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
      </svg>
      <span class="history-title">${escapeHtml(s.title)}</span>
      <button class="delete-session-btn" data-session-id="${escapeHtml(s.id)}" title="删除">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <line x1="18" y1="6" x2="6" y2="18"></line>
          <line x1="6" y1="6" x2="18" y2="18"></line>
        </svg>
      </button>
    </div>
  `).join("");
}

// 事件委托:一次性在 sidebarHistory 上挂监听,避免每次 renderSessionList 都重绑 N 个 listener。
// renderSessionList 走 innerHTML 重建,旧 DOM 节点 GC 时 listener 自动失效;委托后单点处理。
if (sidebarHistory) {
  sidebarHistory.addEventListener("click", (e) => {
    const deleteBtn = e.target.closest(".delete-session-btn");
    if (deleteBtn) {
      e.stopPropagation();
      const sid = deleteBtn.dataset.sessionId;
      if (sid && confirm("确定要删除这个对话吗？")) {
        socket.emit("delete_session", { session_id: sid });
      }
      return;
    }
    const item = e.target.closest(".history-item");
    if (item) {
      const sid = item.dataset.sessionId;
      if (sid && sid !== currentSessionId) {
        socket.emit("switch_session", { session_id: sid });
      }
    }
  });
}

// ============ Socket Events ============

// 页面离开前保存当前会话 ID，用于返回时恢复
window.addEventListener("beforeunload", () => {
  if (currentSessionId) {
    sessionStorage.setItem("last_session_id", currentSessionId);
  }
});

socket.on("connect", () => {
  console.log("Connected");
  // Socket.IO 5.x 自动恢复:断线重连时若 server 端 session 还在,会派发 connect 但
  // recovered=true,此时不能再发 new_session,否则会把用户对话切到一个新空会话。
  if (socket.recovered) {
    socket.emit("get_sessions");
    socket.emit("get_mode");
    if (userConfig) socket.emit("set_user_config", userConfig);
    return;
  }
  const lastSid = sessionStorage.getItem("last_session_id");
  if (lastSid) {
    // 从配置页等返回：恢复之前的会话
    sessionStorage.removeItem("last_session_id");
    socket.emit("switch_session", { session_id: lastSid });
  } else {
    // 首次进入程序：创建新会话
    socket.emit("new_session");
  }
  socket.emit("get_sessions");
  socket.emit("get_mode");
  // 如果有本地用户配置，发送给后端
  if (userConfig) {
    socket.emit("set_user_config", userConfig);
  }
});

socket.on("connect_error", (err) => {
  showToast("连接失败: " + err.message, "error");
});

socket.on("thinking", (data) => {
  showThinking(data.message);
});

socket.on("user_message", (data) => {
  appendMessage(data.message, "user", "你");
  socket.emit("get_sessions");
});

// ============ Streaming Output (typewriter-style 平滑揭示) ============
let currentStreamElement = null;
let streamBuffer = "";           // 已收到的完整文本(用于 stream_end markdown 渲染)
let _streamTextNode = null;
let _pendingChars = "";          // 已到达但还没逐字显示给用户的字符
let _revealRAF = null;           // 平滑揭示循环的 RAF 句柄
let _revealLastTs = 0;
let _streamingActive = false;
let _autoFollow = true;          // 流式时是否自动跟随到底部(用户主动上滚后变 false)
let _sendMessageTimeout = null;  // send_message 超时保护计时器

// 基础揭示速度(每秒多少字符)。50~60 接近 ChatGPT 视觉感受。
const REVEAL_BASE_CPS = 55;

// 监听滚动:用户向上滚停止跟随;回到接近底部恢复跟随
if (chatMessages) {
  chatMessages.addEventListener("wheel", (e) => {
    if (e.deltaY < 0) {
      _autoFollow = false;
    }
  }, { passive: true });
  chatMessages.addEventListener("touchmove", () => {
    // 触屏上滑无法直接判方向,用滚动位置判
    const distFromBottom = chatMessages.scrollHeight - chatMessages.scrollTop - chatMessages.clientHeight;
    if (distFromBottom > 80) _autoFollow = false;
  }, { passive: true });
  chatMessages.addEventListener("scroll", () => {
    const distFromBottom = chatMessages.scrollHeight - chatMessages.scrollTop - chatMessages.clientHeight;
    if (distFromBottom < 40) _autoFollow = true;
  }, { passive: true });
}

function _revealStep(now) {
  if (!_streamTextNode) {
    _revealRAF = null;
    return;
  }
  const last = _revealLastTs || now;
  const dt = now - last;
  _revealLastTs = now;

  const queueLen = _pendingChars.length;
  // 自适应速度:队列越长揭示越快,避免越落越多
  let cps = REVEAL_BASE_CPS;
  if (queueLen > 240) {
    cps = Infinity;                     // 严重落后,直接一次性刷出
  } else if (queueLen > 120) {
    cps = REVEAL_BASE_CPS * 4;
  } else if (queueLen > 60) {
    cps = REVEAL_BASE_CPS * 2;
  } else if (queueLen > 30) {
    cps = REVEAL_BASE_CPS * 1.5;
  }

  let reveal;
  if (cps === Infinity) {
    reveal = queueLen;
  } else {
    reveal = Math.max(1, Math.floor((cps * dt) / 1000));
  }
  if (reveal > queueLen) reveal = queueLen;

  if (reveal > 0) {
    const slice = _pendingChars.slice(0, reveal);
    _pendingChars = _pendingChars.slice(reveal);
    _streamTextNode.appendData(slice);
    if (_autoFollow) {
      chatMessages.scrollTop = chatMessages.scrollHeight;
    }
  }

  if (_pendingChars.length > 0 || _streamingActive) {
    _revealRAF = requestAnimationFrame(_revealStep);
  } else {
    _revealRAF = null;
    _revealLastTs = 0;
  }
}

function _ensureRevealRunning() {
  if (_revealRAF === null) {
    _revealLastTs = 0;
    _revealRAF = requestAnimationFrame(_revealStep);
  }
}

// 流式状态统一复位:错误中断 / 切换会话 / 新建会话 / 删除会话都要调用,
// 否则 _streamTextNode 残留在被 clearMessages 移除的 DOM 节点上,
// 下一次 token_chunk 会往一个游离节点写字,造成"流式回复看不见"。
function resetStreamState() {
  _streamingActive = false;
  if (_revealRAF !== null) {
    cancelAnimationFrame(_revealRAF);
    _revealRAF = null;
    _revealLastTs = 0;
  }
  if (currentStreamElement) {
    currentStreamElement.classList.remove("streaming");
    currentStreamElement = null;
  }
  _streamTextNode = null;
  _pendingChars = "";
  streamBuffer = "";
  if (_sendMessageTimeout) {
    clearTimeout(_sendMessageTimeout);
    _sendMessageTimeout = null;
  }
}

socket.on("stream_start", (data) => {
  clearThinking();
  showEmptyState(false);
  messageCount++;

  const div = document.createElement("div");
  div.className = "message assistant streaming";
  div.innerHTML = `
    <div class="message-inner">
      <div class="message-avatar" title="助手">${ASSISTANT_AVATAR}</div>
      <div class="message-content"></div>
    </div>
  `;
  chatMessages.appendChild(div);
  chatMessages.scrollTop = chatMessages.scrollHeight;
  _autoFollow = true;          // 新一轮回复默认自动跟随到底部

  currentStreamElement = div;
  streamBuffer = "";
  _pendingChars = "";
  _streamingActive = true;
  const contentEl = div.querySelector(".message-content");
  _streamTextNode = document.createTextNode("");
  contentEl.appendChild(_streamTextNode);
});

socket.on("token_chunk", (data) => {
  if (!currentStreamElement || !_streamTextNode) return;
  streamBuffer += data.token;
  _pendingChars += data.token;
  _ensureRevealRunning();
});

socket.on("stream_end", (data) => {
  clearThinking();
  _streamingActive = false;
  if (_sendMessageTimeout) {
    clearTimeout(_sendMessageTimeout);
    _sendMessageTimeout = null;
  }
  // 把还没揭示完的字立刻补完,再用 markdown 替换
  if (_pendingChars.length > 0 && _streamTextNode) {
    _streamTextNode.appendData(_pendingChars);
    _pendingChars = "";
  }
  if (_revealRAF !== null) {
    cancelAnimationFrame(_revealRAF);
    _revealRAF = null;
    _revealLastTs = 0;
  }
  if (currentStreamElement) {
    const contentEl = currentStreamElement.querySelector(".message-content");
    if (contentEl) {
      const finalText = data.message || streamBuffer;
      contentEl.innerHTML = safeRenderMarkdown(finalText);
      contentEl.querySelectorAll("pre code").forEach(block => {
        hljs.highlightElement(block);
      });
    }
    currentStreamElement.classList.remove("streaming");
    currentStreamElement = null;
  }
  _streamTextNode = null;
  streamBuffer = "";
  awaitingReview = data.awaiting_review;
  updateInputHint();
  isProcessing = false;
  sendBtn.style.display = "flex";
  sendBtn.disabled = !chatInput.value.trim();
  if (stopBtn) stopBtn.style.display = "none";
  chatInput.focus();
});

socket.on("token_usage", (data) => {
  // 更新最后一条 AI 消息的 token footer
  const msgs = chatMessages.querySelectorAll(".message.assistant");
  const lastMsg = msgs[msgs.length - 1];
  if (!lastMsg) return;
  const footer = lastMsg.querySelector(".token-footer");
  if (!footer) return;
  footer.classList.add("show");
  const pt = footer.querySelector(".prompt-tokens");
  const ct = footer.querySelector(".completion-tokens");
  const tt = footer.querySelector(".total-tokens");
  const cost = footer.querySelector(".cost-usd");
  const dur = footer.querySelector(".duration-ms");
  if (pt) pt.textContent = (data.prompt_tokens || 0).toLocaleString();
  if (ct) ct.textContent = (data.completion_tokens || 0).toLocaleString();
  if (tt) tt.textContent = (data.total_tokens || 0).toLocaleString();
  if (cost) cost.textContent = (data.cost_usd || 0).toFixed(6);
  if (dur) dur.textContent = data.duration_ms || 0;
});

socket.on("bot_history", (data) => {
  appendMessage(data.message, "assistant", data.sender);
});

socket.on("review_history", (data) => {
  appendMessage(data.review_result, "review", "审查者");
});

socket.on("history_restored", (data) => {
  if (data.awaiting_review) {
    awaitingReview = true;
    updateInputHint();
  }
});

socket.on("review_complete", (data) => {
  clearThinking();
  appendMessage(data.review_result, "review", "审查者");
  awaitingReview = false;
  updateInputHint();
  isProcessing = false;
  sendBtn.style.display = "flex";
  sendBtn.disabled = !chatInput.value.trim();
  if (stopBtn) stopBtn.style.display = "none";
});

socket.on("message_failed", (data) => {
  // 发送失败：删除已显示的最后一条用户消息，恢复输入框内容
  const userMessages = chatMessages.querySelectorAll(".message.user");
  if (userMessages.length > 0) {
    userMessages[userMessages.length - 1].remove();
    messageCount--;
  }
  // 把失败的消息重新填回输入框，方便用户重试
  if (data.message && !chatInput.value.trim()) {
    chatInput.value = data.message;
    autoResize(chatInput);
  }
  showToast(data.error || "发送失败", "error");
  isProcessing = false;
  sendBtn.style.display = "flex";
  sendBtn.disabled = !chatInput.value.trim();
  if (stopBtn) stopBtn.style.display = "none";
  clearThinking();
  resetStreamState();
});

socket.on("error", (data) => {
  showToast(data.message, "error");
  isProcessing = false;
  sendBtn.style.display = "flex";
  sendBtn.disabled = !chatInput.value.trim();
  if (stopBtn) stopBtn.style.display = "none";
  clearThinking();
  resetStreamState();
});

// Session events
socket.on("sessions_list", (data) => {
  sessions = data.sessions || [];
  const current = sessions.find(s => s.is_current);
  if (current) currentSessionId = current.id;
  renderSessionList();
});

socket.on("session_created", (data) => {
  resetStreamState();
  sessions = data.sessions || [];
  currentSessionId = data.session_id;
  clearMessages();
  showEmptyState(true);
  awaitingReview = false;
  updateInputHint();
  renderSessionList();
  showToast("已开启新对话");
});

socket.on("session_switched", (data) => {
  resetStreamState();
  sessions = data.sessions || [];
  currentSessionId = data.session_id;
  clearMessages();
  awaitingReview = false;
  updateInputHint();
  renderSessionList();
  // History will be sent via history_restored
});

socket.on("session_deleted", (data) => {
  resetStreamState();
  sessions = data.sessions || [];
  const current = sessions.find(s => s.is_current);
  if (current) currentSessionId = current.id;
  clearMessages();
  showEmptyState(true);
  awaitingReview = false;
  updateInputHint();
  renderSessionList();
  // 清理附件等会话相关前端状态
  attachedFile = null;
  hideFileAttachment();
  // History will be sent via history_restored for the new current session
});

socket.on("model_info", (data) => {
  const name = data.name || data.provider_name || data.provider;
  currentModelText.textContent = name;
  modelHint.textContent = (data.provider_name || data.provider) + " · " + data.model;
  // 只有当服务器没有任何配置且浏览器本地也没有配置时，才提示配置
  if (!data.server_has_config && !userConfig) {
    // 等待一小段时间再弹出，避免页面刚加载就弹窗
    setTimeout(() => {
      if (!userConfig) showUserConfigDialog();
    }, 500);
  }
});

socket.on("config_activated", (data) => {
  currentModelText.textContent = data.name;
  modelHint.textContent = data.provider_name + " · " + data.model;
  showToast(data.message);
  closeModelSwitcher();
});

socket.on("configs_list", (data) => {
  renderModelSwitcherList(data.configs || [], data.activeConfigId);
});

socket.on("user_config_set", (data) => {
  if (data.success) {
    currentModelText.textContent = data.name;
    modelHint.textContent = data.provider_name + " · " + data.model;
    showToast("配置已保存");
  }
});

socket.on("config_error", (data) => {
  showToast(data.message, "error");
  showUserConfigDialog(true);
});

socket.on("config_required", (data) => {
  resetToIdle();
  // 如果本地有用户配置，重新发送给后端
  if (userConfig) {
    socket.emit("set_user_config", userConfig);
    showToast("正在同步配置...", "info");
    return;
  }
  // 否则弹出用户配置对话框
  showUserConfigDialog();
});

socket.on("generation_stopped", (data) => {
  resetToIdle();
  showToast("生成已停止", "info");
});

socket.on("generation_stopping", (data) => {
  showToast("正在停止...", "info");
});

// ============ Event Handlers ============

const SEND_MESSAGE_TIMEOUT_MS = 120000;

function sendMessage() {
  const message = chatInput.value.trim();
  if (!message || isProcessing) return;

  isProcessing = true;
  sendBtn.style.display = "none";
  if (stopBtn) stopBtn.style.display = "flex";
  chatInput.value = "";
  chatInput.style.height = "auto";

  const payload = { message };
  if (attachedFile) {
    payload.document_context = attachedFile.content;
    hideFileAttachment();
  }

  socket.emit("send_message", payload);

  _sendMessageTimeout = setTimeout(() => {
    showToast("请求超时，请重试", "error");
    resetStreamState();
    resetToIdle();
  }, SEND_MESSAGE_TIMEOUT_MS);
}

sendBtn.addEventListener("click", sendMessage);

if (stopBtn) {
  stopBtn.addEventListener("click", () => {
    socket.emit("stop_generation");
    resetToIdle();
  });
}

function resetToIdle() {
  isProcessing = false;
  sendBtn.style.display = "flex";
  sendBtn.disabled = true;
  if (stopBtn) stopBtn.style.display = "none";
  clearThinking();
}

if (reviewBarBtn) {
  reviewBarBtn.addEventListener("click", () => {
    if (!awaitingReview || isProcessing) return;
    isProcessing = true;
    sendBtn.disabled = true;
    if (reviewBar) reviewBar.style.display = "none";
    showThinking("审查者正在检查回答...");
    socket.emit("trigger_review");
  });
}

chatInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

// ============ Model Switcher ============

function openModelSwitcher() {
  modelSwitcherOverlay.style.display = "flex";
  socket.emit("get_configs");
}

function closeModelSwitcher() {
  modelSwitcherOverlay.style.display = "none";
}

function renderModelSwitcherList(configs, activeId) {
  const PROVIDER_ICONS = {
    ollama: "🦙", deepseek: "🔵", qwen: "🐰", minimax: "⚡", doubao: "🦜",
    glm: "📊", ernie: "🟠", hunyuan: "🐧", spark: "✨", kimi: "🌙",
    siliconflow: "🌊", "kimi-code": "💻", openai: "🤖", anthropic: "🧠",
    gemini: "💎", grok: "🐦", mistral: "🌬️", cohere: "🔗", perplexity: "🔍",
    groq: "⚡", together: "🤝", azure: "☁️"
  };

  const PROVIDER_NAMES = {
    ollama: "Ollama 本地", deepseek: "DeepSeek", qwen: "阿里 Qwen",
    minimax: "MiniMax", doubao: "字节豆包", glm: "智谱 GLM",
    ernie: "百度文心", hunyuan: "腾讯混元", spark: "讯飞星火",
    kimi: "月之暗面 Kimi", siliconflow: "硅基流动", "kimi-code": "Kimi Code",
    openai: "OpenAI", anthropic: "Anthropic", gemini: "Google Gemini",
    grok: "xAI Grok", mistral: "Mistral AI", cohere: "Cohere",
    perplexity: "Perplexity", groq: "Groq", together: "Together AI",
    azure: "Azure OpenAI"
  };

  if (configs.length === 0) {
    modelSwitcherList.innerHTML = `
      <div class="model-switcher-empty">
        <div class="icon">⚙️</div>
        <div>还没有保存的模型配置</div>
        <a href="/config" style="color: var(--accent); margin-top: 8px; display: inline-block;">去添加配置</a>
      </div>
    `;
    return;
  }

  modelSwitcherList.innerHTML = configs.map(cfg => {
    const isActive = cfg.id === activeId;
    const icon = PROVIDER_ICONS[cfg.provider] || "🤖";
    const pName = PROVIDER_NAMES[cfg.provider] || cfg.provider;
    return `
      <div class="model-switcher-item ${isActive ? 'active' : ''}" data-config-id="${escapeHtml(cfg.id)}">
        <div class="model-switcher-icon">${icon}</div>
        <div class="model-switcher-info">
          <div class="model-switcher-name">${escapeHtml(cfg.name)}</div>
          <div class="model-switcher-detail">${pName} · ${escapeHtml(cfg.model)}</div>
        </div>
        ${isActive ? '<div class="model-switcher-check">✓</div>' : ''}
      </div>
    `;
  }).join("");

  modelSwitcherList.querySelectorAll(".model-switcher-item").forEach(item => {
    item.addEventListener("click", () => {
      const configId = item.dataset.configId;
      if (configId !== activeId) {
        socket.emit("activate_config", { configId });
      } else {
        closeModelSwitcher();
      }
    });
  });
}

if (modelSwitcherBtn) {
  modelSwitcherBtn.addEventListener("click", openModelSwitcher);
}

if (modelSwitcherClose) {
  modelSwitcherClose.addEventListener("click", closeModelSwitcher);
}

if (modelSwitcherOverlay) {
  modelSwitcherOverlay.addEventListener("click", (e) => {
    if (e.target === modelSwitcherOverlay) closeModelSwitcher();
  });
}

// ============ User Config (LAN multi-user) ============

const PROVIDER_MODEL_HINTS = {
  deepseek: "deepseek-chat, deepseek-reasoner",
  openai: "gpt-4o, gpt-4o-mini, gpt-3.5-turbo",
  anthropic: "claude-sonnet-4, claude-opus-4, claude-3-7-sonnet",
  gemini: "gemini-2.0-flash, gemini-1.5-pro",
  qwen: "qwen-plus, qwen-max, qwen-turbo",
  kimi: "kimi-k2.6, kimi-latest",
  siliconflow: "deepseek-ai/DeepSeek-V3, Qwen/Qwen2.5-72B-Instruct",
  glm: "glm-4-plus, glm-4-flash",
  ollama: "llama3.2, llama3.1:8b, qwen2.5",
  grok: "grok-3, grok-3-mini",
  mistral: "mistral-large-latest, mistral-small-latest",
  groq: "llama-3.3-70b-versatile, mixtral-8x7b-32768",
  azure: "gpt-4o, gpt-4o-mini"
};

function saveUserConfig(cfg) {
  userConfig = cfg;
  sessionStorage.setItem("user_llm_config", JSON.stringify(cfg));
  if (socket) {
    socket.auth = userConfig ? { api_key: userConfig.apiKey } : {};
  }
}

function clearUserConfig() {
  userConfig = null;
  sessionStorage.removeItem("user_llm_config");
  if (socket) {
    socket.auth = {};
  }
}

function sendUserConfigToServer() {
  if (userConfig && socket.connected) {
    socket.emit("set_user_config", userConfig);
  }
}

function showUserConfigDialog(prefill = false) {
  userConfigOverlay.style.display = "flex";
  if (prefill && userConfig) {
    userCfgProvider.value = userConfig.provider || "deepseek";
    userCfgModel.value = userConfig.model || "";
    userCfgApiKey.value = userConfig.apiKey || "";
  } else {
    userCfgProvider.value = "deepseek";
    userCfgModel.value = "";
    userCfgApiKey.value = "";
  }
  updateModelHint();
  userCfgModel.focus();
}

function hideUserConfigDialog() {
  userConfigOverlay.style.display = "none";
}

function updateModelHint() {
  const provider = userCfgProvider.value;
  const hint = PROVIDER_MODEL_HINTS[provider] || "";
  const hintEl = document.getElementById("userCfgModelHint");
  if (hintEl) {
    hintEl.textContent = hint ? `💡 常见模型：${hint}` : "";
  }
  // Ollama 不需要 API Key
  const keyHint = document.getElementById("userCfgKeyHint");
  if (keyHint) {
    if (provider === "ollama") {
      keyHint.innerHTML = "🦙 Ollama 本地运行无需 API Key";
      userCfgApiKey.placeholder = "无需 API Key";
    } else {
      keyHint.innerHTML = "🔒 仅保存在本地浏览器，不会上传到服务器";
      userCfgApiKey.placeholder = "输入你的 API Key";
    }
  }
}

function handleUserConfigSave() {
  const provider = userCfgProvider.value;
  const model = userCfgModel.value.trim();
  const apiKey = userCfgApiKey.value.trim();

  if (!model) {
    showToast("请填写模型名称", "error");
    return;
  }
  if (provider !== "ollama" && !apiKey) {
    showToast("请输入 API Key", "error");
    return;
  }

  const cfg = { provider, model, apiKey };
  saveUserConfig(cfg);
  sendUserConfigToServer();
  hideUserConfigDialog();
}

// User config dialog events
if (userCfgProvider) {
  userCfgProvider.addEventListener("change", updateModelHint);
}

if (userCfgSave) {
  userCfgSave.addEventListener("click", handleUserConfigSave);
}

if (userCfgCancel) {
  userCfgCancel.addEventListener("click", hideUserConfigDialog);
}

if (userCfgToggleKey) {
  userCfgToggleKey.addEventListener("click", () => {
    userCfgApiKey.type = userCfgApiKey.type === "password" ? "text" : "password";
  });
}

if (userConfigOverlay) {
  userConfigOverlay.addEventListener("click", (e) => {
    if (e.target === userConfigOverlay) hideUserConfigDialog();
  });
}

// ============ Extensions Panel ============

const extensionsBtn = document.getElementById("extensionsBtn");
const extensionsOverlay = document.getElementById("extensionsOverlay");
const extensionsContent = document.getElementById("extensionsContent");
const extensionsClose = document.getElementById("extensionsClose");

function showExtensionsPanel() {
  if (extensionsOverlay) extensionsOverlay.style.display = "flex";
  loadExtensionsStatus();
}

function hideExtensionsPanel() {
  if (extensionsOverlay) extensionsOverlay.style.display = "none";
}

async function loadExtensionsStatus() {
  if (!extensionsContent) return;
  extensionsContent.innerHTML = '<div style="text-align:center; color: var(--text-secondary);">加载中...</div>';

  try {
    const [pluginsRes, cacheRes, routerRes, ragRes] = await Promise.all([
      fetch("/api/plugins").then(r => r.json()).catch(() => ({ success: false })),
      fetch("/api/cache/stats").then(r => r.json()).catch(() => ({ success: false })),
      fetch("/api/router/status").then(r => r.json()).catch(() => ({ success: false })),
      fetch("/api/rag/stats").then(r => r.json()).catch(() => ({ success: false })),
    ]);

    let html = '';

    // 插件系统
    html += '<div style="margin-bottom: 20px; padding: 12px; background: var(--bg-secondary); border-radius: 8px; border: 1.5px solid var(--input-border);">';
    html += '<h4 style="margin: 0 0 10px 0; color: var(--accent);">🔌 插件系统</h4>';
    if (pluginsRes.success && pluginsRes.plugins) {
      if (pluginsRes.plugins.length === 0) {
        html += '<p style="margin:0; color: var(--text-secondary); font-size: 13px;">暂无插件。将 .py 文件放入 plugins/ 文件夹后重启即可加载。</p>';
      } else {
        pluginsRes.plugins.forEach(p => {
          const btnText = p.enabled ? '禁用' : '启用';
          const btnColor = p.enabled ? '#ff6b6b' : '#51cf66';
          const safeName = escapeHtml(p.name);
          const safeDesc = escapeHtml(p.description);
          html += `<div style="display:flex; justify-content:space-between; align-items:center; margin: 6px 0; padding: 8px; background: var(--bg-primary); border-radius: 4px;">`;
          html += `<div style="flex:1; min-width:0;"><div style="font-size: 13px; font-weight:500;">${safeName}</div><div style="font-size: 12px; color: var(--text-secondary); overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">${safeDesc}</div></div>`;
          html += `<button onclick="togglePlugin('${safeName.replace(/'/g, "\\'")}', ${!p.enabled})" style="margin-left:8px; padding: 4px 10px; background: ${btnColor}; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 12px; white-space:nowrap;">${btnText}</button>`;
          html += `</div>`;
        });
      }
    } else {
      html += '<p style="margin:0; color: var(--text-secondary); font-size: 13px;">插件系统不可用</p>';
    }
    html += '</div>';

    // 缓存层
    html += '<div style="margin-bottom: 20px; padding: 12px; background: var(--bg-secondary); border-radius: 8px; border: 1.5px solid var(--input-border);">';
    html += '<h4 style="margin: 0 0 10px 0; color: var(--accent);">💾 响应缓存</h4>';
    if (cacheRes.success && cacheRes.stats) {
      const s = cacheRes.stats;
      const toggleText = s.enabled ? '禁用' : '启用';
      const toggleColor = s.enabled ? '#ff6b6b' : '#51cf66';
      html += `<div style="display:flex; justify-content:space-between; align-items:center; margin: 4px 0;"><span style="font-size: 13px;">状态</span><strong style="font-size: 13px;">${s.enabled ? '已启用' : '已禁用'}</strong></div>`;
      html += `<div style="display:flex; justify-content:space-between; align-items:center; margin: 4px 0;"><span style="font-size: 13px;">缓存条目</span><span style="font-size: 13px;">${s.total_entries}</span></div>`;
      html += `<div style="display:flex; justify-content:space-between; align-items:center; margin: 4px 0;"><span style="font-size: 13px;">命中次数</span><span style="font-size: 13px;">${s.total_hits}</span></div>`;
      html += `<div style="display:flex; justify-content:space-between; align-items:center; margin: 4px 0;"><span style="font-size: 13px;">TTL</span><span style="font-size: 13px;">${s.ttl_hours}h</span></div>`;
      html += `<div style="margin-top: 10px; display:flex; gap: 8px;">`;
      html += `<button onclick="toggleCache(${!s.enabled})" style="padding: 5px 12px; background: ${toggleColor}; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 12px;">${toggleText}缓存</button>`;
      html += `<button onclick="clearCache()" style="padding: 5px 12px; background: #868e96; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 12px;">清空</button>`;
      html += `</div>`;
    } else {
      html += '<p style="margin:0; color: var(--text-secondary); font-size: 13px;">缓存系统不可用</p>';
    }
    html += '</div>';

    // 模型路由
    html += '<div style="margin-bottom: 20px; padding: 12px; background: var(--bg-secondary); border-radius: 8px; border: 1.5px solid var(--input-border);">';
    html += '<h4 style="margin: 0 0 10px 0; color: var(--accent);">🚦 模型路由</h4>';
    if (routerRes.success) {
      const toggleText = routerRes.enabled ? '禁用' : '启用';
      const toggleColor = routerRes.enabled ? '#ff6b6b' : '#51cf66';
      html += `<div style="display:flex; justify-content:space-between; align-items:center; margin: 4px 0;"><span style="font-size: 13px;">状态</span><button onclick="toggleRouter(${!routerRes.enabled})" style="padding: 4px 10px; background: ${toggleColor}; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 12px;">${toggleText}</button></div>`;
      if (routerRes.tiers) {
        Object.entries(routerRes.tiers).forEach(([tier, cfg]) => {
          html += `<div style="display:flex; justify-content:space-between; align-items:center; margin: 4px 0;"><span style="font-size: 13px; text-transform:capitalize;">${tier}</span><span style="font-size: 12px; color: var(--text-secondary);">${cfg.provider}/${cfg.model}</span></div>`;
        });
      }
    } else {
      html += '<p style="margin:0; color: var(--text-secondary); font-size: 13px;">路由系统不可用</p>';
    }
    html += '</div>';

    // RAG 向量存储
    html += '<div style="margin-bottom: 20px; padding: 12px; background: var(--bg-secondary); border-radius: 8px; border: 1.5px solid var(--input-border);">';
    html += '<h4 style="margin: 0 0 10px 0; color: var(--accent);">📚 知识库 (RAG)</h4>';
    if (ragRes.success) {
      html += `<div style="display:flex; justify-content:space-between; align-items:center; margin: 4px 0;"><span style="font-size: 13px;">文档块数</span><span style="font-size: 13px;">${ragRes.total_chunks || 0}</span></div>`;
      html += `<div style="display:flex; justify-content:space-between; align-items:center; margin: 4px 0;"><span style="font-size: 13px;">当前后端</span><span style="font-size: 13px;">${ragRes.current_backend || 'numpy'}</span></div>`;
      if (ragRes.available_backends) {
        ragRes.available_backends.forEach(b => {
          if (b.available && b.name !== ragRes.current_backend) {
            html += `<div style="margin-top: 8px;"><button onclick="switchRagBackend('${b.name}')" style="padding: 5px 12px; background: var(--accent); color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 12px;">切换到 ${b.name}</button></div>`;
          }
        });
      }
    } else {
      html += '<p style="margin:0; color: var(--text-secondary); font-size: 13px;">RAG 系统不可用</p>';
    }
    html += '</div>';

    // 管理入口
    html += '<div style="margin-bottom: 20px; padding: 12px; background: var(--bg-secondary); border-radius: 8px; border: 1.5px solid var(--input-border);">';
    html += '<h4 style="margin: 0 0 12px 0; color: var(--accent); font-size: 14px;">⚙️ 管理</h4>';
    html += '<div style="display: flex; flex-direction: column; gap: 10px;">';
    html += '<a href="/config" style="display: flex; align-items: center; gap: 12px; padding: 14px 16px; background: var(--bg-primary); border-radius: 10px; text-decoration: none; color: var(--text-primary); font-size: 15px; font-weight: 500; transition: all 0.2s; border: 1px solid transparent;" onmouseover="this.style.background=\'var(--accent)\';this.style.color=\'white\';this.style.borderColor=\'var(--accent)\'" onmouseout="this.style.background=\'var(--bg-primary)\';this.style.color=\'var(--text-primary)\';this.style.borderColor=\'transparent\'">';
    html += '<span style="font-size: 22px;">⚙️</span><div><div>模型配置</div><div style="font-size: 12px; opacity: 0.7; font-weight: 400;">切换模型、API Key、路由档位</div></div></a>';
    html += '<a href="/knowledge" style="display: flex; align-items: center; gap: 12px; padding: 14px 16px; background: var(--bg-primary); border-radius: 10px; text-decoration: none; color: var(--text-primary); font-size: 15px; font-weight: 500; transition: all 0.2s; border: 1px solid transparent;" onmouseover="this.style.background=\'var(--accent)\';this.style.color=\'white\';this.style.borderColor=\'var(--accent)\'" onmouseout="this.style.background=\'var(--bg-primary)\';this.style.color=\'var(--text-primary)\';this.style.borderColor=\'transparent\'">';
    html += '<span style="font-size: 22px;">📚</span><div><div>知识库</div><div style="font-size: 12px; opacity: 0.7; font-weight: 400;">上传文档、管理向量库</div></div></a>';
    html += '<a href="/plugins" style="display: flex; align-items: center; gap: 12px; padding: 14px 16px; background: var(--bg-primary); border-radius: 10px; text-decoration: none; color: var(--text-primary); font-size: 15px; font-weight: 500; transition: all 0.2s; border: 1px solid transparent;" onmouseover="this.style.background=\'var(--accent)\';this.style.color=\'white\';this.style.borderColor=\'var(--accent)\'" onmouseout="this.style.background=\'var(--bg-primary)\';this.style.color=\'var(--text-primary)\';this.style.borderColor=\'transparent\'">';
    html += '<span style="font-size: 22px;">🔧</span><div><div>插件管理</div><div style="font-size: 12px; opacity: 0.7; font-weight: 400;">安装、启用/禁用插件</div></div></a>';
    html += '<a href="/mcp" style="display: flex; align-items: center; gap: 12px; padding: 14px 16px; background: var(--bg-primary); border-radius: 10px; text-decoration: none; color: var(--text-primary); font-size: 15px; font-weight: 500; transition: all 0.2s; border: 1px solid transparent;" onmouseover="this.style.background=\'var(--accent)\';this.style.color=\'white\';this.style.borderColor=\'var(--accent)\'" onmouseout="this.style.background=\'var(--bg-primary)\';this.style.color=\'var(--text-primary)\';this.style.borderColor=\'transparent\'">';
    html += '<span style="font-size: 22px;">🔌</span><div><div>MCP 服务器</div><div style="font-size: 12px; opacity: 0.7; font-weight: 400;">连接外部 MCP 工具</div></div></a>';
    html += '</div>';
    html += '</div>';

    extensionsContent.innerHTML = html;
  } catch (err) {
    extensionsContent.innerHTML = `<div style="color: var(--danger);">加载失败: ${escapeHtml(err.message)}</div>`;
  }
}

async function togglePlugin(name, enable) {
  try {
    const endpoint = enable ? `/api/plugins/${name}/enable` : `/api/plugins/${name}/disable`;
    const res = await fetch(endpoint, { method: "POST" });
    const data = await res.json();
    showToast(data.message, data.success ? "success" : "error");
    if (data.success) loadExtensionsStatus();
  } catch (err) {
    showToast("操作失败: " + err.message, "error");
  }
}

async function toggleCache(enable) {
  try {
    const res = await fetch("/api/cache/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ enabled: enable, ttl_hours: 24 }),
    });
    const data = await res.json();
    showToast(data.message, data.success ? "success" : "error");
    if (data.success) loadExtensionsStatus();
  } catch (err) {
    showToast("操作失败: " + err.message, "error");
  }
}

async function clearCache() {
  try {
    const res = await fetch("/api/cache/clear", { method: "POST" });
    const data = await res.json();
    showToast(data.message || "缓存已清空", data.success ? "success" : "error");
    loadExtensionsStatus();
  } catch (err) {
    showToast("清空缓存失败: " + err.message, "error");
  }
}

async function toggleRouter(enable) {
  try {
    const res = await fetch("/api/router/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ enabled: enable }),
    });
    const data = await res.json();
    showToast(data.message, data.success ? "success" : "error");
    if (data.success) loadExtensionsStatus();
  } catch (err) {
    showToast("操作失败: " + err.message, "error");
  }
}

async function switchRagBackend(backend) {
  try {
    const res = await fetch("/api/rag/backend", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ backend }),
    });
    const data = await res.json();
    showToast(data.message, data.success ? "success" : "error");
    if (data.success) loadExtensionsStatus();
  } catch (err) {
    showToast("切换失败: " + err.message, "error");
  }
}

if (extensionsBtn) {
  extensionsBtn.addEventListener("click", showExtensionsPanel);
}
if (extensionsClose) {
  extensionsClose.addEventListener("click", hideExtensionsPanel);
}
if (extensionsOverlay) {
  extensionsOverlay.addEventListener("click", (e) => {
    if (e.target === extensionsOverlay) hideExtensionsPanel();
  });
}

// ============ Init ============

// Fast mode only — no mode toggle, no agent panel

socket.emit("get_model_info");
socket.emit("get_mode");
