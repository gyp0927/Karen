"""聊天记录导出功能 - 支持 Markdown、JSON、PDF 格式。"""

import json
import logging
import os
import tempfile
from datetime import datetime

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

logger = logging.getLogger(__name__)


def export_markdown(messages: list[BaseMessage], title: str = "聊天记录") -> str:
    """导出为 Markdown 格式"""
    lines = [f"# {title}\n"]
    lines.append(f"> 导出时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append(f"> 消息数: {len(messages)}\n\n---\n")

    for msg in messages:
        role = "用户" if isinstance(msg, HumanMessage) else "AI"
        sender = getattr(msg, "name", role)
        lines.append(f"\n## {sender}\n")
        lines.append(msg.content)
        lines.append("\n")

    return "\n".join(lines)


def export_json(messages: list[BaseMessage], title: str = "聊天记录") -> str:
    """导出为 JSON 格式"""
    data = {
        "title": title,
        "exported_at": datetime.now().isoformat(),
        "message_count": len(messages),
        "messages": [],
    }
    for msg in messages:
        data["messages"].append({
            "role": "user" if isinstance(msg, HumanMessage) else "assistant",
            "sender": getattr(msg, "name", ""),
            "content": msg.content,
            "timestamp": datetime.now().isoformat(),
        })
    return json.dumps(data, ensure_ascii=False, indent=2)


def export_html(messages: list[BaseMessage], title: str = "聊天记录") -> str:
    """导出为 HTML 格式（可用于 PDF 转换）"""
    lines = [
        "<!DOCTYPE html>",
        '<html lang="zh-CN">',
        "<head>",
        f'  <meta charset="UTF-8"><title>{title}</title>',
        "  <style>",
        "    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; line-height: 1.6; color: #333; }",
        "    h1 { border-bottom: 2px solid #eee; padding-bottom: 10px; }",
        "    .meta { color: #666; font-size: 14px; margin-bottom: 20px; }",
        "    .message { margin: 20px 0; padding: 15px; border-radius: 8px; }",
        "    .user { background: #f0f7ff; border-left: 4px solid #4a9eff; }",
        "    .assistant { background: #f5f5f5; border-left: 4px solid #888; }",
        "    .review { background: #fff8f0; border-left: 4px solid #ff9500; }",
        "    .sender { font-weight: bold; margin-bottom: 8px; color: #555; }",
        "    .content { white-space: pre-wrap; }",
        "    pre { background: #f8f8f8; padding: 10px; border-radius: 4px; overflow-x: auto; }",
        "    code { font-family: 'Consolas', 'Monaco', monospace; font-size: 14px; }",
        "  </style>",
        "</head>",
        "<body>",
        f"  <h1>{title}</h1>",
        f"  <div class='meta'>导出时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 消息数: {len(messages)}</div>",
    ]

    for msg in messages:
        if isinstance(msg, HumanMessage):
            css_class = "user"
            sender = "用户"
        elif getattr(msg, "name", "") == "reviewer":
            css_class = "review"
            sender = "审查者"
        else:
            css_class = "assistant"
            sender = getattr(msg, "name", "AI") or "AI"

        # 简单转义 HTML
        content = (msg.content
                   .replace("&", "&amp;")
                   .replace("<", "&lt;")
                   .replace(">", "&gt;")
                   .replace('"', "&quot;")
                   .replace("'", "&#x27;"))
        # 保留换行
        content = content.replace("\n", "<br>")

        lines.append(f'  <div class="message {css_class}">')
        lines.append(f'    <div class="sender">{sender}</div>')
        lines.append(f'    <div class="content">{content}</div>')
        lines.append("  </div>")

    lines.extend(["</body>", "</html>"])
    return "\n".join(lines)


def export_pdf(messages: list[BaseMessage], title: str = "聊天记录") -> tuple[bytes, str]:
    """导出为 PDF 格式。

    返回: (pdf_bytes, error_message)
    如果成功，error_message 为空。
    """
    # 先生成 HTML，再尝试转换为 PDF
    html_content = export_html(messages, title)

    # 尝试使用 weasyprint
    try:
        from weasyprint import HTML
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
            f.write(html_content)
            html_path = f.name
        pdf_bytes = HTML(filename=html_path).write_pdf()
        os.remove(html_path)
        return pdf_bytes, ""
    except ImportError:
        logger.warning("weasyprint not installed, falling back to HTML")
        return b"", "PDF 导出需要安装 weasyprint（pip install weasyprint），已回退到 HTML 格式"
    except Exception as e:
        logger.exception("PDF export failed")
        return b"", f"PDF 导出失败: {str(e)}"


def get_export_filename(title: str, fmt: str) -> str:
    """生成导出文件名"""
    safe_title = "".join(c for c in title if c.isalnum() or c in "._-").strip() or "chat"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ext_map = {"md": "md", "json": "json", "html": "html", "pdf": "pdf"}
    ext = ext_map.get(fmt, fmt)
    return f"{safe_title}_{timestamp}.{ext}"
