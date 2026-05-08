import os
import io
from typing import Optional


def parse_document(file_path: str) -> str:
    """解析文档，返回纯文本内容"""
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        return _parse_pdf(file_path)
    elif ext in (".docx", ".doc"):
        return _parse_docx(file_path)
    elif ext in (".txt", ".md", ".py", ".js", ".html", ".css", ".json", ".xml", ".yaml", ".yml"):
        return _parse_text(file_path)
    else:
        # 尝试按文本读取
        try:
            return _parse_text(file_path)
        except (OSError, UnicodeDecodeError, ValueError) as e:
            return f"[无法解析文件类型: {ext} - {e}]"


def _parse_pdf(file_path: str) -> str:
    """解析 PDF"""
    try:
        import PyPDF2
        text = ""
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text.strip() or "[PDF 内容为空或无法提取文本]"
    except Exception as e:
        return f"[PDF 解析失败: {str(e)}]"


def _parse_docx(file_path: str) -> str:
    """解析 Word"""
    try:
        import docx
        doc = docx.Document(file_path)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n".join(paragraphs)
    except Exception as e:
        return f"[Word 解析失败: {str(e)}]"


def _parse_text(file_path: str) -> str:
    """解析文本文件"""
    encodings = ["utf-8", "gbk", "gb2312", "latin-1"]
    for encoding in encodings:
        try:
            with open(file_path, "r", encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    return "[文本解码失败]"


def truncate_text(text: str, max_chars: int = 6000) -> str:
    """截断文本，避免超出模型上下文限制"""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n\n[文档已截断，原始长度: {len(text)} 字符]"


def format_document_context(filename: str, content: str) -> str:
    """将文档内容格式化为系统提示"""
    truncated = truncate_text(content)
    return f"""以下是你需要参考的文档内容（文件名: {filename}）:

---
{truncated}
---

请在回答用户问题时参考以上文档内容。"""
