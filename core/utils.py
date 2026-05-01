"""通用工具函数"""
import re


def detect_language(text: str) -> str:
    """检测文本的主要语言。返回 'zh', 'en', 'ja', 'ko' 等 ISO 代码。

    基于字符集统计，无需额外依赖：
    - 中日韩统一表意文字 (CJK) → zh
    - 平假名/片假名 → ja
    - 韩文音节 → ko
    - 主要是 ASCII → en
    - 否则默认 zh
    """
    if not text or not text.strip():
        return "zh"

    cleaned = re.sub(r"[\s\.\,\!\?\;\:\'\"\(\)\[\]\{\}\\/\-\_\@\#\$\%\&\*\+\=\|\<\>\`\~]", "", text)
    if not cleaned:
        return "zh"

    zh_chars = len(re.findall(r"[一-鿿]", cleaned))
    ja_chars = len(re.findall(r"[぀-ゟ゠-ヿ]", cleaned))
    ko_chars = len(re.findall(r"[가-힯]", cleaned))
    total = len(cleaned)

    if total == 0:
        return "zh"

    scores = {
        "zh": zh_chars / total,
        "ja": ja_chars / total,
        "ko": ko_chars / total,
    }
    best_lang = max(scores, key=scores.get)
    if scores[best_lang] > 0.25:
        return best_lang

    ascii_chars = sum(1 for c in cleaned if ord(c) < 128)
    if ascii_chars / total > 0.6:
        return "en"

    return "zh"
