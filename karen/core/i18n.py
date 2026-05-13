"""国际化工具 - 语言指令和名称映射"""

# 语言指令映射（用于 responder prompt）
LANG_INSTRUCTIONS = {
    "zh": "\n\n重要：你必须用中文回答。",
    "en": "\n\nIMPORTANT: You must respond entirely in English.",
    "ja": "\n\n重要：あなたは日本語で回答しなければなりません。",
    "ko": "\n\n중요: 한국어로 답변해야 합니다.",
    "fr": "\n\nIMPORTANT: Vous devez répondre entièrement en français.",
    "de": "\n\nWICHTIG: Sie müssen vollständig auf Deutsch antworten.",
    "es": "\n\nIMPORTANTE: Debe responder completamente en español.",
    "ru": "\n\nВАЖНО: Вы должны отвечать полностью на русском языке.",
    "ar": "\n\nمهم: يجب أن ترد باللغة العربية بالكامل.",
}

# 语言显示名称
LANG_NAMES = {
    "zh": "中文",
    "en": "English",
    "ja": "日本語",
    "ko": "한국어",
    "fr": "Français",
    "de": "Deutsch",
    "es": "Español",
    "ru": "Русский",
    "ar": "العربية",
}


def get_lang_instruction(lang: str) -> str:
    """根据语言代码生成 responder prompt 中的语言指令。"""
    return LANG_INSTRUCTIONS.get(lang, LANG_INSTRUCTIONS.get("en", ""))
