"""智能模型路由 - 根据问题复杂度自动选择合适的模型档位。

支持三档模型：
- light: 轻量模型（简单问答、问候）
- default: 默认模型（一般问题）
- powerful: 强力模型（复杂推理、代码、深度分析）
"""

import json
import logging
import os
from typing import Optional

from langchain_core.messages import BaseMessage, HumanMessage

logger = logging.getLogger(__name__)

_CONFIG_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "state", "model_tiers.json")

# 复杂度评分权重
_COMPLEXITY_WEIGHTS = {
    "code_keywords": 3,
    "analysis_keywords": 2,
    "comparison_keywords": 2,
    "creative_keywords": 2,
    "greeting_keywords": -2,
    "simple_keywords": -1,
    "length_factor": 0.01,  # 每超出一个字符加多少分
    "turn_factor": 0.5,     # 每多一轮对话加多少分
}

# 关键词分类
_KEYWORD_PATTERNS = {
    "code_keywords": [
        "代码", "编程", "程序", "debug", "调试", "算法", "函数", "类", "接口",
        "写代码", "python", "java", "javascript", "c++", "go", "rust",
        "报错", "异常", "error", "bug", "fix", "实现", "重构",
    ],
    "analysis_keywords": [
        "分析", "解释", "说明", "为什么", "原因", "原理", "机制",
        "evaluate", "analyze", "explain", "reason", "cause",
    ],
    "comparison_keywords": [
        "比较", "对比", "区别", "差异", "vs", "versus",
        "compare", "difference", "better", "worse",
    ],
    "creative_keywords": [
        "写", "创作", "生成", "设计", "创意", "故事", "文章", "诗歌",
        "write", "create", "generate", "design", "creative", "story",
    ],
    "greeting_keywords": [
        "你好", "您好", "嗨", "hello", "hi", "hey", "早上好", "下午好", "晚上好",
        "再见", "拜拜", "bye", "谢谢", "感谢",
    ],
    "simple_keywords": [
        "是", "否", "对", "错", "好的", "ok", "行", "可以",
        "yes", "no", "ok", "sure", "maybe",
    ],
}


class ComplexityAnalyzer:
    """问题复杂度分析器"""

    def __init__(self, weights: dict = None):
        self.weights = weights or _COMPLEXITY_WEIGHTS

    def analyze(self, user_message: str, history_turns: int = 0) -> dict:
        """分析用户问题的复杂度。

        Returns:
            {
                "score": float,        # 复杂度评分（越高越复杂）
                "tier": str,           # 推荐档位: "light" | "default" | "powerful"
                "factors": dict,       # 各因子得分明细
            }
        """
        message_lower = user_message.lower()
        score = 0.0
        factors = {}

        # 关键词评分
        for category, keywords in _KEYWORD_PATTERNS.items():
            weight = self.weights.get(category, 0)
            matched = sum(1 for kw in keywords if kw in message_lower)
            category_score = matched * weight
            factors[category] = category_score
            score += category_score

        # 消息长度评分
        length_threshold = 50
        if len(user_message) > length_threshold:
            length_score = (len(user_message) - length_threshold) * self.weights.get("length_factor", 0)
            factors["length"] = round(length_score, 2)
            score += length_score

        # 对话轮数评分
        turn_score = history_turns * self.weights.get("turn_factor", 0)
        factors["history_turns"] = round(turn_score, 2)
        score += turn_score

        # 确定档位
        if score <= 0:
            tier = "light"
        elif score <= 3:
            tier = "default"
        else:
            tier = "powerful"

        return {
            "score": round(score, 2),
            "tier": tier,
            "factors": factors,
        }


class ModelRouter:
    """模型路由器"""

    # powerful 档目前指向编码专用模型（kimi-for-coding），
    # 对非编码问题会静默返回空响应。所以 powerful 档要再加一层"是否真编码意图"判断。
    _CODING_VERBS = (
        "写代码", "写一段", "写一个", "写个", "实现", "重构", "调试", "debug",
        "修复", "修一下", "修一个", "改一下", "改个 bug",
        "优化", "测试用例", "单元测试", "重写", "解析", "parse",
        "implement", "rewrite", "refactor", "optimize", "fix this",
    )
    _ERROR_KWS = (
        "报错", "异常", "栈追踪", "stacktrace", "traceback",
        "exception", "core dump", "segfault",
    )
    # 出现这些"讨论性"词汇时，即使匹配到了 python/java 等语言名，也判定为非编码
    _DISCUSSION_MARKERS = (
        "哪个", "哪种", "更好", "更适合", "对比", "区别", "差异",
        "vs", "versus", "compare", "difference", "better than", "worse than",
        "推荐", "建议", "选哪个", "选什么",
    )

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.analyzer = ComplexityAnalyzer()
        self._tiers: dict = {}
        self._tiers_mtime: float = 0.0
        self._load_tiers()

    def _is_coding_intent(self, message: str) -> bool:
        """更严格的编码意图判断。

        powerful 档当前指向 kimi-for-coding，该模型对非编码问题会返回 HTTP 200 + 空响应，
        所以只有真正动手写/调/改代码的请求才允许进入 powerful 档。

        规则:
        - 包含代码块标记 ``` 或报错栈关键词 → 编码
        - 包含明确的编码动作词（写/实现/调试/重构/修复...）→ 编码
        - 否则即使提到 python/java 之类的语言名，也不算编码（避免把"X 和 Y 哪个好"误判）
        """
        msg = message.lower()
        if "```" in msg:
            return True
        if any(kw in msg for kw in self._ERROR_KWS):
            # 出现讨论性词汇时判为非编码（如"stacktrace 是什么"）
            if any(d in msg for d in self._DISCUSSION_MARKERS):
                return False
            return True
        if any(v in msg for v in self._CODING_VERBS):
            # 但若同时出现讨论性词汇，仍判为非编码（如"实现 X 和实现 Y 哪个更好"）
            if any(d in msg for d in self._DISCUSSION_MARKERS):
                return False
            return True
        return False

    def _load_tiers(self) -> None:
        """从配置文件加载模型档位（带 mtime 缓存刷新）。"""
        defaults = {
            "light": {"provider": "ollama", "model": "llama3.2", "description": "轻量模型（简单问答）"},
            "default": {"provider": "ollama", "model": "llama3.2", "description": "默认模型"},
            "powerful": {"provider": "ollama", "model": "llama3.2", "description": "强力模型（复杂任务）"},
        }
        mtime = 0.0
        if os.path.exists(_CONFIG_FILE):
            try:
                mtime = os.path.getmtime(_CONFIG_FILE)
                if mtime <= self._tiers_mtime and self._tiers:
                    return  # 文件未修改，跳过
                with open(_CONFIG_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for tier in ["light", "default", "powerful"]:
                    if tier in data:
                        defaults[tier].update(data[tier])
            except (OSError, json.JSONDecodeError) as e:
                logger.warning(f"Failed to load model tier config: {e}")
        self._tiers = defaults
        self._tiers_mtime = mtime

    def save_tiers(self):
        """保存模型档位配置。"""
        try:
            os.makedirs(os.path.dirname(_CONFIG_FILE), exist_ok=True)
            with open(_CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(self._tiers, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save model tiers: {e}")

    def set_tier(self, tier: str, provider: str, model: str, api_key: str = "", base_url: str = ""):
        """设置指定档位的模型。"""
        if tier not in self._tiers:
            raise ValueError(f"Invalid tier: {tier}. Must be one of: light, default, powerful")
        self._tiers[tier] = {
            "provider": provider,
            "model": model,
            "apiKey": api_key,
            "baseUrl": base_url,
            "description": self._tiers[tier].get("description", ""),
        }
        self.save_tiers()

    def get_tier_config(self, tier: str) -> dict:
        """获取指定档位的模型配置。

        如果配置文件中 apiKey 为空，尝试从环境变量读取：
        - siliconflow → SILICONFLOW_API_KEY
        - kimi / kimi-code → KIMI_API_KEY
        """
        self._load_tiers()  # 自动刷新缓存
        cfg = self._tiers.get(tier, self._tiers["default"]).copy()
        if not cfg.get("apiKey"):
            provider = cfg.get("provider", "")
            env_key = None
            if provider == "siliconflow":
                env_key = os.getenv("SILICONFLOW_API_KEY")
            elif provider in ("kimi", "kimi-code"):
                env_key = os.getenv("KIMI_API_KEY")
            if env_key:
                cfg["apiKey"] = env_key
        return cfg

    def get_all_tiers(self) -> dict:
        """获取所有档位配置（API Key 脱敏）。"""
        self._load_tiers()
        result = {}
        for tier, cfg in self._tiers.items():
            result[tier] = dict(cfg)
            api_key = result[tier].get("apiKey")
            if api_key:
                if len(api_key) <= 4:
                    result[tier]["apiKey"] = "****"
                else:
                    result[tier]["apiKey"] = "****" + api_key[-4:]
            # baseUrl 中可能包含凭据（如 https://user:pass@host），需要脱敏
            base_url = result[tier].get("baseUrl", "")
            if base_url and "@" in base_url:
                from urllib.parse import urlparse, urlunparse
                parsed = urlparse(base_url)
                if parsed.password:
                    # 替换密码部分为 ****
                    netloc = parsed.hostname or ""
                    if parsed.port:
                        netloc = f"{netloc}:{parsed.port}"
                    if parsed.username:
                        netloc = f"{parsed.username}:****@{netloc}"
                    result[tier]["baseUrl"] = urlunparse(
                        (parsed.scheme, netloc, parsed.path, parsed.params, parsed.query, parsed.fragment)
                    )
        return result

    def route(self, user_message: str, history_turns: int = 0) -> dict:
        """根据问题复杂度选择模型档位。

        Returns:
            {
                "tier": str,           # 选中的档位
                "config": dict,        # 该档位的模型配置
                "analysis": dict,      # 复杂度分析结果
            }
        """
        if not self.enabled:
            return {
                "tier": "default",
                "config": self.get_tier_config("default"),
                "analysis": {"score": 0, "tier": "default", "factors": {}},
            }

        analysis = self.analyzer.analyze(user_message, history_turns)
        tier = analysis["tier"]

        # powerful 档当前是编码专用模型，对非编码问题会返回空。
        # 复杂但非编码的问题降级到 default，让它走通用模型。
        if tier == "powerful" and not self._is_coding_intent(user_message):
            logger.info(
                f"Powerful tier requested but non-coding intent detected, "
                f"downgrading to default. message='{user_message[:50]}...'"
            )
            tier = "default"

        config = self.get_tier_config(tier)

        logger.info(f"Model routing: tier={tier}, score={analysis['score']}, message='{user_message[:50]}...'")

        return {
            "tier": tier,
            "config": config,
            "analysis": analysis,
        }


# 全局路由器实例
_router_instance: Optional[ModelRouter] = None


def get_router() -> ModelRouter:
    """获取全局模型路由器。"""
    global _router_instance
    if _router_instance is None:
        _router_instance = ModelRouter()
    return _router_instance


def configure_router(enabled: bool = True):
    """配置模型路由器。"""
    global _router_instance
    _router_instance = ModelRouter(enabled=enabled)
