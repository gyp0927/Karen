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
        "compare", "difference", "versus", "better", "worse",
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

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.analyzer = ComplexityAnalyzer()
        self._tiers = self._load_tiers()

    def _load_tiers(self) -> dict:
        """从配置文件加载模型档位。"""
        defaults = {
            "light": {"provider": "ollama", "model": "llama3.2", "description": "轻量模型（简单问答）"},
            "default": {"provider": "ollama", "model": "llama3.2", "description": "默认模型"},
            "powerful": {"provider": "ollama", "model": "llama3.2", "description": "强力模型（复杂任务）"},
        }
        if os.path.exists(_CONFIG_FILE):
            try:
                with open(_CONFIG_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for tier in ["light", "default", "powerful"]:
                    if tier in data:
                        defaults[tier].update(data[tier])
                return defaults
            except (OSError, json.JSONDecodeError) as e:
                logger.warning(f"Failed to load model tier config: {e}")
        return defaults

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
        """获取指定档位的模型配置。"""
        return self._tiers.get(tier, self._tiers["default"]).copy()

    def get_all_tiers(self) -> dict:
        """获取所有档位配置（API Key 脱敏）。"""
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
