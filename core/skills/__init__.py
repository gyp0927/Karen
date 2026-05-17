"""技能系统 - Skill loader, trigger, and state management."""

from .loader import Skill, SkillLoader

# 全局 SkillLoader 实例（模块级单例，延迟加载）
_skill_loader: SkillLoader | None = None


def get_skill_loader() -> SkillLoader:
    """获取全局 SkillLoader 实例（懒加载）。"""
    global _skill_loader
    if _skill_loader is None:
        _skill_loader = SkillLoader()
    return _skill_loader


def match_skill(text: str) -> Skill | None:
    """匹配用户输入到 skill trigger。

    使用全局 SkillLoader 实例进行匹配。
    """
    return get_skill_loader().match(text)


def get_skill(name: str) -> Skill | None:
    """按名称获取 skill。"""
    return get_skill_loader().get(name)


__all__ = ["Skill", "SkillLoader", "get_skill_loader", "match_skill", "get_skill"]
