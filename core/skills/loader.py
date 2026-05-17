"""Skill loader: parse, validate, cache skill definitions."""

from __future__ import annotations

import functools
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

SKILLS_DIR = Path(__file__).resolve().parents[2] / "skills"


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    triggers: list[str]
    category: str
    model: str = "default"
    temperature: float | None = None
    max_tokens: int | None = None
    tools: list[str] = field(default_factory=list)
    requires_mcp: list[str] = field(default_factory=list)
    body: str = ""  # markdown body after frontmatter

    def to_prompt_text(self) -> str:
        """将技能内容转换为可注入 system prompt 的文本。"""
        lines = [
            f"## 当前激活技能: {self.name}",
            f"描述: {self.description}",
            "",
            self.body,
        ]
        return "\n".join(lines)


class SkillLoader:
    def __init__(self, skills_dir: Path | str | None = None) -> None:
        if skills_dir is None:
            env_dir = os.getenv("KAREN_SKILLS_DIR")
            skills_dir = Path(env_dir) if env_dir else SKILLS_DIR
        self.skills_dir = Path(skills_dir)
        self._cache: dict[str, Skill] = {}  # name -> Skill
        self._trigger_index: dict[str, str] = {}  # trigger -> skill_name
        self._load_all()

    def _load_all(self) -> None:
        """Load all skills from disk into memory."""
        if not self.skills_dir.exists():
            logger.warning(f"Skills directory not found: {self.skills_dir}")
            return

        for skill_file in self.skills_dir.rglob("SKILL.md"):
            skill = self._parse_skill(skill_file)
            if skill:
                self._cache[skill.name] = skill
                for trigger in skill.triggers:
                    # 最长 trigger 优先：如果已有短 trigger，被长 trigger 覆盖
                    existing = self._trigger_index.get(trigger.lower())
                    if existing is None or len(trigger) > len(self._trigger_index.get(trigger.lower(), "")):
                        self._trigger_index[trigger.lower()] = skill.name

        logger.info(f"Loaded {len(self._cache)} skills from {self.skills_dir}")

    def _parse_skill(self, path: Path) -> Skill | None:
        """Parse a single SKILL.md file.

        Format: YAML frontmatter between --- markers, then markdown body.
        """
        try:
            content = path.read_text(encoding="utf-8")
        except OSError as e:
            logger.warning(f"Failed to read skill file {path}: {e}")
            return None

        # 分离 frontmatter 和 body
        match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", content, re.DOTALL)
        if not match:
            logger.warning(f"Invalid skill file format (missing frontmatter): {path}")
            return None

        frontmatter_text, body = match.groups()

        try:
            meta: dict[str, Any] = yaml.safe_load(frontmatter_text) or {}
        except yaml.YAMLError as e:
            logger.warning(f"Invalid YAML frontmatter in {path}: {e}")
            return None

        # 验证必填字段
        required = ["name", "description", "triggers", "category"]
        missing = [f for f in required if not meta.get(f)]
        if missing:
            logger.warning(f"Skill {path} missing required fields: {missing}")
            return None

        # 标准化 triggers 为列表
        triggers = meta.get("triggers", [])
        if isinstance(triggers, str):
            triggers = [triggers]

        return Skill(
            name=str(meta["name"]),
            description=str(meta["description"]),
            triggers=[str(t) for t in triggers],
            category=str(meta["category"]),
            model=str(meta.get("model", "default")),
            temperature=float(meta["temperature"]) if meta.get("temperature") is not None else None,
            max_tokens=int(meta["max_tokens"]) if meta.get("max_tokens") is not None else None,
            tools=[str(t) for t in meta.get("tools", [])],
            requires_mcp=[str(t) for t in meta.get("requires_mcp", [])],
            body=body.strip(),
        )

    @functools.lru_cache(maxsize=256)
    def get(self, name: str) -> Skill | None:
        """Get skill by name."""
        return self._cache.get(name)

    def match(self, text: str) -> Skill | None:
        """Match user input against trigger keywords.

        Priority:
        1. Exact match (case-insensitive)
        2. Contains match (case-insensitive)
        3. Longest trigger wins tie-breaking
        """
        if not text:
            return None

        text_lower = text.lower().strip()

        # 1. 完全匹配
        skill_name = self._trigger_index.get(text_lower)
        if skill_name:
            return self.get(skill_name)

        # 2. contains 匹配 — 最长 trigger 优先
        best_match: tuple[int, str] | None = None
        for trigger, skill_name in self._trigger_index.items():
            if trigger in text_lower:
                if best_match is None or len(trigger) > best_match[0]:
                    best_match = (len(trigger), skill_name)

        if best_match:
            return self.get(best_match[1])

        return None

    def list_skills(self) -> list[Skill]:
        """Return all loaded skills."""
        return list(self._cache.values())

    def reload(self) -> None:
        """Hot reload all skills from disk."""
        self._cache.clear()
        self._trigger_index.clear()
        self.get.cache_clear()
        self._load_all()
        logger.info("Skills hot-reloaded")

    def get_trigger_summary(self) -> dict[str, list[str]]:
        """返回每个 skill 的 trigger 列表（用于调试）。"""
        result: dict[str, list[str]] = {}
        for trigger, skill_name in self._trigger_index.items():
            result.setdefault(skill_name, []).append(trigger)
        return result
