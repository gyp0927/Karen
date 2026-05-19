"""Tests for skill system."""

from core.skills.loader import SkillLoader


class TestSkillLoader:
    def test_load_single_skill(self, tmp_path):
        """Test parsing a single skill file."""
        skill_dir = tmp_path / "test-category" / "test-skill"
        skill_dir.mkdir(parents=True)
        skill_file = skill_dir / "SKILL.md"
        skill_file.write_text(
            "---\n"
            "name: test-skill\n"
            "description: A test skill\n"
            "triggers:\n"
            "  - test trigger\n"
            "category: test-category\n"
            "model: default\n"
            "temperature: 0.5\n"
            "---\n"
            "\n"
            "# Test Skill\n"
            "\n"
            "This is the skill body.\n",
            encoding="utf-8",
        )

        loader = SkillLoader(skills_dir=tmp_path)
        skill = loader.get("test-skill")

        assert skill is not None
        assert skill.name == "test-skill"
        assert skill.description == "A test skill"
        assert skill.triggers == ["test trigger"]
        assert skill.category == "test-category"
        assert skill.model == "default"
        assert skill.temperature == 0.5
        assert "Test Skill" in skill.body

    def test_trigger_exact_match(self, tmp_path):
        """Test exact trigger matching."""
        skill_dir = tmp_path / "cat" / "skill-a"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: skill-a\ndescription: Skill A\ntriggers:\n  - hello\ncategory: cat\n---\n",
            encoding="utf-8",
        )

        loader = SkillLoader(skills_dir=tmp_path)
        matched = loader.match("hello")
        assert matched is not None
        assert matched.name == "skill-a"

    def test_trigger_contains_match(self, tmp_path):
        """Test contains trigger matching."""
        skill_dir = tmp_path / "cat" / "skill-b"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: skill-b\ndescription: Skill B\ntriggers:\n  - review code\ncategory: cat\n---\n",
            encoding="utf-8",
        )

        loader = SkillLoader(skills_dir=tmp_path)
        matched = loader.match("please review code for me")
        assert matched is not None
        assert matched.name == "skill-b"

    def test_trigger_priority_longest_wins(self, tmp_path):
        """Test that longest trigger wins on overlap."""
        # Skill A with short trigger
        dir_a = tmp_path / "cat" / "skill-short"
        dir_a.mkdir(parents=True)
        (dir_a / "SKILL.md").write_text(
            "---\nname: skill-short\ndescription: Short\ntriggers:\n  - code\ncategory: cat\n---\n",
            encoding="utf-8",
        )

        # Skill B with longer trigger
        dir_b = tmp_path / "cat" / "skill-long"
        dir_b.mkdir(parents=True)
        (dir_b / "SKILL.md").write_text(
            "---\nname: skill-long\ndescription: Long\ntriggers:\n  - review code\ncategory: cat\n---\n",
            encoding="utf-8",
        )

        loader = SkillLoader(skills_dir=tmp_path)
        matched = loader.match("please review code")
        assert matched is not None
        assert matched.name == "skill-long"

    def test_trigger_case_insensitive(self, tmp_path):
        """Test case-insensitive matching."""
        skill_dir = tmp_path / "cat" / "skill-c"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: skill-c\ndescription: Skill C\ntriggers:\n  - Code Review\ncategory: cat\n---\n",
            encoding="utf-8",
        )

        loader = SkillLoader(skills_dir=tmp_path)
        matched = loader.match("CODE REVIEW")
        assert matched is not None
        assert matched.name == "skill-c"

    def test_invalid_skill_skipped(self, tmp_path, caplog):
        """Test that invalid skills are skipped with warning."""
        skill_dir = tmp_path / "cat" / "bad-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: bad-skill\ndescription: Missing triggers and category\n---\n",
            encoding="utf-8",
        )

        import logging

        with caplog.at_level(logging.DEBUG):
            loader = SkillLoader(skills_dir=tmp_path)

        assert loader.get("bad-skill") is None
        assert "missing required fields" in caplog.text

    def test_invalid_yaml_skipped(self, tmp_path, caplog):
        """Test that invalid YAML is skipped."""
        skill_dir = tmp_path / "cat" / "bad-yaml"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: test\n  invalid: yaml: [\n---\n",
            encoding="utf-8",
        )

        import logging

        with caplog.at_level(logging.WARNING):
            loader = SkillLoader(skills_dir=tmp_path)

        assert len(loader.list_skills()) == 0
        assert "Invalid YAML" in caplog.text

    def test_reload(self, tmp_path):
        """Test hot reload picks up new skills."""
        loader = SkillLoader(skills_dir=tmp_path)
        assert len(loader.list_skills()) == 0

        # Add a new skill after loader is created
        skill_dir = tmp_path / "cat" / "new-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: new-skill\ndescription: New skill\ntriggers:\n  - new\ncategory: cat\n---\n",
            encoding="utf-8",
        )

        loader.reload()
        assert len(loader.list_skills()) == 1
        assert loader.get("new-skill") is not None

    def test_list_skills(self, tmp_path):
        """Test listing all loaded skills."""
        for name in ("skill-1", "skill-2"):
            skill_dir = tmp_path / "cat" / name
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(
                f"---\nname: {name}\ndescription: {name}\ntriggers:\n  - {name}\ncategory: cat\n---\n",
                encoding="utf-8",
            )

        loader = SkillLoader(skills_dir=tmp_path)
        skills = loader.list_skills()
        assert len(skills) == 2
        names = {s.name for s in skills}
        assert names == {"skill-1", "skill-2"}

    def test_empty_directory(self, tmp_path):
        """Test loading from empty directory."""
        loader = SkillLoader(skills_dir=tmp_path)
        assert len(loader.list_skills()) == 0
        assert loader.match("anything") is None

    def test_skill_to_prompt_text(self, tmp_path):
        """Test Skill.to_prompt_text() output."""
        skill_dir = tmp_path / "cat" / "prompt-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\n"
            "name: prompt-skill\n"
            "description: Prompt test\n"
            "triggers:\n"
            "  - prompt\n"
            "category: cat\n"
            "---\n"
            "\n"
            "## Role\n"
            "You are a test assistant.\n",
            encoding="utf-8",
        )

        loader = SkillLoader(skills_dir=tmp_path)
        skill = loader.get("prompt-skill")
        assert skill is not None
        prompt = skill.to_prompt_text()
        assert "当前激活技能: prompt-skill" in prompt
        assert "描述: Prompt test" in prompt
        assert "## Role" in prompt

    def test_single_string_trigger(self, tmp_path):
        """Test that a single string trigger is normalized to list."""
        skill_dir = tmp_path / "cat" / "str-trigger"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: str-trigger\ndescription: String trigger\ntriggers: single-trigger\ncategory: cat\n---\n",
            encoding="utf-8",
        )

        loader = SkillLoader(skills_dir=tmp_path)
        skill = loader.get("str-trigger")
        assert skill is not None
        assert skill.triggers == ["single-trigger"]
