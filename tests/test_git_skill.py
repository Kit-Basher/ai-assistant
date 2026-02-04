import os
import unittest

from agent.skills_loader import SkillLoader


class TestGitSkill(unittest.TestCase):
    def setUp(self) -> None:
        skills_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "skills"))
        self.skills = SkillLoader(skills_path).load_all()

    def test_git_status_no_approval(self) -> None:
        git_skill = self.skills["git"]
        result = git_skill.functions["git_status"].handler({"db": None})
        self.assertIn(result["status"], {"ok", "error"})

    def test_git_commit_requires_approval(self) -> None:
        git_skill = self.skills["git"]
        result = git_skill.functions["git_commit"].handler(
            {"db": None}, message="test", approval_token="NOPE"
        )
        self.assertEqual(result["status"], "blocked")

    def test_git_tag_requires_approval(self) -> None:
        git_skill = self.skills["git"]
        result = git_skill.functions["git_tag"].handler(
            {"db": None}, name="v0.0.0", message="test", approval_token="NOPE"
        )
        self.assertEqual(result["status"], "blocked")

    def test_git_push_requires_approval(self) -> None:
        git_skill = self.skills["git"]
        result = git_skill.functions["git_push"].handler(
            {"db": None}, approval_token="NOPE"
        )
        self.assertEqual(result["status"], "blocked")


if __name__ == "__main__":
    unittest.main()
