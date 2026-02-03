import os
import unittest

from agent.skills_loader import SkillLoader


class TestSkillLoader(unittest.TestCase):
    def test_loads_core_skill(self) -> None:
        skills_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "skills"))
        loader = SkillLoader(skills_path)
        skills = loader.load_all()
        self.assertIn("core", skills)
        self.assertIn("remember_note", skills["core"].functions)
        self.assertIn("set_reminder", skills["core"].functions)


if __name__ == "__main__":
    unittest.main()
