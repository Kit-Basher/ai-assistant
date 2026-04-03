import os
import tempfile
import unittest

from agent.filesystem_skill import FileSystemSkill


class TestFileSystemSkill(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.safe_root = os.path.join(self.tmpdir.name, "safe")
        self.private_root = os.path.join(self.safe_root, "private")
        self.outside_root = os.path.join(self.tmpdir.name, "outside")
        os.makedirs(self.safe_root, exist_ok=True)
        os.makedirs(self.private_root, exist_ok=True)
        os.makedirs(self.outside_root, exist_ok=True)
        self.skill = FileSystemSkill(
            allowed_roots=[self.safe_root],
            base_dir=self.safe_root,
            sensitive_roots=[self.private_root],
        )

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_list_directory_inside_allowed_root(self) -> None:
        notes_dir = os.path.join(self.safe_root, "notes")
        os.makedirs(notes_dir, exist_ok=True)
        with open(os.path.join(notes_dir, "a.txt"), "w", encoding="utf-8") as handle:
            handle.write("a")

        result = self.skill.list_directory(notes_dir)

        self.assertTrue(result["ok"])
        self.assertEqual("list_directory", result["action"])
        self.assertEqual(notes_dir, result["resolved_path"])
        self.assertEqual("a.txt", result["entries"][0]["name"])

    def test_stat_path_inside_allowed_root(self) -> None:
        file_path = os.path.join(self.safe_root, "note.txt")
        with open(file_path, "w", encoding="utf-8") as handle:
            handle.write("hello")

        result = self.skill.stat_path(file_path)

        self.assertTrue(result["ok"])
        self.assertEqual("file", result["type"])
        self.assertEqual(5, result["size"])
        self.assertTrue(result["readable"])

    def test_read_text_file_reads_utf8_text(self) -> None:
        file_path = os.path.join(self.safe_root, "note.txt")
        with open(file_path, "w", encoding="utf-8") as handle:
            handle.write("hello world")

        result = self.skill.read_text_file(file_path, max_bytes=64)

        self.assertTrue(result["ok"])
        self.assertEqual("hello world", result["text"])
        self.assertFalse(result["truncated"])

    def test_read_text_file_truncates_bounded_reads(self) -> None:
        file_path = os.path.join(self.safe_root, "long.txt")
        with open(file_path, "w", encoding="utf-8") as handle:
            handle.write("abcdefghij")

        result = self.skill.read_text_file(file_path, max_bytes=4)

        self.assertTrue(result["ok"])
        self.assertEqual("abcd", result["text"])
        self.assertEqual(4, result["bytes_read"])
        self.assertTrue(result["truncated"])

    def test_binary_files_are_rejected(self) -> None:
        file_path = os.path.join(self.safe_root, "blob.bin")
        with open(file_path, "wb") as handle:
            handle.write(b"\x00\x01\x02")

        result = self.skill.read_text_file(file_path)

        self.assertFalse(result["ok"])
        self.assertEqual("binary_file_not_supported", result["error_kind"])

    def test_path_traversal_outside_allowed_root_is_blocked(self) -> None:
        file_path = os.path.join(self.outside_root, "outside.txt")
        with open(file_path, "w", encoding="utf-8") as handle:
            handle.write("outside")

        result = self.skill.read_text_file("../outside/outside.txt")

        self.assertFalse(result["ok"])
        self.assertEqual("outside_allowed_roots", result["error_kind"])

    def test_symlink_escape_into_sensitive_root_is_blocked(self) -> None:
        secret_path = os.path.join(self.private_root, "secret.txt")
        with open(secret_path, "w", encoding="utf-8") as handle:
            handle.write("secret")
        link_path = os.path.join(self.safe_root, "secret-link.txt")
        os.symlink(secret_path, link_path)

        result = self.skill.read_text_file(link_path)

        self.assertFalse(result["ok"])
        self.assertEqual("sensitive_path_blocked", result["error_kind"])

    def test_sensitive_paths_are_blocked_even_within_allowed_root(self) -> None:
        secret_path = os.path.join(self.private_root, "secret.txt")
        with open(secret_path, "w", encoding="utf-8") as handle:
            handle.write("secret")

        result = self.skill.stat_path(secret_path)

        self.assertFalse(result["ok"])
        self.assertEqual("sensitive_path_blocked", result["error_kind"])

    def test_missing_paths_return_grounded_not_found(self) -> None:
        result = self.skill.stat_path(os.path.join(self.safe_root, "missing.txt"))

        self.assertFalse(result["ok"])
        self.assertEqual("not_found", result["error_kind"])

    def test_search_filenames_works_inside_allowed_root(self) -> None:
        docs_dir = os.path.join(self.safe_root, "docs")
        os.makedirs(docs_dir, exist_ok=True)
        readme_path = os.path.join(docs_dir, "PROJECT_STATUS.md")
        with open(readme_path, "w", encoding="utf-8") as handle:
            handle.write("status")

        result = self.skill.search_filenames(self.safe_root, "project_status", max_results=10, max_depth=3)

        self.assertTrue(result["ok"])
        self.assertEqual(1, len(result["results"]))
        self.assertEqual(readme_path, result["results"][0]["path"])

    def test_search_text_works_for_small_text_files(self) -> None:
        docs_dir = os.path.join(self.safe_root, "docs")
        os.makedirs(docs_dir, exist_ok=True)
        readme_path = os.path.join(docs_dir, "README.md")
        with open(readme_path, "w", encoding="utf-8") as handle:
            handle.write("first line\nTODO: finish this\n")

        result = self.skill.search_text(self.safe_root, "TODO", max_results=10, max_files=10, max_bytes_per_file=256)

        self.assertTrue(result["ok"])
        self.assertEqual(1, len(result["results"]))
        self.assertEqual(readme_path, result["results"][0]["path"])
        self.assertIn("TODO", result["results"][0]["snippet"])
        self.assertEqual(2, result["results"][0]["line_number"])

    def test_search_text_skips_binary_files(self) -> None:
        file_path = os.path.join(self.safe_root, "blob.bin")
        with open(file_path, "wb") as handle:
            handle.write(b"\x00TODO\x01")

        result = self.skill.search_text(self.safe_root, "TODO", max_results=10, max_files=10, max_bytes_per_file=256)

        self.assertFalse(result["ok"])
        self.assertEqual("no_matches", result["error_kind"])
        self.assertEqual(1, result["files_examined"])

    def test_search_filenames_respects_max_depth(self) -> None:
        level1 = os.path.join(self.safe_root, "a")
        level2 = os.path.join(level1, "b")
        os.makedirs(level2, exist_ok=True)
        target_path = os.path.join(level2, "target.txt")
        with open(target_path, "w", encoding="utf-8") as handle:
            handle.write("deep")

        result = self.skill.search_filenames(self.safe_root, "target", max_results=10, max_depth=0)

        self.assertFalse(result["ok"])
        self.assertEqual("no_matches", result["error_kind"])

    def test_search_filenames_respects_max_results(self) -> None:
        for index in range(3):
            with open(os.path.join(self.safe_root, f"note-{index}.txt"), "w", encoding="utf-8") as handle:
                handle.write("x")

        result = self.skill.search_filenames(self.safe_root, "note", max_results=2, max_depth=2)

        self.assertTrue(result["ok"])
        self.assertEqual(2, len(result["results"]))
        self.assertTrue(result["truncated"])
        self.assertEqual("search_limit_reached", result["stop_reason"])

    def test_search_text_respects_max_files(self) -> None:
        for index in range(3):
            with open(os.path.join(self.safe_root, f"file-{index}.txt"), "w", encoding="utf-8") as handle:
                handle.write(f"line {index}\n")

        result = self.skill.search_text(self.safe_root, "missing", max_results=10, max_files=1, max_bytes_per_file=256)

        self.assertFalse(result["ok"])
        self.assertEqual("no_matches", result["error_kind"])
        self.assertEqual(1, result["files_examined"])
        self.assertTrue(result["truncated"])
        self.assertEqual("search_limit_reached", result["stop_reason"])

    def test_sensitive_paths_are_not_searched(self) -> None:
        secret_path = os.path.join(self.private_root, "secret.txt")
        with open(secret_path, "w", encoding="utf-8") as handle:
            handle.write("secret")

        result = self.skill.search_filenames(self.private_root, "secret", max_results=10, max_depth=2)

        self.assertFalse(result["ok"])
        self.assertEqual("sensitive_path_blocked", result["error_kind"])

    def test_search_skips_symlink_escape_targets(self) -> None:
        secret_path = os.path.join(self.private_root, "secret.txt")
        with open(secret_path, "w", encoding="utf-8") as handle:
            handle.write("TODO secret")
        link_path = os.path.join(self.safe_root, "secret-link.txt")
        os.symlink(secret_path, link_path)

        filename_result = self.skill.search_filenames(self.safe_root, "secret-link", max_results=10, max_depth=2)
        text_result = self.skill.search_text(self.safe_root, "TODO", max_results=10, max_files=10, max_bytes_per_file=256)

        self.assertFalse(filename_result["ok"])
        self.assertEqual("no_matches", filename_result["error_kind"])
        self.assertFalse(text_result["ok"])
        self.assertEqual("no_matches", text_result["error_kind"])
