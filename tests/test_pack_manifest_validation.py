from __future__ import annotations

import json
import os
import tempfile
import unittest

from agent.packs.manifest import PackManifestError, compute_permissions_hash, load_manifest


class TestPackManifestValidation(unittest.TestCase):
    def test_invalid_manifest_missing_pack_id_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = os.path.join(tmpdir, "pack.json")
            with open(manifest_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "version": "0.1.0",
                        "entrypoints": ["skills.sample:handler"],
                        "trust": "trusted",
                        "permissions": {"ifaces": ["sample.run"]},
                    },
                    handle,
                    ensure_ascii=True,
                )
            with self.assertRaises(PackManifestError):
                load_manifest(manifest_path)

    def test_invalid_manifest_bad_trust_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = os.path.join(tmpdir, "pack.json")
            with open(manifest_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "pack_id": "sample_pack",
                        "version": "0.1.0",
                        "entrypoints": ["skills.sample:handler"],
                        "trust": "dangerous",
                        "permissions": {"ifaces": ["sample.run"]},
                    },
                    handle,
                    ensure_ascii=True,
                )
            with self.assertRaises(PackManifestError):
                load_manifest(manifest_path)

    def test_valid_manifest_hash_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = os.path.join(tmpdir, "pack.json")
            payload = {
                "pack_id": "sample_pack",
                "version": "0.1.0",
                "title": "Sample",
                "description": "test",
                "entrypoints": ["skills.sample:handler"],
                "trust": "trusted",
                "permissions": {
                    "ifaces": ["sample.run", "sample.read"],
                    "fs": {"read": ["/tmp/**"], "write": []},
                    "net": {"allow_domains": ["example.com"], "deny": ["*"]},
                    "proc": {"spawn": []},
                },
            }
            with open(manifest_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=True)
            manifest = load_manifest(manifest_path)
            first_hash = compute_permissions_hash(manifest.permissions)
            second_hash = compute_permissions_hash(manifest.permissions)
            self.assertEqual(first_hash, second_hash)
            self.assertEqual("sample_pack", manifest.pack_id)


if __name__ == "__main__":
    unittest.main()
