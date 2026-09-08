from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from validate_model_snapshot import REQUIRED_JSON_FILES, validate_snapshot  # noqa: E402


class ModelSnapshotValidationTests(unittest.TestCase):
    def _snapshot(self) -> tuple[tempfile.TemporaryDirectory[str], Path]:
        temporary_directory = tempfile.TemporaryDirectory()
        root = Path(temporary_directory.name)
        shard_name = "model-00001-of-00001.safetensors"
        for name in REQUIRED_JSON_FILES:
            content = {"weight_map": {"tensor": shard_name}} if name.endswith("index.json") else {}
            (root / name).write_text(json.dumps(content), encoding="utf-8")
        (root / shard_name).write_bytes(b"original model bytes")
        return temporary_directory, root

    def test_manifest_hashes_detect_same_size_tampering(self) -> None:
        temporary_directory, root = self._snapshot()
        self.addCleanup(temporary_directory.cleanup)

        manifest = validate_snapshot(root, "model/repo", "revision-sha")
        (root / "model-00001-of-00001.safetensors").write_bytes(
            b"tampered model bytes"
        )
        changed = validate_snapshot(root, "model/repo", "revision-sha")

        self.assertNotEqual(manifest, changed)
        self.assertEqual(
            manifest["shards"][0]["size_bytes"],
            changed["shards"][0]["size_bytes"],
        )

    def test_rejects_unexpected_root_file(self) -> None:
        temporary_directory, root = self._snapshot()
        self.addCleanup(temporary_directory.cleanup)
        (root / "modeling_remote.py").write_text("raise SystemExit", encoding="utf-8")

        with self.assertRaisesRegex(SystemExit, "Unexpected file"):
            validate_snapshot(root, "model/repo", "revision-sha")


if __name__ == "__main__":
    unittest.main()
