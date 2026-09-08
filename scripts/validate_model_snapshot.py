#!/usr/bin/env python3
"""Validate the minimal Hugging Face snapshot baked into the backend image."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any


REQUIRED_JSON_FILES = (
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Invalid JSON file {path}: {exc}") from exc


def _required_file(root: Path, relative_name: str) -> Path:
    path = (root / relative_name).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise SystemExit(f"Model index contains an unsafe path: {relative_name}") from exc
    if not path.is_file() or path.stat().st_size <= 0:
        raise SystemExit(f"Required model file is missing or empty: {path}")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path, name: str) -> dict[str, int | str]:
    return {
        "name": name,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def validate_snapshot(
    root: Path,
    expected_repo: str | None,
    expected_revision: str | None,
) -> dict[str, Any]:
    root = root.resolve()
    if not root.is_dir():
        raise SystemExit(f"Model snapshot directory does not exist: {root}")

    for filename in REQUIRED_JSON_FILES:
        _required_file(root, filename)
        _read_json(root / filename)

    index_path = root / "model.safetensors.index.json"
    index = _read_json(index_path)
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        raise SystemExit(f"Model index has no non-empty weight_map: {index_path}")

    shard_values = list(weight_map.values())
    if not shard_values or not all(isinstance(name, str) for name in shard_values):
        raise SystemExit(f"Model index contains invalid shard names: {index_path}")
    shard_names = sorted(set(shard_values))
    shard_pattern = re.compile(r"model-\d{5}-of-\d{5}\.safetensors")
    for shard_name in shard_names:
        shard_path = PurePosixPath(shard_name)
        if shard_path.name != shard_name or not shard_pattern.fullmatch(shard_name):
            raise SystemExit(f"Model index contains an unexpected shard path: {shard_name!r}")

    allowed_names = set(REQUIRED_JSON_FILES) | set(shard_names) | {
        ".complete",
        "manifest.json",
    }
    for entry in root.iterdir():
        if entry.is_symlink() or not entry.is_file():
            raise SystemExit(f"Unexpected non-file in model snapshot: {entry}")
        if entry.name not in allowed_names:
            raise SystemExit(f"Unexpected file in model snapshot: {entry}")

    shards: list[dict[str, int | str]] = []
    total_bytes = 0
    for shard_name in shard_names:
        shard_path = _required_file(root, shard_name)
        size_bytes = shard_path.stat().st_size
        total_bytes += size_bytes
        shards.append(_file_record(shard_path, shard_name))

    metadata_files = [
        _file_record(root / filename, filename) for filename in REQUIRED_JSON_FILES
    ]

    return {
        "schema_version": 1,
        "repository": expected_repo,
        "revision": expected_revision,
        "metadata_files": metadata_files,
        "shard_count": len(shards),
        "total_shard_bytes": total_bytes,
        "shards": shards,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--expected-repo")
    parser.add_argument("--expected-revision")
    parser.add_argument("--write-manifest", type=Path)
    parser.add_argument("--require-manifest", action="store_true")
    args = parser.parse_args()

    manifest = validate_snapshot(
        args.snapshot,
        expected_repo=args.expected_repo,
        expected_revision=args.expected_revision,
    )

    manifest_path = args.snapshot.resolve() / "manifest.json"
    if args.require_manifest:
        recorded = _read_json(manifest_path)
        if recorded != manifest:
            raise SystemExit(
                f"Model manifest does not match the downloaded snapshot: {manifest_path}"
            )

    if args.write_manifest:
        output_path = args.write_manifest.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    print(
        "Validated model snapshot: "
        f"{manifest['shard_count']} shards, {manifest['total_shard_bytes']} bytes"
    )


if __name__ == "__main__":
    main()
