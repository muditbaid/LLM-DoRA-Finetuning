#!/usr/bin/env python3
"""
Create and submit OpenAI Batch API jobs for skill tagging and multilabel classification.

Outputs:
- batch_files/skills_requests.jsonl
- batch_files/labels_requests.jsonl
- batch_ids.txt (JSONL with batch ids)
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List

import requests
from skill_parsing import STRICT_SKILLS_OUTPUT_CONTRACT


SKILLS_PROMPT_TEMPLATE = """You are tagging which conceptual skills are expressed in a social media post for hate/offense/bullying/threat detection.

Allowed skills (you may ONLY choose from this list and MUST copy each name EXACTLY):
{skills}

{output_contract}

POST: {post}
"""

LABELS_PROMPT_TEMPLATE = """You are a social content moderation expert with strong domain knowledge on hate, bully, offensive and threatening speech.
Your task is to give one or more best suited labels among [hate, bully, offense, threat] to every social media post.
If the post is neither of those labels, then output none.

Output format (exactly):
Labels: label_one,label_two
or, when empty:
Labels: none

POST: {post}
"""


def read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_skills(skills_path: Path) -> str:
    skills = [line.strip() for line in skills_path.read_text().splitlines() if line.strip()]
    return "\n".join(skills)


def build_requests(
    rows: List[Dict],
    model: str,
    skills_text: str,
    max_output_tokens: int,
    reasoning_effort: str,
    verbosity: str,
) -> Dict[str, List[Dict]]:
    skills_requests: List[Dict] = []
    labels_requests: List[Dict] = []

    for rec in rows:
        post = rec.get("input", "")
        rid = rec.get("id") or rec.get("dataset", "row")

        skills_prompt = SKILLS_PROMPT_TEMPLATE.format(
            skills=skills_text,
            output_contract=STRICT_SKILLS_OUTPUT_CONTRACT.strip(),
            post=post,
        )
        labels_prompt = LABELS_PROMPT_TEMPLATE.format(post=post)

        skills_requests.append(
            {
                "custom_id": f"skills:{rid}",
                "method": "POST",
                "url": "/v1/responses",
                "body": {
                    "model": model,
                    "input": [
                        {
                            "role": "system",
                            "content": [{"type": "input_text", "text": "You are a careful tagging assistant."}],
                        },
                        {"role": "user", "content": [{"type": "input_text", "text": skills_prompt}]},
                    ],
                    "max_output_tokens": max_output_tokens,
                    "reasoning": {"effort": reasoning_effort},
                    "text": {"verbosity": verbosity},
                },
            }
        )

        labels_requests.append(
            {
                "custom_id": f"labels:{rid}",
                "method": "POST",
                "url": "/v1/responses",
                "body": {
                    "model": model,
                    "input": [
                        {
                            "role": "system",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": "You are a careful and unbiased moderation assistant.",
                                }
                            ],
                        },
                        {"role": "user", "content": [{"type": "input_text", "text": labels_prompt}]},
                    ],
                    "max_output_tokens": max_output_tokens,
                    "reasoning": {"effort": reasoning_effort},
                    "text": {"verbosity": verbosity},
                },
            }
        )

    return {"skills": skills_requests, "labels": labels_requests}


def upload_file(path: Path, api_key: str) -> str:
    url = "https://api.openai.com/v1/files"
    headers = {"Authorization": f"Bearer {api_key}"}
    with path.open("rb") as f:
        files = {"file": (path.name, f)}
        data = {"purpose": "batch"}
        resp = requests.post(url, headers=headers, files=files, data=data, timeout=120)
    resp.raise_for_status()
    return resp.json()["id"]


def create_batch(input_file_id: str, api_key: str, endpoint: str) -> str:
    url = "https://api.openai.com/v1/batches"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"input_file_id": input_file_id, "endpoint": endpoint, "completion_window": "24h"}
    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["id"]


def append_batch_id(path: Path, record: Dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_api_key(env_path: Path | None = None) -> str | None:
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        return api_key
    candidates = [env_path] if env_path else []
    candidates.append(Path("symbolic-moe/.env"))
    for path in candidates:
        if not path or not path.exists():
            continue
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() == "OPENAI_API_KEY":
                return value.strip().strip("\"'")
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Create and submit OpenAI batch jobs.")
    parser.add_argument("--input", type=Path, default=Path("symbolic-moe/validation_pool.jsonl"))
    parser.add_argument("--skills-file", type=Path, default=Path("symbolic-moe/skills.txt"))
    parser.add_argument("--model", type=str, default="gpt-5.1")
    parser.add_argument("--batch-dir", type=Path, default=Path("symbolic-moe/batch_files"))
    parser.add_argument("--batch-ids", type=Path, default=Path("symbolic-moe/batch_files/batch_ids.txt"))
    parser.add_argument("--endpoint", type=str, default="/v1/responses")
    parser.add_argument("--max-output-tokens", type=int, default=120)
    parser.add_argument("--reasoning-effort", type=str, default="medium")
    parser.add_argument("--verbosity", type=str, default="low")
    parser.add_argument("--env-file", type=Path, default=Path("symbolic-moe/.env"))
    parser.add_argument("--skills-only", action="store_true", help="Submit only skills batch.")
    parser.add_argument("--labels-only", action="store_true", help="Submit only labels batch.")
    args = parser.parse_args()

    api_key = load_api_key(args.env_file)
    if not api_key:
        raise SystemExit("Missing OPENAI_API_KEY environment variable.")

    rows = read_jsonl(args.input)
    skills_text = load_skills(args.skills_file)
    requests_by_kind = build_requests(
        rows,
        args.model,
        skills_text,
        args.max_output_tokens,
        args.reasoning_effort,
        args.verbosity,
    )

    args.batch_dir.mkdir(parents=True, exist_ok=True)
    skills_path = args.batch_dir / "skills_requests.jsonl"
    labels_path = args.batch_dir / "labels_requests.jsonl"
    write_jsonl(skills_path, requests_by_kind["skills"])
    write_jsonl(labels_path, requests_by_kind["labels"])

    args.batch_ids.parent.mkdir(parents=True, exist_ok=True)

    skills_file_id = None
    labels_file_id = None
    if not args.labels_only:
        skills_file_id = upload_file(skills_path, api_key)
    if not args.skills_only:
        labels_file_id = upload_file(labels_path, api_key)

    if skills_file_id:
        skills_batch_id = create_batch(skills_file_id, api_key, args.endpoint)
        append_batch_id(
            args.batch_ids,
            {
                "kind": "skills",
                "batch_id": skills_batch_id,
                "input_file_id": skills_file_id,
                "requests_path": str(skills_path),
            },
        )
        print(f"[batch] Submitted skills batch: {skills_batch_id}")

    if labels_file_id:
        labels_batch_id = create_batch(labels_file_id, api_key, args.endpoint)
        append_batch_id(
            args.batch_ids,
            {
                "kind": "labels",
                "batch_id": labels_batch_id,
                "input_file_id": labels_file_id,
                "requests_path": str(labels_path),
            },
        )
        print(f"[batch] Submitted labels batch: {labels_batch_id}")
    print(f"[batch] Logged to {args.batch_ids}")


if __name__ == "__main__":
    main()
