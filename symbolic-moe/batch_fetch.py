#!/usr/bin/env python3
"""
Fetch OpenAI Batch API results, archive output/error files, and postprocess into pipeline JSONL.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import requests
from skill_parsing import parse_skills_from_text


RAW_TO_CANON = {
    "insult": "directed_insult",
    "insulting": "directed_insult",
    "insulting_language": "directed_insult",
    "insulting_tone": "directed_insult",
    "insulting_remark": "directed_insult",
    "verbal_abuse": "personal_attack",
    "personal_insult": "personal_attack",
    "personal_attack": "personal_attack",
    "bullying": "peer_aggression",
    "mocking": "mockery",
    "mocking_tone": "mockery",
    "teasing": "mockery",
    "sarcasm": "sarcastic_insult",
    "sarcastic": "sarcastic_insult",
    "sarcastic_tone": "sarcastic_insult",
    "swearing": "profanity_tone",
    "swear_words": "profanity_tone",
    "cursing": "profanity_tone",
    "curse_words": "profanity_tone",
    "offensive_language": "toxic_tone",
    "aggressive_tone": "toxic_tone",
    "rude_tone": "toxic_tone",
    "hostile_tone": "hostile_sentiment",
    "anger": "hostile_sentiment",
    "hostility": "hostile_sentiment",
    "threat": "explicit_threat",
    "threatening": "explicit_threat",
    "violent_threat": "explicit_threat",
}


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


def load_skills(skills_path: Path) -> List[str]:
    return [line.strip() for line in skills_path.read_text().splitlines() if line.strip()]


def parse_skills(text: str, allowed: List[str]) -> List[str]:
    return parse_skills_from_text(text, allowed, RAW_TO_CANON)


def parse_labels(text: str) -> List[str]:
    match = re.search(r"Labels\\s*:(.*)", text, flags=re.IGNORECASE)
    if not match:
        return []
    remainder = match.group(1).strip().lower()
    if not remainder or remainder == "none":
        return []
    allowed = {"hate", "bully", "offense", "threat"}
    labels: List[str] = []
    for token in remainder.split(","):
        label = token.strip()
        if label in allowed:
            labels.append(label)
    return list(dict.fromkeys(labels))


def extract_content(body: Dict) -> str:
    if not body:
        return ""
    if "choices" in body and body["choices"]:
        msg = body["choices"][0].get("message", {})
        content = msg.get("content", "")
        if isinstance(content, str):
            return content
    if "output" in body:
        chunks: List[str] = []
        for item in body.get("output", []):
            for part in item.get("content", []):
                if part.get("type") == "output_text":
                    chunks.append(part.get("text", ""))
        return "\n".join(chunks)
    return ""


def download_file(file_id: str, api_key: str, out_path: Path) -> None:
    url = f"https://api.openai.com/v1/files/{file_id}/content"
    headers = {"Authorization": f"Bearer {api_key}"}
    resp = requests.get(url, headers=headers, timeout=120)
    resp.raise_for_status()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(resp.content)


def fetch_batch(batch_id: str, api_key: str) -> Dict:
    url = f"https://api.openai.com/v1/batches/{batch_id}"
    headers = {"Authorization": f"Bearer {api_key}"}
    resp = requests.get(url, headers=headers, timeout=60)
    resp.raise_for_status()
    return resp.json()


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
    parser = argparse.ArgumentParser(description="Fetch batch outputs and postprocess.")
    parser.add_argument("--batch-ids", type=Path, default=Path("symbolic-moe/batch_files/batch_ids.txt"))
    parser.add_argument("--input", type=Path, default=Path("symbolic-moe/validation_pool.jsonl"))
    parser.add_argument("--skills-file", type=Path, default=Path("symbolic-moe/skills.txt"))
    parser.add_argument("--batch-dir", type=Path, default=Path("symbolic-moe/batch_files"))
    parser.add_argument("--skills-out", type=Path, default=Path("symbolic-moe/validation_pool_skills_gpt.jsonl"))
    parser.add_argument("--labels-out", type=Path, default=Path("symbolic-moe/validation_pool_multilabel_gpt.jsonl"))
    parser.add_argument("--env-file", type=Path, default=Path("symbolic-moe/.env"))
    args = parser.parse_args()

    api_key = load_api_key(args.env_file)
    if not api_key:
        raise SystemExit("Missing OPENAI_API_KEY environment variable.")

    batch_records = read_jsonl(args.batch_ids)
    base_rows = read_jsonl(args.input)
    by_id = {row.get("id"): row for row in base_rows}
    skills_vocab = load_skills(args.skills_file)

    output_dir = args.batch_dir / "output_files"
    error_dir = args.batch_dir / "error_files"

    responses: Dict[str, Dict[str, str]] = {"skills": {}, "labels": {}}

    for record in batch_records:
        kind = record.get("kind")
        batch_id = record.get("batch_id")
        if not kind or not batch_id:
            continue
        info = fetch_batch(batch_id, api_key)
        status = info.get("status")
        if status != "completed":
            print(f"[batch] {batch_id} ({kind}) not completed: {status}")
            continue

        output_file_id = info.get("output_file_id")
        error_file_id = info.get("error_file_id")

        if output_file_id:
            out_path = output_dir / f"{batch_id}.jsonl"
            download_file(output_file_id, api_key, out_path)
        if error_file_id:
            err_path = error_dir / f"{batch_id}.jsonl"
            download_file(error_file_id, api_key, err_path)

        if not output_file_id:
            continue
        batch_rows = read_jsonl(output_dir / f"{batch_id}.jsonl")
        for row in batch_rows:
            custom_id = row.get("custom_id", "")
            resp = row.get("response", {})
            body = resp.get("body", {})
            content = extract_content(body).strip()
            if not custom_id:
                continue
            if custom_id.startswith("skills:"):
                rid = custom_id.split(":", 1)[1]
                responses["skills"][rid] = content
            elif custom_id.startswith("labels:"):
                rid = custom_id.split(":", 1)[1]
                responses["labels"][rid] = content

    skills_rows: List[Dict] = []
    labels_rows: List[Dict] = []

    for rid, base in by_id.items():
        if rid is None:
            continue
        skills_text = responses["skills"].get(rid, "")
        labels_text = responses["labels"].get(rid, "")
        if skills_text:
            skills_rows.append(
                {
                    **base,
                    "predicted_skills": parse_skills(skills_text, skills_vocab),
                    "keyword_responses": [skills_text],
                }
            )
        if labels_text:
            labels_rows.append(
                {
                    **base,
                    "predicted_labels": parse_labels(labels_text),
                    "label_response": labels_text,
                }
            )

    if skills_rows:
        write_jsonl(args.skills_out, skills_rows)
        print(f"[batch] Wrote skills output to {args.skills_out}")
    if labels_rows:
        write_jsonl(args.labels_out, labels_rows)
        print(f"[batch] Wrote labels output to {args.labels_out}")


if __name__ == "__main__":
    main()
