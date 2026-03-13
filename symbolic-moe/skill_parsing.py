"""
Shared skill output contract + tolerant parser used across inference paths.
"""
from __future__ import annotations

import json
import re
from typing import Mapping, Sequence


STRICT_SKILLS_OUTPUT_CONTRACT = """Return EXACTLY one line in this format:
Skills: ["skill_one","skill_two"]

Rules:
- Output only skills from the allowed list.
- If none apply, output exactly: Skills: []
- No explanation.
- No extra text.
- No markdown.
"""


def parse_skills_from_text(
    text: str,
    allowed_skills: Sequence[str],
    raw_to_canon: Mapping[str, str] | None = None,
) -> list[str]:
    allowed = [s.strip().lower().replace(" ", "_") for s in allowed_skills if s and s.strip()]
    allowed_set = set(allowed)
    alias_map = {
        k.strip().lower().replace(" ", "_"): v.strip().lower().replace(" ", "_")
        for k, v in (raw_to_canon or {}).items()
    }

    # Prefer parsing from an explicit Skills: line.
    match = re.search(r"^\s*Skills\s*:\s*(.*)$", text, flags=re.IGNORECASE | re.MULTILINE)
    if not match:
        return []

    remainder = match.group(1).strip()
    if not remainder or remainder.lower() in {"none", "[]"}:
        return []

    parsed_tokens: list[str] = []

    if remainder.startswith("[") and remainder.endswith("]"):
        # Strict JSON first.
        try:
            parsed = json.loads(remainder)
            if isinstance(parsed, list):
                parsed_tokens = [str(item) for item in parsed if isinstance(item, str)]
        except json.JSONDecodeError:
            # Tolerate bracketed, non-JSON output like [skill_a, skill_b].
            inner = remainder[1:-1].strip()
            if inner:
                parsed_tokens = [token.strip().strip("\"'") for token in inner.split(",") if token.strip()]
    else:
        # Tolerate CSV after "Skills:".
        parsed_tokens = [token.strip().strip("\"'") for token in remainder.split(",") if token.strip()]

    out: list[str] = []
    for token in parsed_tokens:
        norm = token.lower().replace(" ", "_")
        norm = alias_map.get(norm, norm)
        if norm in allowed_set:
            out.append(norm)

    # Final fallback: if we still parsed nothing, scan full text for exact skills.
    if not out:
        normalized_text = re.sub(r"[^a-z0-9_\s]", " ", text.lower().replace(" ", "_"))
        for skill in allowed:
            if re.search(rf"\b{re.escape(skill)}\b", normalized_text):
                out.append(skill)

    return list(dict.fromkeys(out))
