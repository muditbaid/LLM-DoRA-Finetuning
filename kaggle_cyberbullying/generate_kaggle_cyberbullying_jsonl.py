import argparse
import json
from pathlib import Path
import pandas as pd


DEFAULT_PROMPT = (
    "You review social media posts for bullying. "
    "Reply with a single line in the format: label: bully|not_bully; type: age|gender|ethnicity|religion|none. "
    "Use type: none whenever the post is not bullying."
)


def normalize_label(x: str) -> str:
    x = (x or "").strip().lower()
    if x == "bully":
        return "bully"
    return "not_bully"


def normalize_type(x: str, label: str) -> str:
    x = (x or "").strip().lower()
    if label == "not_bully":
        return "none"
    # map dataset values to allowed set
    if x in {"age", "gender", "ethnicity", "religion"}:
        return x
    # fallback for any unexpected values
    return "none"


def row_to_record(text: str, label: str, ctype: str) -> dict:
    response = (
        f"label: {label}; type: {ctype}"
    )
    return {
        "instruction": DEFAULT_PROMPT,
        "input": text.strip(),
        "output": response,
        "system": DEFAULT_PROMPT,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path, help="Path to kaggle_cyberbullying.csv")
    ap.add_argument("out_dir", type=Path, help="Output directory (e.g., LLaMA-Factory/data)")
    ap.add_argument("--val_ratio", type=float, default=0.1, help="Validation split ratio")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prompt", type=str, default=None, help="Prompt text to use (overrides defaults).")
    ap.add_argument("--prompt-file", type=Path, default=None, help="Read prompt text from file.")
    ap.add_argument(
        "--prompt-mode",
        choices=("both", "system_only", "instruction_only"),
        default="system_only",
        help="Where to place the shared prompt guidance (default: system_only).",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    # Basic cleaning
    df = df[["tweet_text", "cyberbullying_type", "class_label"]].dropna()
    df = df.rename(columns={"tweet_text": "text", "cyberbullying_type": "type", "class_label": "label"})

    # Normalize
    df["label"] = df["label"].map(normalize_label)
    df["type"] = [normalize_type(t, lbl) for t, lbl in zip(df["type"], df["label"])]

    # Shuffle + stratified by label (simple approach)
    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    # Split
    n_total = len(df)
    n_val = int(n_total * args.val_ratio)
    val_df = df.iloc[:n_val]
    train_df = df.iloc[n_val:]

    # Resolve prompt overrides
    prompt_text = DEFAULT_PROMPT
    if args.prompt_file and args.prompt_file.exists():
        prompt_text = args.prompt_file.read_text(encoding="utf-8").strip()
    elif args.prompt is not None:
        prompt_text = args.prompt.strip()

    instruction_text = prompt_text if args.prompt_mode != "system_only" else ""
    system_text = prompt_text if args.prompt_mode != "instruction_only" else ""

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.out_dir / "kaggle_cyberbullying_train.jsonl"
    val_path = args.out_dir / "kaggle_cyberbullying_validation.jsonl"

    with train_path.open("w", encoding="utf-8") as f:
        for _, r in train_df.iterrows():
            rec = row_to_record(r["text"], r["label"], r["type"])
            rec["instruction"] = instruction_text
            rec["system"] = system_text
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with val_path.open("w", encoding="utf-8") as f:
        for _, r in val_df.iterrows():
            rec = row_to_record(r["text"], r["label"], r["type"])
            rec["instruction"] = instruction_text
            rec["system"] = system_text
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("Wrote:")
    print(train_path)
    print(val_path)


if __name__ == "__main__":
    main()
