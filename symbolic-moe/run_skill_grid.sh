#!/usr/bin/env bash
set -euo pipefail

INPUT="symbolic-moe/validation_pool.jsonl"
OUTDIR="symbolic-moe/grid_runs"
mkdir -p "$OUTDIR"

RESULTS_CSV="$OUTDIR/summary.csv"
echo "runs,temp,top_p,rows,empty,empty_rate,pos_rows,pos_empty,pos_empty_rate,neg_rows,neg_empty,neg_empty_rate,avg_skills" > "$RESULTS_CSV"

for RUNS in 5 7 9; do
  for TEMP in 0.15 0.2; do
    for TOPP in 0.6 0.7; do
      TAG="r${RUNS}_t${TEMP}_p${TOPP}"
      OUT="$OUTDIR/validation_pool_skills_${TAG}.jsonl"
      LOG="$OUTDIR/skill_inference_${TAG}.log"

      echo "[grid] $TAG"
      python symbolic-moe/skill_inference.py \
        --input "$INPUT" \
        --output "$OUT" \
        --runs "$RUNS" \
        --min-count 2 \
        --temperature "$TEMP" \
        --top-p "$TOPP" \
        > "$LOG" 2>&1

      python - <<PY >> "$RESULTS_CSV"
import json
from pathlib import Path

p = Path("$OUT")
rows = empty = 0
pos_rows = pos_empty = 0
neg_rows = neg_empty = 0
skill_total = 0

for line in p.open("r", encoding="utf-8"):
    s = line.strip()
    if not s:
        continue
    o = json.loads(s)
    rows += 1
    skills = o.get("predicted_skills") or []
    skill_total += len(skills)

    gold = (o.get("label") or o.get("output") or "").strip().lower()
    is_neg = gold.startswith("not ") or gold.startswith("not_")
    if is_neg:
        neg_rows += 1
    else:
        pos_rows += 1

    if not skills:
        empty += 1
        if is_neg:
            neg_empty += 1
        else:
            pos_empty += 1

def r(a,b): return (a/b) if b else 0.0

print(f"${RUNS},${TEMP},${TOPP},{rows},{empty},{r(empty,rows):.6f},{pos_rows},{pos_empty},{r(pos_empty,pos_rows):.6f},{neg_rows},{neg_empty},{r(neg_empty,neg_rows):.6f},{(skill_total/rows) if rows else 0.0:.6f}")
PY
    done
  done
done

echo
echo "[done] raw summary: $RESULTS_CSV"
echo "[done] ranked by pos_empty_rate then avg_skills:"
python - <<'PY'
import csv
from pathlib import Path
p=Path("symbolic-moe/grid_runs/summary.csv")
rows=list(csv.DictReader(p.open()))
rows.sort(key=lambda x:(float(x["pos_empty_rate"]), float(x["avg_skills"])))
for r in rows:
    print(f'runs={r["runs"]} temp={r["temp"]} top_p={r["top_p"]} | '
          f'pos_empty_rate={float(r["pos_empty_rate"]):.4f} '
          f'neg_empty_rate={float(r["neg_empty_rate"]):.4f} '
          f'empty_rate={float(r["empty_rate"]):.4f} '
          f'avg_skills={float(r["avg_skills"]):.3f}')
PY
