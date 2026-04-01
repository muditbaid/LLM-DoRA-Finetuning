#!/usr/bin/env bash
set -euo pipefail

TABLES_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREVIEW_DIR="$TABLES_DIR/preview"
VISUALS_DIR="$TABLES_DIR/visuals"
DIFF_DIR="$TABLES_DIR/diff"

mkdir -p "$VISUALS_DIR"
mkdir -p "$DIFF_DIR"

LATEX_ENGINE=""
if command -v pdflatex >/dev/null 2>&1; then
  LATEX_ENGINE="pdflatex"
elif command -v tectonic >/dev/null 2>&1; then
  LATEX_ENGINE="tectonic"
elif command -v latexmk >/dev/null 2>&1; then
  LATEX_ENGINE="latexmk"
fi

if [[ -z "$LATEX_ENGINE" ]]; then
  cat <<'EOF'
No LaTeX engine was found on PATH.

Checked for:
  - pdflatex
  - tectonic
  - latexmk

Note:
  Installing the Python package named "pdflatex" is not enough. It is only a wrapper
  and still requires a real TeX engine to be installed on the system.
EOF
  exit 1
fi

cd "$PREVIEW_DIR"

for tex in *_preview.tex; do
  name="$(basename "$tex" .tex)"

  case "$LATEX_ENGINE" in
    pdflatex)
      pdflatex -interaction=nonstopmode -halt-on-error -output-directory "$DIFF_DIR" "$tex"
      ;;
    tectonic)
      tectonic --outdir "$DIFF_DIR" "$tex"
      ;;
    latexmk)
      latexmk -pdf -interaction=nonstopmode -halt-on-error -output-directory="$DIFF_DIR" "$tex"
      ;;
  esac

  cp "$DIFF_DIR/${name}.pdf" "$VISUALS_DIR/${name}.pdf"

  if command -v pdftoppm >/dev/null 2>&1; then
    pdftoppm -png "$VISUALS_DIR/${name}.pdf" "$VISUALS_DIR/${name}"
  fi
done

echo "Rendered previews to $VISUALS_DIR"
echo "Build artifacts stored in $DIFF_DIR"
