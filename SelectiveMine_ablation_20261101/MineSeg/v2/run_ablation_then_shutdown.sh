#!/bin/bash
# Run the frozen-weight ablation, write the four paper files, mail the
# results, then stop the container. A failed step does not shut the machine down.
set -u

ROOT=/root/autodl-tmp/MineSeg/v2
LOG=$ROOT/runs/selective_pa3_ablation/pipeline.log
PAPER=$ROOT/paper_ablation
MAIL=/root/autodl-tmp/MineSeg/mail/report_progress.py
mkdir -p "$(dirname "$LOG")" "$PAPER"
exec >> "$LOG" 2>&1

echo "===== $(date '+%Y-%m-%d %H:%M:%S') ablation start ====="
if ! python3 "$ROOT/ablate_selective.py"; then
  echo "ABLATION_FAILED"
  exit 1
fi

echo "===== $(date '+%Y-%m-%d %H:%M:%S') write paper ====="
if ! python3 "$PAPER/build_paper.py"; then
  echo "PAPER_FAILED"
  exit 1
fi

cd "$PAPER"
if ! xelatex -interaction=nonstopmode paper.tex; then
  echo "XELATEX_FAILED"
  exit 1
fi
if ! xelatex -interaction=nonstopmode paper.tex; then
  echo "XELATEX_FAILED"
  exit 1
fi
if [ ! -s "$PAPER/paper.pdf" ] || [ ! -s "$PAPER/paper.md" ] || [ ! -s "$PAPER/paper.docx" ] || [ ! -s "$PAPER/paper.tex" ]; then
  echo "PAPER_FILES_MISSING"
  exit 1
fi

echo "===== $(date '+%Y-%m-%d %H:%M:%S') send result mail ====="
sent=0
for attempt in 1 2 3 4 5; do
  if python3 "$MAIL" --ablation-only; then
    sent=1
    break
  fi
  echo "mail attempt $attempt failed"
  sleep 20
done
if [ "$sent" -ne 1 ]; then
  echo "MAIL_FAILED"
  exit 1
fi

sync
echo "===== $(date '+%Y-%m-%d %H:%M:%S') shutdown ====="
exec /usr/bin/shutdown
