#!/bin/bash
# Retrain the Geometry Relay controls, write the paper, and mail the record.
# Errors send a failure mail and leave the machine on. This script does not shut down.
set -u
PY=/root/miniconda3/bin/python3
MAILPY=/usr/bin/python3
ROOT=/root/autodl-tmp/MineSeg/v2/relay_retrain
RUN=/root/autodl-tmp/MineSeg/v2/runs/relay_retrain
PAPER=/root/autodl-tmp/MineSeg/v2/paper_relay_retrain
MAIL=/root/autodl-tmp/MineSeg/mail/report_progress.py
LOG=$RUN/pipeline.log
ZIP=/root/autodl-tmp/MineSeg/v2/relay_retrain_package.zip
mkdir -p "$RUN" "$PAPER" "$ROOT/patched"
exec 9>/tmp/relay-retrain.lock
/usr/bin/flock -n 9 || { echo "relay retrain already running"; exit 1; }
exec >> "$LOG" 2>&1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

fail() {
  echo "FAIL $1"
  "$MAILPY" "$MAIL" --relay-only --failure "$1" || true
  sync
  echo "===== $(date '+%Y-%m-%d %H:%M:%S') failed, machine stays on ====="
  exit 1
}

echo "===== $(date '+%Y-%m-%d %H:%M:%S') relay retrain start ====="
crontab -l > /tmp/cron.relay.bak 2>/dev/null || true
grep -v "report_progress.py" /tmp/cron.relay.bak > /tmp/cron.relay.new || true
echo "0 * * * * /usr/bin/flock -n /tmp/mineseg-mail.lock /usr/bin/python3 /root/autodl-tmp/MineSeg/mail/report_progress.py --relay-only >> /root/autodl-tmp/MineSeg/mail/cron.log 2>&1" >> /tmp/cron.relay.new
crontab /tmp/cron.relay.new
# cron and the monitor must not inherit fd 9, or they keep the lock after this script exits.
/usr/sbin/cron 9>&- || true

"$PY" "$ROOT/disk_monitor.py" >> "$RUN/disk_monitor.log" 2>&1 9>&- &
echo $! > "$RUN/disk_monitor.pid"

if [ -s "$RUN/protocol.json" ]; then
  echo "===== $(date '+%Y-%m-%d %H:%M:%S') keep frozen protocol ====="
else
  "$PY" "$ROOT/prepare_groups.py" || fail "prepare_groups"
fi
"$MAILPY" "$MAIL" --relay-only || echo "start mail failed, training continues"

trained=$("$PY" - << 'PY'
import json
from pathlib import Path
root = Path("/root/autodl-tmp/MineSeg/v2/runs/relay_retrain")
variants = ["full", "no_relay", "const_valid", "no_neighbor", "raw_depth", "no_geometry"]
seeds = [2026, 2027, 2028]
ok = 0
for variant in variants:
    for seed in seeds:
        folder = root / variant / f"seed_{seed}"
        selection = folder / "selection.json"
        history = folder / "history.jsonl"
        if not selection.is_file() or not history.is_file() or not (folder / "best.pth").is_file():
            continue
        payload = json.loads(selection.read_text())
        epochs = sum(1 for line in history.read_text().splitlines() if line.strip())
        if payload.get("complete") and int(payload.get("epochs_ran", 0)) == 12 and epochs >= 12:
            ok += 1
print(ok)
PY
)

if [ "$trained" = "18" ]; then
  echo "===== $(date '+%Y-%m-%d %H:%M:%S') 18 trainings already complete, skip smoke and training ====="
else
  "$PY" "$ROOT/train_relay_ablation.py" --smoke || fail "smoke"
  for seed in 2026 2027 2028; do
    for variant in full no_relay const_valid no_neighbor raw_depth no_geometry; do
      echo "===== $(date '+%Y-%m-%d %H:%M:%S') train $variant seed $seed ====="
      "$PY" "$ROOT/train_relay_ablation.py" "$variant" "$seed" || fail "train $variant $seed"
    done
  done
fi

echo "===== $(date '+%Y-%m-%d %H:%M:%S') evaluate ====="
"$PY" "$ROOT/evaluate_relay.py" || fail "evaluate"
echo "===== $(date '+%Y-%m-%d %H:%M:%S') paper ====="
"$PY" "$ROOT/build_paper.py" || fail "paper"
cd "$PAPER"
xelatex -interaction=nonstopmode paper.tex || fail "xelatex"
xelatex -interaction=nonstopmode paper.tex || fail "xelatex"
if [ ! -s "$PAPER/paper.pdf" ] || [ ! -s "$PAPER/paper.md" ] || [ ! -s "$PAPER/paper.docx" ] || [ ! -s "$PAPER/paper.tex" ]; then
  fail "paper files missing"
fi

cd /root/autodl-tmp/MineSeg/v2
rm -f "$ZIP"
find relay_retrain paper_relay_retrain runs/relay_retrain \
  -type f \
  ! -name '*.pth' \
  ! -name '*.partial' \
  ! -path '*__pycache__*' \
  > /tmp/relay_zip_list.txt
zip -q "$ZIP" -@ < /tmp/relay_zip_list.txt || fail "zip"
if [ ! -s "$ZIP" ]; then
  fail "zip missing"
fi
echo "ZIP $ZIP"

sent=0
for attempt in 1 2 3 4 5; do
  if "$MAILPY" "$MAIL" --relay-only; then
    sent=1
    break
  fi
  echo "mail attempt $attempt failed"
  sleep 20
done
if [ "$sent" -ne 1 ]; then
  fail "mail"
fi

sync
echo "===== $(date '+%Y-%m-%d %H:%M:%S') finished, machine stays on ====="
echo "ZIP $ZIP"
