#!/bin/bash
# Shut down only after this run has been seen working, then either finishes
# the package or exits with an error. Starting this script alone does not
# shut the machine down.
set -u
RUN=/root/autodl-tmp/MineSeg/v2/runs/relay_retrain
PAPER=/root/autodl-tmp/MineSeg/v2/paper_relay_retrain
ZIP=/root/autodl-tmp/MineSeg/v2/relay_retrain_package.zip
PIPELINE_LOG=$RUN/pipeline.log
LOG=$RUN/shutdown_watch.log
MAIL=/usr/bin/python3
REPORT=/root/autodl-tmp/MineSeg/mail/report_progress.py
STALL_SECONDS=10800
mkdir -p "$RUN"
exec 8>/tmp/relay-shutdown-watch.lock
/usr/bin/flock -n 8 || { echo "shutdown watch already running"; exit 0; }
exec >> "$LOG" 2>&1

echo "===== $(date '+%Y-%m-%d %H:%M:%S') watch start ====="

ready() {
  [ -s "$RUN/results.json" ] || return 1
  [ -s "$PAPER/paper.md" ] || return 1
  [ -s "$PAPER/paper.tex" ] || return 1
  [ -s "$PAPER/paper.docx" ] || return 1
  [ -s "$PAPER/paper.pdf" ] || return 1
  [ -s "$ZIP" ] || return 1
  return 0
}

busy() {
  local entry cmd
  for entry in /proc/[0-9]*; do
    cmd=$(tr '\0' ' ' < "$entry/cmdline" 2>/dev/null || true)
    case "$cmd" in
      *"/relay_retrain/run_relay_then_shutdown.sh"*|\
      *"/relay_retrain/train_relay_ablation.py"*|\
      *"/relay_retrain/evaluate_relay.py"*|\
      *"/relay_retrain/build_paper.py"*|\
      *"/relay_retrain/prepare_groups.py"*|\
      *"xelatex -interaction=nonstopmode paper.tex"*)
        return 0
        ;;
    esac
  done
  return 1
}

log_age() {
  if [ ! -f "$PIPELINE_LOG" ]; then
    echo 0
    return
  fi
  echo $(( $(date +%s) - $(stat -c %Y "$PIPELINE_LOG") ))
}

stop_machine() {
  echo "===== $(date '+%Y-%m-%d %H:%M:%S') $1 ====="
  sync
  exec /usr/bin/shutdown
}

seen=0
while true; do
  age=$(log_age)
  if busy; then
    seen=1
    if [ "$age" -ge "$STALL_SECONDS" ]; then
      "$MAIL" "$REPORT" --relay-only --failure "pipeline log had no update for ${age} seconds, shutting down" || true
      stop_machine "stalled, shutdown"
    fi
    echo "$(date '+%Y-%m-%d %H:%M:%S') running age=${age}s"
  elif [ "$seen" -eq 1 ] && ready; then
    sleep 30
    if ! busy && ready; then
      echo "ZIP $ZIP"
      stop_machine "all tasks finished, shutdown"
    fi
  elif [ "$seen" -eq 1 ]; then
    sleep 45
    if ! busy && ! ready; then
      "$MAIL" "$REPORT" --relay-only --failure "run ended before the result package was ready, shutting down" || true
      stop_machine "error, shutdown"
    fi
  else
    echo "$(date '+%Y-%m-%d %H:%M:%S') run not seen yet, staying on"
  fi
  sleep 60
done
