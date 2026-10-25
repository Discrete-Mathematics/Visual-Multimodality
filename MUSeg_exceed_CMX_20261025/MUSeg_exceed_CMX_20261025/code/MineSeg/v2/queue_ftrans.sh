#!/bin/bash
# Wait for the other experiment to release the GPU, then train on the
# FTransUNet train/val lists. The test list is not touched here.
set -u
LOG=/root/autodl-tmp/MineSeg/v2/runs/queue_ftrans.log
mkdir -p /root/autodl-tmp/MineSeg/v2/runs
exec >>"$LOG" 2>&1
exec 9>/tmp/museg-queue.lock
if ! flock -n 9; then
  echo "$(date) queue already running"
  exit 0
fi
echo "$(date) queue start"

gpu_busy() {
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | grep -E 'python|pt_main' || true
}

while true; do
  busy="$(gpu_busy)"
  if [ -n "$busy" ]; then
    echo "$(date) waiting, GPU busy: ${busy//$'\n'/ | }"
    sleep 30
    continue
  fi
  sleep 15
  busy="$(gpu_busy)"
  if [ -n "$busy" ]; then
    continue
  fi
  echo "$(date) GPU idle"
  break
done

cd /root/autodl-tmp/MineSeg/v2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

INIT=/root/autodl-tmp/MineSeg/v2/runs/selective_hi/best.pth
if [ ! -f "$INIT" ]; then
  INIT=/root/autodl-tmp/SelectiveMine_best_20261024/weights/best_inference.pth
fi
echo "$(date) selective init $INIT"

python finetune_selective.py \
  --epochs 16 \
  --batch-size 4 \
  --height 768 \
  --width 896 \
  --lr 1e-5 \
  --workers 4 \
  --rare-crop 0.45 \
  --lovasz-weight 0.25 \
  --checkpoint "$INIT" \
  --outdir /root/autodl-tmp/MineSeg/v2/runs/selective_ftrans
echo "$(date) selective exit $?"

python train.py \
  --model debf \
  --variant b5 \
  --epochs 30 \
  --batch-size 4 \
  --height 768 \
  --width 896 \
  --lr 4e-5 \
  --head-lr 4e-4 \
  --workers 6 \
  --paste \
  --rare-sampler \
  --outdir /root/autodl-tmp/MineSeg/v2/runs/debf_b5_ftrans
debf_status=$?
echo "$(date) debf exit $debf_status"
if [ "$debf_status" -ne 0 ]; then
  echo "$(date) retry DEBF at 640x768"
  python train.py \
    --model debf \
    --variant b5 \
    --epochs 36 \
    --batch-size 4 \
    --height 640 \
    --width 768 \
    --lr 4e-5 \
    --head-lr 4e-4 \
    --workers 6 \
    --paste \
    --rare-sampler \
    --outdir /root/autodl-tmp/MineSeg/v2/runs/debf_b5_640
  echo "$(date) debf retry exit $?"
fi
echo "$(date) queue done"
