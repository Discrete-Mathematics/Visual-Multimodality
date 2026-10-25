#!/bin/bash
# Start complement training only after the other agent's GPU job has released memory.
cd /root/autodl-tmp/MineSeg/v3
LOG=/root/autodl-tmp/MineSeg/v3/launch.log
exec >> "$LOG" 2>&1
echo "===== waiter start $(date -Is) ====="
while true; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
  echo "$(date -Is) gpu_mem_mib=${used}"
  if [ "${used:-99999}" -lt 12000 ]; then
    break
  fi
  sleep 30
done
CKPT=/root/autodl-tmp/MineSeg/v2/runs/selective_ftrans/best.pth
if [ ! -f "$CKPT" ]; then
  CKPT=/root/autodl-tmp/MineSeg/v2/runs/selective_hi/best.pth
fi
echo "selective checkpoint ${CKPT}"
python3 -u train_selective.py \
  --checkpoint "$CKPT" \
  --epochs 8 \
  --batch-size 4 \
  --height 768 \
  --width 896 \
  --lr 1.2e-5 \
  --workers 4 \
  --outdir /root/autodl-tmp/MineSeg/v3/runs/selective_complement
echo "selective train exit $?"
python3 -u train_selective.py \
  --checkpoint /root/autodl-tmp/MineSeg/v3/runs/selective_complement/best.pth \
  --eval-test --tta \
  --batch-size 2 \
  --height 768 \
  --width 896 \
  --workers 2 \
  --outdir /root/autodl-tmp/MineSeg/v3/runs/selective_complement
echo "selective test exit $?"
python3 -u train_debf.py \
  --teacher "$CKPT" \
  --epochs 10 \
  --batch-size 2 \
  --height 640 \
  --width 768 \
  --workers 2 \
  --outdir /root/autodl-tmp/MineSeg/v3/runs/debf_complement
echo "debf train exit $?"
python3 -u train_debf.py \
  --init /root/autodl-tmp/MineSeg/v3/runs/debf_complement/best.pth \
  --eval-test \
  --batch-size 2 \
  --height 640 \
  --width 768 \
  --workers 2 \
  --outdir /root/autodl-tmp/MineSeg/v3/runs/debf_complement
echo "debf test exit $?"
echo "===== waiter done $(date -Is) ====="
