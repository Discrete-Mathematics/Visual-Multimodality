#!/bin/bash
set -euo pipefail
cd /root/autodl-tmp/MineSeg/v2
echo "OPTIMIZE_START $(date -Is)"

python finetune_selective.py \
  --epochs 20 \
  --batch-size 4 \
  --height 768 \
  --width 896 \
  --workers 4 \
  --lr 2e-5 \
  --rare-crop 0.3 \
  --outdir /root/autodl-tmp/MineSeg/v2/runs/selective_hi

python finetune_selective.py \
  --eval-test \
  --height 768 \
  --width 896 \
  --batch-size 4 \
  --workers 4 \
  --checkpoint /root/autodl-tmp/MineSeg/v2/runs/selective_hi/best.pth \
  --outdir /root/autodl-tmp/MineSeg/v2/runs/selective_hi

python train.py \
  --model debf \
  --variant b5 \
  --epochs 36 \
  --batch-size 6 \
  --height 640 \
  --width 768 \
  --with-background \
  --simple-loss \
  --rare-sampler \
  --workers 4 \
  --lr 4e-5 \
  --head-lr 4e-4 \
  --outdir /root/autodl-tmp/MineSeg/v2/runs/debf_b5_bg

python evaluate.py \
  --checkpoint /root/autodl-tmp/MineSeg/v2/runs/debf_b5_bg/best.pth \
  --tag official_test \
  --batch-size 6 \
  --workers 4

echo "PIPELINE_DONE $(date -Is)"
