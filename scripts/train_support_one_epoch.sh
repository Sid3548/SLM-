#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/support_one_epoch_local.yaml}

python train.py --config "$CONFIG" --build-data-only
python train.py --config "$CONFIG" --epochs 1
python sample.py \
  --config "$CONFIG" \
  --ckpt checkpoints/epoch_1.pt \
  --prompt "I need to raise a support ticket and talk to a human." \
  --max-new 80
