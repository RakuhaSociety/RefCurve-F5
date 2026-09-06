#!/bin/bash
# Smoke test: 1 GPU, 120 successful updates, verify LR trajectory

set -euo pipefail

SMOKE_DIR="ckpts/smoke_scheduler_1gpu_120_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$SMOKE_DIR"

# Small dummy dataset (reuse calibration dataset but limit samples)
DATASET_DIR="/data/visualnovel/aggregates/wave-abc-v1-exclude-conflicts"

# Minimal model config for speed
python3 -m accelerate.commands.launch \
  --num_processes 1 \
  --mixed_precision bf16 \
  src/f5_tts/train/train.py \
    --config-name F5TTS_v1_JA_Base \
    ++trainer.epochs=999 \
    ++trainer.max_updates=120 \
    ++trainer.num_warmup_updates=100 \
    ++trainer.save_per_updates=120 \
    ++trainer.keep_last_n_checkpoints=1 \
    ++trainer.checkpoint_path="$SMOKE_DIR" \
    ++trainer.lr_trace_path="$SMOKE_DIR/lr_trace.jsonl" \
    ++trainer.logger=null \
    ++datasets.batch_size_per_gpu=3200 \
    ++datasets.max_samples=4 \
    ++datasets.data_dir="$DATASET_DIR"

echo ""
echo "=== 1 GPU Smoke Complete ==="
echo "LR trace: $SMOKE_DIR/lr_trace.jsonl"
echo ""
echo "Expected LR checkpoints (global updates):"
echo "  update 1: ~1.10e-6 (warmup start)"
echo "  update 50: ~5.05e-6 (mid-warmup)"
echo "  update 100: 1.00e-05 (warmup peak)"
echo "  update 101: ~9.90e-06 (decay start)"
echo "  update 120: ~9.73e-06 (slight decay)"
echo ""
python3 -c "
import json
with open('$SMOKE_DIR/lr_trace.jsonl') as f:
    records = [json.loads(line) for line in f if 'global_update' in json.loads(line)]
    for r in [records[0], records[49], records[99], records[100], records[-1]]:
        print(f\"update {r['global_update']:3d}: {r['lr']:.2e}\")
"
