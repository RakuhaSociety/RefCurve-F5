#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
ACCELERATE="${ACCELERATE:-accelerate}"
DATASET_DIR="${VISUALNOVEL_CALIBRATION_DATASET:-$ROOT/data/visualnovel_sharded_ja}"
RUN_ROOT="${VISUALNOVEL_CALIBRATION_RUN_ROOT:-$ROOT/ckpts/visualnovel_calibration_ja}"
# Smoke 模式：只提前停步，不改 max_updates，因此 scheduler horizon 与真实 run 完全一致。
SMOKE_UPDATES="${VISUALNOVEL_CALIBRATION_SMOKE_UPDATES:-}"
if [[ -n "$SMOKE_UPDATES" ]]; then
  [[ "$SMOKE_UPDATES" =~ ^[1-9][0-9]*$ ]] || {
    printf 'VISUALNOVEL_CALIBRATION_SMOKE_UPDATES must be a positive integer: %s\n' "$SMOKE_UPDATES" >&2
    exit 1
  }
  RUN_PREFIX="smoke${SMOKE_UPDATES}-"
else
  RUN_PREFIX=""
fi

# world size 旋钮：只为 smoke 存在。1 卡与 8 卡必须走同一条启动路径，
# 否则"world-size 等价性"比较的就是两套代码，而不是同一套代码的两种规模。
NUM_PROCESSES="${VISUALNOVEL_CALIBRATION_NUM_PROCESSES:-8}"
[[ "$NUM_PROCESSES" =~ ^[1-9][0-9]*$ ]] || {
  printf 'VISUALNOVEL_CALIBRATION_NUM_PROCESSES must be a positive integer: %s\n' "$NUM_PROCESSES" >&2
  exit 1
}
# 正式 calibration 的 8 卡 BF16 是不可协商的协议；只有 smoke 允许改 world size。
if [[ -z "$SMOKE_UPDATES" && "$NUM_PROCESSES" != "8" ]]; then
  printf 'Calibration protocol requires 8 processes; NUM_PROCESSES may only be overridden for smoke runs.\n' >&2
  exit 1
fi
if [[ -n "$SMOKE_UPDATES" ]]; then
  RUN_PREFIX="${RUN_PREFIX}gpu${NUM_PROCESSES}-"
fi
if [[ "$NUM_PROCESSES" -gt 1 ]]; then
  LAUNCH_PARALLEL="--multi_gpu --num_processes $NUM_PROCESSES"
else
  LAUNCH_PARALLEL="--num_processes 1"
fi

if [[ -n "${VISUALNOVEL_CALIBRATION_RUN_NAME+x}" ]]; then
  RUN_NAME="$VISUALNOVEL_CALIBRATION_RUN_NAME"
  EXPLICIT_RUN_NAME=1
else
  # smoke run 名字带前缀，永远不会被误当成正式 calibration run。
  RUN_NAME="$RUN_PREFIX$(date -u +%Y%m%dT%H%M%SZ)-$$"
  EXPLICIT_RUN_NAME=0
fi
RUN_DIR="$RUN_ROOT/$RUN_NAME"
VOCAB="${VISUALNOVEL_JA_VOCAB:-$ROOT/src/f5_tts/configs/vocab/F5TTS_v1_JA_Base/vocab.txt}"
CONTRACT="${VISUALNOVEL_JA_CONTRACT:-$ROOT/src/f5_tts/configs/vocab/F5TTS_v1_JA_Base/contract.json}"
SOURCE_CHECKPOINT="${VISUALNOVEL_CALIBRATION_SOURCE_CHECKPOINT:-$ROOT/ckpts/F5TTS_v1_Base/model_1250000.safetensors}"
SOURCE_VOCAB="${VISUALNOVEL_SOURCE_VOCAB:-$ROOT/src/f5_tts/configs/vocab/F5TTS_v1_Base/vocab.txt}"
SOURCE_CONTRACT="${VISUALNOVEL_SOURCE_CONTRACT:-$ROOT/src/f5_tts/configs/vocab/F5TTS_v1_Base/contract.json}"
SELECTION_V1="${VISUALNOVEL_SELECTION_V1:-$ROOT/src/f5_tts/train/datasets/visualnovel_selection_v1.json}"
OPERATION_LOCK_ROOT="${VISUALNOVEL_OPERATION_LOCK_ROOT:-$ROOT/.visualnovel-operation-locks}"
LOCK_OWNER="calibration-$RUN_NAME-$$"

for required in \
  "$DATASET_DIR/READY" \
  "$DATASET_DIR/manifest.json" \
  "$DATASET_DIR/eval_manifest.json" \
  "$DATASET_DIR/aggregate_provenance.json" \
  "$DATASET_DIR/aggregate_report.json" \
  "$DATASET_DIR/audio_roots.json" \
  "$DATASET_DIR/deployment_audio_roots.json" \
  "$VOCAB" "$CONTRACT" "$SOURCE_CHECKPOINT" "$SOURCE_VOCAB" "$SOURCE_CONTRACT" "$SELECTION_V1"; do
  [[ -f "$required" ]] || { printf 'Required file not found: %s\n' "$required" >&2; exit 1; }
done

if [[ -e "$RUN_DIR" ]]; then
  if [[ "$EXPLICIT_RUN_NAME" != 1 ]]; then
    printf 'Automatically generated calibration run directory already exists: %s\n' "$RUN_DIR" >&2
    exit 1
  fi
  [[ -d "$RUN_DIR" ]] || { printf 'Calibration run path is not a directory: %s\n' "$RUN_DIR" >&2; exit 1; }
  [[ -f "$RUN_DIR/run_manifest.json" ]] || {
    printf 'Existing calibration run has no immutable manifest: %s\n' "$RUN_DIR" >&2
    exit 1
  }
  if [[ ! -f "$RUN_DIR/model_last.pt" ]] && ! compgen -G "$RUN_DIR/model_[0-9]*.pt" >/dev/null; then
    printf 'Existing calibration run has no resumable checkpoint: %s\n' "$RUN_DIR" >&2
    exit 1
  fi
  printf 'Resuming explicitly named calibration run: %s\n' "$RUN_DIR"
fi

"$PYTHON" - "$NUM_PROCESSES" <<'PY'
import sys

import torch

required = int(sys.argv[1])
available = torch.cuda.device_count()
if not torch.cuda.is_available() or available < required:
    raise RuntimeError(f"run requires at least {required} visible CUDA GPUs; found {available}")
for index in range(required):
    if not torch.cuda.is_bf16_supported(index):
        raise RuntimeError(f"GPU {index} does not support BF16")
print(f"preflight: torch={torch.__version__} cuda={torch.version.cuda} gpus_required={required} available={available} bf16=ok")
PY

# Deliberately no free-form Hydra arguments: calibration settings are an immutable protocol.
exec "$PYTHON" -m f5_tts.train.operation_lock_run \
  --lock-root "$OPERATION_LOCK_ROOT" \
  --operation training \
  --owner "$LOCK_OWNER" \
  -- \
  "$ACCELERATE" launch $LAUNCH_PARALLEL --mixed_precision bf16 \
  src/f5_tts/train/train.py --config-name F5TTS_v1_JA_Base.yaml \
  "datasets.name=$DATASET_DIR" \
  datasets.dataset_type=ShardedArrowDataset \
  datasets.num_workers=1 \
  optim.max_updates=5000 \
  ${SMOKE_UPDATES:+"+optim.stop_after_updates=$SMOKE_UPDATES"} \
  "+ckpts.lr_trace_path=$RUN_DIR/lr_trace.jsonl" \
  "ckpts.save_dir=$RUN_DIR" \
  ckpts.save_per_updates=500 \
  ckpts.last_per_updates=100 \
  ckpts.keep_last_n_checkpoints=10 \
  "model.tokenizer_path=$VOCAB" \
  "model.tokenizer_contract_path=$CONTRACT" \
  "ckpts.pretrained_init.checkpoint_path=$SOURCE_CHECKPOINT" \
  "ckpts.pretrained_init.source_vocab_path=$SOURCE_VOCAB" \
  "ckpts.pretrained_init.source_vocab_contract_path=$SOURCE_CONTRACT" \
  +ckpts.run_manifest.enabled=true \
  +ckpts.run_manifest.required=true \
  "+ckpts.run_manifest.evidence_paths.selection_v1=$SELECTION_V1" \
  "+ckpts.run_manifest.evidence_paths.dataset_ready=$DATASET_DIR/READY" \
  "+ckpts.run_manifest.evidence_paths.aggregate_provenance=$DATASET_DIR/aggregate_provenance.json" \
  "+ckpts.run_manifest.evidence_paths.aggregate_report=$DATASET_DIR/aggregate_report.json" \
  "+ckpts.run_manifest.evidence_paths.train_manifest=$DATASET_DIR/manifest.json" \
  "+ckpts.run_manifest.evidence_paths.eval_manifest=$DATASET_DIR/eval_manifest.json" \
  "+ckpts.run_manifest.evidence_paths.audio_roots=$DATASET_DIR/audio_roots.json" \
  "+ckpts.run_manifest.evidence_paths.deployment_audio_roots=$DATASET_DIR/deployment_audio_roots.json"
