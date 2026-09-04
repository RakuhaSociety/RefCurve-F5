#!/usr/bin/env bash
set -euo pipefail

REPO="${F5TTS_REPO:-/root/F5-TTS}"
VENV="${F5TTS_VENV:-/root/f5-tts-env}"
PYTHON="${PYTHON:-$VENV/bin/python}"
ACCELERATE="${ACCELERATE:-$VENV/bin/accelerate}"
MODE="${1:-preflight}"
shift || true

DATASET_DIR="${JMICA_SMOKE_DATASET:-$REPO/data/JmicaSingleSpeaker_custom}"
CONFIG_NAME="F5TTS_v1_JA_Base.yaml"
CONFIG_PATH="$REPO/src/f5_tts/configs/$CONFIG_NAME"
CHECKPOINT="$REPO/ckpts/F5TTS_v1_Base/model_1250000.safetensors"
SOURCE_VOCAB="$REPO/src/f5_tts/configs/vocab/F5TTS_v1_Base/vocab.txt"
SOURCE_CONTRACT="$REPO/src/f5_tts/configs/vocab/F5TTS_v1_Base/contract.json"
TARGET_VOCAB="$REPO/src/f5_tts/configs/vocab/F5TTS_v1_JA_Base/vocab.txt"
TARGET_CONTRACT="$REPO/src/f5_tts/configs/vocab/F5TTS_v1_JA_Base/contract.json"

SMOKE_MAX_UPDATES="${SMOKE_MAX_UPDATES:-2}"
RESUME_MAX_UPDATES="${RESUME_MAX_UPDATES:-10}"
SMOKE_1GPU_SAVE_DIR="${SMOKE_1GPU_SAVE_DIR:-ckpts/F5TTS_v1_JA_Base_smoke_1gpu}"
SMOKE_8GPU_SAVE_DIR="${SMOKE_8GPU_SAVE_DIR:-ckpts/F5TTS_v1_JA_Base_smoke_8gpu}"
FULL_SAVE_DIR="${FULL_SAVE_DIR:-ckpts/F5TTS_v1_JA_Base}"
MIN_DATA_FREE_GIB="${MIN_DATA_FREE_GIB:-20}"
REQUIRED_GPU_COUNT="${REQUIRED_GPU_COUNT:-8}"

usage() {
  printf 'Usage: %s preflight|smoke-1gpu|resume-1gpu|smoke-8gpu|resume-8gpu|full [Hydra overrides...]\n' "$0" >&2
}

require_file() {
  [[ -f "$1" ]] || { printf 'Required file not found: %s\n' "$1" >&2; exit 1; }
}

check_sha256() {
  local path="$1"
  local expected="$2"
  local actual
  actual="$(sha256sum "$path" | cut -d ' ' -f 1)"
  [[ "$actual" == "$expected" ]] || {
    printf 'SHA256 mismatch: %s\n  expected: %s\n  actual:   %s\n' "$path" "$expected" "$actual" >&2
    exit 1
  }
  printf 'sha256 ok: %s\n' "$path"
}

preflight() {
  command -v sha256sum >/dev/null || { printf 'sha256sum is required\n' >&2; exit 1; }
  [[ -x "$PYTHON" ]] || { printf 'Python not executable: %s\n' "$PYTHON" >&2; exit 1; }
  [[ -x "$ACCELERATE" ]] || { printf 'accelerate not executable: %s\n' "$ACCELERATE" >&2; exit 1; }

  local path
  for path in \
    "$CONFIG_PATH" \
    "$DATASET_DIR/raw.arrow" \
    "$DATASET_DIR/duration.json" \
    "$CHECKPOINT" \
    "$SOURCE_VOCAB" \
    "$SOURCE_CONTRACT" \
    "$TARGET_VOCAB" \
    "$TARGET_CONTRACT"; do
    require_file "$path"
  done

  check_sha256 "$CHECKPOINT" "670900fd14e6c458b95da6e9ed317cdb20dbaf7a1c02ac06a05475a9d32b6a38"
  check_sha256 "$SOURCE_VOCAB" "2a05f992e00af9b0bd3800a8d23e78d520dbd705284ed2eedb5f4bd29398fa3c"
  check_sha256 "$SOURCE_CONTRACT" "2b72038bbba10bd4cd790815f2ff966b1e830ac38a115657756126e605bdf3a6"
  check_sha256 "$TARGET_VOCAB" "f405cceeeaf2461b8ee2247118f93140db617ddc25d1447790d7e93a761f89e3"
  check_sha256 "$TARGET_CONTRACT" "2954cd0d4fadd98d5845c059d8179a6cea93fa1b684ff76dfb0e44fcd4d2eba0"

  REPO="$REPO" SOURCE_VOCAB="$SOURCE_VOCAB" SOURCE_CONTRACT="$SOURCE_CONTRACT" \
    TARGET_VOCAB="$TARGET_VOCAB" TARGET_CONTRACT="$TARGET_CONTRACT" \
    REQUIRED_GPU_COUNT="$REQUIRED_GPU_COUNT" "$PYTHON" - <<'PY'
import importlib
import os
import sys

repo = os.environ["REPO"]
sys.path.insert(0, os.path.join(repo, "src"))

required_modules = (
    "accelerate", "datasets", "ema_pytorch", "hydra", "librosa", "numpy",
    "safetensors", "soundfile", "torch", "torchaudio", "torchdiffeq",
    "transformers", "vocos", "wandb", "x_transformers",
)
missing = []
for name in required_modules:
    try:
        importlib.import_module(name)
    except Exception as exc:
        missing.append(f"{name}: {exc}")
if missing:
    raise RuntimeError("missing or broken training dependencies:\n  " + "\n  ".join(missing))

import torch
from f5_tts.model.vocab_contract import validate_vocabulary_contract

validate_vocabulary_contract(os.environ["SOURCE_VOCAB"], os.environ["SOURCE_CONTRACT"])
validate_vocabulary_contract(
    os.environ["TARGET_VOCAB"],
    os.environ["TARGET_CONTRACT"],
    expected_identity="refcurve-f5tts-v1-ja-base-v1",
)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available")
gpu_count = torch.cuda.device_count()
required = int(os.environ["REQUIRED_GPU_COUNT"])
if gpu_count < required:
    raise RuntimeError(f"expected at least {required} GPUs, found {gpu_count}")
for index in range(gpu_count):
    if not torch.cuda.is_bf16_supported(index):
        raise RuntimeError(f"GPU {index} does not support BF16")

print(f"torch={torch.__version__} cuda={torch.version.cuda} gpus={gpu_count} bf16=ok")
print("vocabulary contracts=ok")
PY

  local data_probe="/data"
  [[ -d "$data_probe" ]] || data_probe="$DATASET_DIR"
  local available_kib required_kib
  available_kib="$(df -Pk "$data_probe" | tail -n 1 | tr -s ' ' | cut -d ' ' -f 4)"
  required_kib=$((MIN_DATA_FREE_GIB * 1024 * 1024))
  if (( available_kib < required_kib )); then
    printf 'Insufficient free space on %s: need >= %s GiB, have %s GiB\n' \
      "$data_probe" "$MIN_DATA_FREE_GIB" "$((available_kib / 1024 / 1024))" >&2
    exit 1
  fi
  printf 'data capacity ok: %s GiB free on %s\n' "$((available_kib / 1024 / 1024))" "$data_probe"
  printf 'preflight passed\n'
}

hydra_overrides() {
  COMMON_OVERRIDES=(
    "datasets.name=$DATASET_DIR"
    "model.tokenizer_path=$TARGET_VOCAB"
    "model.tokenizer_contract_path=$TARGET_CONTRACT"
    "ckpts.pretrained_init.checkpoint_path=$CHECKPOINT"
    "ckpts.pretrained_init.source_vocab_path=$SOURCE_VOCAB"
    "ckpts.pretrained_init.source_vocab_contract_path=$SOURCE_CONTRACT"
    "ckpts.logger=null"
    "ckpts.log_samples=false"
  )
}

launch() {
  local processes="$1"
  local save_dir="$2"
  local max_updates="$3"
  shift 3
  local launcher=("$ACCELERATE" launch --num_processes "$processes" --mixed_precision bf16)
  if [[ "$processes" == "8" ]]; then
    launcher+=(--multi_gpu)
  fi
  hydra_overrides
  cd "$REPO"
  if [[ "$processes" == "1" ]]; then
    CUDA_VISIBLE_DEVICES=0 exec "${launcher[@]}" src/f5_tts/train/train.py \
      --config-name "$CONFIG_NAME" \
      "${COMMON_OVERRIDES[@]}" \
      "ckpts.save_dir=$save_dir" \
      "optim.max_updates=$max_updates" \
      "$@"
  else
    exec "${launcher[@]}" src/f5_tts/train/train.py \
      --config-name "$CONFIG_NAME" \
      "${COMMON_OVERRIDES[@]}" \
      "ckpts.save_dir=$save_dir" \
      "optim.max_updates=$max_updates" \
      "$@"
  fi
}

print_full_entrypoint() {
  hydra_overrides
  local command=(
    "$ACCELERATE" launch --multi_gpu --num_processes 8 --mixed_precision bf16
    src/f5_tts/train/train.py --config-name "$CONFIG_NAME"
    "${COMMON_OVERRIDES[@]}"
    "ckpts.save_dir=$FULL_SAVE_DIR"
    "optim.max_updates=${FULL_MAX_UPDATES:-null}"
    "$@"
  )
  printf 'Full training is not started automatically. Review and execute this entrypoint manually:\ncd %q && ' "$REPO"
  printf '%q ' "${command[@]}"
  printf '\n'
}

case "$MODE" in
  preflight)
    preflight
    ;;
  smoke-1gpu)
    preflight
    launch 1 "$SMOKE_1GPU_SAVE_DIR" "$SMOKE_MAX_UPDATES" "$@"
    ;;
  resume-1gpu)
    preflight
    launch 1 "$SMOKE_1GPU_SAVE_DIR" "$RESUME_MAX_UPDATES" "$@"
    ;;
  smoke-8gpu)
    preflight
    launch 8 "$SMOKE_8GPU_SAVE_DIR" "$SMOKE_MAX_UPDATES" "$@"
    ;;
  resume-8gpu)
    preflight
    launch 8 "$SMOKE_8GPU_SAVE_DIR" "$RESUME_MAX_UPDATES" "$@"
    ;;
  full)
    preflight
    print_full_entrypoint "$@"
    ;;
  *)
    usage
    exit 2
    ;;
esac
