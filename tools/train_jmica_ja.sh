#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODE="${1:-smoke}"
shift || true
DATASET_DIR="${JMICA_JA_DATASET:-$ROOT/data/JmicaSingleSpeaker_custom}"
PRETRAINED="${JMICA_JA_PRETRAINED:-$ROOT/ckpts/F5TTS_JA_Jmica/model_21999120.pt}"
VOCAB="${JMICA_JA_VOCAB:-$ROOT/ckpts/F5TTS_JA_Jmica/vocab_japanese.txt}"

for required in "$DATASET_DIR/raw.arrow" "$DATASET_DIR/duration.json" "$PRETRAINED" "$VOCAB"; do
  if [[ ! -e "$required" ]]; then
    printf 'Required file not found: %s\n' "$required" >&2
    exit 1
  fi
done

python - <<'PY'
import torch

print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
for index in range(torch.cuda.device_count()):
    print(index, torch.cuda.get_device_name(index))
assert torch.__version__.startswith("2.4."), "expected torch 2.4.x"
assert torch.version.cuda == "12.4", "expected CUDA 12.4 wheels"
assert torch.cuda.device_count() == 8, "expected eight GPUs"
PY

COMMON=(
  --multi_gpu
  --num_processes 8
  --mixed_precision bf16
  src/f5_tts/train/train.py
  --config-name F5TTS_Jmica_JA_SingleSpeaker.yaml
  "datasets.name=$DATASET_DIR"
  "model.tokenizer_path=$VOCAB"
  "ckpts.pretrained_init=$PRETRAINED"
)

case "$MODE" in
  smoke)
    exec accelerate launch "${COMMON[@]}" \
      optim.epochs=1 \
      optim.num_warmup_updates=1 \
      datasets.batch_size_per_gpu=1200 \
      datasets.max_samples=2 \
      datasets.num_workers=2 \
      ckpts.logger=null \
      ckpts.log_samples=false \
      ckpts.save_dir=ckpts/F5TTS_Jmica_JA_SingleSpeaker_smoke \
      ckpts.save_per_updates=2 \
      ckpts.last_per_updates=1 \
      ckpts.keep_last_n_checkpoints=1 \
      "$@"
    ;;
  full)
    exec accelerate launch "${COMMON[@]}" "$@"
    ;;
  *)
    printf 'Usage: %s [smoke|full] [Hydra overrides...]\n' "$0" >&2
    exit 2
    ;;
esac
