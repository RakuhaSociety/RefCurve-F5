#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODE="${1:-preflight}"
shift || true
DATASET_DIR="${VISUALNOVEL_SHARDED_JA_DATASET:-$ROOT/data/visualnovel_sharded_ja}"
VOCAB="${VISUALNOVEL_JA_VOCAB:-$ROOT/src/f5_tts/configs/vocab/F5TTS_v1_JA_Base/vocab.txt}"
CONTRACT="${VISUALNOVEL_JA_CONTRACT:-$ROOT/src/f5_tts/configs/vocab/F5TTS_v1_JA_Base/contract.json}"

for required in "$DATASET_DIR/READY" "$DATASET_DIR/manifest.json" "$DATASET_DIR/aggregate_provenance.json" "$DATASET_DIR/aggregate_report.json" "$DATASET_DIR/audio_roots.json" "$DATASET_DIR/deployment_audio_roots.json" "$VOCAB" "$CONTRACT"; do
  [[ -f "$required" ]] || { printf 'Required file not found: %s\n' "$required" >&2; exit 1; }
done

expected_gpus=0
case "$MODE" in
  preflight) expected_gpus=0 ;;
  smoke-1gpu) expected_gpus=1; export CUDA_VISIBLE_DEVICES=0 ;;
  smoke-8gpu) expected_gpus=8 ;;
  *) printf 'Usage: %s preflight|smoke-1gpu|smoke-8gpu [Hydra overrides...]\n' "$0" >&2; exit 2 ;;
esac

python - "$DATASET_DIR" "$VOCAB" "$CONTRACT" "$expected_gpus" <<'PY'
import json
import sys
from pathlib import Path

import torch
from f5_tts.model.sharded_dataset import ShardedArrowDataset, load_audio_root_registry, load_manifest, open_frame_index
from f5_tts.model.vocab_contract import validate_vocabulary_contract

root, vocab, contract, expected = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4])
manifest = load_manifest(root / "manifest.json")
roots = load_audio_root_registry(root / "audio_roots.json", root / "deployment_audio_roots.json")
rows = ShardedArrowDataset(manifest, audio_roots=roots)
frames = open_frame_index(manifest)
tokens, vocab_contract = validate_vocabulary_contract(vocab, contract)
provenance = json.loads((root / "aggregate_provenance.json").read_text(encoding="utf-8"))
report = json.loads((root / "aggregate_report.json").read_text(encoding="utf-8"))
assert provenance["schema_version"] == "visualnovel-aggregate-v2"
assert report["conflict_policy"] == "exclude_audio_object"
assert provenance["vocabulary"]["identity"] == vocab_contract.identity == manifest.vocabulary_identity
assert report["train_rows"] == len(rows) == len(frames) == manifest.total_rows > 0
eval_manifest = load_manifest(root / "eval_manifest.json")
assert report["eval_rows"] == eval_manifest.total_rows
assert rows[0]["audio_root_id"] and rows[-1]["relative_audio_path"]
print("dataset rows:", len(rows), "shards:", len(manifest.shards), "vocab tokens:", len(tokens))
print("torch:", torch.__version__, "CUDA:", torch.version.cuda, "GPUs:", torch.cuda.device_count())
if expected:
    assert torch.cuda.is_available(), "CUDA is required for smoke training"
    assert torch.cuda.device_count() == expected, f"expected exactly {expected} visible GPU(s)"
PY

[[ "$MODE" == preflight ]] && exit 0

processes=1
multi=()
if [[ "$MODE" == smoke-8gpu ]]; then
  processes=8
  multi=(--multi_gpu)
elif [[ "$MODE" == smoke-1gpu ]]; then
  export CUDA_VISIBLE_DEVICES=0
fi
SAVE_DIR="${VISUALNOVEL_SMOKE_SAVE_ROOT:-ckpts/visualnovel_sharded_ja_smoke}/${MODE}-$(date +%Y%m%d-%H%M%S)-$$"

exec accelerate launch "${multi[@]}" --num_processes "$processes" --mixed_precision bf16 \
  src/f5_tts/train/train.py --config-name F5TTS_v1_JA_Base.yaml \
  "datasets.name=$DATASET_DIR" \
  datasets.dataset_type=ShardedArrowDataset \
  datasets.batch_size_per_gpu=1200 \
  datasets.max_samples=2 \
  datasets.num_workers=1 \
  optim.epochs=1 \
  optim.max_updates=2 \
  optim.num_warmup_updates=1 \
  ckpts.logger=null \
  ckpts.log_samples=false \
  "ckpts.save_dir=$SAVE_DIR" \
  ckpts.save_per_updates=2 \
  ckpts.last_per_updates=1 \
  ckpts.keep_last_n_checkpoints=1 \
  "$@"
