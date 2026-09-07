#!/usr/bin/env bash
# 远端 8×4090 服务器（183.147.142.130:9000）专用包装：把所有写路径钉在 /data。
#
# 为什么需要它：该机根盘 /dev/sda2 445G 长期接近写满，/data（/dev/sdb 3.5T）才是
# 容量所在。训练涉及五个独立的写路径，缺任何一个都会失败，且四种失败**都不像**
# 磁盘问题：
#
#   1. run root      → PytorchStreamWriter ... unexpected pos   （像 torch 序列化 bug）
#   2. $TMPDIR       → No usable temporary directory found      （挂在 import torch）
#   3. numba cache   → cannot cache function: no locator available（挂在 librosa）
#   4. hydra.run.dir → OSError: [Errno 28] No space left        （相对路径落在仓库里）
#   5. 日志重定向    → bash: No space left on device            （连 nohup 都写不出）
#
# 用法与 tools/train_visualnovel_calibration_ja.sh 相同，环境变量照样可覆盖：
#
#   bash tools/train_calibration_remote_data_volume.sh                       # 正式 run
#   VISUALNOVEL_CALIBRATION_SMOKE_UPDATES=120 \
#   VISUALNOVEL_CALIBRATION_NUM_PROCESSES=1 \
#     bash tools/train_calibration_remote_data_volume.sh                     # smoke
#
# smoke 只提前停步，不改 optim.max_updates=5000，所以 LR 轨迹与正式 run 一致。

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_VOLUME="${F5_DATA_VOLUME:-/data}"
DATA_ROOT="${F5_DATA_ROOT:-$DATA_VOLUME/f5-ckpts}"

[[ -d "$DATA_VOLUME" ]] || { printf 'Data volume not found: %s\n' "$DATA_VOLUME" >&2; exit 1; }

mkdir -p "$DATA_ROOT/visualnovel_calibration_ja" "$DATA_ROOT/hydra" "$DATA_VOLUME/f5-tmp/numba-cache"

export TMPDIR="${TMPDIR:-$DATA_VOLUME/f5-tmp}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-$DATA_VOLUME/f5-tmp/numba-cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$DATA_VOLUME/f5-tmp/mpl}"
export HF_HOME="${HF_HOME:-$DATA_VOLUME/f5-tmp/huggingface}"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR"

export VISUALNOVEL_CALIBRATION_RUN_ROOT="${VISUALNOVEL_CALIBRATION_RUN_ROOT:-$DATA_ROOT/visualnovel_calibration_ja}"
export VISUALNOVEL_OPERATION_LOCK_ROOT="${VISUALNOVEL_OPERATION_LOCK_ROOT:-$DATA_ROOT/.visualnovel-operation-locks}"
export VISUALNOVEL_CALIBRATION_DATASET="${VISUALNOVEL_CALIBRATION_DATASET:-$DATA_VOLUME/visualnovel/aggregates/wave-abc-v1-exclude-conflicts}"

# 该机的训练解释器不是系统 python3（后者没装 torch/pytest）。
export PYTHON="${PYTHON:-/root/f5-tts-env/bin/python}"
export ACCELERATE="${ACCELERATE:-/root/f5-tts-env/bin/accelerate}"

# hydra.run.dir 在 config 里是仓库内相对路径，run root 指到大盘也管不到它，
# 所以 launcher 专门暴露了这一个路径旋钮（只接受纯路径，不接受自由 Hydra 参数）。
export VISUALNOVEL_CALIBRATION_HYDRA_RUN_DIR="${VISUALNOVEL_CALIBRATION_HYDRA_RUN_DIR:-$DATA_ROOT/hydra}"

printf 'remote data-volume wrapper:\n'
printf '  run root   : %s\n' "$VISUALNOVEL_CALIBRATION_RUN_ROOT"
printf '  lock root  : %s\n' "$VISUALNOVEL_OPERATION_LOCK_ROOT"
printf '  dataset    : %s\n' "$VISUALNOVEL_CALIBRATION_DATASET"
printf '  TMPDIR     : %s\n' "$TMPDIR"
printf '  numba cache: %s\n' "$NUMBA_CACHE_DIR"
printf '  hydra dir  : %s\n' "$DATA_ROOT/hydra"
printf '  python     : %s\n' "$PYTHON"
df -BG --output=avail "$DATA_ROOT" | tail -1 | xargs printf '  avail      : %sG\n'
printf '\n'

exec bash "$ROOT/tools/train_visualnovel_calibration_ja.sh" "$@"
