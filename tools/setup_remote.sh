#!/usr/bin/env bash
# 远端服务器 F5-TTS 训练环境补全脚本
# 复用服务器已有 PyTorch；不下载、不升级、不替换 torch/torchaudio。
set -euo pipefail

VENV="${F5TTS_VENV:-/root/f5-tts-env}"
REPO="${F5TTS_REPO:-/root/F5-TTS}"

if [[ ! -x "$VENV/bin/python" ]]; then
    printf 'Existing virtual environment not found: %s\n' "$VENV" >&2
    printf 'Create it and install the server CUDA-compatible torch/torchaudio first.\n' >&2
    exit 1
fi

echo "=== [1/4] 验证已有 PyTorch/CUDA（不会重装 torch）==="
"$VENV/bin/python" - <<'PY'
import torch
import torchaudio

print(f"torch {torch.__version__}")
print(f"torchaudio {torchaudio.__version__}")
print(f"torch CUDA {torch.version.cuda}")
print(f"cuda available: {torch.cuda.is_available()}")
print(f"gpu count: {torch.cuda.device_count()}")
if not torch.cuda.is_available():
    raise RuntimeError("existing PyTorch cannot access CUDA")
PY

echo "=== [2/4] 升级打包工具 ==="
"$VENV/bin/python" -m pip install --upgrade pip setuptools wheel -q

echo "=== [3/4] 显式补齐训练关键依赖 ==="
"$VENV/bin/python" -m pip install \
    'accelerate>=0.33.0' \
    cached_path \
    datasets \
    einops \
    'ema-pytorch>=0.5.2' \
    hydra-core \
    librosa \
    'numpy==1.26.4' \
    omegaconf \
    pypinyin \
    rjieba \
    safetensors \
    soundfile \
    torchdiffeq \
    tqdm \
    transformers \
    transformers_stream_generator \
    unidecode \
    vocos \
    wandb \
    'x-transformers>=1.31.14' \
    -q

if [[ ! -d "$REPO" ]]; then
    printf 'Repository not found: %s\n' "$REPO" >&2
    exit 1
fi

echo "=== [4/4] 安装本仓库（--no-deps 防止解析并替换 PyTorch）==="
"$VENV/bin/python" -m pip install -e "$REPO" --no-deps -q

echo ""
echo "=== 完成！先运行：bash $REPO/tools/train_f5tts_v1_ja.sh preflight ==="
