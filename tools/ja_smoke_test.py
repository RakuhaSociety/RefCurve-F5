"""Phase 0 冒烟测试：Jmica 日文模型能否正常加载 + 单参考日文推理。可反复运行。

没有日文参考音频时，用合成的静音+正弦波不行（模型需要真实语音做参考），
所以本测试分两级：
  Level 1: 模型加载 + vocab/架构匹配（无需参考音频）
  Level 2: 若 tests/ja_ref/ 下有用户放置的日文参考 wav+txt，跑完整推理
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

print("=== Level 1: 模型加载 ===")
from f5_tts.infer.ja_model import load_ja_model

model, vocoder, device = load_ja_model()
print(f"device: {device}")
print(f"model type: {type(model).__name__}")
print(f"vocab size: {len(model.vocab_char_map)}")

# 假名在 vocab_char_map 里可查
probe = "ワタシハアイウエオ"
missing = [c for c in probe if c not in model.vocab_char_map]
print(f"假名查表: {'OK 全部命中' if not missing else f'MISSING {missing}'}")

print()
print("=== Level 2: 完整推理（需要日文参考音频）===")
ref_dir = ROOT / "tests" / "ja_ref"
wavs = sorted(ref_dir.glob("*.wav")) if ref_dir.exists() else []
if not wavs:
    print(f"未找到参考音频。把日文参考 wav + 同名 txt（转写）放到 {ref_dir} 后重跑本脚本。")
    print("Level 1 通过，模型本身就绪。")
    sys.exit(0)

import soundfile as sf

from f5_tts.infer.ja_frontend import ja_to_kana, check_vocab_coverage
from f5_tts.infer.utils_infer import infer_process

VOCAB = str(ROOT / "ckpts" / "F5TTS_JA_Jmica" / "vocab_japanese.txt")
GEN_JA = "こんな所で、諦めるわけにはいかない！"

wav_path = wavs[0]
txt_path = wav_path.with_suffix(".txt")
ref_text_raw = txt_path.read_text(encoding="utf-8").strip() if txt_path.exists() else ""
if not ref_text_raw:
    print(f"缺少转写 {txt_path.name}，跳过推理。")
    sys.exit(0)

ref_kana = ja_to_kana(ref_text_raw)
gen_kana = ja_to_kana(GEN_JA)
print(f"参考: {wav_path.name}")
print(f"转写(假名): {ref_kana}")
print(f"生成(假名): {gen_kana}")
for label, s in (("ref", ref_kana), ("gen", gen_kana)):
    miss = check_vocab_coverage(s, VOCAB)
    if miss:
        print(f"警告 {label} 有 vocab 外字符: {miss}")

wav, sr, _ = infer_process(
    str(wav_path), ref_kana, gen_kana, model, vocoder,
    nfe_step=32, seed=777, show_info=lambda *a, **k: None,
)
out = ROOT / "tests" / "ja_ref" / "_smoke_output.wav"
sf.write(str(out), wav, sr)
print(f"生成成功: {out}  duration={len(wav)/sr:.2f}s")
