"""Jmica 日文 F5-TTS 模型加载器（Phase 0 基线测试用）。

与 gradio_mix_demo._load_default_model 平行的日文版本：
- 架构：旧版 F5TTS_Base（非 v1！text_mask_padding=False, pe_attn_head=1）
- checkpoint：ckpts/F5TTS_JA_Jmica/model_21999120.pt（Jmica JA_21999120，7.1k 小时日文）
- vocab：ckpts/F5TTS_JA_Jmica/vocab_japanese.txt（假名字符级，无汉字，无拼音转换）
- 许可：cc-by-nc-4.0（非商用，仅实验）

注意两点：
1. 日文 vocab 是字符级 tokenizer，文本直接逐字符查表——不要过 convert_char_to_pinyin
   的拼音逻辑（它对非中文字符原样透传，纯假名文本恰好安全，但混入汉字会悄悄查不到）。
   输入前必须先用 ja_frontend.ja_to_kana 转假名。
2. 模型是 .pt 训练 checkpoint（含 EMA），用 load_model 的 use_ema=True 加载。
"""

from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import OmegaConf

from f5_tts.infer.utils_infer import load_model, load_vocoder
from f5_tts.model import DiT

_ja_model_cache = None
_ja_vocoder_cache = None
_ja_device = None

_REPO_ROOT = Path(__file__).resolve().parents[3]


def load_ja_model(
    ckpt_path: str | None = None,
    vocab_path: str | None = None,
    vocoder_path: str | None = None,
):
    """加载 Jmica 日文模型 + vocos 声码器。返回 (model, vocoder, device)。"""
    global _ja_model_cache, _ja_vocoder_cache, _ja_device
    if _ja_model_cache is not None:
        return _ja_model_cache, _ja_vocoder_cache, _ja_device

    ckpt = Path(ckpt_path or _REPO_ROOT / "ckpts" / "F5TTS_JA_Jmica" / "model_21999120.pt")
    vocab = Path(vocab_path or _REPO_ROOT / "ckpts" / "F5TTS_JA_Jmica" / "vocab_japanese.txt")
    vocoder_dir = Path(vocoder_path or _REPO_ROOT / "ckpts" / "vocos-mel-24khz")
    cfg = _REPO_ROOT / "src" / "f5_tts" / "configs" / "F5TTS_Base.yaml"

    missing = [str(p) for p in (ckpt, vocab, vocoder_dir, cfg) if not p.exists()]
    if missing:
        raise FileNotFoundError("日文模型文件缺失:\n" + "\n".join(missing))

    model_cfg = OmegaConf.load(cfg)

    _ja_device = "cuda" if torch.cuda.is_available() else "cpu"

    _ja_vocoder_cache = load_vocoder(
        vocoder_name="vocos",
        is_local=True,
        local_path=str(vocoder_dir),
        device=_ja_device,
    )

    _ja_model_cache = load_model(
        model_cls=DiT,
        model_cfg=model_cfg.model.arch,   # 旧版 Base 架构（text_mask_padding=False 等）
        ckpt_path=str(ckpt),
        mel_spec_type="vocos",
        vocab_file=str(vocab),            # 日文假名 vocab
        ode_method="euler",
        use_ema=True,
        device=_ja_device,
    )
    return _ja_model_cache, _ja_vocoder_cache, _ja_device
