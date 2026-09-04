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

# Cache by every input that affects model weights.  The old singleton returned the
# first checkpoint forever, which made checkpoint sweeps silently evaluate the
# wrong weights (and made online/EMA comparisons impossible).
_ja_model_cache: dict[tuple[str, str, str, str, str, str, str, str, int, int, bool, str], object] = {}
_ja_vocoder_cache: dict[tuple[str, str], object] = {}


def clear_ja_model_cache(*, include_vocoder: bool = False) -> None:
    """Release cached Japanese checkpoint models between matrix systems.

    The Gradio path keeps its existing cache behaviour. Evaluation can call this
    after each system so online/EMA checkpoint sweeps never retain several large
    models on the accelerator at once.
    """
    _ja_model_cache.clear()
    if include_vocoder:
        _ja_vocoder_cache.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


_REPO_ROOT = Path(__file__).resolve().parents[3]


def load_ja_model(
    ckpt_path: str | None = None,
    vocab_path: str | None = None,
    vocoder_path: str | None = None,
    *,
    use_ema: bool = True,
    device: str | None = None,
    config_path: str | None = None,
    vocab_contract_path: str | None = None,
    model_family: str = "legacy_jmica",
    expected_vocab_identity: str | None = None,
    expected_token_sequence_sha256: str | None = None,
    expected_embedding_rows: int | None = None,
    expected_embedding_width: int | None = None,
):
    """加载 Jmica 日文模型 + vocos 声码器。返回 (model, vocoder, device)。

    ``use_ema`` is keyword-only so the Gradio defaults remain unchanged. The
    optional compatibility arguments are explicit for evaluation callers; when
    omitted, the historical Jmica config and vocabulary behaviour is preserved.
    Cache keys include every declared contract input that can affect model
    construction or weights.
    """
    ckpt = Path(ckpt_path or _REPO_ROOT / "ckpts" / "F5TTS_JA_Jmica" / "model_21999120.pt").resolve()
    vocab = Path(vocab_path or _REPO_ROOT / "ckpts" / "F5TTS_JA_Jmica" / "vocab_japanese.txt").resolve()
    vocoder_dir = Path(vocoder_path or _REPO_ROOT / "ckpts" / "vocos-mel-24khz").resolve()
    cfg = Path(config_path).resolve() if config_path else (_REPO_ROOT / "src" / "f5_tts" / "configs" / "F5TTS_Base.yaml").resolve()
    contract = Path(vocab_contract_path).resolve() if vocab_contract_path else None

    required = [ckpt, vocab, vocoder_dir, cfg]
    if contract is not None:
        required.append(contract)
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("日文模型文件缺失:\n" + "\n".join(missing))

    if contract is not None and (expected_vocab_identity or expected_token_sequence_sha256):
        from f5_tts.model.vocab_contract import validate_vocabulary_contract

        _, vocab_contract = validate_vocabulary_contract(
            vocab,
            contract,
            expected_identity=expected_vocab_identity,
        )
        if expected_token_sequence_sha256 and vocab_contract.token_sequence_sha256 != expected_token_sequence_sha256:
            raise ValueError(
                f"vocabulary token hash mismatch: expected {expected_token_sequence_sha256}, "
                f"got {vocab_contract.token_sequence_sha256}"
            )
    model_cfg = OmegaConf.load(cfg)
    mel = model_cfg.model.mel_spec
    supported_mel = {
        "target_sample_rate": 24000,
        "n_mel_channels": 100,
        "hop_length": 256,
        "win_length": 1024,
        "n_fft": 1024,
        "mel_spec_type": "vocos",
    }
    actual_mel = {key: getattr(mel, key) for key in supported_mel}
    if actual_mel != supported_mel:
        raise ValueError(f"unsupported Japanese inference mel contract: {actual_mel}")
    model_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_key = (
        str(ckpt),
        str(vocab),
        str(cfg),
        str(contract) if contract else "",
        model_family,
        str(model_cfg.model.arch),
        expected_vocab_identity or "",
        expected_token_sequence_sha256 or "",
        int(expected_embedding_rows or -1),
        int(expected_embedding_width or -1),
        use_ema,
        model_device,
    )
    vocoder_key = (str(vocoder_dir), model_device)

    if vocoder_key not in _ja_vocoder_cache:
        _ja_vocoder_cache[vocoder_key] = load_vocoder(
            vocoder_name="vocos",
            is_local=True,
            local_path=str(vocoder_dir),
            device=model_device,
        )

    if model_key not in _ja_model_cache:
        loaded = load_model(
            model_cls=DiT,
            model_cfg=model_cfg.model.arch,  # 旧版 Base 架构（text_mask_padding=False 等）
            ckpt_path=str(ckpt),
            mel_spec_type="vocos",
            vocab_file=str(vocab),  # 日文假名 vocab
            ode_method="euler",
            use_ema=use_ema,
            device=model_device,
        )
        if expected_embedding_rows is not None or expected_embedding_width is not None:
            from f5_tts.model.checkpoint_init import checkpoint_state_for_source, verify_checkpoint_state_loaded

            expected_state = checkpoint_state_for_source(ckpt, "ema" if use_ema else "online")
            loaded_digest = verify_checkpoint_state_loaded(expected_state, loaded)
            loaded._evaluation_state_sha256 = loaded_digest
        _ja_model_cache[model_key] = loaded
    return _ja_model_cache[model_key], _ja_vocoder_cache[vocoder_key], model_device
