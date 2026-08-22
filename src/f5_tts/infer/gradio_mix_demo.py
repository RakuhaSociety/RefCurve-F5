import os
import tempfile
from importlib.resources import files
from pathlib import Path
from typing import Optional

import gradio as gr
import torch
from omegaconf import OmegaConf
import torchaudio

from f5_tts.infer.utils_infer import (
    cfg_strength,
    cross_fade_duration,
    fix_duration,
    infer_process,
    load_model,
    load_vocoder,
    mel_spec_type,
    nfe_step,
    preprocess_ref_audio_text,
    speed,
    sway_sampling_coef,
    target_rms,
)
from f5_tts.model import DiT, UNetT

# ----------------------------
# Model loader (lazy singleton)
# ----------------------------

_model_cache = None
_vocoder_cache = None
_model_device = None
_paraformer_asr = None


def _resolve_path(rel: str) -> Path:
    """Search for a relative path under (a) repo root, (b) package resources."""
    candidates = []
    # repo root guess
    repo_root = Path(__file__).resolve().parents[3]
    candidates.append(repo_root / rel)
    # package resource
    candidates.append(Path(files("f5_tts").joinpath(rel)))
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


def _load_default_model(model_name: str = "F5TTS_v1_Base"):
    """Lazy load default model and vocoder from ckpts/. Prioritize repo root ckpts, fallback to package."""

    global _model_cache, _vocoder_cache, _model_device
    if _model_cache is not None and _vocoder_cache is not None:
        return _model_cache, _vocoder_cache, _model_device

    cfg_path = _resolve_path(f"configs/{model_name}.yaml")
    ckpt_path = _resolve_path(f"ckpts/{model_name}/model_1250000.safetensors")
    vocab_path = _resolve_path(f"ckpts/{model_name}/vocab.txt")
    vocoder_local = _resolve_path("ckpts/vocos-mel-24khz")

    missing = []
    if not cfg_path.exists():
        missing.append(str(cfg_path))
    if not ckpt_path.exists():
        missing.append(str(ckpt_path))
    if not vocab_path.exists():
        missing.append(str(vocab_path))
    if missing:
        raise gr.Error("找不到模型文件，请检查路径:\n" + "\n".join(missing))

    model_cfg = OmegaConf.load(cfg_path)
    backbone_name = model_cfg.model.backbone
    model_cls = DiT if "DiT" in backbone_name else UNetT

    _model_device = "cuda" if torch.cuda.is_available() else "cpu"

    _vocoder_cache = load_vocoder(
        vocoder_name="vocos",
        is_local=True,
        local_path=str(vocoder_local),
        device=_model_device,
    )

    _model_cache = load_model(
        model_cls=model_cls,
        model_cfg=model_cfg.model.arch,
        ckpt_path=str(ckpt_path),
        mel_spec_type="vocos",
        vocab_file=str(vocab_path),
        ode_method="euler",
        use_ema=True,
        device=_model_device,
    )
    return _model_cache, _vocoder_cache, _model_device


def _get_paraformer_asr(device: str):
    """Lazy-load FunASR Paraformer ASR. Raises a user-facing error if missing."""
    global _paraformer_asr
    if _paraformer_asr is not None:
        return _paraformer_asr

    try:
        from funasr import AutoModel
    except ImportError:
        raise gr.Error("未安装 FunASR。请先安装：pip install funasr modelscope")

    try:
        # paraformer-zh: general Mandarin model; trust_remote_code required
        _paraformer_asr = AutoModel(
            model="paraformer-zh",
            trust_remote_code=True,
            device=device,
        )
    except Exception as e:
        raise gr.Error(
            "加载 FunASR Paraformer 失败，请确认已安装依赖，并可访问模型：" + str(e)
        )
    return _paraformer_asr


def _clean_cn_text(text: str) -> str:
    """Collapse spaces and ensure a closing punctuation for Chinese text."""
    if not text:
        return text
    # Remove all ASCII spaces
    t = text.replace(" ", "")
    t = t.strip()
    if not t:
        return t
    if t[-1] not in "。.!？！?":
        t = t + "。"
    return t


# ----------------------------
# Inference wrapper for Gradio
# ----------------------------


def run_inference(
    ref_audio_a,
    ref_text_a,
    ref_audio_b,
    ref_text_b,
    gen_text,
    steps,
    cfg,
    sway_coef,
    speed_val,
    mix_method,
    mix_schedule,
    mix_a_start,
    mix_a_end,
    mix_2d_mode,
    n_schedule,
    n_a_start,
    n_a_end,
    allow_extrapolation,
    seed,
    use_asr,
):
    if not ref_audio_a or not ref_audio_b:
        raise gr.Error("请提供两段参考音频 (A/B)")

    # gradio Audio(type="filepath") returns a path string
    ref_a_path = Path(ref_audio_a)
    ref_b_path = Path(ref_audio_b)
    if not ref_a_path.exists() or not ref_b_path.exists():
        raise gr.Error("音频文件不存在，请重新上传")

    # Load model early to get device for ASR
    model, vocoder, model_device = _load_default_model()

    # Preprocess references (silence trimming). If use_asr=False and text empty, skip ASR by providing a dot.
    if use_asr:
        asr = _get_paraformer_asr(device=model_device)
        if not ref_text_a.strip():
            res = asr.generate(input=str(ref_a_path), batch_size=1)
            ref_text_a = res[0].get("text", "") if isinstance(res, list) else res.get("text", "")
            ref_text_a = _clean_cn_text(ref_text_a)
        if not ref_text_b.strip():
            res = asr.generate(input=str(ref_b_path), batch_size=1)
            ref_text_b = res[0].get("text", "") if isinstance(res, list) else res.get("text", "")
            ref_text_b = _clean_cn_text(ref_text_b)
    else:
        ref_text_a = ref_text_a if ref_text_a.strip() else "."
        ref_text_b = ref_text_b if ref_text_b.strip() else "."

    ref_a_path, ref_text_a = preprocess_ref_audio_text(str(ref_a_path), ref_text_a or "")
    ref_b_path, ref_text_b = preprocess_ref_audio_text(str(ref_b_path), ref_text_b or "")

    # gradio Radio may return "None" (str); normalize
    if n_schedule in ("none", "None", None):
        n_schedule = None

    # Seed: empty -> None (random); otherwise cast to int
    seed_val = None
    if seed not in (None, "", "None"):
        try:
            seed_val = int(seed)
        except ValueError:
            raise gr.Error("Seed 需要是整数或留空")

    audio_np, sr, _ = infer_process(
        ref_a_path,
        ref_text_a,
        ref_b_path,
        ref_text_b,
        gen_text,
        model,
        vocoder,
        mel_spec_type="vocos",
        target_rms=target_rms,
        cross_fade_duration=cross_fade_duration,
        nfe_step=steps,
        cfg_strength=cfg,
        sway_sampling_coef=sway_coef,
        speed=speed_val,
        fix_duration=fix_duration,
        device=model_device,
        allow_extrapolation=allow_extrapolation,
        seed=seed_val,
        mix_method=mix_method,
        mix_schedule=mix_schedule,
        mix_a_start=mix_a_start,
        mix_a_end=mix_a_end,
        mix_2d_mode=mix_2d_mode,
        n_schedule=n_schedule,
        n_a_start=n_a_start,
        n_a_end=n_a_end,
    )

    return (sr, audio_np)


def _prepare_transfer_audio(ref_a_path: Path, ref_b_path: Path, ref_c_path: Path, scale: float = 1.0):
    """
    Construct an augmented reference: C + scale * (B - A).
    Returns path to augmented wav and aligned C wav (both trimmed to min length).
    """
    def load_audio(path: Path):
        audio, sr = torchaudio.load(str(path))
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        if sr != 24000:
            audio = torchaudio.transforms.Resample(sr, 24000)(audio)
        return audio

    a = load_audio(ref_a_path)
    b = load_audio(ref_b_path)
    c = load_audio(ref_c_path)
    min_len = min(a.shape[-1], b.shape[-1], c.shape[-1])
    a = a[..., :min_len]
    b = b[..., :min_len]
    c = c[..., :min_len]
    aug = c + scale * (b - a)

    tmp_c = Path(tempfile.NamedTemporaryFile(delete=False, suffix="_c.wav").name)
    tmp_aug = Path(tempfile.NamedTemporaryFile(delete=False, suffix="_aug.wav").name)
    torchaudio.save(str(tmp_c), c, 24000)
    torchaudio.save(str(tmp_aug), aug, 24000)
    return tmp_c, tmp_aug


def run_inference_transfer(
    ref_audio_a,
    ref_text_a,
    ref_audio_b,
    ref_text_b,
    ref_audio_c,
    ref_text_c,
    gen_text,
    steps,
    cfg,
    sway_coef,
    speed_val,
    mix_method,
    mix_schedule,
    mix_a_start,
    mix_a_end,
    mix_2d_mode,
    n_schedule,
    n_a_start,
    n_a_end,
    allow_extrapolation,
    seed,
    use_asr,
    diff_scale,
):
    if not ref_audio_a or not ref_audio_b or not ref_audio_c:
        raise gr.Error("请提供三段参考音频 (A/B/C)")

    pa = Path(ref_audio_a)
    pb = Path(ref_audio_b)
    pc = Path(ref_audio_c)
    for p in (pa, pb, pc):
        if not p.exists():
            raise gr.Error("音频文件不存在，请重新上传")

    model, vocoder, model_device = _load_default_model()

    # ASR for missing texts
    if use_asr:
        asr = _get_paraformer_asr(device=model_device)
        if not ref_text_a.strip():
            res = asr.generate(input=str(pa), batch_size=1)
            ref_text_a = res[0].get("text", "") if isinstance(res, list) else res.get("text", "")
            ref_text_a = _clean_cn_text(ref_text_a)
        if not ref_text_b.strip():
            res = asr.generate(input=str(pb), batch_size=1)
            ref_text_b = res[0].get("text", "") if isinstance(res, list) else res.get("text", "")
            ref_text_b = _clean_cn_text(ref_text_b)
        if not ref_text_c.strip():
            res = asr.generate(input=str(pc), batch_size=1)
            ref_text_c = res[0].get("text", "") if isinstance(res, list) else res.get("text", "")
            ref_text_c = _clean_cn_text(ref_text_c)
    else:
        ref_text_a = ref_text_a if ref_text_a.strip() else "."
        ref_text_b = ref_text_b if ref_text_b.strip() else "."
        ref_text_c = ref_text_c if ref_text_c.strip() else "."

    # Preprocess A/B/C (silence trim etc.)
    pa_proc, ref_text_a = preprocess_ref_audio_text(str(pa), ref_text_a)
    pb_proc, ref_text_b = preprocess_ref_audio_text(str(pb), ref_text_b)
    pc_proc, ref_text_c = preprocess_ref_audio_text(str(pc), ref_text_c)

    # Build augmented reference: C + scale*(B-A)
    c_path_aligned, aug_path = _prepare_transfer_audio(Path(pa_proc), Path(pb_proc), Path(pc_proc), scale=diff_scale)

    # n_schedule normalize
    if n_schedule in ("none", "None", None):
        n_schedule = None

    seed_val = None
    if seed not in (None, "", "None"):
        try:
            seed_val = int(seed)
        except ValueError:
            raise gr.Error("Seed 需要是整数或留空")

    audio_np, sr, _ = infer_process(
        str(c_path_aligned),
        ref_text_c,
        str(aug_path),
        ref_text_c,
        gen_text,
        model,
        vocoder,
        mel_spec_type="vocos",
        target_rms=target_rms,
        cross_fade_duration=cross_fade_duration,
        nfe_step=steps,
        cfg_strength=cfg,
        sway_sampling_coef=sway_coef,
        speed=speed_val,
        fix_duration=fix_duration,
        device=model_device,
        allow_extrapolation=allow_extrapolation,
        seed=seed_val,
        mix_method=mix_method,
        mix_schedule=mix_schedule,
        mix_a_start=mix_a_start,
        mix_a_end=mix_a_end,
        mix_2d_mode=mix_2d_mode,
        n_schedule=n_schedule,
        n_a_start=n_a_start,
        n_a_end=n_a_end,
    )

    # cleanup temp files
    for p in (c_path_aligned, aug_path):
        try:
            Path(p).unlink(missing_ok=True)
        except Exception:
            pass

    return (sr, audio_np)


# ----------------------------
# Gradio UI
# ----------------------------


def build_interface():
    with gr.Blocks(title="Sound characteristics Mix Controller") as demo:
        gr.Markdown("## 声音特征 混合实验台")
        with gr.Tabs():
            with gr.Tab("双参考混合"):
                with gr.Row():
                    with gr.Column():
                        ref_audio_a = gr.Audio(label="参考音频 A", type="filepath")
                        ref_text_a = gr.Textbox(label="参考文本 A (可留空自动转写)")
                        ref_audio_b = gr.Audio(label="参考音频 B", type="filepath")
                        ref_text_b = gr.Textbox(label="参考文本 B (可留空自动转写)")
                        gen_text = gr.Textbox(label="生成文本", value="让我们一起说中文。")
                        use_asr = gr.Checkbox(value=False, label="启用自动转写（FunASR Paraformer，无需 ffmpeg）")
                    with gr.Column():
                        steps = gr.Slider(8, 64, value=nfe_step, step=1, label="NFE steps")
                        cfg = gr.Slider(0.0, 5.0, value=cfg_strength, step=0.1, label="CFG strength")
                        sway_coef = gr.Slider(-2.0, 2.0, value=sway_sampling_coef, step=0.1, label="Sway coef")
                        speed_val = gr.Slider(0.3, 2.0, value=speed, step=0.05, label="语速倍率")
                        seed = gr.Number(value=None, label="Seed（留空随机）", precision=0)
                        gr.Markdown("### 混合参数")
                        mix_method = gr.Radio(["lerp", "slerp", "log"], value="slerp", label="混合算法")
                        mix_schedule = gr.Radio(["linear", "cosine", "sigmoid"], value="linear", label="t 维度曲线")
                        mix_a_start = gr.Slider(-1.0, 2.0, value=0.5, step=0.05, label="t=0 A 权重")
                        mix_a_end = gr.Slider(-1.0, 2.0, value=0.5, step=0.05, label="t=1 A 权重")
                        mix_2d_mode = gr.Radio(["t_only", "n_only", "multiply", "add", "max", "min"], value="t_only", label="2D 组合模式")
                        n_schedule = gr.Radio(["none", "linear", "cosine", "sigmoid"], value="linear", label="n 维度曲线")
                        n_a_start = gr.Slider(-1.0, 2.0, value=0.5, step=0.05, label="n=0 A 权重", interactive=True)
                        n_a_end = gr.Slider(-1.0, 2.0, value=0.5, step=0.05, label="n=1 A 权重", interactive=True)
                        allow_extrapolation = gr.Checkbox(value=True, label="允许权重超出[0,1]")

                run_btn = gr.Button("生成（双参考）")
                out_audio = gr.Audio(label="生成音频", type="numpy")

                run_btn.click(
                    fn=run_inference,
                    inputs=[
                        ref_audio_a,
                        ref_text_a,
                        ref_audio_b,
                        ref_text_b,
                        gen_text,
                        steps,
                        cfg,
                        sway_coef,
                        speed_val,
                        mix_method,
                        mix_schedule,
                        mix_a_start,
                        mix_a_end,
                        mix_2d_mode,
                        n_schedule,
                        n_a_start,
                        n_a_end,
                        allow_extrapolation,
                        seed,
                        use_asr,
                    ],
                    outputs=[out_audio],
                )

            with gr.Tab("三参考情绪迁移"):
                with gr.Row():
                    with gr.Column():
                        ref_audio_a3 = gr.Audio(label="参考音频 A (情绪基准)", type="filepath")
                        ref_text_a3 = gr.Textbox(label="参考文本 A (可留空自动转写)")
                        ref_audio_b3 = gr.Audio(label="参考音频 B (情绪目标)", type="filepath")
                        ref_text_b3 = gr.Textbox(label="参考文本 B (可留空自动转写)")
                        ref_audio_c3 = gr.Audio(label="参考音频 C (声线基底)", type="filepath")
                        ref_text_c3 = gr.Textbox(label="参考文本 C (可留空自动转写)")
                        gen_text3 = gr.Textbox(label="生成文本", value="让我们一起说中文。")
                        use_asr3 = gr.Checkbox(value=False, label="启用自动转写（FunASR Paraformer，无需 ffmpeg）")
                        diff_scale = gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="差分缩放 (B-A)")
                    with gr.Column():
                        steps3 = gr.Slider(8, 64, value=nfe_step, step=1, label="NFE steps")
                        cfg3 = gr.Slider(0.0, 5.0, value=cfg_strength, step=0.1, label="CFG strength")
                        sway_coef3 = gr.Slider(-2.0, 2.0, value=sway_sampling_coef, step=0.1, label="Sway coef")
                        speed_val3 = gr.Slider(0.3, 2.0, value=speed, step=0.05, label="语速倍率")
                        seed3 = gr.Number(value=None, label="Seed（留空随机）", precision=0)
                        gr.Markdown("### 混合参数 (控制情绪注入强度)")
                        mix_method3 = gr.Radio(["lerp", "slerp", "log"], value="lerp", label="混合算法")
                        mix_schedule3 = gr.Radio(["linear", "cosine", "sigmoid"], value="linear", label="t 维度曲线")
                        mix_a_start3 = gr.Slider(-3.0, 3.0, value=0.0, step=0.05, label="t=0 情绪强度 (可放大/反转)")
                        mix_a_end3 = gr.Slider(-3.0, 3.0, value=1.0, step=0.05, label="t=1 情绪强度 (可放大/反转)")
                        mix_2d_mode3 = gr.Radio(["t_only", "n_only", "multiply", "add", "max", "min"], value="t_only", label="2D 组合模式")
                        n_schedule3 = gr.Radio(["none", "linear", "cosine", "sigmoid"], value="none", label="n 维度曲线")
                        n_a_start3 = gr.Slider(-3.0, 3.0, value=1.0, step=0.05, label="n=0 情绪强度 (可放大/反转)", interactive=True)
                        n_a_end3 = gr.Slider(-3.0, 3.0, value=1.0, step=0.05, label="n=1 情绪强度 (可放大/反转)", interactive=True)
                        allow_extrapolation3 = gr.Checkbox(value=True, label="允许权重超出[0,1] (放大/反转时需开启)")

                run_btn3 = gr.Button("生成（三参考情绪迁移）")
                out_audio3 = gr.Audio(label="生成音频", type="numpy")

                run_btn3.click(
                    fn=run_inference_transfer,
                    inputs=[
                        ref_audio_a3,
                        ref_text_a3,
                        ref_audio_b3,
                        ref_text_b3,
                        ref_audio_c3,
                        ref_text_c3,
                        gen_text3,
                        steps3,
                        cfg3,
                        sway_coef3,
                        speed_val3,
                        mix_method3,
                        mix_schedule3,
                        mix_a_start3,
                        mix_a_end3,
                        mix_2d_mode3,
                        n_schedule3,
                        n_a_start3,
                        n_a_end3,
                        allow_extrapolation3,
                        seed3,
                        use_asr3,
                        diff_scale,
                    ],
                    outputs=[out_audio3],
                )

    return demo


def main():
    demo = build_interface()
    demo.launch()


if __name__ == "__main__":
    main()
