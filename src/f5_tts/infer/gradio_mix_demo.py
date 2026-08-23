import os
from importlib.resources import files
from pathlib import Path
from typing import Optional

import gradio as gr
import torch
from omegaconf import OmegaConf

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
    mix_on,
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
        gen_text,
        model,
        vocoder,
        ref_audio_2=ref_b_path,
        ref_text_2=ref_text_b,
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
        mix_on=mix_on,
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
                        mix_on = gr.Radio(
                            ["cond", "pred", "output"],
                            value="pred",
                            label="混合模式",
                            info=(
                                "pred（推荐）: 每步分别用 A / B 各推理一次再混合速度场，"
                                "开销翻倍但音素结构不被破坏、情绪控制最准确，且每个分支用"
                                "自己的转写（文本与音频自洽）。适合绝大多数场景。"
                                "cond: 逐帧混合 mel 条件后推理 1 次，快但两参考共用一条文本"
                                "（按权重二选一），较短参考主导时可能吞字。"
                                "output: 两条参考各自完整生成一遍，再用 DTW 在 mel 域对齐后混合。"
                                "⚠️ 跨说话人时会听出明显叠音——DTW 在 mel 帧上找不准音素对应，"
                                "且成品频谱直接平均会抹平共振峰。仅适合同说话人不同语速、"
                                "或确实需要节奏随权重插值的场景。"
                            ),
                        )
                        mix_method = gr.Radio(
                            ["lerp", "slerp", "log"],
                            value="slerp",
                            label="混合算法",
                            info=(
                                "cond / output 模式下生效。pred 模式混合的是速度场"
                                "（切空间向量，线性可加），恒用线性混合，此项被忽略。"
                            ),
                        )
                        mix_schedule = gr.Radio(
                            ["linear", "cosine", "sigmoid"],
                            value="linear",
                            label="t 维度曲线（仅 cond / pred）",
                            info="output 模式没有扩散时间轴，此项被忽略。",
                        )
                        mix_a_start = gr.Slider(
                            -1.0,
                            2.0,
                            value=0.5,
                            step=0.05,
                            label="t=0 A 权重",
                            info=(
                                "output 模式下：n 维度曲线为 none 时它是唯一的标量 A 权重；"
                                "无论曲线开关，它都决定节奏（输出时长）偏向 A 还是 B。"
                            ),
                        )
                        mix_a_end = gr.Slider(
                            -1.0, 2.0, value=0.5, step=0.05, label="t=1 A 权重（仅 cond / pred）"
                        )
                        mix_2d_mode = gr.Radio(
                            ["t_only", "n_only", "multiply", "add", "max", "min"],
                            value="t_only",
                            label="2D 组合模式（仅 cond / pred）",
                            info="output 模式无 t / n 两维可组合，此项被忽略。",
                        )
                        n_schedule = gr.Radio(
                            ["none", "linear", "cosine", "sigmoid"],
                            value="linear",
                            label="n 维度曲线",
                            info=(
                                "三种模式均生效。output 模式下选 none 时用「t=0 A 权重」作为标量权重；"
                                "选其他曲线则改用 n=0 / n=1 权重做帧级控制（如开头偏 A、结尾偏 B），"
                                "此时「t=0 A 权重」仅用于决定节奏偏向。"
                            ),
                        )
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
                        mix_on,
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

    return demo


def main():
    demo = build_interface()
    demo.launch()


if __name__ == "__main__":
    main()
