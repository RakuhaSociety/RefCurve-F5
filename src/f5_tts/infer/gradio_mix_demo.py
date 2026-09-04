import math
import os
from importlib.resources import files
from pathlib import Path
from typing import Optional

import gradio as gr
import torch
from omegaconf import OmegaConf

from f5_tts.infer import asr_backends
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
    trim_generated_silence,
)
from f5_tts.model import DiT, UNetT
from f5_tts.infer.ja_frontend import ja_to_kana
from f5_tts.infer.ja_model import load_ja_model

# ----------------------------
# Model loader (lazy singleton)
# ----------------------------

_model_cache = None
_vocoder_cache = None
_model_device = None


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
    """Lazy-load FunASR Paraformer（中文）。实现在 asr_backends，这里只包装成 gr.Error。

    保留此名字是因为 tools/mix_curve_server.py 从本模块 import 它。
    """
    try:
        return asr_backends.get_paraformer_asr(device=device)
    except asr_backends.AsrUnavailable as e:
        raise gr.Error(str(e))


def _clean_cn_text(text: str) -> str:
    """转发到 asr_backends.clean_cn_text（同上，保留名字供 mix_curve_server import）。"""
    return asr_backends.clean_cn_text(text)


def _transcribe(audio_path, lang: str, device: str) -> str:
    """按语言自动转写参考音频，失败时抛 gr.Error 给 UI。"""
    try:
        return asr_backends.transcribe(str(audio_path), lang=lang, device=device)
    except asr_backends.AsrUnavailable as e:
        raise gr.Error(str(e))


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
    lang_mode="zh_en",
    trim_edges=True,  # ✅ 裁掉生成音频的首尾静音
    max_internal_silence=0.0,  # ✅ >0 时把内部空白压到该上限（秒），0 表示不动
):
    if not ref_audio_a or not ref_audio_b:
        raise gr.Error("请提供两段参考音频 (A/B)")

    # gradio Audio(type="filepath") returns a path string
    ref_a_path = Path(ref_audio_a)
    ref_b_path = Path(ref_audio_b)
    if not ref_a_path.exists() or not ref_b_path.exists():
        raise gr.Error("音频文件不存在，请重新上传")

    # Load model early to get device for ASR
    # 按语言模式加载对应模型：日文走 Jmica（旧架构 + 假名 vocab），中英文走默认 v1 模型
    asr_lang = "ja" if lang_mode == "ja" else "zh"
    if lang_mode == "ja":
        try:
            model, vocoder, model_device = load_ja_model()
        except FileNotFoundError as e:
            raise gr.Error(str(e))
    else:
        model, vocoder, model_device = _load_default_model()

    # Preprocess references (silence trimming). If use_asr=False and text empty, skip ASR by providing a dot.
    if use_asr:
        if not ref_text_a.strip():
            ref_text_a = _transcribe(ref_a_path, lang=asr_lang, device=model_device)
        if not ref_text_b.strip():
            ref_text_b = _transcribe(ref_b_path, lang=asr_lang, device=model_device)
    else:
        ref_text_a = ref_text_a if ref_text_a.strip() else "."
        ref_text_b = ref_text_b if ref_text_b.strip() else "."

    ref_a_path, ref_text_a = preprocess_ref_audio_text(str(ref_a_path), ref_text_a or "")
    ref_b_path, ref_text_b = preprocess_ref_audio_text(str(ref_b_path), ref_text_b or "")

    # 日文模式：汉字 → 假名（Jmica vocab 只含假名，汉字会被查表为 -1 丢失）
    # preprocess_ref_audio_text 之后再转，避免它把已转好的假名再做 strip 截断。
    #
    # ⚠️ 占位符"."绝对不能传给模型：模型看到 1 个句点要对应整段参考音频（~200帧），
    # 文本-音频对齐完全断裂，开头会产生韩语般的杂音，只有 gen_text 区域正确。
    # 解决方案：占位符统一清成 ""，走 text_list=[gen_text] 分支，模型只用音频条件。
    if lang_mode == "ja":
        try:
            if ref_text_a.strip() in ("", "."):
                ref_text_a = ""
            else:
                ref_text_a = ja_to_kana(ref_text_a)
            if ref_text_b.strip() in ("", "."):
                ref_text_b = ""
            else:
                ref_text_b = ja_to_kana(ref_text_b)
            gen_text = ja_to_kana(gen_text)
        except RuntimeError as e:
            raise gr.Error(str(e))

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

    # 日文字符级前端：逐字符分割，不走 convert_char_to_pinyin 的拼音路径。
    # Jmica vocab 是假名字符级，直接分字即可；汉字已被 ja_to_kana 提前转掉。
    def _ja_frontend(text_list):
        return [list(t) for t in text_list]

    text_frontend = _ja_frontend if lang_mode == "ja" else None

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
        text_frontend=text_frontend,
    )

    # ✅ 生成时长在推理前就按字节比例算定（模型拿到定长画布、必须填满，没有"说完就停"
    # 这个选项），而实测同 mora 数的台词真实时长能差 4-6 倍，估不准的余量就成了静音。
    # 这里做事后整理：首尾裁剪默认开，内部空白压缩要显式给上限。
    # NaN 的比较恒为 False，若不显式挡掉会静默跳过压缩（不崩但无声），所以宁可报错。
    cap = float(max_internal_silence or 0.0)
    if not math.isfinite(cap):
        raise gr.Error("内部空白上限必须是有限数")
    cap = min(max(cap, 0.0), 2.0)
    if trim_edges or cap > 0:
        audio_np, _ = trim_generated_silence(
            audio_np,
            sr,
            trim_edges=bool(trim_edges),
            max_internal_silence_s=cap if cap > 0 else None,
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
                        use_asr = gr.Checkbox(value=False, label="启用自动转写（参考文本留空时生效）")
                        lang_mode = gr.Radio(
                            choices=[
                                ("中文 / 英文（F5TTS_v1_Base + FunASR）", "zh_en"),
                                ("日文（Jmica JA_21999120 + Visual-novel-whisper）", "ja"),
                            ],
                            value="zh_en",
                            label="合成语言 / 模型",
                            info=(
                                "切换语言会加载不同模型：中英文用 F5TTS_v1_Base（中英双语），"
                                "日文用 Jmica JA_21999120（假名 vocab，7.1k 小时 Galgame 语料）。"
                                "两个模型共用同一套 vocos 声码器，无需重复加载。"
                            ),
                        )
                    with gr.Column():
                        steps = gr.Slider(8, 64, value=nfe_step, step=1, label="NFE steps")
                        cfg = gr.Slider(0.0, 5.0, value=cfg_strength, step=0.1, label="CFG strength")
                        sway_coef = gr.Slider(-2.0, 2.0, value=sway_sampling_coef, step=0.1, label="Sway coef")
                        speed_val = gr.Slider(0.3, 2.0, value=speed, step=0.05, label="语速倍率")
                        seed = gr.Number(value=None, label="Seed（留空随机）", precision=0)
                        trim_edges = gr.Checkbox(
                            value=True,
                            label="裁掉首尾静音",
                            info=(
                                "生成长度在推理前就按「参考帧数 / 参考文本字节 × 目标文本字节 / 语速」"
                                "算定，模型拿到定长画布必须填满，没有「说完就停」这个选项；而实测同 mora "
                                "数的台词真实时长能差 4-6 倍（停顿、拖腔不在文本里），估不准的余量就成了"
                                "静音。此项做事后裁剪，首尾各留 50ms，不动内部。"
                            ),
                        )
                        max_internal_silence = gr.Slider(
                            0.0,
                            2.0,
                            value=0.0,
                            step=0.05,
                            label="内部空白上限（秒，0=不处理）",
                            info=(
                                "把句中过长的空白压到该上限，两端对称保留以保住尾音衰减与起音。"
                                "⚠️ 句读处的停顿（逗号、问号）是正常语调，压太狠会让语速发急；"
                                "真正要压的是 cond / pred 吞字留下的 1.5s+ 空位——那种情况更该直接换 "
                                "two_stage。建议先留 0，确有需要再设 0.5-0.8。"
                            ),
                        )
                        gr.Markdown("### 混合参数")
                        mix_on = gr.Radio(
                            ["cond", "pred", "two_stage", "output"],
                            value="pred",
                            label="混合模式",
                            info=(
                                "two_stage（两条参考长度差得多时用这个）: 先让两条参考各自"
                                "单独合成一遍目标文本，再拿这两条产出做 pred 混合。实测"
                                "CER 0.056、字数与目标一致，权重响应覆盖两个单参考锚点跨距"
                                "约 76%。代价是推理三次，约 3 倍耗时。"
                                "pred: 每步分别用 A / B 各推理一次再混合速度场，开销翻倍，"
                                "音素结构不被破坏且每个分支用自己的转写。"
                                "cond: 逐帧混合 mel 条件后推理 1 次，最快，但两参考共用一条"
                                "文本（按权重二选一）。"
                                "⚠️ cond 与 pred 都要求两条参考长度接近。实测 3.4s + 5.0s 的"
                                "一对参考、17 字目标文本，两者都只念出 12 字（CER 0.389 / "
                                "0.333）——长度差让并集 prompt 区留下没有文本对应的空洞，"
                                "开头被挪用后又被截掉。裁到等长即恢复正常，或直接用 two_stage。"
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
                                "cond / output 模式下生效。pred 与 two_stage 混合的是速度场"
                                "（切空间向量，线性可加），恒用线性混合，此项被忽略。"
                            ),
                        )
                        mix_schedule = gr.Radio(
                            ["linear", "cosine", "sigmoid"],
                            value="linear",
                            label="t 维度曲线（仅 cond / pred / two_stage）",
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
                            -1.0, 2.0, value=0.5, step=0.05, label="t=1 A 权重（仅 cond / pred / two_stage）"
                        )
                        mix_2d_mode = gr.Radio(
                            ["t_only", "n_only", "multiply", "add", "max", "min"],
                            value="t_only",
                            label="2D 组合模式（仅 cond / pred / two_stage）",
                            info="output 模式无 t / n 两维可组合，此项被忽略。",
                        )
                        n_schedule = gr.Radio(
                            ["none", "linear", "cosine", "sigmoid"],
                            value="linear",
                            label="n 维度曲线",
                            info=(
                                "四种模式均生效。output 模式下选 none 时用「t=0 A 权重」作为标量权重；"
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
                        lang_mode,
                        trim_edges,
                        max_internal_silence,
                    ],
                    outputs=[out_audio],
                )

    return demo


def main():
    demo = build_interface()
    demo.launch()


if __name__ == "__main__":
    main()
