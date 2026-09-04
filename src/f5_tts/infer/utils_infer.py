# A unified script for inference process
# Make adjustments inside functions, and consider both gradio and cli scripts if need to change func output format
import os
import sys
from concurrent.futures import ThreadPoolExecutor


os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # for MPS device compatibility
sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/../../third_party/BigVGAN/")

import hashlib
import re
import tempfile
from importlib.resources import files

import matplotlib


matplotlib.use("Agg")

import matplotlib.pylab as plt
import numpy as np
import torch
import torchaudio
import tqdm
from huggingface_hub import hf_hub_download
from pydub import AudioSegment, silence
from vocos import Vocos

from f5_tts.model import CFM
from f5_tts.model.cfm import log_domain_blend, slerp_with_norm
from f5_tts.model.utils import convert_char_to_pinyin, get_tokenizer


_ref_audio_cache = {}
_ref_text_cache = {}

device = (
    "cuda"
    if torch.cuda.is_available()
    else "xpu"
    if torch.xpu.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

tempfile_kwargs = {"delete_on_close": False} if sys.version_info >= (3, 12) else {"delete": False}

# -----------------------------------------

target_sample_rate = 24000
n_mel_channels = 100
hop_length = 256
win_length = 1024
n_fft = 1024
mel_spec_type = "vocos"
target_rms = 0.1
cross_fade_duration = 0.15
ode_method = "euler"
nfe_step = 32  # 16, 32
cfg_strength = 2.0
sway_sampling_coef = -1.0
speed = 1.0
fix_duration = None

# -----------------------------------------


# ============================================================
# mix_on="output" 专用：mel 域 DTW 对齐
#
# 背景：cond / pred 两种模式都把两条参考的特征按"帧号"直接对位混合，
# 但两条参考念的内容不同、语速不同，第 k 帧的音素并不对应。更关键的是
# CFM.sample() 只接受一条 text，所以另一条参考的音素序列从未进入模型，
# 混出的 mel 与文本条件自相矛盾。
#
# output 模式改为：两条参考各自独立完成一次单参考推理（各自的 mel 与
# 各自的文本完全自洽），再把两段"念同一句话"的结果做 DTW 对齐后混合。
# 对齐基准不是 A 也不是 B，而是按 alpha 插值出的中间时间轴，这样节奏
# 本身也参与混合，而不是把 B 的韵律抹平到 A 上。
# ============================================================


def _mel_dtw_path(mel_a, mel_b):
    """
    在 mel 域上计算 A、B 的 DTW 对齐路径。

    mel_a: [T_a, C]，mel_b: [T_b, C]，均为 np.ndarray（float32/64）
    返回 (idx_a, idx_b)：两个等长 np.ndarray，路径按时间递增排序。

    用余弦距离而非欧氏：mel 的整体能量差（响度/说话人差异）会主导欧氏距离，
    而我们要对齐的是频谱形状（音素身份），余弦对能量缩放不敏感。

    注：曾试过限制步长斜率为 (1,1)/(1,2)/(2,1) 来消除"冻结段"（默认步集允许一侧连续
    停住，实测 B 曾冻结 98 步 ≈ 1 秒）。冻结段确实消失了，但真实音频的字数反而变差 ——
    说明冻结只是症状：两段独立生成的 mel 逐帧余弦相似度本就只有 0.23（中心化后），
    10 分位为负，即"被判为同一音素"的帧对常常毫不相似。强行限制斜率只是把错误的
    对齐摊得更均匀。根因是 mel 帧级特征不足以跨说话人定位音素，故保留默认步集。
    """
    import librosa

    # librosa.sequence.dtw 期望 [C, T]，且对特征做转置后按列比较
    x = np.asarray(mel_a, dtype=np.float64).T  # [C, T_a]
    y = np.asarray(mel_b, dtype=np.float64).T  # [C, T_b]

    # 退化情形：任一侧过短，无法对齐，退回线性映射
    if x.shape[1] < 2 or y.shape[1] < 2:
        n = max(x.shape[1], y.shape[1])
        idx_a = np.linspace(0, x.shape[1] - 1, n).round().astype(np.int64)
        idx_b = np.linspace(0, y.shape[1] - 1, n).round().astype(np.int64)
        return idx_a, idx_b

    _, wp = librosa.sequence.dtw(X=x, Y=y, metric="cosine")
    # librosa 返回的 wp 是从终点回溯的，倒序即为时间递增
    wp = wp[::-1]
    return wp[:, 0].astype(np.int64), wp[:, 1].astype(np.int64)


def _dtw_common_timeline(mel_a, mel_b, alpha):
    """
    构造一条"中间时间轴"，并给出 A、B 各自到它的采样位置。

    alpha=1.0 → 时间轴完全等于 A 的节奏；alpha=0.0 → 完全等于 B 的节奏；
    中间值则按比例插值，于是节奏本身也随 alpha 连续变化。

    返回 (pos_a, pos_b, n_out)：
      pos_a / pos_b 是 float 数组，长度 n_out，表示输出第 k 帧应从
      A / B 的哪个（可能是小数的）帧位置采样。

    ✅ 关键不变式：pos_a[k] 与 pos_b[k] 必须是 DTW 路径上的同一点，否则输出第 k 帧
    会把 A 的某个音素位置和 B 的另一个音素位置叠加起来 —— 听感就是叠音/回声。
    旧实现分别对两侧的采样位置做 alpha 插值（pos_a = α·ia + (1-α)·ia_from_b，
    pos_b 同理），只有当 warp 是仿射时才满足该不变式；语音的 warp 是分段非线性的，
    f(中点) ≠ 中点(f)，于是中间 alpha 处配对崩坏。实测注入"完全正确"的对齐路径后，
    残差仍在 alpha=0.5 达到峰值（配对误差 2.4 帧 ≈ 26ms），两端却几乎为 0，
    与"极端权重干净、50/50 叠音"的听感完全吻合。

    现在改为沿路径参数化：alpha 只决定"沿路径推进的时钟快慢"（即最终节奏偏向谁），
    两侧位置恒取自同一路径点，配对由构造保证。
    """
    idx_a, idx_b = _mel_dtw_path(mel_a, mel_b)

    t_a = int(mel_a.shape[0])
    t_b = int(mel_b.shape[0])

    pa = idx_a.astype(np.float64)
    pb = idx_b.astype(np.float64)

    # 混合时钟：沿路径单调推进。alpha=1 按 A 的帧号计时，alpha=0 按 B 的。
    clock = np.maximum.accumulate(alpha * pa + (1.0 - alpha) * pb)

    # 路径含水平/垂直段（一对多），同一时钟值对应多个路径点。按时钟值分组取均值
    # 聚合，既让 np.interp 的横轴严格单调，也保留了旧实现在两端的均值语义
    # （alpha=0 时 B 的每帧对上的是所有匹配 A 帧的平均，而非第一个）。
    uniq, inv = np.unique(clock, return_inverse=True)
    if uniq.size < 2:
        # 退化：整条路径压成一个时钟点（两侧都极短），退回线性映射
        n_out = max(2, max(t_a, t_b))
        pos_a = np.linspace(0.0, t_a - 1, n_out)
        pos_b = np.linspace(0.0, t_b - 1, n_out)
        return pos_a, pos_b, n_out

    cnt = np.bincount(inv).astype(np.float64)
    pa_u = np.maximum.accumulate(np.bincount(inv, weights=pa) / cnt)
    pb_u = np.maximum.accumulate(np.bincount(inv, weights=pb) / cnt)

    # 输出长度：A、B 时长按 alpha 插值，节奏随之伸缩
    n_out = int(round(alpha * t_a + (1.0 - alpha) * t_b))
    n_out = max(2, min(n_out, max(t_a, t_b) * 2))

    # 沿时钟均匀取 n_out 个点，两侧位置来自同一路径点
    cs = np.linspace(uniq[0], uniq[-1], n_out, dtype=np.float64)
    pos_a = np.clip(np.interp(cs, uniq, pa_u), 0.0, t_a - 1)
    pos_b = np.clip(np.interp(cs, uniq, pb_u), 0.0, t_b - 1)
    return pos_a, pos_b, n_out


def _resample_mel(mel, pos):
    """
    按（小数）帧位置 pos 线性重采样 mel。

    mel: [T, C] np.ndarray；pos: [n_out] float
    返回 [n_out, C]。逐通道 np.interp 比构造稀疏矩阵更直接，
    n_out 量级在千帧、C=100，开销可忽略。
    """
    mel = np.asarray(mel, dtype=np.float64)
    grid = np.arange(mel.shape[0], dtype=np.float64)
    out = np.empty((pos.shape[0], mel.shape[1]), dtype=np.float64)
    for c in range(mel.shape[1]):
        out[:, c] = np.interp(pos, grid, mel[:, c])
    return out


def chunk_text(text, max_chars=135):
    """
    Splits the input text into chunks, each with a maximum number of characters.

    Args:
        text (str): The text to be split.
        max_chars (int): The maximum number of characters per chunk.

    Returns:
        List[str]: A list of text chunks.
    """
    chunks = []
    current_chunk = ""
    # Split the text into sentences based on punctuation followed by whitespace
    sentences = re.split(r"(?<=[;:,.!?])\s+|(?<=[；：，。！？])", text)

    for sentence in sentences:
        if not sentence:
            continue
        if len(current_chunk.encode("utf-8")) + len(sentence.encode("utf-8")) <= max_chars:
            current_chunk += sentence + " " if sentence and len(sentence[-1].encode("utf-8")) == 1 else sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " " if sentence and len(sentence[-1].encode("utf-8")) == 1 else sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


# load vocoder
def load_vocoder(vocoder_name="vocos", is_local=False, local_path="", device=device, hf_cache_dir=None):
    if vocoder_name == "vocos":
        # vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device)
        if is_local:
            print(f"Load vocos from local path {local_path}")
            config_path = f"{local_path}/config.yaml"
            model_path = f"{local_path}/pytorch_model.bin"
        else:
            print("Download Vocos from huggingface charactr/vocos-mel-24khz")
            repo_id = "charactr/vocos-mel-24khz"
            config_path = hf_hub_download(repo_id=repo_id, cache_dir=hf_cache_dir, filename="config.yaml")
            model_path = hf_hub_download(repo_id=repo_id, cache_dir=hf_cache_dir, filename="pytorch_model.bin")
        vocoder = Vocos.from_hparams(config_path)
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        from vocos.feature_extractors import EncodecFeatures

        if isinstance(vocoder.feature_extractor, EncodecFeatures):
            encodec_parameters = {
                "feature_extractor.encodec." + key: value
                for key, value in vocoder.feature_extractor.encodec.state_dict().items()
            }
            state_dict.update(encodec_parameters)
        vocoder.load_state_dict(state_dict)
        vocoder = vocoder.eval().to(device)
    elif vocoder_name == "bigvgan":
        try:
            from third_party.BigVGAN import bigvgan
        except ImportError:
            print("You need to follow the README to init submodule and change the BigVGAN source code.")
        if is_local:
            # download generator from https://huggingface.co/nvidia/bigvgan_v2_24khz_100band_256x/tree/main
            vocoder = bigvgan.BigVGAN.from_pretrained(local_path, use_cuda_kernel=False)
        else:
            vocoder = bigvgan.BigVGAN.from_pretrained(
                "nvidia/bigvgan_v2_24khz_100band_256x", use_cuda_kernel=False, cache_dir=hf_cache_dir
            )

        vocoder.remove_weight_norm()
        vocoder = vocoder.eval().to(device)
    return vocoder


# load asr pipeline

asr_pipe = None


def initialize_asr_pipeline(device: str = device, dtype=None):
    from transformers import pipeline

    if dtype is None:
        dtype = (
            torch.float16
            if "cuda" in device
            and torch.cuda.get_device_properties(device).major >= 7
            and not torch.cuda.get_device_name().endswith("[ZLUDA]")
            else torch.float32
        )
    global asr_pipe
    asr_pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v3-turbo",
        torch_dtype=dtype,
        device=device,
    )


# transcribe


def transcribe(ref_audio, language=None):
    global asr_pipe
    if asr_pipe is None:
        initialize_asr_pipeline(device=device)
    return asr_pipe(
        ref_audio,
        chunk_length_s=30,
        batch_size=128,
        generate_kwargs={"task": "transcribe", "language": language} if language else {"task": "transcribe"},
        return_timestamps=False,
    )["text"].strip()


# load model checkpoint for inference


def load_checkpoint(model, ckpt_path, device: str, dtype=None, use_ema=True):
    if dtype is None:
        dtype = (
            torch.float16
            if "cuda" in device
            and torch.cuda.get_device_properties(device).major >= 7
            and not torch.cuda.get_device_name().endswith("[ZLUDA]")
            else torch.float32
        )
    model = model.to(dtype)

    ckpt_type = ckpt_path.split(".")[-1]
    if ckpt_type == "safetensors":
        from safetensors.torch import load_file

        checkpoint = load_file(ckpt_path, device=device)
    else:
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)

    if use_ema:
        ema_state = checkpoint if ckpt_type == "safetensors" else checkpoint["ema_model_state_dict"]
        model_state = {
            k.removeprefix("ema_model."): v
            for k, v in ema_state.items()
            if k not in ["initted", "step", "update"]
        }

        # patch for backward compatibility, 305e3ea
        for key in ["mel_spec.mel_stft.mel_scale.fb", "mel_spec.mel_stft.spectrogram.window"]:
            model_state.pop(key, None)

        model.load_state_dict(model_state)
    else:
        model_state = checkpoint if ckpt_type == "safetensors" else checkpoint["model_state_dict"]
        model.load_state_dict(model_state)

    del checkpoint
    torch.cuda.empty_cache()

    return model.to(device)


# load model for inference


def load_model(
    model_cls,
    model_cfg,
    ckpt_path,
    mel_spec_type=mel_spec_type,
    vocab_file="",
    ode_method=ode_method,
    use_ema=True,
    device=device,
):
    if vocab_file == "":
        vocab_file = str(files("f5_tts").joinpath("infer/examples/vocab.txt"))
    tokenizer = "custom"

    print("\nvocab : ", vocab_file)
    print("token : ", tokenizer)
    print("model : ", ckpt_path, "\n")

    vocab_char_map, vocab_size = get_tokenizer(vocab_file, tokenizer)
    model = CFM(
        transformer=model_cls(**model_cfg, text_num_embeds=vocab_size, mel_dim=n_mel_channels),
        mel_spec_kwargs=dict(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mel_channels=n_mel_channels,
            target_sample_rate=target_sample_rate,
            mel_spec_type=mel_spec_type,
        ),
        odeint_kwargs=dict(
            method=ode_method,
        ),
        vocab_char_map=vocab_char_map,
    ).to(device)

    dtype = torch.float32 if mel_spec_type == "bigvgan" else None
    model = load_checkpoint(model, ckpt_path, device, dtype=dtype, use_ema=use_ema)

    return model


def remove_silence_edges(audio, silence_threshold=-42):
    # Remove silence from the start
    non_silent_start_idx = silence.detect_leading_silence(audio, silence_threshold=silence_threshold)
    audio = audio[non_silent_start_idx:]

    # Remove silence from the end
    reversed_audio = audio.reverse()
    non_silent_end_idx = silence.detect_leading_silence(reversed_audio, silence_threshold=silence_threshold)
    if non_silent_end_idx > 0:
        trimmed_audio = audio[: len(audio) - non_silent_end_idx]
    else:
        trimmed_audio = audio

    return trimmed_audio


# preprocess reference audio and text


def preprocess_ref_audio_text(ref_audio_orig, ref_text, show_info=print):
    show_info("Converting audio...")

    # Compute a hash of the reference audio file
    with open(ref_audio_orig, "rb") as audio_file:
        audio_data = audio_file.read()
        audio_hash = hashlib.md5(audio_data).hexdigest()

    global _ref_audio_cache

    if audio_hash in _ref_audio_cache:
        show_info("Using cached preprocessed reference audio...")
        ref_audio = _ref_audio_cache[audio_hash]

    else:  # first pass, do preprocess
        with tempfile.NamedTemporaryFile(suffix=".wav", **tempfile_kwargs) as f:
            temp_path = f.name

        aseg = AudioSegment.from_file(ref_audio_orig)

        # 1. try to find long silence for clipping
        non_silent_segs = silence.split_on_silence(
            aseg, min_silence_len=1000, silence_thresh=-50, keep_silence=1000, seek_step=10
        )
        non_silent_wave = AudioSegment.silent(duration=0)
        for non_silent_seg in non_silent_segs:
            if len(non_silent_wave) > 6000 and len(non_silent_wave + non_silent_seg) > 12000:
                show_info("Audio is over 12s, clipping short. (1)")
                break
            non_silent_wave += non_silent_seg

        # 2. try to find short silence for clipping if 1. failed
        if len(non_silent_wave) > 12000:
            non_silent_segs = silence.split_on_silence(
                aseg, min_silence_len=100, silence_thresh=-40, keep_silence=1000, seek_step=10
            )
            non_silent_wave = AudioSegment.silent(duration=0)
            for non_silent_seg in non_silent_segs:
                if len(non_silent_wave) > 6000 and len(non_silent_wave + non_silent_seg) > 12000:
                    show_info("Audio is over 12s, clipping short. (2)")
                    break
                non_silent_wave += non_silent_seg

        aseg = non_silent_wave

        # 3. if no proper silence found for clipping
        if len(aseg) > 12000:
            aseg = aseg[:12000]
            show_info("Audio is over 12s, clipping short. (3)")

        aseg = remove_silence_edges(aseg) + AudioSegment.silent(duration=50)
        aseg.export(temp_path, format="wav")
        ref_audio = temp_path

        # Cache the processed reference audio
        _ref_audio_cache[audio_hash] = ref_audio

    if not ref_text.strip():
        global _ref_text_cache
        if audio_hash in _ref_text_cache:
            # Use cached asr transcription
            show_info("Using cached reference text...")
            ref_text = _ref_text_cache[audio_hash]
        else:
            show_info("No reference text provided, transcribing reference audio...")
            ref_text = transcribe(ref_audio)
            # Cache the transcribed text (not caching custom ref_text, enabling users to do manual tweak)
            _ref_text_cache[audio_hash] = ref_text
    else:
        show_info("Using custom reference text...")

    # Ensure ref_text ends with a proper sentence-ending punctuation
    if not ref_text.endswith(". ") and not ref_text.endswith("。"):
        if ref_text.endswith("."):
            ref_text += " "
        else:
            ref_text += ". "

    print("\nref_text  ", ref_text)

    return ref_audio, ref_text


# infer process: chunk text -> infer batches [i.e. infer_batch_process()]


def infer_process(
    ref_audio,
    ref_text,
    gen_text,
    model_obj,
    vocoder,
    mel_spec_type=mel_spec_type,
    show_info=print,
    progress=tqdm,
    target_rms=target_rms,
    cross_fade_duration=cross_fade_duration,
    nfe_step=nfe_step,
    cfg_strength=cfg_strength,
    sway_sampling_coef=sway_sampling_coef,
    speed=speed,
    fix_duration=fix_duration,
    device=device,
    allow_extrapolation=False,
    seed=None,
    # ========== 改造版：第二/三参考（关键字可选，不传则退化为单参考） ==========
    # 放在签名末尾，避免挤占上游的位置参数顺序
    ref_audio_2=None,
    ref_text_2="",
    # ========== 阶段一 & 阶段二：混合控制参数 ==========
    mix_on="cond",       # "cond" 或 "pred"：混合发生在条件空间还是预测空间
    # "two_stage"：先各自单参考生成，再以两条产出为参考做 pred 混合。
    # 两条产出说的是同一句话，长度天然接近，因此不会出现"参考长度不等 → 并集
    # mask 留下无文本空洞 → 开头被挪用后截掉"的吞字问题。
    second_stage_prompt="repeat",  # two_stage 专用，目前只接受 "repeat"
    mix_method="lerp",
    mix_schedule="linear",
    log_blend_mode="logmel",
    n_normalize_to_ref=True,
    mix_a_start=0.9,
    mix_a_end=0.9,
    mix_2d_mode="t_only",
    mix_2d_weights=None,
    n_schedule=None,
    n_a_start=None,
    n_a_end=None,
    text_frontend=None,   # ✅ 自定义分词前端，替代 convert_char_to_pinyin；日文等语言用
):
    # Split the input text into batches
    audio_a, sr_a = torchaudio.load(ref_audio)

    single_ref = ref_audio_2 is None
    if single_ref:
        # 单参考模式：不要在这里把 A 克隆成 B。克隆会让下游 cond_b 非 None，
        # 使 CFM 的 single_ref 快捷路径失效，从而走完整混合路径（A 与 A 混合）：
        # cond 模式下 slerp/log 引入数值误差，pred 模式下变成两次独立 forward，
        # 结果都会偏离上游单参考基线。这里保持 None 原样传下去。
        audio_b, sr_b = None, None
        ref_text_2 = ref_text
    else:
        audio_b, sr_b = torchaudio.load(ref_audio_2)

    if mix_on == "output" and not single_ref:
        # output 模式两条支路各自独立推理，但必须共用同一套分块才能逐块对齐。
        # 各自算一次 max_chars 取较小者：用较宽松的那个会让短参考那一支超出
        # 它自己的 30 秒生成上限，导致两支路块数虽同、内容却被截断错位。
        secs_a = audio_a.shape[-1] / sr_a
        secs_b = audio_b.shape[-1] / sr_b
        chars_a = max(1, len(ref_text.encode("utf-8")))
        chars_b = max(1, len(ref_text_2.encode("utf-8")))
        max_chars = min(
            int(chars_a / secs_a * (22 - secs_a) * speed),
            int(chars_b / secs_b * (22 - secs_b) * speed),
        )
        max_chars = max(1, max_chars)
    else:
        ref_secs = audio_a.shape[-1] / sr_a if single_ref else max(audio_a.shape[-1] / sr_a, audio_b.shape[-1] / sr_b)
        ref_chars = max(len(ref_text.encode("utf-8")), len(ref_text_2.encode("utf-8")))

        max_chars = int(ref_chars / ref_secs * (22 - ref_secs) * speed)

    gen_text_batches = chunk_text(gen_text, max_chars=max_chars)
    for i, gen_text_i in enumerate(gen_text_batches):
        print(f"gen_text {i}", gen_text_i)
    print("\n")

    show_info(f"Generating audio in {len(gen_text_batches)} batches...")

    if not gen_text_batches:
        show_info("No text batches to generate.")
        return None, target_sample_rate, None

    return next(
        infer_batch_process(
            (audio_a, sr_a),
            ref_text,
            gen_text_batches,
            model_obj,
            vocoder,
            mel_spec_type=mel_spec_type,
            progress=progress,
            target_rms=target_rms,
            cross_fade_duration=cross_fade_duration,
            nfe_step=nfe_step,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
            speed=speed,
            fix_duration=fix_duration,
            device=device,
            allow_extrapolation=allow_extrapolation,
            seed=seed,
            
            # ✅新增：第二参考传下去（单参考时保持 None，触发下游 single_ref 快捷路径）
            ref_audio_2=None if single_ref else (audio_b, sr_b),
            ref_text_2=ref_text_2,

            # ✅阶段一 & 阶段二：混合控制参数
            mix_on=mix_on,
            second_stage_prompt=second_stage_prompt,
            mix_method=mix_method,
            mix_schedule=mix_schedule,
            log_blend_mode=log_blend_mode,
            n_normalize_to_ref=n_normalize_to_ref,
            mix_a_start=mix_a_start,
            mix_a_end=mix_a_end,
            mix_2d_mode=mix_2d_mode,
            mix_2d_weights=mix_2d_weights,
            n_schedule=n_schedule,
            n_a_start=n_a_start,
            n_a_end=n_a_end,
            text_frontend=text_frontend,
        )
    )


# infer batches


def infer_batch_process(
    ref_audio,
    ref_text,
    gen_text_batches,
    model_obj,
    vocoder,
    mel_spec_type="vocos",
    progress=tqdm,
    target_rms=0.1,
    cross_fade_duration=0.15,
    nfe_step=32,
    cfg_strength=2.0,
    sway_sampling_coef=-1,
    speed=1,
    fix_duration=None,
    device=None,
    streaming=False,
    chunk_size=2048,
    allow_extrapolation=False,
    seed=None,
    # ===== 新增：第二参考 =====
    ref_audio_2=None,   # (audio2, sr2)；不传则退化为单参考
    ref_text_2="",
    # ===== 阶段一 & 阶段二：混合控制参数 =====
    mix_on="cond",
    second_stage_prompt="repeat",  # two_stage 专用，目前只接受 "repeat"
    mix_method="lerp",
    mix_schedule="linear",
    log_blend_mode="logmel",
    n_normalize_to_ref=True,
    mix_a_start=0.9,
    mix_a_end=0.9,
    mix_2d_mode="t_only",
    mix_2d_weights=None,
    n_schedule=None,
    n_a_start=None,
    n_a_end=None,
    # ✅ 自定义文本前端：替代 convert_char_to_pinyin（日文等语言用）
    # 签名与 convert_char_to_pinyin 一致：接受 list[str]，返回 list[list[str]]
    text_frontend=None,
):
    # ----------------------------
    # 0) unpack & prepare 2 or 3 audios
    # ----------------------------
    audio_a, sr_a = ref_audio

    # 单参考时仍克隆一份 B 供 _prep_audio / duration 估计复用，但传给 model.sample
    # 时必须还原为 None，否则 CFM 的 single_ref 快捷路径失效（详见下方 cond_b=）。
    single_ref = ref_audio_2 is None
    if single_ref:
        audio_b, sr_b = audio_a.clone(), sr_a
        # 单参考时没有可混合的第二路，output / two_stage 均无意义：降级为 cond 走上游基线路径。
        # 不降级的话 mix_on="output" 会原样传到 CFM.sample() 并触发那里的校验报错。
        if mix_on in ("output", "two_stage"):
            mix_on = "cond"
    else:
        audio_b, sr_b = ref_audio_2

    def _prep_audio(audio, sr):
        if audio is None:
            return None, None
        # mono
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)

        # rms normalize (keep original rms for restoring output loudness)
        rms = torch.sqrt(torch.mean(torch.square(audio)))
        if rms < target_rms:
            audio = audio * target_rms / rms

        # resample
        if sr != target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
            audio = resampler(audio)

        audio = audio.to(device)
        return audio, rms

    audio_a, rms_a = _prep_audio(audio_a, sr_a)
    audio_b, rms_b = _prep_audio(audio_b, sr_b)

    # ----------------------------
    # 1) merge ref texts -> prompt_text
    # ----------------------------
    def _ensure_end_punc(t: str) -> str:
        t = (t or "").strip()
        if not t:
            return ""
        if not t.endswith(". ") and not t.endswith("。"):
            if t.endswith("."):
                t += " "
            else:
                t += ". "
        return t

    ref_text = _ensure_end_punc(ref_text)
    ref_text_2 = _ensure_end_punc(ref_text_2)

    def _finalize_prompt(t: str) -> str:
        t = t.strip()
        # 和原版一致：如果最后一个字符是 ascii（len==1），补个空格
        if t and len(t[-1].encode("utf-8")) == 1:
            t = t + " "
        return t

    # 根据 mix_a_start 权重决定用哪个文本作为 prompt
    # 这样交换 A/B 并交换权重时，选中的文本也会相应切换，保持对称
    if mix_a_start >= 0.5:
        prompt_text = ref_text.strip()  # A 权重更大，用 A 的文本
    else:
        prompt_text = ref_text_2.strip()  # B 权重更大，用 B 的文本

    prompt_text = _finalize_prompt(prompt_text)

    # output 模式下两条参考各跑一次独立推理，各自使用自己的文本，
    # 不再二选一——这正是该模式相对 cond/pred 的核心优势。
    prompt_text_a = _finalize_prompt(ref_text)
    prompt_text_b = _finalize_prompt(ref_text_2)

    # ✅ pred 模式也需要各自的文本：它每步分别用 cond_a/cond_b 各跑一次 forward，
    # 两次若喂同一条文本，B 支路就是"B 的音频 + A 的转写"——文本描述的不是它
    # 自己的参考。之前按 mix_a_start >= 0.5 二选一，导致 alpha 在 0.49/0.51 跨过
    # 阈值时整个 prompt 硬跳变、B 支路被喂错误转写。
    # cond 模式仍二选一（只有一个文本序列），所以 prompt_text 仍留着。

    generated_waves = []
    spectrograms = []

    # ----------------------------
    # 2) per-batch inference
    # ----------------------------
    # 上游修复：把 fix_duration 按各分块文本长度分摊，避免每块都用整句时长导致 N 倍膨胀
    # 改造版：参考时长取两段（或三段）参考的较大值（与下面 union prompt 的裁剪保持一致）
    if fix_duration is not None and len(gen_text_batches) > 1:
        ref_audio_len_frames = max(audio_a.shape[-1], audio_b.shape[-1]) // hop_length
        ref_sec = ref_audio_len_frames * hop_length / target_sample_rate
        target_total = fix_duration - ref_sec
        weights = [len(c.encode("utf-8")) for c in gen_text_batches]
        total_w = sum(weights)
        allocatable = target_total + cross_fade_duration * (len(gen_text_batches) - 1)
        fix_durations = [ref_sec + allocatable * w / total_w for w in weights]
    else:
        fix_durations = [fix_duration] * len(gen_text_batches)

    def _local_speed_of(gen_text):
        return 0.3 if len(gen_text.encode("utf-8")) < 10 else speed

    def _natural_gen_frames(audio, ref_txt, gen_text):
        """第一阶段单参考推理会自然生成多少帧（纯算术，不做推理）。

        用于让 A / B 两路共用同一个生成长度：语速不同会让两路自然长度差到 36%，
        若事后 pad 补齐，log-mel 的 0 并非静音（真实 mel 均值 -1.9、静音约 -9，
        0 反而在均值之上），补出来的是一段宽带噪声。
        """
        ref_audio_len = audio.shape[-1] // hop_length
        ref_text_len = max(1, len(ref_txt.encode("utf-8")))
        gen_text_len = len(gen_text.encode("utf-8"))
        return int(ref_audio_len / ref_text_len * gen_text_len / _local_speed_of(gen_text))

    def _mel_onset(mel, rel_thresh=0.25):
        """mel 里语音真正开始的帧索引。

        log-mel 的静音底在 -9 附近、语音均值约 -1.9，两者差距足够大，用帧均值
        相对自身动态范围取阈值即可，不需要绝对阈值（不同 speaker 的底噪不同）。
        """
        frame_energy = mel[0].float().mean(dim=-1)  # [T]
        lo = torch.quantile(frame_energy, 0.1)
        hi = torch.quantile(frame_energy, 0.9)
        if not torch.isfinite(lo) or not torch.isfinite(hi) or hi <= lo:
            return 0
        above = (frame_energy > lo + rel_thresh * (hi - lo)).nonzero()
        return int(above[0].item()) if above.numel() else 0

    def _infer_one_ref_mel(audio, ref_txt, prompt_txt, gen_text, fix_dur, gen_frames=None):
        """两阶段模式的第一阶段：单参考推理，返回 mel tensor（不转 numpy）。

        与 _infer_one_ref 的区别只是返回类型：这里保留 torch tensor [1, T, C]，
        因为第二阶段要把它当作 cond 直接喂回 model.sample()，走 CFM 里
        to_mel 对 3 维输入的直通分支，从而避开 vocode → 重算 mel 的量化往返。
        """
        local_speed = _local_speed_of(gen_text)
        text_list = [prompt_txt + gen_text] if prompt_txt else [gen_text]
        _tok = text_frontend if text_frontend is not None else convert_char_to_pinyin
        final_text_list = _tok(text_list)

        ref_audio_len = audio.shape[-1] // hop_length
        if fix_dur is not None:
            duration = int(fix_dur * target_sample_rate / hop_length)
        elif gen_frames is not None:
            # 两阶段模式：A / B 共用同一生成长度，产出天生等长，无需事后 pad。
            duration = ref_audio_len + gen_frames
        else:
            ref_text_len = max(1, len(ref_txt.encode("utf-8")))
            gen_text_len = len(gen_text.encode("utf-8"))
            duration = ref_audio_len + int(ref_audio_len / ref_text_len * gen_text_len / local_speed)

        with torch.inference_mode():
            generated, _ = model_obj.sample(
                cond=audio,
                cond_b=None,
                text=final_text_list,
                duration=duration,
                steps=nfe_step,
                cfg_strength=cfg_strength,
                sway_sampling_coef=sway_sampling_coef,
                seed=seed,
                allow_extrapolation=allow_extrapolation,
            )
        # 切掉自己的 prompt，只留生成段：[1, T_gen, C]
        return generated.to(torch.float32)[:, ref_audio_len:, :]

    def _infer_two_stage(gen_text, fix_dur):
        """两阶段推理：先各自单参考生成，再以两条产出为参考做 pred 混合。

        动机：直接把两条**内容不同、长度不等**的参考送进混合会吞字。实测同一句
        目标文本（17 字），mix_on="cond" 与 "pred" 都只念出 12 字、CER 0.389，
        而把两条参考裁到等长后 CER 回到 0.111 —— 说明问题出在两条参考的长度差，
        而不是混合算法本身。（cfm.py 里让 pred 的两路各用自己的 mask 而非并集
        mask 已经改掉，那是正确的，但单独改它不解决吞字。）

        两阶段绕开这一点：第一阶段各自以单参考跑正常推理，两条产出说的是同一句
        gen_text；第二阶段以它们为参考做 pred 混合，此时两条参考内容相同、长度
        经处理后一致，ref_text 也就是 gen_text 本身。实测 alpha ∈ {0.9,0.5,0.1}
        下 CER 均为 0.056、字数 17，与单参考持平。

        代价是推理三次而非一次。

        混合保真度（5 seeds，seed 内以两个单参考锚点归一化）：pos ≈ alpha，
        可达区间覆盖锚点跨距约 76%，delta 正常穿过零点。残余偏差 err ≤ +0.09
        （SEM ≈ 0.05）方向上偏向被 shared_gen_frames 拉长的那一路，但量级在噪声
        内，未建立。若日后要消除，正确做法是让两路各按自然语速生成、再用 DTW
        warp 到共同时间轴，而不是靠强制同长度。
        """
        # 两路共用同一生成长度，让产出天生等长。
        #
        # 不能先各自按自己语速生成、再 pad 补齐：log-mel 的 0 不是静音。实测真实
        # mel 取值域 [-9.64, 4.96]、均值 -1.93，静音在 -9 附近，而 0 落在均值之上
        # 0.54 sigma —— 补出来的是一段中等强度宽带噪声。两条参考语速差可达 36%
        # （本例 284 vs 443 帧），补零就是往尾部塞 150 帧噪声，且它会以 mix_a_start
        # 的权重进入混合：A 权重高时尾部噪声清晰可闻。
        #
        # 取两者较大值而非较小值：较小值会压缩慢speaker的生成空间，重新引入吞字；
        # 较大值只会让快speaker多出一点自然静音尾，而模型生成的静音就在 -9 附近。
        shared_gen_frames = max(
            _natural_gen_frames(audio_a, ref_text, gen_text),
            _natural_gen_frames(audio_b, ref_text_2, gen_text),
        )
        mel_a = _infer_one_ref_mel(audio_a, ref_text, prompt_text_a, gen_text, fix_dur, shared_gen_frames)
        mel_b = _infer_one_ref_mel(audio_b, ref_text_2, prompt_text_b, gen_text, fix_dur, shared_gen_frames)

        # 起音对齐：两路产出的前导静音长度不同（实测 A 约 1.05s、B 约 0.51s），
        # 而 score-space 混合是逐帧进行的 —— 帧 k 上 A 还在静音、B 已经在发音时，
        # 混出来的不是"两种音色的中间态"，而是"一个声音叠在另一个的静音上"。
        # 表现为 alpha=0.5 时相似度不居中（实测偏向 A +0.098 而非 ~0）。
        #
        # 裁掉各自的前导静音使发音位置对齐。裁的是静音不是内容，所以无损；
        # 保留少量共同前导，避免 prompt 从爆音开始。
        onset_a = _mel_onset(mel_a)
        onset_b = _mel_onset(mel_b)
        lead = min(onset_a, onset_b, 8)
        mel_a = mel_a[:, onset_a - lead :, :]
        mel_b = mel_b[:, onset_b - lead :, :]

        # 裁掉前导静音后两路长度不再相等，补齐到较长者。
        #
        # 不能截到较短者：那会从较长那路的**尾部**切掉真实语音（实测切掉 86 帧
        # 约 0.92s，正好是句尾"九点半"），于是 prompt 音频停在句中、prompt 文本
        # 却是完整句子 —— 模型先补出缺失的字再重新开始，表现为输出开头多字
        # （ASR 20~22 字 vs 目标 17 字，CER 0.056 -> 0.278/0.333）。
        #
        # 补齐用该路自己的末帧复制，而不是 0：末帧是它自然的收尾静音（log-mel
        # 约 -9），复制等于延长静音；而 0 落在 mel 均值之上 0.54 sigma，是一段
        # 中等强度宽带噪声，会按 mix_a_start 的权重混进输出。
        target_len = max(mel_a.shape[1], mel_b.shape[1])

        def _extend_with_own_tail(mel):
            missing = target_len - mel.shape[1]
            if missing <= 0:
                return mel
            return torch.cat([mel, mel[:, -1:, :].expand(-1, missing, -1)], dim=1)

        mel_a = _extend_with_own_tail(mel_a)
        mel_b = _extend_with_own_tail(mel_b)

        # 第二阶段两条参考各自说完整的 gen_text，所以 prompt 文本就是 gen_text，
        # 文本序列为 gen_text+gen_text：前半描述参考音频，后半是生成目标。
        #
        # 这里没有"不带 prompt"的选项。曾经实现过 second_stage_prompt="none"
        # （ref_len2=0、文本只放一份 gen_text），它结构上不成立：cond 的全部内容
        # 都在 prompt 区，lens 取 0 等于没有条件可用，取默认全长则 ODE 自由空间
        # = duration - lens = 0，而 CFM 采样结束后会执行
        # out = torch.where(cond_mask, final_cond, out)，把整个输出替换成
        # fused_cond —— 即 mel_a 与 mel_b 的插值。那退化成 mix_on="output" 的
        # mel 域混合，alpha≈0.5 时必然叠音（实测 CER 0.889、ASR 读出 31 字且
        # 内容重复），正是 two_stage 要避开的失效模式。
        if second_stage_prompt != "repeat":
            raise ValueError(
                f"second_stage_prompt must be 'repeat', got {second_stage_prompt!r}. "
                "The 'none' variant was removed: it leaves the ODE no generation space "
                "and degenerates into mel-domain interpolation."
            )
        stage2_prompt = _finalize_prompt(gen_text)
        text_list2 = [stage2_prompt + gen_text]
        text_list2_b = [stage2_prompt + gen_text]
        ref_len2 = target_len

        _tok = text_frontend if text_frontend is not None else convert_char_to_pinyin
        final_text_list2 = _tok(text_list2)
        final_text_list2_b = _tok(text_list2_b)

        if fix_dur is not None:
            duration2 = int(fix_dur * target_sample_rate / hop_length)
        else:
            # 生成段长度已由第一阶段确定，第二阶段只需复现同样长度
            duration2 = ref_len2 + target_len

        with torch.inference_mode():
            local_mix_2d = mix_2d_weights
            if local_mix_2d is not None and not torch.is_tensor(local_mix_2d):
                local_mix_2d = torch.tensor(local_mix_2d, device=mel_a.device, dtype=mel_a.dtype)

            generated, _ = model_obj.sample(
                cond=mel_a,
                cond_b=mel_b,
                text=final_text_list2,
                text_b=final_text_list2_b,
                duration=duration2,
                steps=nfe_step,
                cfg_strength=cfg_strength,
                sway_sampling_coef=sway_sampling_coef,
                seed=seed,
                allow_extrapolation=allow_extrapolation,
                mix_on="pred",
                mix_method=mix_method,
                mix_schedule=mix_schedule,
                log_blend_mode=log_blend_mode,
                n_normalize_to_ref=n_normalize_to_ref,
                mix_a_start=mix_a_start,
                mix_a_end=mix_a_end,
                mix_2d_mode=mix_2d_mode,
                mix_2d_weights=local_mix_2d,
                n_schedule=n_schedule,
                n_a_start=n_a_start,
                n_a_end=n_a_end,
            )

        generated = generated.to(torch.float32)[:, ref_len2:, :]
        generated = generated.permute(0, 2, 1)  # [1, C, T]

        if mel_spec_type == "vocos":
            generated_wave = vocoder.decode(generated)
        elif mel_spec_type == "bigvgan":
            generated_wave = vocoder(generated)

        alpha_time = min(1.0, max(0.0, float(mix_a_start)))
        weighted_rms = alpha_time * rms_a + (1 - alpha_time) * rms_b
        if weighted_rms < target_rms:
            generated_wave = generated_wave * weighted_rms / target_rms

        generated_wave = generated_wave.squeeze().cpu().numpy()
        return generated_wave, generated

    def _infer_one_ref(audio, ref_txt, prompt_txt, gen_text, fix_dur):
        """
        output 模式的单支路：只用一条参考做一次纯单参考推理。

        与 _infer_basic 的区别是 cond_b 恒为 None（走 CFM 的 single_ref 快捷路径），
        且 duration / prompt 裁剪都基于这一条参考自己的长度，因此 mel 与文本
        条件完全自洽，不存在两条参考音素叠加的问题。

        返回切掉 prompt 前缀后的 mel：[T, C]，np.float64。
        """
        local_speed = _local_speed_of(gen_text)
        text_list = [prompt_txt + gen_text] if prompt_txt else [gen_text]
        _tok = text_frontend if text_frontend is not None else convert_char_to_pinyin
        final_text_list = _tok(text_list)

        ref_audio_len = audio.shape[-1] // hop_length
        if fix_dur is not None:
            duration = int(fix_dur * target_sample_rate / hop_length)
        else:
            ref_text_len = max(1, len(ref_txt.encode("utf-8")))
            gen_text_len = len(gen_text.encode("utf-8"))
            duration = ref_audio_len + int(ref_audio_len / ref_text_len * gen_text_len / local_speed)

        with torch.inference_mode():
            generated, _ = model_obj.sample(
                cond=audio,
                cond_b=None,
                text=final_text_list,
                duration=duration,
                steps=nfe_step,
                cfg_strength=cfg_strength,
                sway_sampling_coef=sway_sampling_coef,
                seed=seed,
                allow_extrapolation=allow_extrapolation,
            )
        generated = generated.to(torch.float32)[:, ref_audio_len:, :]  # 切掉自己的 prompt
        return generated[0].cpu().numpy().astype(np.float64)

    def _infer_output_mode(gen_text, fix_dur):
        """
        output 模式：两条参考各自独立生成同一段目标文本，DTW 对齐后在 mel 域混合。

        对齐基准是按 alpha 插值出的中间时间轴（见 _dtw_common_timeline），
        所以 alpha 不仅控制音色权重，也控制最终节奏偏向 A 还是 B。
        """
        mel_a = _infer_one_ref(audio_a, ref_text, prompt_text_a, gen_text, fix_dur)
        mel_b = _infer_one_ref(audio_b, ref_text_2, prompt_text_b, gen_text, fix_dur)

        # output 模式没有 ODE 时间轴可用，t 维度曲线在此无意义；
        # 用 mix_a_start 作为标量权重（与 UI 的"A 权重"语义一致）。
        alpha_scalar = float(mix_a_start)
        # 时间轴插值需要 [0,1] 内的比例，即使允许外推也要夹紧，
        # 否则 n_out 会算出负数或超长
        alpha_time = min(1.0, max(0.0, alpha_scalar))

        pos_a, pos_b, n_out = _dtw_common_timeline(mel_a, mel_b, alpha_time)
        wa = _resample_mel(mel_a, pos_a)  # [n_out, C]
        wb = _resample_mel(mel_b, pos_b)

        ta = torch.from_numpy(wa).to(device=device, dtype=torch.float32)
        tb = torch.from_numpy(wb).to(device=device, dtype=torch.float32)

        # 帧级权重：仅当 n 维度曲线真正启用时才用它覆盖标量权重，
        # 使"开头偏 A、结尾偏 B"这类控制在 output 模式下同样可用。
        # 必须同时检查 n_schedule —— UI 的 n_a_start/n_a_end 是 Slider，恒有值，
        # 只判非 None 会在曲线关闭时把权重钉死成 n_a_start，丢掉 mix_a_start。
        if n_schedule is not None and n_a_start is not None and n_a_end is not None:
            ratio = torch.linspace(0, 1, n_out, device=device, dtype=torch.float32)
            if callable(n_schedule):
                r01 = n_schedule(ratio)
            elif n_schedule == "cosine":
                r01 = 0.5 - 0.5 * torch.cos(torch.pi * ratio)
            elif n_schedule == "sigmoid":
                r01 = torch.sigmoid(12.0 * (ratio - 0.5))
            else:
                r01 = ratio
            a = n_a_start * (1 - r01) + n_a_end * r01
            if not allow_extrapolation:
                a = a.clamp(0.0, 1.0)
            a = a.view(-1, 1)
        else:
            a = torch.full((1, 1), alpha_scalar, device=device, dtype=torch.float32)
            if not allow_extrapolation:
                a = a.clamp(0.0, 1.0)

        # 复用与 cond 模式相同的三种混合算法，语义保持一致
        if mix_method == "lerp":
            fused = a * ta + (1 - a) * tb
        elif mix_method == "slerp":
            fused = slerp_with_norm(ta.unsqueeze(0), tb.unsqueeze(0), a.unsqueeze(0)).squeeze(0)
        elif mix_method == "log":
            fused = log_domain_blend(ta.unsqueeze(0), tb.unsqueeze(0), a.unsqueeze(0), mode=log_blend_mode).squeeze(0)
        else:
            raise ValueError(f"unknown mix_method: {mix_method}, expected 'lerp', 'slerp', or 'log'")

        generated = fused.unsqueeze(0).permute(0, 2, 1)  # [1, C, T] for vocoder
        with torch.inference_mode():
            if mel_spec_type == "vocos":
                generated_wave = vocoder.decode(generated)
            elif mel_spec_type == "bigvgan":
                generated_wave = vocoder(generated)
            else:
                raise ValueError(f"Unknown mel_spec_type: {mel_spec_type}")

        weighted_rms = alpha_time * rms_a + (1 - alpha_time) * rms_b
        if weighted_rms < target_rms:
            generated_wave = generated_wave * weighted_rms / target_rms

        generated_wave = generated_wave.squeeze().cpu().numpy()
        # 必须返回 [1, C, T]（与 _infer_basic 一致）：调用方取 [0] 后按 axis=1
        # 拼接各块 mel，返回 [1, T, C] 会让多块拼接因频率维不匹配而报错。
        return generated_wave, generated

    def _infer_basic(gen_text, fix_dur):
        if mix_on == "output" and not single_ref:
            return _infer_output_mode(gen_text, fix_dur)
        if mix_on == "two_stage" and not single_ref:
            return _infer_two_stage(gen_text, fix_dur)

        local_speed = speed
        if len(gen_text.encode("utf-8")) < 10:
            local_speed = 0.3

        # Prepare text
        if prompt_text:
            text_list = [prompt_text + gen_text]
        else:
            text_list = [gen_text]
        _tok = text_frontend if text_frontend is not None else convert_char_to_pinyin
        final_text_list = _tok(text_list)

        # ✅ pred 模式需要 B 支路自己的文本列表（双参考时才有意义）
        if mix_on == "pred" and not single_ref:
            text_list_b = [prompt_text_b + gen_text] if prompt_text_b else [gen_text]
            final_text_list_b = _tok(text_list_b)
        else:
            final_text_list_b = None

        # duration 必须留出 "并集 prompt 长度 + 生成空间"。CFM 里 cond_mask 是
        # mask_a | mask_b（或 mask_a | mask_b | mask_c），即 prompt 区域恒为参考
        # 的较长者；而每字帧数只能按主参考（A，也就是 prompt_text 的来源）估。
        # 两者混用会出两种事故：
        #   · 生成空间不足：B 主导且 B 远短于 A 时，按 B 的帧/字比估出的总长
        #     还不到并集长度，被 CFM 的 duration 下限抬平后只剩 1 帧可生成，
        #     输出被截断（"吞字"）。
        #   · 曾经这里把 ref_audio_len 直接取 max 来回避上一条，但那让模型以为
        #     前 max 帧都对应 A 的文本，于是在输出开头复述另一条参考的尾部
        #     （听感即"开头多了一截别的内容"）。
        # 解决方案：ref_len_union 取并集（与 cond_mask 一致，与下面 prompt 裁剪一致），
        # 但 pace 要按 prompt_text 实际所属的那条参考（见上面按 mix_a_start
        # 选文本的分支）。写死成 A 会在 prompt_text 来自 B 时用 A 的语速去估
        # B 的文本，估短了就没有生成空间。
        len_a = audio_a.shape[-1] // hop_length
        len_b = audio_b.shape[-1] // hop_length
        ref_len_union = max(len_a, len_b)

        # 帧/字比必须取自 prompt_text 实际所属的那条参考（见上面按 mix_a_start
        # 选文本的分支）。写死成 A 会在 prompt_text 来自 B 时用 A 的语速去估
        # B 的文本，估短了就没有生成空间。
        if mix_a_start >= 0.5:
            pace_frames, pace_text = len_a, ref_text
        else:
            pace_frames, pace_text = len_b, ref_text_2

        # 两条参考语速可能差近一倍。只按被选中那条估，遇到快语速参考
        # 会把生成区压得装不下目标文本（听感是开头整段缺失）。取所有参考中较慢的
        # 作下限：宁可多留静音（后面 remove_silence 可清），也不要截掉内容。
        def _pace(frames, text):
            # 当参考文本是占位符 "." 时跳过（返回 0 表示"不参与 max"）。
            # 否则 1-byte 分母会把帧/字比炸到数百倍，生成 duration 超出
            # DiT max_seq_len=8192 而报 shape 不匹配。
            effective = len(text.strip().encode("utf-8"))
            if effective <= 1:
                return 0.0
            return frames / effective

        _pace_vals = [
            _pace(pace_frames, pace_text),
            _pace(len_a, ref_text),
            _pace(len_b, ref_text_2),
        ]
        _pace_valid = [p for p in _pace_vals if p > 0]
        # 所有参考文本均为占位符时，退回约 10 帧/字节（≈ 0.32 秒/日文字，正常语速下限）
        pace_ratio = max(_pace_valid) if _pace_valid else 10.0

        # 上游修复：用分摊后的 fix_dur，而非整句 fix_duration
        if fix_dur is not None:
            duration = int(fix_dur * target_sample_rate / hop_length)
        else:
            gen_text_len = len(gen_text.encode("utf-8"))
            gen_frames = int(pace_ratio * gen_text_len / local_speed)
            duration = ref_len_union + gen_frames

        # 下面切 prompt 用并集长度：CFM 的 cond_mask 是 mask_a | mask_b，实际被
        # prompt 占用的就是这一段。用 A 自己的长度会把较长那条参考的尾部留在
        # 输出开头（听感即"开头多了一截别的内容"）。
        ref_audio_len = ref_len_union

        with torch.inference_mode():
            local_mix_2d = mix_2d_weights
            # allow list/np input for 2d weights
            if local_mix_2d is not None and not torch.is_tensor(local_mix_2d):
                local_mix_2d = torch.tensor(local_mix_2d, device=audio_a.device, dtype=audio_a.dtype)

            generated, _ = model_obj.sample(
                cond=audio_a,
                # ✅第二参考音频；单参考时传 None 以触发 CFM 的 single_ref 快捷路径，
                # 保证结果与上游逐位一致（传 audio_a 的克隆会走完整混合路径）
                cond_b=None if single_ref else audio_b,
                text=final_text_list,
                text_b=final_text_list_b,  # ✅ pred 模式：B 支路自己的文本
                duration=duration,
                steps=nfe_step,
                cfg_strength=cfg_strength,
                sway_sampling_coef=sway_sampling_coef,
                seed=seed,
                allow_extrapolation=allow_extrapolation,
                # ✅阶段一 & 阶段二：混合控制参数
                mix_on=mix_on,
                mix_method=mix_method,
                mix_schedule=mix_schedule,
                log_blend_mode=log_blend_mode,
                n_normalize_to_ref=n_normalize_to_ref,
                mix_a_start=mix_a_start,
                mix_a_end=mix_a_end,
                mix_2d_mode=mix_2d_mode,
                mix_2d_weights=local_mix_2d,
                n_schedule=n_schedule,
                n_a_start=n_a_start,
                n_a_end=n_a_end,
            )
            del _

            generated = generated.to(torch.float32)   # mel: [1, T, C]
            generated = generated[:, ref_audio_len:, :]  # ✅切掉 union prompt
            generated = generated.permute(0, 2, 1)    # -> [1, C, T] for vocoder

            if mel_spec_type == "vocos":
                generated_wave = vocoder.decode(generated)
            elif mel_spec_type == "bigvgan":
                generated_wave = vocoder(generated)
            else:
                raise ValueError(f"Unknown mel_spec_type: {mel_spec_type}")

            # restore loudness based on weighted RMS of both refs
            # 使用 mix_a_start 作为权重，使响度恢复更对称
            weighted_rms = mix_a_start * rms_a + (1 - mix_a_start) * rms_b
            if weighted_rms < target_rms:
                generated_wave = generated_wave * weighted_rms / target_rms

            # wav -> numpy
            generated_wave = generated_wave.squeeze().cpu().numpy()

        return generated_wave, generated

    def infer_single_process(gen_text, fix_dur):
        generated_wave, generated = _infer_basic(gen_text, fix_dur)
        generated_cpu = generated[0].cpu().numpy()
        del generated
        return generated_wave, generated_cpu

    def infer_single_process_streaming(gen_text, fix_dur):
        # for src/f5_tts/socket_server.py
        generated_wave, generated = _infer_basic(gen_text, fix_dur)
        del generated
        for j in range(0, len(generated_wave), chunk_size):
            yield generated_wave[j : j + chunk_size], target_sample_rate

    # ----------------------------
    # 3) output aggregation (same as original)
    # ----------------------------
    if streaming:
        batches_iter = progress.tqdm(gen_text_batches) if progress is not None else gen_text_batches
        for gen_text, fix_dur in zip(batches_iter, fix_durations):
            for chunk in infer_single_process_streaming(gen_text, fix_dur):
                yield chunk
    else:
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(infer_single_process, gt, fd) for gt, fd in zip(gen_text_batches, fix_durations)]
            for future in progress.tqdm(futures) if progress is not None else futures:
                result = future.result()
                if result:
                    generated_wave, generated_mel_spec = result
                    generated_waves.append(generated_wave)
                    spectrograms.append(generated_mel_spec)

        if generated_waves:
            if cross_fade_duration <= 0:
                # Simply concatenate
                final_wave = np.concatenate(generated_waves)
            else:
                # Combine all generated waves with cross-fading
                final_wave = generated_waves[0]
                for i in range(1, len(generated_waves)):
                    prev_wave = final_wave
                    next_wave = generated_waves[i]

                    # Calculate cross-fade samples, ensuring it does not exceed wave lengths
                    cross_fade_samples = int(cross_fade_duration * target_sample_rate)
                    cross_fade_samples = min(cross_fade_samples, len(prev_wave), len(next_wave))

                    if cross_fade_samples <= 0:
                        # No overlap possible, concatenate
                        final_wave = np.concatenate([prev_wave, next_wave])
                        continue

                    # Overlapping parts
                    prev_overlap = prev_wave[-cross_fade_samples:]
                    next_overlap = next_wave[:cross_fade_samples]

                    # Fade out and fade in
                    fade_out = np.linspace(1, 0, cross_fade_samples)
                    fade_in = np.linspace(0, 1, cross_fade_samples)

                    # Cross-faded overlap
                    cross_faded_overlap = prev_overlap * fade_out + next_overlap * fade_in

                    # Combine
                    new_wave = np.concatenate(
                        [prev_wave[:-cross_fade_samples], cross_faded_overlap, next_wave[cross_fade_samples:]]
                    )
                    final_wave = new_wave

            # Create a combined spectrogram
            combined_spectrogram = np.concatenate(spectrograms, axis=1)

            yield final_wave, target_sample_rate, combined_spectrogram
        else:
            yield None, target_sample_rate, None


# remove silence from generated wav


def remove_silence_for_generated_wav(filename):
    aseg = AudioSegment.from_file(filename)
    non_silent_segs = silence.split_on_silence(
        aseg, min_silence_len=1000, silence_thresh=-50, keep_silence=500, seek_step=10
    )
    non_silent_wave = AudioSegment.silent(duration=0)
    for non_silent_seg in non_silent_segs:
        non_silent_wave += non_silent_seg
    aseg = non_silent_wave
    aseg.export(filename, format="wav")


# save spectrogram


def save_spectrogram(spectrogram, path):
    plt.figure(figsize=(12, 4))
    plt.imshow(spectrogram, origin="lower", aspect="auto")
    plt.colorbar()
    plt.savefig(path)
    plt.close()
