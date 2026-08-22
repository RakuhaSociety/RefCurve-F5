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


# chunk text into smaller pieces


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
        if ckpt_type == "safetensors":
            checkpoint = {"ema_model_state_dict": checkpoint}
        checkpoint["model_state_dict"] = {
            k.replace("ema_model.", ""): v
            for k, v in checkpoint["ema_model_state_dict"].items()
            if k not in ["initted", "step"]
        }

        # patch for backward compatibility, 305e3ea
        for key in ["mel_spec.mel_stft.mel_scale.fb", "mel_spec.mel_stft.spectrogram.window"]:
            if key in checkpoint["model_state_dict"]:
                del checkpoint["model_state_dict"][key]

        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        if ckpt_type == "safetensors":
            checkpoint = {"model_state_dict": checkpoint}
        model.load_state_dict(checkpoint["model_state_dict"])

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
    # ========== 改造版：第二参考（关键字可选，不传则退化为单参考） ==========
    # 放在签名末尾，避免挤占上游的位置参数顺序
    ref_audio_2=None,
    ref_text_2="",
    # ========== 阶段一 & 阶段二：混合控制参数 ==========
    mix_method="lerp",
    mix_schedule="linear",
    log_blend_mode="signed_magnitude",
    n_normalize_to_ref=False,
    mix_a_start=0.9,
    mix_a_end=0.9,
    mix_2d_mode="t_only",
    mix_2d_weights=None,
    n_schedule=None,
    n_a_start=None,
    n_a_end=None,
):
    # Split the input text into batches
    audio_a, sr_a = torchaudio.load(ref_audio)

    if ref_audio_2 is None:
        # 单参考模式：退化为原版行为，第二参考复用第一参考
        audio_b, sr_b = audio_a.clone(), sr_a
        ref_text_2 = ref_text
    else:
        audio_b, sr_b = torchaudio.load(ref_audio_2)

    ref_secs = max(audio_a.shape[-1] / sr_a, audio_b.shape[-1] / sr_b)
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
            
            # ✅新增：第二参考传下去
            ref_audio_2=(audio_b, sr_b),
            ref_text_2=ref_text_2,
            
            # ✅阶段一 & 阶段二：混合控制参数
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
    mix_method="lerp",
    mix_schedule="linear",
    log_blend_mode="signed_magnitude",
    n_normalize_to_ref=False,
    mix_a_start=0.9,
    mix_a_end=0.9,
    mix_2d_mode="t_only",
    mix_2d_weights=None,
    n_schedule=None,
    n_a_start=None,
    n_a_end=None,
):
    # ----------------------------
    # 0) unpack & prepare 2 audios
    # ----------------------------
    audio_a, sr_a = ref_audio

    if ref_audio_2 is None:
        audio_b, sr_b = audio_a.clone(), sr_a
    else:
        audio_b, sr_b = ref_audio_2

    def _prep_audio(audio, sr):
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

    # 根据 mix_a_start 权重决定用哪个文本作为 prompt
    # 这样交换 A/B 并交换权重时，选中的文本也会相应切换，保持对称
    if mix_a_start >= 0.5:
        prompt_text = ref_text.strip()  # A 权重更大，用 A 的文本
    else:
        prompt_text = ref_text_2.strip()  # B 权重更大，用 B 的文本

    # 和原版一致：如果最后一个字符是 ascii（len==1），补个空格
    if prompt_text and len(prompt_text[-1].encode("utf-8")) == 1:
        prompt_text = prompt_text + " "

    generated_waves = []
    spectrograms = []

    # ----------------------------
    # 2) per-batch inference
    # ----------------------------
    # 上游修复：把 fix_duration 按各分块文本长度分摊，避免每块都用整句时长导致 N 倍膨胀
    # 改造版：参考时长取两段参考的较大值（与下面 union prompt 的裁剪保持一致）
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

    def _infer_basic(gen_text, fix_dur):
        local_speed = speed
        if len(gen_text.encode("utf-8")) < 10:
            local_speed = 0.3

        # Prepare text
        if prompt_text:
            text_list = [prompt_text + gen_text]
        else:
            text_list = [gen_text]
        final_text_list = convert_char_to_pinyin(text_list)

        # union prompt length (frames) so we cut correctly
        ref_len_a = audio_a.shape[-1] // hop_length
        ref_len_b = audio_b.shape[-1] // hop_length
        ref_audio_len = max(ref_len_a, ref_len_b)

        # 上游修复：用分摊后的 fix_dur，而非整句 fix_duration
        if fix_dur is not None:
            duration = int(fix_dur * target_sample_rate / hop_length)
        else:
            # duration estimation uses the longer of the two reference texts to avoid underestimating
            ref_text_len = max(1, len(ref_text.encode("utf-8")), len(ref_text_2.encode("utf-8")))
            gen_text_len = len(gen_text.encode("utf-8"))
            duration = ref_audio_len + int(ref_audio_len / ref_text_len * gen_text_len / local_speed)

        with torch.inference_mode():
            local_mix_2d = mix_2d_weights
            # allow list/np input for 2d weights
            if local_mix_2d is not None and not torch.is_tensor(local_mix_2d):
                local_mix_2d = torch.tensor(local_mix_2d, device=audio_a.device, dtype=audio_a.dtype)

            generated, _ = model_obj.sample(
                cond=audio_a,
                cond_b=audio_b,  # ✅第二参考音频
                text=final_text_list,
                duration=duration,
                steps=nfe_step,
                cfg_strength=cfg_strength,
                sway_sampling_coef=sway_sampling_coef,
                seed=seed,
                allow_extrapolation=allow_extrapolation,
                # ✅阶段一 & 阶段二：混合控制参数
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
