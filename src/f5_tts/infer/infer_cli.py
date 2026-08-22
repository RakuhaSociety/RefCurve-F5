import argparse
import codecs
import os
import random
import re
import sys
from datetime import datetime
from importlib.resources import files
from pathlib import Path

import numpy as np
import torch
import soundfile as sf
import tomli
from cached_path import cached_path
from hydra.utils import get_class
from omegaconf import OmegaConf
from unidecode import unidecode

# Prefer local source over installed package when running directly
CURRENT_DIR = Path(__file__).resolve()
REPO_SRC = CURRENT_DIR.parents[2]
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

from f5_tts.infer.utils_infer import (
    cfg_strength,
    cross_fade_duration,
    device,
    fix_duration,
    infer_process,
    load_model,
    load_vocoder,
    mel_spec_type,
    nfe_step,
    preprocess_ref_audio_text,
    remove_silence_for_generated_wav,
    speed,
    sway_sampling_coef,
    target_rms,
)


parser = argparse.ArgumentParser(
    prog="python3 infer-cli.py",
    description="Commandline interface for E2/F5 TTS with Advanced Batch Processing.",
    epilog="Specify options above to override one or more settings from config.",
)
parser.add_argument(
    "-c",
    "--config",
    type=str,
    default=os.path.join(files("f5_tts").joinpath("infer/examples/basic"), "basic.toml"),
    help="The configuration file, default see infer/examples/basic/basic.toml",
)


# Note. Not to provide default value here in order to read default from config file

parser.add_argument(
    "-m",
    "--model",
    type=str,
    help="The model name: F5TTS_v1_Base | F5TTS_Base | E2TTS_Base | etc.",
)
parser.add_argument(
    "-mc",
    "--model_cfg",
    type=str,
    help="The path to F5-TTS model config file .yaml",
)
parser.add_argument(
    "-p",
    "--ckpt_file",
    type=str,
    help="The path to model checkpoint .pt, leave blank to use default",
)
parser.add_argument(
    "-v",
    "--vocab_file",
    type=str,
    help="The path to vocab file .txt, leave blank to use default",
)
parser.add_argument(
    "-r",
    "--ref_audio",
    type=str,
    help="The reference audio file.",
)
parser.add_argument(
    "-s",
    "--ref_text",
    type=str,
    help="The transcript/subtitle for the reference audio",
)
parser.add_argument(
    "-r2",
    "--ref_audio_2",
    type=str,
    help="The reference audio file.",
)
parser.add_argument(
    "-s2",
    "--ref_text_2",
    type=str,
    help="The transcript/subtitle for the reference audio",
)
parser.add_argument(
    "-t",
    "--gen_text",
    type=str,
    help="The text to make model synthesize a speech",
)
parser.add_argument(
    "-f",
    "--gen_file",
    type=str,
    help="The file with text to generate, will ignore --gen_text",
)
parser.add_argument(
    "-o",
    "--output_dir",
    type=str,
    help="The path to output folder",
)
parser.add_argument(
    "-w",
    "--output_file",
    type=str,
    help="The name of output file",
)
parser.add_argument(
    "--save_chunk",
    action="store_true",
    help="To save each audio chunks during inference",
)
parser.add_argument(
    "--no_legacy_text",
    action="store_false",
    help="Not to use lossy ASCII transliterations of unicode text in saved file names.",
)
parser.add_argument(
    "--remove_silence",
    action="store_true",
    help="To remove long silence found in ouput",
)
parser.add_argument(
    "--load_vocoder_from_local",
    action="store_true",
    help="To load vocoder from local dir, auto-detected under ckpts/ (see --vocoder_local_path)",
)
parser.add_argument(
    "--vocoder_local_path",
    type=str,
    default=None,
    help="Explicit local vocoder dir; overrides auto-detection under ckpts/ or ../checkpoints/",
)
parser.add_argument(
    "--vocoder_name",
    type=str,
    choices=["vocos", "bigvgan"],
    help=f"Used vocoder name: vocos | bigvgan, default {mel_spec_type}",
)
parser.add_argument(
    "--target_rms",
    type=float,
    help=f"Target output speech loudness normalization value, default {target_rms}",
)
parser.add_argument(
    "--cross_fade_duration",
    type=float,
    help=f"Duration of cross-fade between audio segments in seconds, default {cross_fade_duration}",
)
parser.add_argument(
    "--nfe_step",
    type=int,
    help=f"The number of function evaluation (denoising steps), default {nfe_step}",
)
parser.add_argument(
    "--cfg_strength",
    type=float,
    help=f"Classifier-free guidance strength, default {cfg_strength}",
)
parser.add_argument(
    "--sway_sampling_coef",
    type=float,
    help=f"Sway Sampling coefficient, default {sway_sampling_coef}",
)
parser.add_argument(
    "--speed",
    type=float,
    help=f"The speed of the generated audio, default {speed}",
)
parser.add_argument(
    "--fix_duration",
    type=float,
    help=f"Fix the total duration (ref and gen audios) in seconds, default {fix_duration}",
)
parser.add_argument(
    "--device",
    type=str,
    default="cuda" if torch.cuda.is_available() else "cpu",
    help="Specify the device to run on",
)
parser.add_argument(
    "--seed",
    type=int,
    default=1234,
    help="Random seed for reproducible inference",
)
parser.add_argument(
    "--allow_extrapolation",
    action=argparse.BooleanOptionalAction,
    default=None,
    help="Allow mix weights to go beyond [0,1] for cond blending (use --no-allow_extrapolation to disable)",
)

# ========== 阶段一 & 阶段二：混合控制参数 ==========
parser.add_argument(
    "--mix_method",
    type=str,
    choices=["lerp", "slerp", "log"],
    default="slerp",
    help="Mixing method: lerp (linear), slerp (spherical), log (geometric mean)",
)
parser.add_argument(
    "--log_blend_mode",
    type=str,
    choices=["signed_magnitude", "logmel"],
    default="signed_magnitude",
    help="Log blend sub-mode: signed_magnitude (legacy, overflow-prone) or logmel (mathematically correct for log-mel input)",
)
parser.add_argument(
    "--n_normalize_to_ref",
    action="store_true",
    help="Normalize n-dimension curve to reference length instead of max_duration (experimental)",
)
parser.add_argument(
    "--mix_schedule",
    type=str,
    choices=["linear", "cosine", "sigmoid"],
    default="linear",
    help="Schedule for t-dimension mixing weight curve",
)
parser.add_argument(
    "--mix_a_start",
    type=float,
    default=0.5,
    help="Weight of ref_a at t=0 (denoising start)",
)
parser.add_argument(
    "--mix_a_end",
    type=float,
    default=0.5,
    help="Weight of ref_a at t=1 (denoising end)",
)
parser.add_argument(
    "--mix_2d_mode",
    type=str,
    choices=["t_only", "n_only", "multiply", "add", "max", "min"],
    default="t_only",
    help="How to combine t-dimension and n-dimension weights",
)
parser.add_argument(
    "--n_schedule",
    type=str,
    choices=["linear", "cosine", "sigmoid"],
    default="linear",
    help="Schedule for n-dimension (mel frame position) mixing weight curve",
)
parser.add_argument(
    "--n_a_start",
    type=float,
    default=0.5,
    help="Weight of ref_a at n=0 (audio start), None to disable n-dimension",
)
parser.add_argument(
    "--n_a_end",
    type=float,
    default=0.5,
    help="Weight of ref_a at n=1 (audio end)",
)

args = parser.parse_args()


# config file

config = tomli.load(open(args.config, "rb"))


def _get_ckpt_cache_dir() -> Path:
    """Prefer caching under repo ckpts to avoid default C: user cache."""
    env_dir = os.getenv("F5TTS_CKPT_CACHE")
    if env_dir:
        return Path(env_dir)

    script_path = Path(__file__).resolve()
    for parent in script_path.parents:
        candidate = parent / "ckpts"
        if candidate.exists():
            return candidate

    return script_path.parent / "ckpts"


def _default_vocoder_path(vocoder_name: str) -> str:
    """Locate a local vocoder dir, preferring the repo's ckpts/ over the upstream ../checkpoints/ layout."""
    subdirs = {
        "vocos": ("vocos-mel-24khz", "charactr/vocos-mel-24khz"),
        "bigvgan": ("bigvgan_v2_24khz_100band_256x", "nvidia/bigvgan_v2_24khz_100band_256x"),
    }.get(vocoder_name, ())

    roots = [_get_ckpt_cache_dir()]
    script_path = Path(__file__).resolve()
    roots += [parent / "checkpoints" for parent in script_path.parents[:5]]
    roots.append(Path("../checkpoints"))  # 上游默认布局，向后兼容

    for root in roots:
        for sub in subdirs:
            candidate = root / sub
            if (candidate / "config.yaml").exists() or (candidate / "bigvgan_generator.pt").exists():
                return str(candidate)

    # 都没找到时返回 ckpts/ 下的首选路径，让报错信息指向正确的位置
    return str(_get_ckpt_cache_dir() / subdirs[0]) if subdirs else ""


def _resolve_example_path(path_str: str) -> str:
    """Resolve example assets preferring repo files over installed package."""
    p = Path(path_str)
    if p.is_absolute():
        return str(p)

    script_path = Path(__file__).resolve()
    for parent in script_path.parents:
        candidate = parent / path_str
        if candidate.exists():
            return str(candidate)

    try:
        pkg_path = files("f5_tts").joinpath(path_str)
        if pkg_path.is_file():
            return str(pkg_path)
    except Exception:
        pass

    return path_str


# command-line interface parameters

model = args.model or config.get("model", "F5TTS_v1_Base")
ckpt_file = args.ckpt_file or config.get("ckpt_file", "")
vocab_file = args.vocab_file or config.get("vocab_file", "")

ref_audio = args.ref_audio or config.get("ref_audio", "infer/examples/basic/basic_ref_en.wav")
ref_text = (
    args.ref_text
    if args.ref_text is not None
    else config.get("ref_text", "Some call me nature, others call me mother nature.")
)
ref_audio_2 = args.ref_audio_2 or config.get("ref_audio_2", None)
ref_text_2 = (
    args.ref_text_2
    if args.ref_text_2 is not None
    else config.get("ref_text_2", "Some call me nature, others call me mother nature.")
)
gen_text = args.gen_text or config.get("gen_text", "让我们一起说中文。")
gen_file = args.gen_file or config.get("gen_file", "")

output_dir = args.output_dir or config.get("output_dir", "tests")
output_file = args.output_file or config.get(
    "output_file", f"infer_cli_{datetime.now().strftime(r'%Y%m%d_%H%M%S')}.wav"
)

save_chunk = args.save_chunk or config.get("save_chunk", False)
use_legacy_text = args.no_legacy_text or config.get("no_legacy_text", False)  # no_legacy_text is a store_false arg
if save_chunk and use_legacy_text:
    print(
        "\nWarning to --save_chunk: lossy ASCII transliterations of unicode text for legacy (.wav) file names, --no_legacy_text to disable.\n"
    )

remove_silence = args.remove_silence or config.get("remove_silence", False)
load_vocoder_from_local = args.load_vocoder_from_local or config.get("load_vocoder_from_local", False)

vocoder_name = args.vocoder_name or config.get("vocoder_name", mel_spec_type)
target_rms = args.target_rms or config.get("target_rms", target_rms)
cross_fade_duration = args.cross_fade_duration or config.get("cross_fade_duration", cross_fade_duration)
nfe_step = args.nfe_step or config.get("nfe_step", nfe_step)
cfg_strength = args.cfg_strength or config.get("cfg_strength", cfg_strength)
sway_sampling_coef = args.sway_sampling_coef or config.get("sway_sampling_coef", sway_sampling_coef)
speed = args.speed or config.get("speed", speed)
fix_duration = args.fix_duration or config.get("fix_duration", fix_duration)
device = args.device or config.get("device", device)
seed = args.seed if args.seed is not None else config.get("seed", None)
allow_extrapolation = (
    args.allow_extrapolation
    if args.allow_extrapolation is not None
    else config.get("allow_extrapolation", False)
)

if seed is not None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# patches for pip pkg user
# patches for pip pkg user
if "infer/examples/" in ref_audio:
    ref_audio = _resolve_example_path(ref_audio)

# ✅新增：ref_audio_2 也要补丁（可为 None，表示单参考）
if ref_audio_2 and "infer/examples/" in ref_audio_2:
    ref_audio_2 = _resolve_example_path(ref_audio_2)

if "infer/examples/" in gen_file:
    gen_file = _resolve_example_path(gen_file)

# ✅voices 里同时补 ref_audio / ref_audio_2
if "voices" in config:
    for voice in config["voices"]:
        for k in ("ref_audio", "ref_audio_2"):
            if k in config["voices"][voice]:
                p = config["voices"][voice][k]
                if isinstance(p, str) and "infer/examples/" in p:
                    config["voices"][voice][k] = _resolve_example_path(p)

# ignore gen_text if gen_file provided

if gen_file:
    gen_text = codecs.open(gen_file, "r", "utf-8").read()


# output path

wave_path = Path(output_dir) / output_file
# spectrogram_path = Path(output_dir) / "infer_cli_out.png"
if save_chunk:
    output_chunk_dir = os.path.join(output_dir, f"{Path(output_file).stem}_chunks")
    if not os.path.exists(output_chunk_dir):
        os.makedirs(output_chunk_dir)


# load vocoder

# 优先级：命令行 --vocoder_local_path > toml 的 vocoder_local_path > 自动探测 ckpts/ 与 ../checkpoints/
vocoder_local_path = (
    args.vocoder_local_path or config.get("vocoder_local_path") or _default_vocoder_path(vocoder_name)
)

vocoder = load_vocoder(
    vocoder_name=vocoder_name, is_local=load_vocoder_from_local, local_path=vocoder_local_path, device=device
)


# load TTS model

model_cfg = OmegaConf.load(
    args.model_cfg or config.get("model_cfg", str(files("f5_tts").joinpath(f"configs/{model}.yaml")))
)
model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
model_arc = model_cfg.model.arch

repo_name, ckpt_step, ckpt_type = "F5-TTS", 1250000, "safetensors"

if model != "F5TTS_Base":
    assert vocoder_name == model_cfg.model.mel_spec.mel_spec_type

# override for previous models
if model == "F5TTS_Base":
    if vocoder_name == "vocos":
        ckpt_step = 1200000
    elif vocoder_name == "bigvgan":
        model = "F5TTS_Base_bigvgan"
        ckpt_type = "pt"
elif model == "E2TTS_Base":
    repo_name = "E2-TTS"
    ckpt_step = 1200000

if not ckpt_file:
    # 改造版：下载到仓库本地 ckpts/ 而非全局 HF 缓存
    ckpt_cache_dir = _get_ckpt_cache_dir()
    ckpt_cache_dir.mkdir(parents=True, exist_ok=True)
    ckpt_file = str(
        cached_path(
            f"hf://SWivid/{repo_name}/{model}/model_{ckpt_step}.{ckpt_type}",
            cache_dir=ckpt_cache_dir,
        )
    )
elif ckpt_file.startswith("hf://"):
    ckpt_file = str(cached_path(ckpt_file, cache_dir=_get_ckpt_cache_dir()))

if vocab_file.startswith("hf://"):
    vocab_file = str(cached_path(vocab_file, cache_dir=_get_ckpt_cache_dir()))

print(f"Using {model}...")
ema_model = load_model(
    model_cls, model_arc, ckpt_file, mel_spec_type=vocoder_name, vocab_file=vocab_file, device=device
)


# inference process


def main():
    main_voice = {"ref_audio": ref_audio, "ref_text": ref_text, "ref_audio_2": ref_audio_2, "ref_text_2": ref_text_2}
    if "voices" not in config:
        voices = {"main": main_voice}
    else:
        voices = config["voices"]
        voices["main"] = main_voice
    for voice in voices:
        print("Voice:", voice)
        print("ref_audio ", voices[voice]["ref_audio"])
        voices[voice]["ref_audio"], voices[voice]["ref_text"] = preprocess_ref_audio_text(
            voices[voice]["ref_audio"], voices[voice]["ref_text"]
        )
        # 第二参考为可选：多音色 toml（如 examples/multi/story.toml）的各 voice
        # 通常没有 ref_audio_2，此时该 voice 退化为单参考
        if voices[voice].get("ref_audio_2"):
            voices[voice]["ref_audio_2"], voices[voice]["ref_text_2"] = preprocess_ref_audio_text(
                voices[voice]["ref_audio_2"], voices[voice].get("ref_text_2", "")
            )
        else:
            voices[voice]["ref_audio_2"], voices[voice]["ref_text_2"] = None, ""
        print("ref_audio_", voices[voice]["ref_audio"], "\n\n")

    generated_audio_segments = []
    reg1 = r"(?=\[\w+\])"
    chunks = re.split(reg1, gen_text)
    reg2 = r"\[(\w+)\]"
    for text in chunks:
        if not text.strip():
            continue
        match = re.match(reg2, text)
        if match:
            voice = match[1]
        else:
            print("No voice tag found, using main.")
            voice = "main"
        if voice not in voices:
            print(f"Voice {voice} not found, using main.")
            voice = "main"
        text = re.sub(reg2, "", text)
        ref_audio_ = voices[voice]["ref_audio"]
        ref_text_ = voices[voice]["ref_text"]
        ref_audio_2_ = voices[voice]["ref_audio_2"]
        ref_text_2_ = voices[voice]["ref_text_2"]
        local_speed = voices[voice].get("speed", speed)
        gen_text_ = text.strip()
        print(f"Voice: {voice}")
        audio_segment, final_sample_rate, spectrogram = infer_process(
            ref_audio_,
            ref_text_,
            gen_text_,
            ema_model,
            vocoder,
            ref_audio_2=ref_audio_2_,
            ref_text_2=ref_text_2_,
            mel_spec_type=vocoder_name,
            target_rms=target_rms,
            cross_fade_duration=cross_fade_duration,
            nfe_step=nfe_step,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
            speed=local_speed,
            fix_duration=fix_duration,
            device=device,
            allow_extrapolation=allow_extrapolation,
            # ✅阶段一 & 阶段二：混合控制参数
            mix_method=args.mix_method,
            mix_schedule=args.mix_schedule,
            log_blend_mode=args.log_blend_mode,
            n_normalize_to_ref=args.n_normalize_to_ref,
            mix_a_start=args.mix_a_start,
            mix_a_end=args.mix_a_end,
            mix_2d_mode=args.mix_2d_mode,
            n_schedule=args.n_schedule,
            n_a_start=args.n_a_start,
            n_a_end=args.n_a_end,
        )
        generated_audio_segments.append(audio_segment)

        if save_chunk:
            if len(gen_text_) > 200:
                gen_text_ = gen_text_[:200] + " ... "
            if use_legacy_text:
                gen_text_ = unidecode(gen_text_)
            sf.write(
                os.path.join(output_chunk_dir, f"{len(generated_audio_segments) - 1}_{gen_text_}.wav"),
                audio_segment,
                final_sample_rate,
            )

    if generated_audio_segments:
        final_wave = np.concatenate(generated_audio_segments)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(wave_path, "wb") as f:
            sf.write(f.name, final_wave, final_sample_rate)
            # Remove silence
            if remove_silence:
                remove_silence_for_generated_wav(f.name)
            print(f.name)


if __name__ == "__main__":
    main()
