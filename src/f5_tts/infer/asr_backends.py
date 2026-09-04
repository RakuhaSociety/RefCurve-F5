"""参考音频自动转写后端：中文 FunASR Paraformer + 日文 Visual-novel-whisper。

两个后端各自 lazy singleton 缓存，首次调用才加载模型。统一入口是
`transcribe(audio_path, lang, device)`，`lang` 取 "zh" / "ja"。

日文用的是 Visual-novel-whisper（kotoba-whisper-v1.1 在 Galgame 语料上的微调），
走 transformers 的 trust_remote_code 自定义 pipeline。它比通用 Whisper 更适合
配音/台词类音频。模型目录默认指向本机的 Visual-novel-whisper 项目，可用环境变量
`VNW_MODEL_DIR` 覆盖，也可放到仓库 `ckpts/Visual-novel-whisper/`。

⚠️ 远程 pipeline 代码硬性 import `punctuators` 与 `stable_whisper`，缺任一个都会
直接 ImportError，关掉 stable_ts / punctuator 开关也绕不过去。安装：

    pip install punctuators
    pip install stable-ts --no-deps     # 必须 --no-deps

stable-ts 依赖的 openai-whisper==20231117 其 setup.py 仍用已被新版 setuptools
移除的 pkg_resources，构建必然失败；我们只要文本，stable-ts 仅用于时间戳精修，
跳过它的依赖不影响使用。
"""

from __future__ import annotations

import os
from pathlib import Path

_paraformer_asr = None
_vnw_asr = None

_REPO_ROOT = Path(__file__).resolve().parents[3]

_VNW_DEFAULT_DIRS = (
    r"D:/Audio_process/Visual-novel-whisper/models/Visual-novel-whisper",
    str(_REPO_ROOT / "ckpts" / "Visual-novel-whisper"),
)


class AsrUnavailable(RuntimeError):
    """后端不可用（缺依赖或缺模型）。调用方决定是否包装成 UI 错误。"""


# ----------------------------
# 中文：FunASR Paraformer
# ----------------------------


def get_paraformer_asr(device: str):
    """Lazy-load FunASR Paraformer（中文）。"""
    global _paraformer_asr
    if _paraformer_asr is not None:
        return _paraformer_asr

    try:
        from funasr import AutoModel
    except ImportError:
        raise AsrUnavailable("未安装 FunASR。请先安装：pip install funasr modelscope")

    try:
        # paraformer-zh: general Mandarin model; trust_remote_code required
        _paraformer_asr = AutoModel(model="paraformer-zh", trust_remote_code=True, device=device)
    except Exception as e:
        raise AsrUnavailable("加载 FunASR Paraformer 失败，请确认已安装依赖，并可访问模型：" + str(e))
    return _paraformer_asr


# ----------------------------
# 日文：Visual-novel-whisper
# ----------------------------


def _resolve_vnw_dir() -> str:
    """按 环境变量 → 本机项目路径 → 仓库 ckpts 的顺序找模型目录。"""
    env = os.environ.get("VNW_MODEL_DIR", "").strip()
    candidates = ([env] if env else []) + list(_VNW_DEFAULT_DIRS)
    for c in candidates:
        if c and Path(c).is_dir():
            return c
    raise AsrUnavailable(
        "找不到 Visual-novel-whisper 模型目录。请设置环境变量 VNW_MODEL_DIR，"
        "或放到 ckpts/Visual-novel-whisper/。已尝试:\n" + "\n".join(c for c in candidates if c)
    )


def _load_vnw_pipeline_class():
    """取 KotobaWhisperPipeline 类。

    模型 config.json 里 custom_pipelines.impl 写的是
    `kotoba-tech/kotoba-whisper-v1.1--kotoba_whisper.KotobaWhisperPipeline`，
    `--` 前是远程仓库名 —— 即便权重在本地，transformers 仍会联网取这份 pipeline 代码。
    HF_HOME 指向仓库内 .huggingface（两个 .bat 都这么设）时缓存对不上，加载必然失败。

    所以这里优先用仓库内自带的本地副本 vendor/kotoba_whisper.py，
    找不到才回退到远程动态加载（需联网或已有对应缓存）。
    """
    vendored = Path(__file__).resolve().parent / "vendor" / "kotoba_whisper.py"
    if vendored.exists():
        import importlib.util

        spec = importlib.util.spec_from_file_location("f5_tts_vendor_kotoba_whisper", vendored)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.KotobaWhisperPipeline

    try:
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        return get_class_from_dynamic_module(
            "kotoba_whisper.KotobaWhisperPipeline", "kotoba-tech/kotoba-whisper-v1.1"
        )
    except Exception as e:
        raise AsrUnavailable(
            "无法加载 KotobaWhisperPipeline：仓库内缺少 "
            f"{vendored}，且远程动态加载失败（需联网）：{e}"
        )


def get_vnw_asr(device: str):
    """Lazy-load Visual-novel-whisper（日文）。"""
    global _vnw_asr
    if _vnw_asr is not None:
        return _vnw_asr

    model_dir = _resolve_vnw_dir()

    try:
        import torch
        from transformers import pipeline
    except ImportError as e:
        raise AsrUnavailable("加载日文 ASR 需要 transformers 与 torch：" + str(e))

    # 远程 pipeline 硬性 import 这两个包，提前给出可操作的报错
    for mod, hint in (("punctuators", "pip install punctuators"), ("stable_whisper", "pip install stable-ts --no-deps")):
        try:
            __import__(mod)
        except ImportError:
            raise AsrUnavailable(
                f"Visual-novel-whisper 的自定义 pipeline 需要 {mod}，但未安装。请执行：{hint}\n"
                "（stable-ts 必须加 --no-deps，其依赖的 openai-whisper 构建脚本在新版 setuptools 下会失败）"
            )

    pipeline_class = _load_vnw_pipeline_class()

    on_cuda = str(device).startswith("cuda")
    kwargs = dict(
        task="automatic-speech-recognition",
        model=model_dir,
        pipeline_class=pipeline_class,  # 显式传类，绕开 config 里的远程 impl 引用
        dtype=torch.float16 if on_cuda else torch.float32,
        device=device if on_cuda else "cpu",
        chunk_length_s=15,
        batch_size=8,
        stable_ts=False,  # 只要文本，不做时间戳精修
    )

    # punctuator 需要联网下载 PunctCapSegModelONNX，且它内部用 hf_hub_download 直连
    # huggingface.co（不认 HF_ENDPOINT 镜像），断网/墙内会失败。补标点只是锦上添花，
    # 默认关掉；想要就设 VNW_PUNCTUATOR=1，失败时自动降级而不是让整个 ASR 挂掉。
    want_punct = os.environ.get("VNW_PUNCTUATOR", "").strip() in ("1", "true", "True")
    if want_punct:
        try:
            _vnw_asr = pipeline(**kwargs, punctuator=True)
            return _vnw_asr
        except Exception as e:
            print(f"[asr] punctuator 加载失败，降级为不补标点（clean_ja_text 仍会补句尾）：{e}")

    try:
        _vnw_asr = pipeline(**kwargs, punctuator=False)
    except Exception as e:
        raise AsrUnavailable("加载 Visual-novel-whisper 失败：" + str(e))
    return _vnw_asr


# ----------------------------
# 文本清洗
# ----------------------------

_CN_ENDINGS = "。.!？！?"
_JA_ENDINGS = "。.!？！?…」』"


def clean_cn_text(text: str) -> str:
    """中文：去掉所有 ASCII 空格，并保证结尾有句读。"""
    if not text:
        return text
    t = text.replace(" ", "").strip()
    if not t:
        return t
    if t[-1] not in _CN_ENDINGS:
        t = t + "。"
    return t


def clean_ja_text(text: str) -> str:
    """日文：折叠空白并保证结尾有句读。

    与中文版的区别：日文不能无条件删空格（罗马字/外语词内部的空格有意义），
    只把连续空白折叠成一个半角空格；全角空格统一成半角；结尾允许以引号收尾
    （台词类音频常见「…」『…』）。
    """
    if not text:
        return text
    t = text.replace("　", " ")
    t = " ".join(t.split())  # 折叠所有连续空白
    t = t.strip()
    if not t:
        return t
    if t[-1] not in _JA_ENDINGS:
        t = t + "。"
    return t


# ----------------------------
# 统一入口
# ----------------------------

LANG_CHOICES = ("zh", "ja")


def transcribe(audio_path: str, lang: str, device: str) -> str:
    """转写单个音频并做对应语言的清洗。lang: "zh" | "ja"。"""
    if lang == "zh":
        asr = get_paraformer_asr(device=device)
        res = asr.generate(input=str(audio_path), batch_size=1)
        text = res[0].get("text", "") if isinstance(res, list) else res.get("text", "")
        return clean_cn_text(text)

    if lang == "ja":
        asr = get_vnw_asr(device=device)
        res = asr(
            str(audio_path),
            return_timestamps=True,
            generate_kwargs={"language": "japanese", "task": "transcribe"},
        )
        text = res.get("text", "") if isinstance(res, dict) else str(res)
        return clean_ja_text(text)

    raise ValueError(f"lang 需为 {LANG_CHOICES} 之一，收到 {lang!r}")
