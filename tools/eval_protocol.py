"""Phase 0 评测协议：情绪相似度 + 说话人相似度 + 日文 CER + α 单调性扫描。

这是 japanese-emotion-roadmap Phase 0 的核心交付物，同一套协议将服务于：
  - Jmica 日文 F5 基线测试（本仓库）
  - GPT-SoVITS 天花板参照（手动生成后放入对比目录）
  - RefCurve-SoVITS 的 M1-M3 验收（方案 §9）

三个指标模型全部走 funasr/modelscope 生态（本仓库既有依赖）：
  情绪  emotion2vec_plus_base   —— utterance 级 embedding 余弦
  说话人 campplus_sv            —— speaker embedding 余弦
  可懂度 whisper large-v3       —— 日文转写 CER（假名归一化后）

用法：
  # 1) 准备测试清单 eval_manifest.json（格式见 EXAMPLE_MANIFEST）
  # 2) python tools/eval_protocol.py run  eval_manifest.json  results.json
  # 3) python tools/eval_protocol.py report results.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

EXAMPLE_MANIFEST = {
    "comment": "cases 里每条 = 一次待评测音频。ref_emotion/ref_speaker 是情绪与音色的目标锚点。",
    "gen_text": "こんな所で、諦めるわけにはいかない！",
    "cases": [
        {
            "id": "alpha_0.5",
            "wav": "tests/ja_eval/out_a0.5.wav",
            "ref_emotion_wav": "tests/ja_ref/angry.wav",
            "ref_speaker_wav": "tests/ja_ref/calm.wav",
            "alpha": 0.5,
        }
    ],
}

# ---------------- 模型加载（懒加载单例） ----------------

_emo = None
_spk = None
_asr = None
_embedding_cache = {}


def emo_model():
    global _emo
    if _emo is None:
        from funasr import AutoModel

        _emo = AutoModel(model="iic/emotion2vec_plus_base", disable_update=True)
    return _emo


def spk_model():
    global _spk
    if _spk is None:
        from funasr import AutoModel

        _spk = AutoModel(model="iic/speech_campplus_sv_zh-cn_16k-common", disable_update=True)
    return _spk


def asr_model():
    global _asr
    if _asr is None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError("日文评测 ASR 需要 faster-whisper；请安装项目的 eval 可选依赖") from exc

        _asr = WhisperModel("large-v3", device="auto", compute_type="default")
    return _asr


# ---------------- 指标 ----------------


def _to_numpy(value):
    """将 FunASR 可能返回的 GPU Tensor / ndarray 统一转成 CPU numpy。"""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return value


def emotion_embedding(wav_path: str):
    """emotion2vec utterance 级 embedding（数值向量）。"""
    import numpy as np

    key = ("emotion", str(Path(wav_path).resolve()))
    if key not in _embedding_cache:
        res = emo_model().generate(wav_path, granularity="utterance", extract_embedding=True)
        _embedding_cache[key] = np.asarray(_to_numpy(res[0]["feats"]), dtype="float32")
    return _embedding_cache[key]


def speaker_embedding(wav_path: str):
    import numpy as np

    key = ("speaker", str(Path(wav_path).resolve()))
    if key not in _embedding_cache:
        res = spk_model().generate(wav_path)
        _embedding_cache[key] = np.asarray(_to_numpy(res[0]["spk_embedding"]), dtype="float32").squeeze()
    return _embedding_cache[key]


def cosine(a, b) -> float:
    import numpy as np

    a = a.flatten()
    b = b.flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def transcribe_ja(wav_path: str) -> str:
    """日文 ASR。优先 Visual-novel-whisper，次之 asr_backends paraformer，最后 faster-whisper。"""
    try:
        from f5_tts.infer.asr_backends import transcribe

        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        return transcribe(wav_path, lang="ja", device=device).strip()
    except Exception as vnw_error:
        # VNW 不可用时尝试 paraformer 直接走日文路径（远端已缓存，无需下载）
        try:
            from f5_tts.infer.asr_backends import get_paraformer_asr, clean_ja_text

            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
            asr = get_paraformer_asr(device=device)
            results = asr.generate(input=wav_path, generate_kwargs={"language": "japanese", "task": "transcribe"})
            raw = "".join(r.get("text", "") for r in results) if isinstance(results, list) else str(results)
            cleaned = clean_ja_text(raw).strip()
            if not cleaned:
                raise ValueError("paraformer returned empty transcription for Japanese audio")
            return cleaned
        except Exception as paraformer_error:
            try:
                segments, _ = asr_model().transcribe(
                    wav_path,
                    language="ja",
                    beam_size=5,
                    best_of=5,
                    temperature=0.0,
                    condition_on_previous_text=False,
                    vad_filter=False,
                )
                return "".join(segment.text for segment in segments).strip()
            except Exception as fallback_error:
                raise RuntimeError(
                    f"日文 ASR 均不可用：Visual-novel-whisper={vnw_error}; "
                    f"paraformer={paraformer_error}; faster-whisper={fallback_error}"
                ) from fallback_error


def audio_diagnostics(wav_path: str, *, silence_db: float = -50.0) -> dict[str, float | int | bool]:
    """Return dependency-light WAV integrity metrics used by matrix hard gates."""
    import numpy as np
    import soundfile as sf

    audio, sample_rate = sf.read(wav_path, always_2d=True, dtype="float32")
    finite = bool(np.isfinite(audio).all())
    safe = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)
    peak = float(np.max(np.abs(safe))) if safe.size else 0.0
    threshold = 10.0 ** (silence_db / 20.0)
    return {
        "sample_rate": int(sample_rate),
        "channels": int(audio.shape[1]),
        "frames": int(audio.shape[0]),
        "duration_seconds": float(audio.shape[0] / sample_rate) if sample_rate else 0.0,
        "peak": peak,
        "clipping_fraction": float(np.mean(np.abs(safe) >= 0.999)) if safe.size else 0.0,
        "silence_fraction": float(np.mean(np.max(np.abs(safe), axis=1) < threshold)) if safe.size else 1.0,
        "has_nan_inf": not finite,
    }


def score_audio(
    wav_path: str,
    *,
    reference_kana: str | None = None,
    speaker_reference: str | None = None,
    emotion_reference: str | None = None,
    metrics: tuple[str, ...] = ("cer", "speaker_sim", "emotion_sim"),
) -> dict[str, object]:
    """Score one clip, loading optional ASR/embedding models only when requested."""
    row: dict[str, object] = audio_diagnostics(wav_path)
    # 先算不依赖 ASR 的 embedding 指标。这样即使本机没装 faster-whisper，
    # speaker/emotion 评分仍能落盘，CER 可以随后补跑而不是整行丢失。
    if "speaker_sim" in metrics and speaker_reference:
        row["speaker_sim"] = cosine(speaker_embedding(wav_path), speaker_embedding(speaker_reference))
    if "emotion_sim" in metrics and emotion_reference:
        row["emotion_sim"] = cosine(emotion_embedding(wav_path), emotion_embedding(emotion_reference))
    if "cer" in metrics and reference_kana is not None:
        hypothesis = transcribe_ja(wav_path)
        row["asr_text"] = hypothesis
        row["cer"] = cer(normalize_kana(hypothesis), normalize_kana(reference_kana, convert=False))
    return row


def normalize_kana(text: str, *, convert: bool = True) -> str:
    """转写归一化：汉字混排 → 片假名，去标点空白，供 CER 对齐。

    Prepared matrix manifests already contain kana; ``convert=False`` lets remote
    scoring consume it without importing pyopenjtalk.
    """
    if convert:
        from f5_tts.infer.ja_frontend import ja_to_kana

        text = ja_to_kana(text)
    puncts = set("、。！？!?，,．.…「」『』（）()　 ・ー")
    # 长音符对 CER 噪声大（ASR 与 g2p 的长音表示常不一致），一并去掉
    return "".join(c for c in text if c not in puncts)


def cer(hyp: str, ref: str) -> float:
    """字符错误率（Levenshtein / len(ref)）。"""
    m, n = len(hyp), len(ref)
    if n == 0:
        return 0.0 if m == 0 else 1.0
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (hyp[i - 1] != ref[j - 1]))
            prev = cur
    return dp[n] / n


# ---------------- 主流程 ----------------


def run(manifest_path: str, out_path: str):
    manifest_file = Path(manifest_path).resolve()
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    base_dir = manifest_file.parent
    gen_text = manifest["gen_text"]
    ref_kana = normalize_kana(gen_text)

    results = []
    for case in manifest["cases"]:
        wav = Path(case["wav"])
        wav = wav if wav.is_absolute() else (base_dir / wav).resolve()
        if not wav.exists():
            print(f"[skip] {case['id']}: {wav} 不存在")
            continue

        row = {"id": case["id"], "alpha": case.get("alpha")}

        emo_ref = case.get("ref_emotion_wav")
        if emo_ref:
            emo_ref = Path(emo_ref)
            emo_ref = emo_ref if emo_ref.is_absolute() else (base_dir / emo_ref).resolve()
        if emo_ref and emo_ref.exists():
            row["emotion_sim"] = round(cosine(emotion_embedding(str(wav)), emotion_embedding(str(emo_ref))), 4)

        spk_ref = case.get("ref_speaker_wav")
        if spk_ref:
            spk_ref = Path(spk_ref)
            spk_ref = spk_ref if spk_ref.is_absolute() else (base_dir / spk_ref).resolve()
        if spk_ref and spk_ref.exists():
            row["speaker_sim"] = round(cosine(speaker_embedding(str(wav)), speaker_embedding(str(spk_ref))), 4)

        hyp = transcribe_ja(str(wav))
        row["asr_text"] = hyp
        row["cer"] = round(cer(normalize_kana(hyp), ref_kana), 4)

        results.append(row)
        print(f"[done] {row['id']}: emo={row.get('emotion_sim', '-')} spk={row.get('speaker_sim', '-')} cer={row['cer']}")

    Path(out_path).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已写入 {out_path}")


def report(results_path: str):
    """单调性报告：按 alpha 排序看指标响应是否单调。"""
    rows = json.loads(Path(results_path).read_text(encoding="utf-8"))
    rows = [r for r in rows if r.get("alpha") is not None]
    rows.sort(key=lambda r: r["alpha"])
    if not rows:
        print("没有带 alpha 的条目")
        return

    print(f"{'alpha':>6}  {'emo_sim':>8}  {'spk_sim':>8}  {'cer':>6}")
    for r in rows:
        print(f"{r['alpha']:>6}  {r.get('emotion_sim', float('nan')):>8}  {r.get('speaker_sim', float('nan')):>8}  {r['cer']:>6}")

    def monotone(vals):
        vals = [v for v in vals if v is not None]
        if len(vals) < 3:
            return "样本不足"
        inc = all(a <= b + 1e-6 for a, b in zip(vals, vals[1:]))
        dec = all(a >= b - 1e-6 for a, b in zip(vals, vals[1:]))
        return "单调递增" if inc else "单调递减" if dec else "非单调"

    print(f"\nemotion_sim 随 alpha: {monotone([r.get('emotion_sim') for r in rows])}")
    print(f"speaker_sim 随 alpha: {monotone([r.get('speaker_sim') for r in rows])}")
    print(f"cer 最大值: {max(r['cer'] for r in rows)}（>0.15 视为内容受损）")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        print("示例 manifest：")
        print(json.dumps(EXAMPLE_MANIFEST, ensure_ascii=False, indent=2))
    elif sys.argv[1] == "run":
        run(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "eval_results.json")
    elif sys.argv[1] == "report":
        report(sys.argv[2])
    else:
        print("用法: eval_protocol.py run <manifest.json> [out.json] | report <results.json>")
