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
        import whisper

        _asr = whisper.load_model("large-v3")
    return _asr


# ---------------- 指标 ----------------


def emotion_embedding(wav_path: str):
    """emotion2vec utterance 级 embedding（数值向量）。"""
    import numpy as np

    res = emo_model().generate(wav_path, granularity="utterance", extract_embedding=True)
    return np.asarray(res[0]["feats"], dtype="float32")


def speaker_embedding(wav_path: str):
    import numpy as np

    res = spk_model().generate(wav_path)
    return np.asarray(res[0]["spk_embedding"], dtype="float32").squeeze()


def cosine(a, b) -> float:
    import numpy as np

    a = a.flatten()
    b = b.flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def transcribe_ja(wav_path: str) -> str:
    res = asr_model().transcribe(wav_path, language="ja")
    return res["text"].strip()


def normalize_kana(text: str) -> str:
    """转写归一化：汉字混排 → 片假名，去标点空白，供 CER 对齐。"""
    from f5_tts.infer.ja_frontend import ja_to_kana

    kana = ja_to_kana(text)
    puncts = set("、。！？!?，,．.…「」『』（）()　 ・ー")
    # 长音符对 CER 噪声大（ASR 与 g2p 的长音表示常不一致），一并去掉
    return "".join(c for c in kana if c not in puncts)


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
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    gen_text = manifest["gen_text"]
    ref_kana = normalize_kana(gen_text)

    results = []
    for case in manifest["cases"]:
        wav = case["wav"]
        if not Path(wav).exists():
            print(f"[skip] {case['id']}: {wav} 不存在")
            continue

        row = {"id": case["id"], "alpha": case.get("alpha")}

        emo_ref = case.get("ref_emotion_wav")
        if emo_ref and Path(emo_ref).exists():
            row["emotion_sim"] = round(cosine(emotion_embedding(wav), emotion_embedding(emo_ref)), 4)

        spk_ref = case.get("ref_speaker_wav")
        if spk_ref and Path(spk_ref).exists():
            row["speaker_sim"] = round(cosine(speaker_embedding(wav), speaker_embedding(spk_ref)), 4)

        hyp = transcribe_ja(wav)
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
