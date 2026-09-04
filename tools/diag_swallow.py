#!/usr/bin/env python
"""Measure how much of gen_text actually survives, single-ref vs dual-ref.

"吞段落" is subjective until it is counted, so this generates the same gen_text
through several configurations, transcribes each output with the same ASR the
試験台 uses, and reports character error rate against gen_text. A config that
swallows content shows a high deletion rate while its duration budget was ample.

The duration arithmetic was already ruled out separately (tools/diag_duration.py):
the dual path allots MORE frames per character than single-ref A. So the suspect
here is prompt-region misalignment — cond_mask spans max(len_a, len_b) while the
text sequence only describes one reference, leaving the length difference
unexplained by any text.

Configs:
  a_only / b_only : single reference, upstream-equivalent path (cond_b=None)
  dual_lerp       : both refs, default cond mixing
  dual_trimmed    : both refs trimmed to equal length, isolating the length gap
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

TARGET_SR = 24000


def levenshtein(a: str, b: str) -> int:
    if not a:
        return len(b)
    previous = list(range(len(a) + 1))
    for i, cb in enumerate(b, start=1):
        current = [i]
        for j, ca in enumerate(a, start=1):
            current.append(min(previous[j] + 1, current[-1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ref-a", required=True)
    parser.add_argument("--ref-b", required=True)
    parser.add_argument("--text-a", required=True)
    parser.add_argument("--text-b", required=True)
    parser.add_argument("--gen-text", required=True)
    parser.add_argument("--out-dir", default="diag_swallow_out")
    parser.add_argument("--configs", default="a_only,b_only,dual_lerp,dual_trimmed")
    parser.add_argument("--mix-a-start", type=float, default=None,
                        help="A 权重；不传则用 infer_process 的默认值 0.9")
    parser.add_argument("--speaker-sim", action="store_true",
                        help="额外测每条输出与 ref A / ref B 的声纹相似度（判断混合是否真的生效）")
    args = parser.parse_args(argv)

    import soundfile as sf
    import torch
    import torchaudio

    from f5_tts.infer.gradio_mix_demo import _clean_cn_text, _get_paraformer_asr, _load_default_model
    from f5_tts.infer.utils_infer import infer_process, preprocess_ref_audio_text

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, vocoder, device = _load_default_model()
    print(f"model loaded on {device}", flush=True)

    ref_a, text_a = preprocess_ref_audio_text(args.ref_a, args.text_a, show_info=lambda *a, **k: None)
    ref_b, text_b = preprocess_ref_audio_text(args.ref_b, args.text_b, show_info=lambda *a, **k: None)

    def seconds(path: str) -> float:
        info = torchaudio.info(path)
        return info.num_frames / info.sample_rate

    sec_a, sec_b = seconds(ref_a), seconds(ref_b)
    print(f"ref A {sec_a:.2f}s  ref B {sec_b:.2f}s  (gap {abs(sec_a - sec_b):.2f}s)", flush=True)

    trimmed_b = None
    if "dual_trimmed" in args.configs:
        # 把较长的一条裁到较短的那条长度，让 mask_a 与 mask_b 完全重合，
        # 从而把"长度差造成的无文本音频"这一个变量单独摘出来。
        keep = min(sec_a, sec_b)
        wav, sr = torchaudio.load(ref_b if sec_b > sec_a else ref_a)
        wav = wav[:, : int(keep * sr)]
        trimmed_b = str(out_dir / "ref_trimmed.wav")
        torchaudio.save(trimmed_b, wav, sr)
        print(f"trimmed longer ref to {keep:.2f}s -> {trimmed_b}", flush=True)

    configs = {
        "a_only":      dict(ref_audio=ref_a, ref_text=text_a, ref_audio_2=None,  ref_text_2="",   mix_on="cond"),
        "b_only":      dict(ref_audio=ref_b, ref_text=text_b, ref_audio_2=None,  ref_text_2="",   mix_on="cond"),
        "dual_cond":   dict(ref_audio=ref_a, ref_text=text_a, ref_audio_2=ref_b, ref_text_2=text_b, mix_on="cond"),
        "dual_pred":   dict(ref_audio=ref_a, ref_text=text_a, ref_audio_2=ref_b, ref_text_2=text_b, mix_on="pred"),
        "two_stage_repeat": dict(
            ref_audio=ref_a, ref_text=text_a, ref_audio_2=ref_b, ref_text_2=text_b,
            mix_on="two_stage", second_stage_prompt="repeat",
        ),
        # two_stage_none 已移除：prompt 区占满整条 cond 时 ODE 自由空间为 0，
        # 输出被 fused_cond 覆盖，退化成 mel 域插值（alpha≈0.5 必然叠音）。
        "dual_lerp":   dict(ref_audio=ref_a, ref_text=text_a, ref_audio_2=ref_b, ref_text_2=text_b, mix_on="cond"),
        "dual_trimmed": dict(
            ref_audio=ref_a if sec_b > sec_a else trimmed_b,
            ref_text=text_a,
            ref_audio_2=trimmed_b if sec_b > sec_a else ref_b,
            ref_text_2=text_b,
            mix_on="cond",
        ),
    }

    asr = _get_paraformer_asr(device=device)

    def transcribe(path: str) -> str:
        result = asr.generate(input=path, batch_size=1)
        raw = result[0].get("text", "") if isinstance(result, list) else result.get("text", "")
        return _clean_cn_text(raw)

    # CER 只衡量内容，衡量不了音色是否真的混合了。声纹相似度补上这一维：
    # 若输出对 A / B 的相似度随 mix_a_start 变化，说明混合确实生效。
    spk_embed = None
    if args.speaker_sim:
        sys.path.insert(0, str(ROOT / "tools"))
        from tools.eval_protocol import cosine, speaker_embedding

        def spk_embed(path):  # noqa: F811
            return speaker_embedding(path)

    target = args.gen_text
    print(f"\ngen_text ({len(target)} chars): {target}", flush=True)
    if args.mix_a_start is not None:
        print(f"mix_a_start = {args.mix_a_start}", flush=True)
    print(flush=True)

    header = f"{'config':<18}{'out_s':>7}{'chars':>7}{'CER':>7}"
    if args.speaker_sim:
        header += f"{'simA':>7}{'simB':>7}{'A-B':>7}"
    header += "  asr_text"
    print(header, flush=True)
    print("-" * (len(header) + 20), flush=True)

    ref_emb_a = spk_embed(ref_a) if args.speaker_sim else None
    ref_emb_b = spk_embed(ref_b) if args.speaker_sim else None

    for name in [c.strip() for c in args.configs.split(",") if c.strip()]:
        kwargs = configs[name]
        if kwargs.get("ref_audio") is None:
            continue
        extra = {}
        if args.mix_a_start is not None:
            extra["mix_a_start"] = args.mix_a_start
            extra["mix_a_end"] = args.mix_a_start
        wave, sr, _ = infer_process(
            kwargs["ref_audio"],
            kwargs["ref_text"],
            target,
            model,
            vocoder,
            ref_audio_2=kwargs["ref_audio_2"],
            ref_text_2=kwargs["ref_text_2"],
            mix_on=kwargs.get("mix_on", "cond"),
            second_stage_prompt=kwargs.get("second_stage_prompt", "repeat"),
            device=device,
            show_info=lambda *a, **k: None,
            progress=None,
            **extra,
        )
        path = out_dir / f"{name}.wav"
        sf.write(path, wave, sr)
        hypothesis = transcribe(str(path))
        # 只关心内容缺失，所以按目标文本长度归一
        clean_target = _clean_cn_text(target)
        rate = levenshtein(hypothesis, clean_target) / max(1, len(clean_target))
        row = f"{name:<18}{len(wave) / sr:>7.2f}{len(hypothesis):>7}{rate:>7.3f}"
        if args.speaker_sim:
            emb = spk_embed(str(path))
            sim_a = cosine(emb, ref_emb_a)
            sim_b = cosine(emb, ref_emb_b)
            row += f"{sim_a:>7.3f}{sim_b:>7.3f}{sim_a - sim_b:>+7.3f}"
        print(f"{row}  {hypothesis}", flush=True)

    print(f"\nwavs in {out_dir.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
