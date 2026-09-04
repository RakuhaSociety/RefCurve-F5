#!/usr/bin/env python
"""Measure how faithfully mix_a_start steers timbre, with seed noise controlled.

Why this exists: comparing "how much of the anchor span does mixing cover" across
separate runs was unreliable — with no fixed seed, the single-reference anchors
themselves moved by 0.06-0.12 cosine between runs, which is as large as the effect
being measured. One run had 诧异's anchor at -0.213 and the next at +0.092 for the
same audio and text.

Two fixes:
  * Fixed seeds, and every config within a seed uses that same seed, so anchors and
    mixes are directly comparable.
  * Report a seed-internal normalised position instead of a raw cosine delta:
        pos = (mix_delta - anchor_b_delta) / (anchor_a_delta - anchor_b_delta)
    where each *_delta is cosine(out, refA) - cosine(out, refB). pos=1 means the
    output sits exactly where the A-only anchor sits, pos=0 exactly at B-only.
    A faithful control surface gives pos ≈ mix_a_start.

Also loads the model and ASR once for the whole sweep, rather than paying that per
configuration as the earlier per-invocation script did.

`--swap` additionally runs the mirrored assignment (B's audio in the A slot) and
reports which *voice* the bias follows. If a bias tracks the voice that gets
time-stretched rather than the A slot, that points at the stretch as the cause; if
it stays in the A slot, the asymmetry is in the mixing code instead.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ref-a", required=True)
    parser.add_argument("--ref-b", required=True)
    parser.add_argument("--text-a", required=True)
    parser.add_argument("--text-b", required=True)
    parser.add_argument("--gen-text", required=True)
    parser.add_argument("--alphas", default="0.9,0.7,0.5,0.3,0.1")
    parser.add_argument("--seeds", default="1234,5678,9012")
    parser.add_argument("--mode", default="two_stage")
    parser.add_argument("--second-stage-prompt", default="repeat")
    parser.add_argument("--swap", action="store_true", help="also run the mirrored A/B assignment")
    parser.add_argument("--out-dir", default="diag_alpha_out")
    args = parser.parse_args(argv)

    import soundfile as sf
    import torch

    from f5_tts.infer.gradio_mix_demo import _clean_cn_text, _get_paraformer_asr, _load_default_model
    from f5_tts.infer.utils_infer import infer_process, preprocess_ref_audio_text
    from tools.eval_protocol import cosine, speaker_embedding

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, vocoder, device = _load_default_model()
    asr = _get_paraformer_asr(device=device)
    print(f"model + asr ready on {device}", flush=True)

    ref_a, text_a = preprocess_ref_audio_text(args.ref_a, args.text_a, show_info=lambda *a, **k: None)
    ref_b, text_b = preprocess_ref_audio_text(args.ref_b, args.text_b, show_info=lambda *a, **k: None)

    # 相似度基准始终是原始两条参考，不随 swap 改变，这样两个朝向的数字可直接比较
    emb_ref_a = speaker_embedding(ref_a)
    emb_ref_b = speaker_embedding(ref_b)

    alphas = [float(x) for x in args.alphas.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]

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

    clean_target = _clean_cn_text(args.gen_text)

    def generate(slot_a, slot_a_text, slot_b, slot_b_text, alpha, seed, mode, tag):
        kwargs = dict(
            device=device,
            show_info=lambda *a, **k: None,
            progress=None,
            seed=seed,
        )
        if mode == "single":
            wave, sr, _ = infer_process(
                slot_a, slot_a_text, args.gen_text, model, vocoder, ref_audio_2=None, ref_text_2="", **kwargs
            )
        else:
            wave, sr, _ = infer_process(
                slot_a,
                slot_a_text,
                args.gen_text,
                model,
                vocoder,
                ref_audio_2=slot_b,
                ref_text_2=slot_b_text,
                mix_on=mode,
                second_stage_prompt=args.second_stage_prompt,
                mix_a_start=alpha,
                mix_a_end=alpha,
                **kwargs,
            )
        path = out_dir / f"{tag}.wav"
        sf.write(path, wave, sr)
        embedding = speaker_embedding(str(path))
        delta = cosine(embedding, emb_ref_a) - cosine(embedding, emb_ref_b)
        result = asr.generate(input=str(path), batch_size=1)
        raw = result[0].get("text", "") if isinstance(result, list) else result.get("text", "")
        hypothesis = _clean_cn_text(raw)
        cer = levenshtein(hypothesis, clean_target) / max(1, len(clean_target))
        return delta, cer, len(wave) / sr

    orientations = [("normal", ref_a, text_a, ref_b, text_b)]
    if args.swap:
        orientations.append(("swapped", ref_b, text_b, ref_a, text_a))

    for name, slot_a, slot_a_text, slot_b, slot_b_text in orientations:
        natural_a = _natural_frames(slot_a, slot_a_text, args.gen_text)
        natural_b = _natural_frames(slot_b, slot_b_text, args.gen_text)
        shared = max(natural_a, natural_b)
        if natural_a < natural_b:
            stretched = f"A slot (+{shared - natural_a}f, +{100 * (shared - natural_a) / natural_a:.0f}%)"
        elif natural_b < natural_a:
            stretched = f"B slot (+{shared - natural_b}f, +{100 * (shared - natural_b) / natural_b:.0f}%)"
        else:
            stretched = "neither (equal natural lengths)"
        print(f"\n{'=' * 78}")
        print(f"orientation: {name}   (time-stretched by shared_gen_frames: {stretched})")
        print(f"{'=' * 78}")

        positions: dict[float, list[float]] = {a: [] for a in alphas}
        for seed in seeds:
            anchor_a, cer_a, _ = generate(slot_a, slot_a_text, None, "", 0.0, seed, "single", f"{name}_s{seed}_anchorA")
            anchor_b, cer_b, _ = generate(slot_b, slot_b_text, None, "", 0.0, seed, "single", f"{name}_s{seed}_anchorB")
            span = anchor_a - anchor_b
            print(
                f"\n seed {seed}: anchorA_delta={anchor_a:+.3f} (cer {cer_a:.3f})   "
                f"anchorB_delta={anchor_b:+.3f} (cer {cer_b:.3f})   span={span:+.3f}"
            )
            if abs(span) < 1e-6:
                print("   span ~ 0, cannot normalise; skipping this seed")
                continue
            print(f"   {'alpha':>6}{'delta':>9}{'pos':>7}{'cer':>7}{'dur_s':>7}")
            for alpha in alphas:
                delta, cer, duration = generate(
                    slot_a,
                    slot_a_text,
                    slot_b,
                    slot_b_text,
                    alpha,
                    seed,
                    args.mode,
                    f"{name}_s{seed}_a{alpha}",
                )
                position = (delta - anchor_b) / span
                positions[alpha].append(position)
                print(f"   {alpha:>6.2f}{delta:>+9.3f}{position:>7.2f}{cer:>7.3f}{duration:>7.2f}")

        print(f"\n {name}: pos vs alpha across {len(seeds)} seeds  (ideal pos == alpha)")
        print(f"   {'alpha':>6}{'pos_mean':>10}{'pos_sd':>8}{'err':>8}")
        for alpha in alphas:
            values = positions[alpha]
            if not values:
                continue
            mean = statistics.mean(values)
            sd = statistics.stdev(values) if len(values) > 1 else 0.0
            print(f"   {alpha:>6.2f}{mean:>10.2f}{sd:>8.2f}{mean - alpha:>+8.2f}")

    print(f"\nwavs in {out_dir.resolve()}")
    return 0


def _natural_frames(path, ref_text, gen_text) -> int:
    """复刻 utils_infer._natural_gen_frames，用来指出哪一路会被 shared_gen_frames 拉长。

    不能只比原始时长：自然生成长度是 ref_frames / ref_text_bytes * gen_text_bytes，
    参考文本长度同样参与，所以更长的音频未必是被拉长的那一路。
    """
    import torchaudio

    info = torchaudio.info(str(path))
    frames = int(info.num_frames * 24000 / info.sample_rate) // 256
    return int(frames / max(1, len(ref_text.encode("utf-8"))) * len(gen_text.encode("utf-8")))


if __name__ == "__main__":
    raise SystemExit(main())
