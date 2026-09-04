#!/usr/bin/env python
"""Reproduce infer_process's duration arithmetic for a given ref pair, without loading the model.

"吞段落" (generated speech stopping before the text finishes) means the allotted
`duration` was too short for `gen_text`. That allotment is pure arithmetic over the
reference lengths and texts, so it can be checked in seconds on CPU rather than by
listening to model output.

Prints, for single-ref A, single-ref B, and the dual-ref union path:
  ref frames, pace ratio (frames per utf-8 byte), gen_frames, total duration,
  and the implied seconds for the generated span.

A healthy result has the dual-ref gen_frames close to the single-ref ones. If the
dual path allots noticeably fewer frames per character than either reference on its
own, that is the swallowing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

TARGET_SR = 24000
HOP = 256


def pace(frames: int, text: str) -> float:
    """utils_infer._pace: frames per utf-8 byte."""
    return frames / max(1, len(text.encode("utf-8")))


def report(label: str, ref_frames: int, pace_ratio: float, gen_text: str, speed: float) -> None:
    gen_bytes = len(gen_text.encode("utf-8"))
    gen_frames = int(pace_ratio * gen_bytes / speed)
    duration = ref_frames + gen_frames
    chars = max(1, len(gen_text))
    print(
        f"  {label:<24} ref={ref_frames:>5}f  pace={pace_ratio:>6.3f} f/byte  "
        f"gen={gen_frames:>5}f ({gen_frames * HOP / TARGET_SR:>5.2f}s)  "
        f"total={duration:>5}f ({duration * HOP / TARGET_SR:>5.2f}s)  "
        f"{gen_frames / chars:>5.1f} f/char"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ref-a", required=True)
    parser.add_argument("--ref-b", required=True)
    parser.add_argument("--text-a", required=True)
    parser.add_argument("--text-b", required=True)
    parser.add_argument("--gen-text", required=True)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--mix-a-start", type=float, default=0.5)
    parser.add_argument("--raw", action="store_true", help="skip preprocess_ref_audio_text, use raw file length")
    args = parser.parse_args(argv)

    import torchaudio

    def prepare(path: str, text: str) -> tuple[int, str]:
        if args.raw:
            info = torchaudio.info(path)
            frames = int(info.num_frames * TARGET_SR / info.sample_rate) // HOP
            return frames, text
        from f5_tts.infer.utils_infer import preprocess_ref_audio_text

        processed_path, processed_text = preprocess_ref_audio_text(path, text, show_info=lambda *a, **k: None)
        wav, sr = torchaudio.load(processed_path)
        if sr != TARGET_SR:
            wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
        return wav.shape[-1] // HOP, processed_text

    len_a, text_a = prepare(args.ref_a, args.text_a)
    len_b, text_b = prepare(args.ref_b, args.text_b)

    print(f"gen_text: {args.gen_text!r}")
    print(f"  chars={len(args.gen_text)}  utf8_bytes={len(args.gen_text.encode('utf-8'))}  speed={args.speed}")
    print(f"ref A: {len_a}f ({len_a * HOP / TARGET_SR:.2f}s)  text={len(text_a)} chars")
    print(f"ref B: {len_b}f ({len_b * HOP / TARGET_SR:.2f}s)  text={len(text_b)} chars")
    print()

    pace_a = pace(len_a, text_a)
    pace_b = pace(len_b, text_b)
    print("single-ref paths (each ref alone, its own pace):")
    report("A alone", len_a, pace_a, args.gen_text, args.speed)
    report("B alone", len_b, pace_b, args.gen_text, args.speed)
    print()

    # utils_infer: prompt_text 按 mix_a_start 二选一，pace 跟着它走
    if args.mix_a_start >= 0.5:
        pace_frames, pace_text, which = len_a, text_a, "A"
    else:
        pace_frames, pace_text, which = len_b, text_b, "B"
    ref_len_union = max(len_a, len_b)
    candidates = [pace(pace_frames, pace_text), pace_a, pace_b]
    valid = [p for p in candidates if p > 0]
    pace_ratio = max(valid) if valid else 10.0

    print(f"dual-ref path (mix_a_start={args.mix_a_start} -> prompt/pace from {which}):")
    print(f"  ref_len_union = max({len_a}, {len_b}) = {ref_len_union}")
    print(f"  pace candidates = {[round(p, 3) for p in candidates]} -> max = {pace_ratio:.3f}")
    report("dual (union)", ref_len_union, pace_ratio, args.gen_text, args.speed)
    print()

    # 关键对比：prompt 区域给了多少帧，而文本只描述了其中一条参考
    implied = ref_len_union / max(1, len(pace_text.encode("utf-8")))
    print("prompt-region consistency:")
    print(f"  frames allotted to prompt : {ref_len_union}")
    print(f"  text describing prompt    : {len(pace_text)} chars ({len(pace_text.encode('utf-8'))} bytes) from {which}")
    print(f"  implied pace of prompt    : {implied:.3f} f/byte")
    print(f"  pace used for gen span    : {pace_ratio:.3f} f/byte")
    skew = implied / pace_ratio if pace_ratio else float("inf")
    print(f"  prompt/gen pace skew      : {skew:.3f}x", end="  ")
    if skew > 1.15:
        print("<-- prompt implies SLOWER speech than gen was allotted; expect truncation")
    elif skew < 0.87:
        print("<-- prompt implies faster speech than gen was allotted")
    else:
        print("(consistent)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
