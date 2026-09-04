#!/usr/bin/env python
"""Score Japanese CER locally with Visual-novel-whisper, for WAVs generated on the remote.

Why this exists: the training remote has transformers 5.15.1 against torch
2.4.0+cu124, so transformers disables its PyTorch backend entirely
(`is_torch_available()` is False) and no transformers Whisper model can load
there at all. Pushing the 2.9 GB VNW weights over would not help. The clips are
only 50 MB, so the audio comes here instead — this machine has torch 2.8.0 and
the VNW weights already on disk.

CER numbers must be comparable with the spec's gates, so normalisation and the
edit distance come from `tools/eval_protocol.py` rather than being reimplemented:
the hypothesis is converted kanji->katakana, the reference kana is used as-is,
and both drop the same punctuation set.

Writes one JSON object per clip to `--out` as JSONL, so an interrupted run keeps
whatever it already scored and `--resume` continues from there.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


def load_scored(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    scored = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        scored[record["clip_id"]] = record
    return scored


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prepared", required=True, help="prepared.json from the eval run")
    parser.add_argument("--audio-root", required=True, help="directory holding <system_id>/<text_id>.wav")
    parser.add_argument("--out", required=True, help="JSONL output path")
    parser.add_argument("--resume", action="store_true", help="skip clips already present in --out")
    parser.add_argument("--limit", type=int, default=0, help="score at most N clips (0 = all)")
    args = parser.parse_args(argv)

    prepared = json.loads(Path(args.prepared).read_text(encoding="utf-8"))
    text_kana = {t["id"]: t["text_kana"] for t in prepared["texts"]}
    text_split = {t["id"]: t.get("split") for t in prepared["texts"]}
    system_ids = [s["id"] for s in prepared["systems"]]

    audio_root = Path(args.audio_root)
    out_path = Path(args.out)
    already = load_scored(out_path) if args.resume else {}

    jobs = []
    for system_id in system_ids:
        for text_id in text_kana:
            clip_id = f"{system_id}::{text_id}"
            if clip_id in already:
                continue
            wav = audio_root / system_id / f"{text_id}.wav"
            if wav.is_file():
                jobs.append((clip_id, system_id, text_id, wav))
    if args.limit:
        jobs = jobs[: args.limit]

    print(f"clips to score: {len(jobs)} (already done: {len(already)})", flush=True)
    if not jobs:
        return 0

    import torch

    from f5_tts.infer.asr_backends import transcribe
    from tools.eval_protocol import cer, normalize_kana

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}", flush=True)

    started = time.monotonic()
    failures = 0
    with out_path.open("a", encoding="utf-8") as handle:
        for index, (clip_id, system_id, text_id, wav) in enumerate(jobs, start=1):
            record = {
                "clip_id": clip_id,
                "system_id": system_id,
                "text_id": text_id,
                "split": text_split.get(text_id),
                "wav": str(wav),
            }
            try:
                hypothesis = transcribe(str(wav), lang="ja", device=device).strip()
                reference = text_kana[text_id]
                # 与 eval_protocol.score_audio 完全一致：假设转写需 kanji->kana，参考已是 kana
                record["asr_text"] = hypothesis
                record["cer"] = cer(normalize_kana(hypothesis), normalize_kana(reference, convert=False))
                record["status"] = "ok"
            except Exception as error:  # noqa: BLE001 - 单条失败不该终止整批
                record["status"] = "error"
                record["error_type"] = type(error).__name__
                record["error"] = str(error)[:300]
                failures += 1
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            if index % 8 == 0 or index == len(jobs):
                rate = index / max(time.monotonic() - started, 1e-9)
                remaining = (len(jobs) - index) / rate if rate else 0
                print(
                    f"  {index}/{len(jobs)}  {rate:.2f} clip/s  eta {remaining / 60:.1f} min  failures {failures}",
                    flush=True,
                )

    print(f"done: {len(jobs) - failures} scored, {failures} failed -> {out_path}")
    return 1 if failures == len(jobs) else 0


if __name__ == "__main__":
    raise SystemExit(main())
