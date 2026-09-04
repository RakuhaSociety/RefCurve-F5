"""Prepare a Japanese CSV/wav dataset for fine-tuning the Jmica F5-TTS model.

CSV format (header required, ``|`` delimiter)::

    audio_file|text
    /absolute/path/to/001.wav|私は元気です。

Text is converted to kana with the repository Japanese frontend. Rows containing
characters absent from the exact Jmica vocabulary are skipped rather than being
silently mapped to the unknown/space token.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import soundfile as sf
from datasets.arrow_writer import ArrowWriter
from tqdm import tqdm

from f5_tts.infer.ja_frontend import ja_to_kana


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_VOCAB_PATH = REPO_ROOT / "ckpts" / "F5TTS_JA_Jmica" / "vocab_japanese.txt"


def load_vocab(vocab_path: str | Path) -> tuple[list[str], set[str]]:
    """Load the Jmica vocabulary without sorting or otherwise changing its indices."""
    path = Path(vocab_path).expanduser().resolve()
    entries = path.read_text(encoding="utf-8").splitlines()
    if not entries or entries[0] != " ":
        raise ValueError(f"Jmica vocab must have space at index 0: {path}")
    return entries, set(entries)


def kana_and_oov(text: str, vocab: set[str]) -> tuple[str, list[str]]:
    """Convert text to Jmica-compatible kana and return any uncovered characters."""
    kana = ja_to_kana(text.strip())
    oov = sorted({char for char in kana if char not in vocab})
    return kana, oov


def read_csv_rows(csv_path: str | Path) -> list[tuple[int, Path, str]]:
    path = Path(csv_path).expanduser().resolve()
    rows = []
    with path.open(mode="r", newline="", encoding="utf-8-sig") as csvfile:
        reader = csv.reader(csvfile, delimiter="|")
        header = next(reader, None)
        if header is None or len(header) < 2 or [part.strip() for part in header[:2]] != ["audio_file", "text"]:
            raise ValueError("CSV header must be: audio_file|text")
        for row_number, row in enumerate(reader, start=2):
            if len(row) < 2 or not row[0].strip() or not row[1].strip():
                continue
            audio_path = Path(row[0].strip()).expanduser()
            if not audio_path.is_absolute():
                raise ValueError(f"audio_file must be an absolute path (row {row_number}): {row[0]}")
            rows.append((row_number, audio_path, row[1].strip()))
    return rows


def prepare_rows(csv_path: str | Path, vocab_path: str | Path) -> tuple[list[dict], list[float], list[dict]]:
    _, vocab = load_vocab(vocab_path)
    prepared = []
    durations = []
    rejected = []

    for row_number, audio_path, text in tqdm(read_csv_rows(csv_path), desc="Preparing Japanese rows"):
        base_error = {"row": row_number, "audio_path": audio_path.as_posix(), "original_text": text}
        if not audio_path.is_file():
            rejected.append({**base_error, "reason": "missing audio"})
            continue
        try:
            duration = sf.info(audio_path).duration
        except Exception as error:
            rejected.append({**base_error, "reason": f"invalid audio: {error}"})
            continue
        if not 0.3 <= duration <= 30:
            rejected.append({**base_error, "reason": f"duration {duration:.3f}s outside [0.3, 30]"})
            continue

        kana, oov = kana_and_oov(text, vocab)
        if not kana:
            rejected.append({**base_error, "reason": "empty kana text"})
            continue
        if oov:
            rejected.append(
                {**base_error, "reason": "OOV", "characters": oov, "kana_text": kana}
            )
            continue

        prepared.append({"audio_path": audio_path.resolve().as_posix(), "text": kana, "duration": duration})
        durations.append(duration)

    return prepared, durations, rejected


def save_dataset(out_dir: str | Path, vocab_path: str | Path, rows: list[dict], durations: list[float], rejected: list[dict]):
    if not rows:
        raise RuntimeError("No valid Japanese rows remained after audio and OOV filtering.")

    output = Path(out_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    with ArrowWriter(path=(output / "raw.arrow").as_posix()) as writer:
        for row in rows:
            writer.write(row)
        writer.finalize()
    (output / "duration.json").write_text(
        json.dumps({"duration": durations}, ensure_ascii=False), encoding="utf-8"
    )
    (output / "rejected.json").write_text(json.dumps(rejected, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copyfile(Path(vocab_path).expanduser().resolve(), output / "vocab.txt")

    print(f"Saved {len(rows)} samples ({sum(durations) / 3600:.2f} hours) to {output}")
    print(f"Duration range: {min(durations):.3f}s - {max(durations):.3f}s")
    print(f"Rejected {len(rejected)} rows; details: {output / 'rejected.json'}")

    oov_rows = [row for row in rejected if row.get("reason") == "OOV"]
    if oov_rows:
        raise RuntimeError(
            f"Found {len(oov_rows)} OOV rows. Fix transcripts or the frontend before training; "
            f"see {output / 'rejected.json'}"
        )


def get_args():
    parser = argparse.ArgumentParser(description="Prepare kana/OOV-filtered data for Jmica Japanese fine-tuning.")
    parser.add_argument("csv_path", help="CSV with header audio_file|text and absolute audio paths")
    parser.add_argument("out_dir", help="Output CustomDatasetPath directory")
    parser.add_argument("--vocab", default=str(DEFAULT_VOCAB_PATH), help="Exact Jmica vocab_japanese.txt")
    return parser.parse_args()


def cli():
    args = get_args()
    rows, durations, rejected = prepare_rows(args.csv_path, args.vocab)
    save_dataset(args.out_dir, args.vocab, rows, durations, rejected)


if __name__ == "__main__":
    cli()
