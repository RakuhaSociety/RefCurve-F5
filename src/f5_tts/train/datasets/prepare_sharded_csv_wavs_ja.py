"""Stream a Japanese ``audio_file|text`` CSV into bounded Arrow shards.

The output directory is published transactionally: every generated file is first
written with a ``.partial`` suffix and moved into place with :func:`os.replace`.
``READY`` is written last and is therefore the commit marker for the dataset.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import unicodedata
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import IO, Any, Iterator

import numpy as np
import soundfile as sf
from datasets.arrow_writer import ArrowWriter

from f5_tts.model.sharded_dataset import file_sha256
from f5_tts.model.vocab_contract import token_sequence_sha256, validate_vocabulary_contract
from f5_tts.train.datasets.visualnovel_package import source_rejection_reason


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_VOCAB_PATH = REPO_ROOT / "src" / "f5_tts" / "configs" / "vocab" / "F5TTS_v1_JA_Base" / "vocab.txt"
DEFAULT_CONTRACT_PATH = (
    REPO_ROOT / "src" / "f5_tts" / "configs" / "vocab" / "F5TTS_v1_JA_Base" / "contract.json"
)
GENERATED_NAMES = {
    "frames.u32",
    "manifest.json",
    "provenance.json",
    "quality.json",
    "eval.jsonl",
    "rejected.jsonl",
    "stats.json",
    "READY",
}
FROZEN_SOURCE_SCHEMAS = {"visualnovel-package-v1", "visualnovel-package-v2"}
FROZEN_SIDECAR_SCHEMAS = {"kana-sidecar-v1", "kana-sidecar-v2"}
ja_to_kana = None  # CSV compatibility hook; frozen-sidecar mode never imports or calls the frontend.


def load_vocab(vocab_path: str | Path, contract_path: str | Path) -> tuple[list[str], set[str], object]:
    entries, contract = validate_vocabulary_contract(vocab_path, contract_path)
    return entries, set(entries), contract


def kana_and_oov(text: str, vocab: set[str]) -> tuple[str, list[str]]:
    global ja_to_kana
    if ja_to_kana is None:
        from f5_tts.infer.ja_frontend import ja_to_kana as frontend

        ja_to_kana = frontend
    kana = ja_to_kana(text.strip())
    return kana, sorted({char for char in kana if char not in vocab})


def _partial_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.partial")


def _write_json_atomic(path: Path, value: object, *, indent: int | None = None) -> None:
    partial = _partial_path(path)
    with partial.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=indent)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)


def _write_text_atomic(path: Path, text: str) -> None:
    partial = _partial_path(path)
    with partial.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)


def _partial_entries(output: Path) -> list[Path]:
    if not output.exists():
        return []
    return sorted(path for path in output.rglob("*.partial") if path.is_file())


def _nonpartial_generated_entries(output: Path) -> list[Path]:
    if not output.exists():
        return []
    entries = [output / name for name in GENERATED_NAMES if (output / name).exists()]
    shards = output / "shards"
    if shards.exists() and any(shards.iterdir()):
        entries.append(shards)
    return sorted(entries)


def _prepare_output(output: Path, *, clean_partials: bool) -> bool:
    """Return ``False`` when an existing READY dataset makes the run a no-op."""
    ready = output / "READY"
    if ready.is_file():
        return False

    partials = _partial_entries(output)
    if partials and not clean_partials:
        names = ", ".join(path.relative_to(output).as_posix() for path in partials[:5])
        raise RuntimeError(f"Partial output exists ({names}); rerun with --clean-partials to remove it safely.")
    for partial in partials:
        partial.unlink()

    existing = _nonpartial_generated_entries(output)
    if existing:
        names = ", ".join(path.relative_to(output).as_posix() for path in existing)
        raise RuntimeError(f"Output has generated files but no READY marker; refusing to overwrite: {names}")

    output.mkdir(parents=True, exist_ok=True)
    (output / "shards").mkdir(exist_ok=True)
    return True


def _reject(row_number: int, row: list[str], reason: str, **details: object) -> dict:
    result: dict[str, object] = {"row": row_number, "reason": reason}
    if row:
        result["audio_path"] = row[0].strip()
    if len(row) > 1:
        result["original_text"] = row[1].strip()
    result.update(details)
    return result


@contextmanager
def _arrow_shard(path: Path) -> Iterator[ArrowWriter]:
    partial = _partial_path(path)
    try:
        with ArrowWriter(path=partial.as_posix()) as writer:
            yield writer
            writer.finalize()
        os.replace(partial, path)
    except BaseException:
        # Keep the partial as an explicit incomplete-run marker.
        raise


def _load_json_object(path: Path, label: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label}: {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or value != value.lower():
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest") from error
    return value


def _contained_package_path(package_dir: Path, relative: object, label: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError(f"{label} must be a non-empty POSIX relative path")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or any(part in ("", ".", "..") for part in pure.parts):
        raise ValueError(f"{label} is not contained in package directory")
    path = package_dir.joinpath(*pure.parts).resolve()
    if not path.is_relative_to(package_dir) or not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} is not a contained regular file")
    return path


def _validate_frozen_kana(kana: object, vocab: set[str], label: str) -> str:
    if not isinstance(kana, str) or not kana:
        raise ValueError(f"{label} must be non-empty")
    han = [char for char in kana if "CJK UNIFIED IDEOGRAPH" in unicodedata.name(char, "")]
    controls = [char for char in kana if unicodedata.category(char).startswith("C")]
    oov = sorted({char for char in kana if char not in vocab})
    if han:
        raise ValueError(f"{label} contains Han characters")
    if controls:
        raise ValueError(f"{label} contains control characters")
    if oov:
        raise ValueError(f"{label} contains OOV characters: {oov}")
    return kana


def _audio_quality(path: Path) -> tuple[dict[str, Any], np.ndarray]:
    try:
        with sf.SoundFile(path) as handle:
            audio = handle.read(dtype="float32", always_2d=True)
            sample_rate, channels, frames = int(handle.samplerate), int(handle.channels), int(handle.frames)
    except Exception as error:
        raise ValueError(f"audio cannot be fully decoded: {path}") from error
    if frames <= 0 or audio.shape != (frames, channels) or not np.isfinite(audio).all():
        raise ValueError(f"audio is empty, truncated, or non-finite: {path}")
    absolute = np.abs(audio.astype(np.float64, copy=False))
    metrics = {
        "frames": frames,
        "sample_rate": sample_rate,
        "channels": channels,
        "duration_seconds": frames / sample_rate,
        "finite": True,
        "peak": float(absolute.max()),
        "rms": float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))),
        "clipping_fraction": float(np.mean(absolute >= 0.999)),
        "silence_fraction": float(np.mean(absolute <= 1e-4)),
    }
    return metrics, audio


def _read_sidecar_rows(path: Path) -> tuple[list[dict], str]:
    content = path.read_bytes()
    rows = []
    for number, line in enumerate(content.splitlines(), start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid sidecar JSONL row {number}") from error
        if not isinstance(row, dict):
            raise ValueError(f"sidecar row {number} must be an object")
        rows.append(row)
    return rows, __import__("hashlib").sha256(content).hexdigest()


def prepare_frozen_sidecar_dataset(
    package_dir: str | Path,
    sidecar_meta_path: str | Path,
    sidecar_jsonl_path: str | Path,
    out_dir: str | Path,
    vocab_path: str | Path,
    contract_path: str | Path,
    *,
    shard_size: int = 32,
    sample_rate: int = 24_000,
    hop_length: int = 256,
    clean_partials: bool = False,
) -> dict:
    """Audit a frozen VisualNovel package/sidecar pair and publish bounded shards."""
    if shard_size <= 0:
        raise ValueError("shard_size must be positive")
    package = Path(package_dir).expanduser().resolve()
    output = Path(out_dir).expanduser().resolve()
    meta_path = Path(sidecar_meta_path).expanduser().resolve()
    rows_path = Path(sidecar_jsonl_path).expanduser().resolve()
    vocab_source = Path(vocab_path).expanduser().resolve()
    contract_source = Path(contract_path).expanduser().resolve()
    if not package.is_dir() or not (package / "READY").is_file():
        raise ValueError("package directory must contain READY")

    entries, vocab, contract = load_vocab(vocab_source, contract_source)
    source_meta_path = package / "metadata.json"
    source = _load_json_object(source_meta_path, "package metadata")
    sidecar = _load_json_object(meta_path, "sidecar metadata")
    rows, rows_hash = _read_sidecar_rows(rows_path)
    if set(sidecar) != {"schema_version", "source", "vocabulary", "frontend_provenance", "rows"}:
        raise ValueError("sidecar metadata does not match kana-sidecar-v1")
    sidecar_schema = sidecar.get("schema_version")
    if source.get("schema_version") not in FROZEN_SOURCE_SCHEMAS or sidecar_schema not in FROZEN_SIDECAR_SCHEMAS:
        raise ValueError("package or sidecar schema version mismatch")
    samples = source.get("samples")
    package_schema = source.get("schema_version")
    package_schema_versions = {"visualnovel-package-v1", "visualnovel-package-v2"}
    required_source_keys = {
        "schema_version", "package_id", "package_sha256", "archive_bytes", "archive_root", "index_sha256",
        "index_bytes", "index_schema", "unindexed_audio", "source_id", "record_count", "speakers", "samples",
    }
    if package_schema not in package_schema_versions or set(source) != required_source_keys:
        raise ValueError("package metadata does not match a supported VisualNovel schema")
    if not isinstance(samples, list) or source.get("record_count") != len(samples):
        raise ValueError("package record_count must match a samples array")
    if not samples:
        raise ValueError("frozen package must contain at least one sample")
    base_sample_keys = {
        "sample_id", "row_index", "speaker_id", "speaker", "voice", "text", "text_sha256",
        "relative_audio_path", "audio",
    }
    optional_sample_keys = {"indexed_audio_path"}
    if package_schema == "visualnovel-package-v2":
        base_sample_keys |= {"source_status", "source_rejection_reason"}
    for number, sample in enumerate(samples):
        if (
            not isinstance(sample, dict)
            or not base_sample_keys <= set(sample)
            or set(sample) - base_sample_keys - optional_sample_keys
        ):
            raise ValueError(f"package sample {number} does not match {package_schema}")
        if hashlib.sha256(sample["text"].encode("utf-8")).hexdigest() != sample["text_sha256"]:
            raise ValueError(f"package text SHA-256 mismatch: {sample.get('sample_id')}")
        if package_schema == "visualnovel-package-v2":
            reason = source_rejection_reason(
                index_schema=source["index_schema"],
                speaker=sample["speaker"],
                indexed_audio_path=sample.get("indexed_audio_path", sample["relative_audio_path"]),
                text=sample["text"],
            )
            status, declared_reason = sample["source_status"], sample["source_rejection_reason"]
            if status == "accepted" and (reason is not None or declared_reason is not None):
                raise ValueError(f"package source status is invalid: {sample.get('sample_id')}")
            if status == "rejected" and (reason is None or declared_reason != reason):
                raise ValueError(f"package source status is invalid: {sample.get('sample_id')}")

    if source.get("index_schema") not in {"Voice", "FilePath"} or not isinstance(source.get("unindexed_audio"), list):
        raise ValueError("package index audit metadata is invalid")
    side_rows_meta = sidecar.get("rows")
    if sidecar_schema == "kana-sidecar-v1":
        required_rows_meta = {
            "source_count", "accepted_count", "rejected_count", "train_count", "eval_count",
            "eval_target_count", "eval_split_policy", "eval_shortfall", "jsonl_sha256",
        }
    else:
        required_rows_meta = {"source_count", "accepted_count", "rejected_count", "jsonl_sha256"}
    if not isinstance(side_rows_meta, dict) or set(side_rows_meta) != required_rows_meta:
        raise ValueError(f"sidecar rows metadata does not match {sidecar_schema}")
    record_count = len(samples)
    if len(rows) != record_count or side_rows_meta["source_count"] != record_count:
        raise ValueError("frozen sidecar must preserve the package row set")

    package_hash = _require_sha256(source.get("package_sha256"), "package_sha256")
    if (package / "READY").read_text(encoding="ascii").strip() != package_hash:
        raise ValueError("package READY hash does not match metadata")
    index_path = package / "index.json"
    if file_sha256(index_path) != _require_sha256(source.get("index_sha256"), "index_sha256"):
        raise ValueError("index.json SHA-256 mismatch")
    if rows_hash != side_rows_meta["jsonl_sha256"]:
        raise ValueError("sidecar rows SHA-256 mismatch")

    side_source = sidecar.get("source")
    source_identity_keys = (
        "schema_version",
        "package_id",
        "package_sha256",
        "archive_bytes",
        "archive_root",
        "index_sha256",
        "index_bytes",
        "index_schema",
        "unindexed_audio",
        "source_id",
        "record_count",
    )
    expected_side_source_keys = set(source_identity_keys)
    if sidecar_schema == "kana-sidecar-v2":
        expected_side_source_keys.add("metadata_sha256")
    if not isinstance(side_source, dict) or set(side_source) != expected_side_source_keys:
        raise ValueError(f"sidecar source identity does not match {sidecar_schema}")
    metadata_hash = file_sha256(source_meta_path)
    if sidecar_schema == "kana-sidecar-v2" and side_source.get("metadata_sha256") != metadata_hash:
        raise ValueError("sidecar source identity mismatch: metadata_sha256")
    for key in source_identity_keys:
        if side_source.get(key) != source.get(key):
            raise ValueError(f"sidecar source identity mismatch: {key}")
    side_vocab = sidecar.get("vocabulary")
    expected_vocab = {
        "identity": contract.identity,
        "token_sequence_sha256": token_sequence_sha256(entries),
        "contract_sha256": file_sha256(contract_source),
        "vocab_file_sha256": file_sha256(vocab_source),
    }
    if not isinstance(side_vocab, dict) or any(side_vocab.get(k) != v for k, v in expected_vocab.items()):
        raise ValueError("sidecar vocabulary identity or hash mismatch")

    source_by_id = {sample.get("sample_id"): sample for sample in samples}
    if None in source_by_id or len(source_by_id) != record_count:
        raise ValueError("package sample IDs are missing or duplicated")
    if {row.get("sample_id") for row in rows} != set(source_by_id):
        raise ValueError("sidecar and package 107-row sample sets differ")

    quality_rows = []
    audited: dict[str, tuple[Path, dict[str, Any]]] = {}
    sample_rate_distribution: dict[str, int] = {}
    channel_distribution: dict[str, int] = {}
    for sample in samples:
        sample_id = sample["sample_id"]
        audio_path = _contained_package_path(package, sample.get("relative_audio_path"), f"sample {sample_id} audio path")
        audio_meta = sample.get("audio")
        if not isinstance(audio_meta, dict) or file_sha256(audio_path) != _require_sha256(audio_meta.get("sha256"), "audio.sha256"):
            raise ValueError(f"audio SHA-256 mismatch: {sample_id}")
        metrics, _ = _audio_quality(audio_path)
        for key in ("frames", "sample_rate", "channels"):
            if audio_meta.get(key) != metrics[key]:
                raise ValueError(f"audio metadata mismatch for {sample_id}: {key}")
        sample_rate_distribution[str(metrics["sample_rate"])] = sample_rate_distribution.get(str(metrics["sample_rate"]), 0) + 1
        channel_distribution[str(metrics["channels"])] = channel_distribution.get(str(metrics["channels"]), 0) + 1
        quality_rows.append({"sample_id": sample_id, "relative_audio_path": sample["relative_audio_path"], **metrics})
        audited[sample_id] = (audio_path, metrics)

    accepted_rows = []
    rejected_rows = []
    for position, row in enumerate(rows):
        sample_id = row.get("sample_id")
        sample = source_by_id.get(sample_id)
        if sample is None or row.get("source_order") != position or row.get("row_index") != sample.get("row_index"):
            raise ValueError("sidecar row ordering/index identity mismatch")
        if row.get("source_text") != sample.get("text"):
            raise ValueError(f"sidecar source text mismatch: {sample_id}")
        if row.get("relative_audio_path") != sample.get("relative_audio_path") or row.get("audio") != sample.get("audio"):
            raise ValueError(f"sidecar audio identity mismatch: {sample_id}")
        status = row.get("status")
        if status == "accepted":
            kana = _validate_frozen_kana(row.get("kana_text"), vocab, f"sidecar kana {sample_id}")
            accepted_rows.append((row, sample, kana))
        elif status == "rejected":
            if row.get("kana_text") is not None:
                raise ValueError("rejected sidecar rows must not contain kana_text")
            rejected_rows.append(row)
        else:
            raise ValueError("sidecar status must be accepted or rejected")

    if side_rows_meta["accepted_count"] != len(accepted_rows) or side_rows_meta["rejected_count"] != len(rejected_rows):
        raise ValueError("sidecar accepted/rejected counts do not match rows")
    if record_count != len(accepted_rows) + len(rejected_rows):
        raise AssertionError("frozen sidecar row conservation failed")
    if not accepted_rows:
        raise RuntimeError("frozen sidecar contains no accepted rows")
    selected_rows = accepted_rows
    eval_rows = []
    if sidecar_schema == "kana-sidecar-v1":
        selected_rows = [item for item in accepted_rows if item[0].get("split") == "train"]
        eval_rows = [item[0] for item in accepted_rows if item[0].get("split") == "eval"]
        if any(item[0].get("split") not in {"train", "eval"} for item in accepted_rows):
            raise ValueError("v1 accepted rows must have train/eval split")
    if not selected_rows:
        raise RuntimeError("frozen sidecar contains no rows selected for Arrow output")

    input_identity = {
        "schema_version": "frozen-sidecar-input-v2",
        "package_id": source["package_id"],
        "package_sha256": package_hash,
        "metadata_sha256": metadata_hash,
        "index_sha256": source["index_sha256"],
        "index_schema": source["index_schema"],
        "unindexed_audio": source["unindexed_audio"],
        "sidecar_schema": sidecar_schema,
        "sidecar_meta_sha256": file_sha256(meta_path),
        "sidecar_rows_sha256": rows_hash,
        "vocabulary": expected_vocab,
        "frontend_provenance": sidecar.get("frontend_provenance"),
        "policy": {
            "shard_size": shard_size,
            "sample_rate": sample_rate,
            "hop_length": hop_length,
        },
    }
    identity_sha256 = hashlib.sha256(
        (json.dumps(input_identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    ).hexdigest()
    ready_path = output / "READY"
    if ready_path.is_file():
        try:
            committed = json.loads(ready_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("existing READY identity is invalid") from error
        if committed != {"identity_sha256": identity_sha256, "input_identity": input_identity}:
            raise ValueError("existing READY input identity does not match current inputs")
        return json.loads((output / "stats.json").read_text(encoding="utf-8"))
    if not _prepare_output(output, clean_partials=clean_partials):
        raise AssertionError("READY appeared while preparing output")

    frames_final = output / "frames.u32"
    rejected_final = output / "rejected.jsonl"
    eval_final = output / "eval.jsonl"
    frames_partial = _partial_path(frames_final)
    rejected_partial = _partial_path(rejected_final)
    eval_partial = _partial_path(eval_final)
    shards = []
    total_duration = 0.0
    audio_root_id = source["package_id"]
    with frames_partial.open("wb") as frame_file:
        for start in range(0, len(selected_rows), shard_size):
            shard_rows = selected_rows[start : start + shard_size]
            shard_path = output / "shards" / f"{len(shards):06d}.arrow"
            with _arrow_shard(shard_path) as writer:
                for row, sample, kana in shard_rows:
                    audio_path, metrics = audited[row["sample_id"]]
                    duration = metrics["duration_seconds"]
                    writer.write(
                        {
                            "audio_root_id": audio_root_id,
                            "relative_audio_path": row["relative_audio_path"],
                            "text": kana,
                            "duration": duration,
                            "sample_id": row["sample_id"],
                            "source_order": row["source_order"],
                            "row_index": row["row_index"],
                            "speaker_id": row["speaker_id"],
                            "speaker": row["speaker"],
                            "voice": row["voice"],
                            "source_text": row["source_text"],
                            "audio_sha256": sample["audio"]["sha256"],
                            "sidecar_rows_sha256": rows_hash,
                            "package_sha256": package_hash,
                        }
                    )
                    np.asarray([max(1, int(duration * sample_rate / hop_length))], dtype="<u4").tofile(frame_file)
                    total_duration += duration
            shards.append({"path": shard_path.relative_to(output).as_posix(), "rows": len(shard_rows), "sha256": file_sha256(shard_path)})
        frame_file.flush()
        os.fsync(frame_file.fileno())
    with rejected_partial.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rejected_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    with eval_partial.open("w", encoding="utf-8", newline="\n") as handle:
        for row in eval_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(frames_partial, frames_final)
    os.replace(rejected_partial, rejected_final)
    os.replace(eval_partial, eval_final)

    manifest = {
        "version": 1,
        "total_rows": len(selected_rows),
        "shards": shards,
        "frame_index": {"path": "frames.u32", "count": len(selected_rows), "dtype": "uint32", "sha256": file_sha256(frames_final)},
        "audio": {"sample_rate": sample_rate, "hop_length": hop_length},
        "vocabulary": {"identity": contract.identity, "token_sequence_sha256": contract.token_sequence_sha256},
    }
    stats = {
        "input_rows": record_count,
        "accepted_rows": len(selected_rows),
        "rejected_rows": len(rejected_rows),
        "eval_rows": len(eval_rows),
        "total_duration_seconds": total_duration,
        "shard_count": len(shards),
        "sample_rate": sample_rate,
        "hop_length": hop_length,
        "vocabulary_identity": contract.identity,
        "vocabulary_token_sequence_sha256": contract.token_sequence_sha256,
        "rejection_reasons": {"sidecar rejected": len(rejected_rows)},
    }
    provenance = {
        "mode": "frozen-sidecar",
        "source_schema": package_schema,
        "sidecar_schema": sidecar_schema,
        "package_id": source["package_id"],
        "source_id": source["source_id"],
        "package_sha256": package_hash,
        "index_sha256": source["index_sha256"],
        "index_schema": source["index_schema"],
        "unindexed_audio": source["unindexed_audio"],
        "metadata_sha256": file_sha256(source_meta_path),
        "sidecar_meta_sha256": file_sha256(meta_path),
        "sidecar_rows_sha256": rows_hash,
        "vocabulary": expected_vocab,
        "source_rows": record_count,
        "output_rows": len(selected_rows),
        "eval_rows": len(eval_rows),
        "rejected_rows": len(rejected_rows),
        "audio_root_id": audio_root_id,
    }
    quality = {
        "schema_version": "visualnovel-audio-quality-v1",
        "audited_rows": len(quality_rows),
        "all_finite": all(row["finite"] for row in quality_rows),
        "sample_rate_distribution": sample_rate_distribution,
        "channel_distribution": channel_distribution,
        "rows": quality_rows,
    }
    _write_json_atomic(output / "manifest.json", manifest, indent=2)
    _write_json_atomic(output / "stats.json", stats, indent=2)
    _write_json_atomic(output / "provenance.json", provenance, indent=2)
    _write_json_atomic(output / "quality.json", quality, indent=2)
    _write_text_atomic(
        output / "READY",
        json.dumps(
            {"identity_sha256": identity_sha256, "input_identity": input_identity},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
    )
    return stats


def prepare_sharded_dataset(
    csv_path: str | Path,
    out_dir: str | Path,
    vocab_path: str | Path,
    contract_path: str | Path,
    *,
    shard_size: int = 10_000,
    sample_rate: int = 24_000,
    hop_length: int = 256,
    min_duration: float = 0.3,
    max_duration: float = 30.0,
    clean_partials: bool = False,
) -> dict:
    """Prepare shards without retaining input rows or frame lengths in memory."""
    if shard_size <= 0:
        raise ValueError("shard_size must be positive")
    if sample_rate <= 0 or hop_length <= 0:
        raise ValueError("sample_rate and hop_length must be positive")

    source = Path(csv_path).expanduser().resolve()
    output = Path(out_dir).expanduser().resolve()
    vocab_source = Path(vocab_path).expanduser().resolve()
    contract_source = Path(contract_path).expanduser().resolve()
    if not _prepare_output(output, clean_partials=clean_partials):
        return json.loads((output / "stats.json").read_text(encoding="utf-8"))

    _, vocab, vocab_contract = load_vocab(vocab_source, contract_source)
    frames_final = output / "frames.u32"
    frames_partial = _partial_path(frames_final)
    rejected_final = output / "rejected.jsonl"
    rejected_partial = _partial_path(rejected_final)

    total_rows = accepted_rows = rejected_rows = 0
    total_duration = 0.0
    rejection_reasons: dict[str, int] = {}
    shards: list[dict] = []
    shard_writer_cm = None
    shard_writer: ArrowWriter | None = None
    rows_in_shard = 0
    shard_path: Path | None = None

    def write_rejection(handle: IO[str], rejection: dict) -> None:
        nonlocal rejected_rows
        rejected_rows += 1
        reason = str(rejection["reason"])
        rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
        handle.write(json.dumps(rejection, ensure_ascii=False) + "\n")

    try:
        with (
            source.open("r", newline="", encoding="utf-8-sig") as csv_file,
            frames_partial.open("wb") as frame_file,
            rejected_partial.open("w", encoding="utf-8", newline="\n") as rejected_file,
        ):
            reader = csv.reader(csv_file, delimiter="|")
            header = next(reader, None)
            if header is None or len(header) < 2 or [part.strip() for part in header[:2]] != ["audio_file", "text"]:
                raise ValueError("CSV header must be: audio_file|text")

            for row_number, row in enumerate(reader, start=2):
                total_rows += 1
                if len(row) < 2 or not row[0].strip() or not row[1].strip():
                    write_rejection(rejected_file, _reject(row_number, row, "malformed row"))
                    continue

                audio_path = Path(row[0].strip()).expanduser()
                text = row[1].strip()
                if not audio_path.is_absolute():
                    write_rejection(rejected_file, _reject(row_number, row, "audio path is not absolute"))
                    continue
                if not audio_path.is_file():
                    write_rejection(rejected_file, _reject(row_number, row, "missing audio"))
                    continue
                try:
                    duration = float(sf.info(audio_path).duration)
                except Exception as error:
                    write_rejection(rejected_file, _reject(row_number, row, "invalid audio", error=str(error)))
                    continue
                if not min_duration <= duration <= max_duration:
                    write_rejection(
                        rejected_file,
                        _reject(row_number, row, "duration out of range", duration=duration),
                    )
                    continue

                kana, oov = kana_and_oov(text, vocab)
                if not kana:
                    write_rejection(rejected_file, _reject(row_number, row, "empty kana text"))
                    continue
                if oov:
                    write_rejection(
                        rejected_file,
                        _reject(row_number, row, "OOV", characters=oov, kana_text=kana),
                    )
                    continue

                if shard_writer is None:
                    shard_path = output / "shards" / f"{len(shards):06d}.arrow"
                    shard_writer_cm = _arrow_shard(shard_path)
                    shard_writer = shard_writer_cm.__enter__()
                    rows_in_shard = 0

                resolved_audio = audio_path.resolve().as_posix()
                shard_writer.write({"audio_path": resolved_audio, "text": kana, "duration": duration})
                frame_count = max(1, int(duration * sample_rate / hop_length))
                np.asarray([frame_count], dtype="<u4").tofile(frame_file)
                rows_in_shard += 1
                accepted_rows += 1
                total_duration += duration

                if rows_in_shard == shard_size:
                    assert shard_writer_cm is not None and shard_path is not None
                    shard_writer_cm.__exit__(None, None, None)
                    shards.append(
                        {
                            "path": shard_path.relative_to(output).as_posix(),
                            "rows": rows_in_shard,
                            "sha256": file_sha256(shard_path),
                        }
                    )
                    shard_writer_cm = None
                    shard_writer = None

            if shard_writer is not None:
                assert shard_writer_cm is not None and shard_path is not None
                shard_writer_cm.__exit__(None, None, None)
                shards.append(
                    {
                        "path": shard_path.relative_to(output).as_posix(),
                        "rows": rows_in_shard,
                        "sha256": file_sha256(shard_path),
                    }
                )
                shard_writer_cm = None
                shard_writer = None

            frame_file.flush()
            os.fsync(frame_file.fileno())
            rejected_file.flush()
            os.fsync(rejected_file.fileno())
    except BaseException as error:
        if shard_writer_cm is not None:
            shard_writer_cm.__exit__(type(error), error, error.__traceback__)
        raise

    if accepted_rows == 0:
        raise RuntimeError("No valid Japanese rows remained after audio and OOV filtering.")
    if total_rows != accepted_rows + rejected_rows:
        raise AssertionError("Input row conservation failed")

    os.replace(frames_partial, frames_final)
    os.replace(rejected_partial, rejected_final)

    manifest = {
        "version": 1,
        "total_rows": accepted_rows,
        "shards": shards,
        "frame_index": {
            "path": frames_final.relative_to(output).as_posix(),
            "count": accepted_rows,
            "dtype": "uint32",
            "sha256": file_sha256(frames_final),
        },
        "audio": {"sample_rate": sample_rate, "hop_length": hop_length},
        "vocabulary": {
            "identity": vocab_contract.identity,
            "token_sequence_sha256": vocab_contract.token_sequence_sha256,
        },
    }
    stats = {
        "input_rows": total_rows,
        "accepted_rows": accepted_rows,
        "rejected_rows": rejected_rows,
        "total_duration_seconds": total_duration,
        "shard_count": len(shards),
        "sample_rate": sample_rate,
        "hop_length": hop_length,
        "vocabulary_identity": vocab_contract.identity,
        "vocabulary_token_sequence_sha256": vocab_contract.token_sequence_sha256,
        "rejection_reasons": rejection_reasons,
    }
    _write_json_atomic(output / "manifest.json", manifest, indent=2)
    _write_json_atomic(output / "stats.json", stats, indent=2)
    _write_text_atomic(output / "READY", "ready\n")
    return stats


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream Japanese data into training-ready Arrow shards.")
    parser.add_argument("source", help="CSV path, or long-lived package directory in frozen-sidecar mode")
    parser.add_argument("out_dir", help="Output sharded dataset directory")
    parser.add_argument("--mode", choices=("csv", "frozen-sidecar"), default="csv")
    parser.add_argument("--sidecar-meta", help="source_kana.meta.json (required in frozen-sidecar mode)")
    parser.add_argument("--sidecar-jsonl", help="source_kana.jsonl (required in frozen-sidecar mode)")
    parser.add_argument("--vocab", default=str(DEFAULT_VOCAB_PATH), help="Exact Japanese vocabulary")
    parser.add_argument("--contract", default=str(DEFAULT_CONTRACT_PATH), help="Strict vocabulary contract JSON")
    parser.add_argument("--shard-size", type=int, default=None, help="CSV default 10000; frozen-sidecar requires 32")
    parser.add_argument("--sample-rate", type=int, default=24_000)
    parser.add_argument("--hop-length", type=int, default=256)
    parser.add_argument("--min-duration", type=float, default=0.3)
    parser.add_argument("--max-duration", type=float, default=30.0)
    parser.add_argument("--clean-partials", action="store_true", help="Delete stale *.partial files before a new run")
    return parser.parse_args()


def cli() -> None:
    args = get_args()
    if args.mode == "frozen-sidecar":
        if not args.sidecar_meta or not args.sidecar_jsonl:
            raise SystemExit("--sidecar-meta and --sidecar-jsonl are required in frozen-sidecar mode")
        stats = prepare_frozen_sidecar_dataset(
            args.source,
            args.sidecar_meta,
            args.sidecar_jsonl,
            args.out_dir,
            args.vocab,
            args.contract,
            shard_size=args.shard_size or 32,
            sample_rate=args.sample_rate,
            hop_length=args.hop_length,
            clean_partials=args.clean_partials,
        )
    else:
        stats = prepare_sharded_dataset(
            args.source,
            args.out_dir,
            args.vocab,
            args.contract,
            shard_size=args.shard_size or 10_000,
            sample_rate=args.sample_rate,
            hop_length=args.hop_length,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
            clean_partials=args.clean_partials,
        )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    cli()
