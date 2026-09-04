"""Deterministically aggregate frozen VisualNovel packages without copying audio."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable

import numpy as np
from datasets.arrow_writer import ArrowWriter

from f5_tts.model.sharded_dataset import file_sha256
from f5_tts.train.datasets.visualnovel_package import source_rejection_reason

SCHEMA_VERSION = "visualnovel-aggregate-v2"
PACKAGE_SCHEMA_VERSIONS = {"visualnovel-package-v1", "visualnovel-package-v2"}
SIDECAR_SCHEMAS = {"kana-sidecar-v1", "kana-sidecar-v2"}
SIDECAR_METADATA_KEYS = {"schema_version", "source", "vocabulary", "frontend_provenance", "rows"}
SIDECAR_SOURCE_KEYS = {
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
}
SIDECAR_ROWS_V1_KEYS = {
    "source_count",
    "accepted_count",
    "rejected_count",
    "train_count",
    "eval_count",
    "eval_target_count",
    "eval_split_policy",
    "eval_shortfall",
    "jsonl_sha256",
}
SIDECAR_ROWS_V2_KEYS = {"source_count", "accepted_count", "rejected_count", "jsonl_sha256"}
EVAL_SPLIT_SIZE = 5
DEDUP_POLICY = "same audio+canonical kana; winner=min(wave,selection_order,package_id,row_index)"
CONFLICT_POLICY = "exclude_audio_object"
SPLIT_POLICY = "lowest sha256(audio_sha256,kana), accepted only, retain >=1 train row per speaker"


class AggregateError(ValueError):
    """Raised when aggregate inputs disagree or violate their frozen contracts."""


class AggregateConflictError(AggregateError):
    """Conflict error carrying the durable, unpublished conflict report."""

    def __init__(self, report_path: Path):
        super().__init__(f"same audio has conflicting canonical kana; report: {report_path}")
        self.report_path = report_path


@dataclass(frozen=True)
class AggregateContext:
    source_revision: str
    catalog_identity: str
    selection_fingerprint: str


@dataclass(frozen=True)
class AggregateInput:
    package_dir: str | Path
    sidecar_meta: str | Path
    sidecar_jsonl: str | Path
    quality: str | Path
    provenance: str | Path
    audio_root_id: str
    wave: str
    selection_order: int


def _canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _fingerprint(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _lower_hex(value: object, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _sha256(value: object) -> bool:
    return _lower_hex(value, 64)


def _commit_sha(value: object) -> bool:
    return _lower_hex(value, 40)


def _json_object(path: Path, label: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AggregateError(f"cannot read {label}: {path}") from error
    if not isinstance(value, dict):
        raise AggregateError(f"{label} must be a JSON object")
    return value


def _jsonl(path: Path) -> tuple[list[dict], str]:
    content = path.read_bytes()
    rows = []
    for number, line in enumerate(content.splitlines(), start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise AggregateError(f"invalid sidecar JSONL row {number}: {path}") from error
        if not isinstance(row, dict):
            raise AggregateError(f"sidecar JSONL row {number} must be an object")
        rows.append(row)
    return rows, hashlib.sha256(content).hexdigest()


def _contained_audio(package: Path, relative: object) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise AggregateError("relative_audio_path must be a non-empty POSIX path")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or any(part in ("", ".", "..") for part in pure.parts):
        raise AggregateError(f"unsafe relative_audio_path: {relative!r}")
    path = package.joinpath(*pure.parts).resolve()
    if not path.is_relative_to(package) or not path.is_file() or path.is_symlink():
        raise AggregateError(f"audio is not a contained regular file: {relative!r}")
    return path


def _load_input(spec: AggregateInput) -> tuple[dict, list[dict], dict, dict]:
    if not spec.wave or type(spec.selection_order) is not int or spec.selection_order < 0:
        raise AggregateError("wave and selection_order are required")
    package = Path(spec.package_dir).expanduser().resolve()
    ready_path = package / "READY"
    if not package.is_dir() or not ready_path.is_file():
        raise AggregateError(f"package must be committed with READY: {package}")
    metadata_path = package / "metadata.json"
    metadata = _json_object(metadata_path, "package metadata")
    package_sha = metadata.get("package_sha256")
    if not _sha256(package_sha) or ready_path.read_text(encoding="ascii").strip() != package_sha:
        raise AggregateError("package READY does not bind package_sha256")
    side_meta_path = Path(spec.sidecar_meta).expanduser().resolve()
    side_rows_path = Path(spec.sidecar_jsonl).expanduser().resolve()
    quality_path = Path(spec.quality).expanduser().resolve()
    provenance_path = Path(spec.provenance).expanduser().resolve()
    side_meta = _json_object(side_meta_path, "sidecar metadata")
    side_rows, rows_sha = _jsonl(side_rows_path)
    quality = _json_object(quality_path, "quality")
    _json_object(provenance_path, "provenance")
    if metadata.get("schema_version") not in PACKAGE_SCHEMA_VERSIONS:
        raise AggregateError("unsupported package schema")
    if metadata.get("index_schema") not in {"Voice", "FilePath"} or not isinstance(metadata.get("unindexed_audio"), list):
        raise AggregateError("package index audit metadata is invalid")
    sidecar_schema = side_meta.get("schema_version")
    if sidecar_schema not in SIDECAR_SCHEMAS or set(side_meta) != SIDECAR_METADATA_KEYS:
        raise AggregateError("unsupported or invalid sidecar metadata schema")
    source = side_meta.get("source")
    expected_source_keys = SIDECAR_SOURCE_KEYS | ({"metadata_sha256"} if sidecar_schema == "kana-sidecar-v2" else set())
    if not isinstance(source, dict) or set(source) != expected_source_keys:
        raise AggregateError(f"sidecar source metadata does not match {sidecar_schema}")
    source_keys = SIDECAR_SOURCE_KEYS if sidecar_schema == "kana-sidecar-v2" else {
        "schema_version", "package_id", "package_sha256", "source_id", "index_sha256"
    }
    if any(source.get(key) != metadata.get(key) for key in source_keys):
        raise AggregateError("sidecar source identity does not match package metadata")
    if sidecar_schema == "kana-sidecar-v2" and source.get("metadata_sha256") != file_sha256(metadata_path):
        raise AggregateError("sidecar source metadata_sha256 does not match package metadata")
    rows_meta = side_meta.get("rows")
    expected_rows_keys = SIDECAR_ROWS_V2_KEYS if sidecar_schema == "kana-sidecar-v2" else SIDECAR_ROWS_V1_KEYS
    if not isinstance(rows_meta, dict) or set(rows_meta) != expected_rows_keys:
        raise AggregateError(f"sidecar rows metadata does not match {sidecar_schema}")
    if rows_meta.get("jsonl_sha256") != rows_sha:
        raise AggregateError("sidecar JSONL hash does not match metadata")
    samples = metadata.get("samples")
    package_v2 = metadata.get("schema_version") == "visualnovel-package-v2"
    if not isinstance(samples, list) or metadata.get("record_count") != len(samples):
        raise AggregateError("package record_count does not match samples")
    base_sample_keys = {"sample_id", "row_index", "speaker_id", "speaker", "voice", "text", "text_sha256", "relative_audio_path", "audio"}
    optional_sample_keys = {"indexed_audio_path"}
    if package_v2:
        base_sample_keys |= {"source_status", "source_rejection_reason"}
    for sample in samples:
        if (
            not isinstance(sample, dict)
            or not base_sample_keys <= set(sample)
            or set(sample) - base_sample_keys - optional_sample_keys
        ):
            raise AggregateError("package sample schema is invalid")
        if not isinstance(sample["text"], str) or hashlib.sha256(sample["text"].encode()).hexdigest() != sample["text_sha256"]:
            raise AggregateError("package sample text hash is invalid")
        if package_v2:
            reason = source_rejection_reason(
                index_schema=metadata["index_schema"],
                speaker=sample["speaker"],
                indexed_audio_path=sample.get("indexed_audio_path", sample["relative_audio_path"]),
                text=sample["text"],
            )
            status, declared_reason = sample["source_status"], sample["source_rejection_reason"]
            if status == "accepted" and (reason is not None or declared_reason is not None):
                raise AggregateError("package source status is invalid")
            if status == "rejected" and (reason is None or declared_reason != reason):
                raise AggregateError("package source status is invalid")
    if len(samples) != len(side_rows) or rows_meta.get("source_count") != len(samples):
        raise AggregateError("package and sidecar cardinality differ")
    accepted_count = sum(row.get("status") == "accepted" for row in side_rows)
    rejected_count = sum(row.get("status") == "rejected" for row in side_rows)
    if accepted_count + rejected_count != len(side_rows):
        raise AggregateError("sidecar rows must have accepted or rejected status")
    if rows_meta.get("accepted_count") != accepted_count or rows_meta.get("rejected_count") != rejected_count:
        raise AggregateError("sidecar accepted/rejected cardinality does not match metadata")
    if sidecar_schema == "kana-sidecar-v1":
        train_count = sum(row.get("status") == "accepted" and row.get("split") == "train" for row in side_rows)
        eval_count = sum(row.get("status") == "accepted" and row.get("split") == "eval" for row in side_rows)
        rejected_split_count = sum(row.get("status") == "rejected" and row.get("split") == "rejected" for row in side_rows)
        if train_count + eval_count != accepted_count or rejected_split_count != rejected_count:
            raise AggregateError("v1 sidecar split cardinality is invalid")
        if rows_meta.get("train_count") != train_count or rows_meta.get("eval_count") != eval_count:
            raise AggregateError("v1 sidecar split cardinality does not match metadata")
    elif any("split" in row for row in side_rows):
        raise AggregateError("v2 sidecar rows must not contain authoritative split")
    sample_by_id = {sample.get("sample_id"): sample for sample in samples if isinstance(sample, dict)}
    if len(sample_by_id) != len(samples):
        raise AggregateError("package sample IDs are missing or duplicated")
    quality_rows = quality.get("rows")
    if not isinstance(quality_rows, list):
        raise AggregateError("quality rows must be an array")
    quality_by_id = {row.get("sample_id"): row for row in quality_rows if isinstance(row, dict)}
    if len(quality_by_id) != len(quality_rows):
        raise AggregateError("quality sample IDs are missing or duplicated")
    accepted, audio_identities = [], []
    for side_row in side_rows:
        if side_row.get("status") != "accepted":
            continue
        sample = sample_by_id.get(side_row.get("sample_id"))
        if package_v2 and sample.get("source_status") != "accepted":
            raise AggregateError("source-rejected package row cannot be accepted by sidecar")
        if sample is None or side_row.get("relative_audio_path") != sample.get("relative_audio_path"):
            raise AggregateError("sidecar sample identity does not match package metadata")
        audio = sample.get("audio")
        if not isinstance(audio, dict) or not _sha256(audio.get("sha256")):
            raise AggregateError("package sample lacks audio SHA-256")
        audio_path = _contained_audio(package, sample["relative_audio_path"])
        if file_sha256(audio_path) != audio["sha256"]:
            raise AggregateError(f"audio SHA-256 mismatch: {sample['sample_id']}")
        duration = quality_by_id.get(sample["sample_id"], {}).get("duration_seconds", audio.get("duration_seconds"))
        kana = side_row.get("kana_text")
        row_index = sample.get("row_index")
        if not isinstance(duration, (int, float)) or duration <= 0 or not isinstance(kana, str) or not kana:
            raise AggregateError(f"accepted row lacks duration or kana: {sample['sample_id']}")
        if type(row_index) is not int or row_index < 0:
            raise AggregateError("package row_index must be a non-negative integer")
        canonical_identity = {
            "wave": spec.wave,
            "selection_order": spec.selection_order,
            "package_id": metadata["package_id"],
            "row_index": row_index,
            "sample_id": sample["sample_id"],
        }
        if metadata.get("package_id") != source["package_id"] or metadata.get("package_sha256") != source["package_sha256"]:
            raise AggregateError("sidecar source identity changed during aggregation")
        accepted.append({
            "audio_sha256": audio["sha256"], "text": kana, "duration": float(duration),
            "speaker_id": side_row.get("speaker_id"), "speaker": side_row.get("speaker"),
            "sample_id": sample["sample_id"], "row_index": row_index,
            "audio_root_id": spec.audio_root_id, "relative_audio_path": sample["relative_audio_path"],
            "package_id": metadata["package_id"], "package_sha256": package_sha,
            "source_id": metadata["source_id"], "wave": spec.wave,
            "selection_order": spec.selection_order, "canonical_identity": canonical_identity,
        })
        audio_identities.append({"sample_id": sample["sample_id"], "row_index": row_index,
                                 "relative_audio_path": sample["relative_audio_path"], "audio_sha256": audio["sha256"]})
    semantic = {
        "audio_root_id": spec.audio_root_id, "wave": spec.wave, "selection_order": spec.selection_order,
        "package_id": metadata["package_id"], "package_sha256": package_sha,
        "source_id": metadata["source_id"], "index_sha256": metadata["index_sha256"],
        "metadata_sha256": file_sha256(metadata_path), "package_ready_sha256": file_sha256(ready_path),
        "sidecar_meta_sha256": file_sha256(side_meta_path), "sidecar_rows_sha256": rows_sha,
        "frontend_provenance": side_meta.get("frontend_provenance"), "vocabulary": side_meta.get("vocabulary"),
        "quality_sha256": file_sha256(quality_path), "provenance_sha256": file_sha256(provenance_path),
        "audio": sorted(audio_identities, key=lambda value: (value["row_index"], value["sample_id"])),
    }
    deployment = {**semantic, "package_dir": package.as_posix()}
    root = {"audio_root_id": spec.audio_root_id, "package_id": metadata["package_id"],
            "package_sha256": package_sha, "ready_sha256": file_sha256(ready_path)}
    return side_meta, accepted, semantic, {"deployment": deployment, "root": root}


def _assign_splits(rows: list[dict], eval_size: int) -> None:
    counts: dict[str, int] = {}
    for row in rows:
        speaker = str(row["speaker_id"])
        counts[speaker] = counts.get(speaker, 0) + 1
    remaining, eval_hashes = dict(counts), set()
    for row in sorted(rows, key=lambda value: (_fingerprint([value["audio_sha256"], value["text"]]), value["audio_sha256"])):
        speaker = str(row["speaker_id"])
        if len(eval_hashes) < eval_size and remaining[speaker] > 1:
            eval_hashes.add(row["audio_sha256"])
            remaining[speaker] -= 1
    for row in rows:
        row["split"] = "eval" if row["audio_sha256"] in eval_hashes else "train"


def _winner_key(row: dict) -> tuple:
    identity = row["canonical_identity"]
    return identity["wave"], identity["selection_order"], identity["package_id"], identity["row_index"]


def _write_view(staging: Path, name: str, rows: list[dict], shard_size: int, sample_rate: int,
                hop_length: int, vocabulary: dict) -> tuple[dict | None, list[dict]]:
    if not rows:
        return None, []
    root = staging if name == "train" else staging / "eval"
    (root / "shards").mkdir(parents=True)
    shards, frames_path = [], root / "frames.u32"
    with frames_path.open("wb") as frame_file:
        for start in range(0, len(rows), shard_size):
            chunk = rows[start:start + shard_size]
            shard_path = root / "shards" / f"{len(shards):06d}.arrow"
            with ArrowWriter(path=shard_path.as_posix()) as writer:
                for row in chunk:
                    writer.write(row)
                    np.asarray([max(1, int(row["duration"] * sample_rate / hop_length))], dtype="<u4").tofile(frame_file)
                writer.finalize()
            shards.append({"path": shard_path.relative_to(root).as_posix(), "rows": len(chunk),
                           "sha256": file_sha256(shard_path)})
        frame_file.flush()
        os.fsync(frame_file.fileno())
    manifest_prefix = "" if name == "train" else "eval/"
    manifest_shards = [{**item, "path": manifest_prefix + item["path"]} for item in shards]
    manifest = {
        "version": 1, "total_rows": len(rows), "shards": manifest_shards,
        "frame_index": {"path": manifest_prefix + "frames.u32", "count": len(rows), "dtype": "uint32", "sha256": file_sha256(frames_path)},
        "audio": {"sample_rate": sample_rate, "hop_length": hop_length},
        "vocabulary": {"identity": vocabulary["identity"], "token_sequence_sha256": vocabulary["token_sequence_sha256"]},
    }
    (staging / ("manifest.json" if name == "train" else "eval_manifest.json")).write_bytes(_canonical_bytes(manifest))
    hashes = [{"view": name, "path": item["path"], "sha256": item["sha256"]} for item in shards]
    hashes.append({"view": name, "path": "frames.u32", "sha256": manifest["frame_index"]["sha256"]})
    return manifest, hashes


def aggregate_visualnovel_packages(inputs: Iterable[AggregateInput], out_dir: str | Path, *,
                                   aggregate_context: AggregateContext, shard_size: int = 10_000,
                                   sample_rate: int = 24_000, hop_length: int = 256,
                                   eval_size: int = EVAL_SPLIT_SIZE) -> dict:
    """Publish manifest.json as train-only and eval_manifest.json as a separate view."""
    if not _commit_sha(aggregate_context.source_revision) or not aggregate_context.catalog_identity \
            or not _sha256(aggregate_context.selection_fingerprint):
        raise AggregateError("aggregate_context revision/catalog/selection fingerprint is incomplete")
    specs = list(inputs)
    if not specs or shard_size <= 0 or sample_rate <= 0 or hop_length <= 0 or eval_size < 0:
        raise ValueError("inputs and numeric settings must be valid and non-empty")
    if len({spec.audio_root_id for spec in specs}) != len(specs) or any(not spec.audio_root_id for spec in specs):
        raise AggregateError("audio_root_id values must be unique non-empty strings")
    if len({(spec.wave, spec.selection_order) for spec in specs}) != len(specs):
        raise AggregateError("wave/selection_order pairs must be unique")
    loaded = [_load_input(spec) for spec in specs]
    vocabularies = [item[0].get("vocabulary") for item in loaded]
    if any(value != vocabularies[0] for value in vocabularies[1:]):
        raise AggregateError("all sidecars must use the same vocabulary identity")
    vocabulary = vocabularies[0]
    if not isinstance(vocabulary, dict) or not vocabulary.get("identity") or not _sha256(vocabulary.get("token_sequence_sha256")):
        raise AggregateError("sidecar vocabulary identity is incomplete")

    grouped_rows: dict[str, list[dict]] = {}
    for _, input_rows, _, _ in loaded:
        for row in input_rows:
            grouped_rows.setdefault(row["audio_sha256"], []).append(row)

    by_audio, duplicates, conflicts = {}, [], []
    conflict_excluded_rows = 0
    for audio_sha256, audio_rows in sorted(grouped_rows.items()):
        rows_by_text: dict[str, list[dict]] = {}
        for row in audio_rows:
            rows_by_text.setdefault(row["text"], []).append(row)
        winners = []
        for text, exact_rows in sorted(rows_by_text.items()):
            winner = min(exact_rows, key=_winner_key)
            winners.append(winner)
            for duplicate in exact_rows:
                if duplicate is not winner:
                    duplicates.append({
                        "audio_sha256": audio_sha256,
                        "canonical_kana": text,
                        "kept_identity": winner["canonical_identity"],
                        "excluded_identity": duplicate["canonical_identity"],
                    })
        if len(rows_by_text) > 1:
            conflict_excluded_rows += len(audio_rows)
            conflicts.append({
                "audio_sha256": audio_sha256,
                "canonical_kana_values": sorted(rows_by_text),
                "rows": sorted(
                    (
                        {"canonical_kana": row["text"], "canonical_identity": row["canonical_identity"]}
                        for row in audio_rows
                    ),
                    key=_canonical_bytes,
                ),
            })
            continue
        by_audio[audio_sha256] = winners[0]
    rows = sorted(by_audio.values(), key=lambda row: (row["audio_sha256"], row["text"]))
    output = Path(out_dir).expanduser().resolve()
    if not rows:
        raise AggregateError("no accepted rows remain after aggregation")
    _assign_splits(rows, eval_size)
    context = {"source_revision": aggregate_context.source_revision, "catalog_identity": aggregate_context.catalog_identity,
               "selection_fingerprint": aggregate_context.selection_fingerprint}
    semantic_inputs = sorted((item[2] for item in loaded), key=lambda value: (value["wave"], value["selection_order"]))
    semantic_rows = [{"audio_sha256": row["audio_sha256"], "text": row["text"], "duration": row["duration"],
                      "speaker_id": row["speaker_id"], "split": row["split"],
                      "canonical_identity": row["canonical_identity"]} for row in rows]
    semantic_fingerprint = _fingerprint({"schema": SCHEMA_VERSION, "aggregate_context": context,
                                         "inputs": semantic_inputs, "rows": semantic_rows,
                                         "audio": {"sample_rate": sample_rate, "hop_length": hop_length},
                                         "vocabulary": vocabulary, "dedup_policy": DEDUP_POLICY,
                                         "conflict_policy": CONFLICT_POLICY, "split_policy": SPLIT_POLICY})
    if output.exists():
        raise FileExistsError(f"aggregate output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.partial-", dir=output.parent))
    try:
        train_rows, eval_rows = [row for row in rows if row["split"] == "train"], [row for row in rows if row["split"] == "eval"]
        _, train_hashes = _write_view(staging, "train", train_rows, shard_size, sample_rate, hop_length, vocabulary)
        eval_manifest, eval_hashes = _write_view(staging, "eval", eval_rows, shard_size, sample_rate, hop_length, vocabulary)
        deployments = sorted((item[3]["deployment"] for item in loaded), key=lambda value: value["audio_root_id"])
        artifacts = train_hashes + eval_hashes
        root_paths = [{"audio_root_id": value["audio_root_id"], "package_dir": value["package_dir"]} for value in deployments]
        deployment_fingerprint = _fingerprint({"semantic_fingerprint": semantic_fingerprint,
                                               "root_registry_paths": root_paths, "artifacts": artifacts})
        report = {"schema_version": SCHEMA_VERSION, "input_packages": len(specs),
                  "input_accepted_rows": sum(len(item[1]) for item in loaded), "output_rows": len(rows),
                  "dedup_policy": DEDUP_POLICY, "conflict_policy": CONFLICT_POLICY,
                  "duplicate_rows": len(duplicates),
                  "conflict_excluded_rows": conflict_excluded_rows,
                  "conflict_excluded_audio_objects": len(conflicts),
                  "train_rows": len(train_rows), "eval_rows": len(eval_rows),
                  "eval_target_rows": eval_size, "speaker_train_floor": 1, "training_manifest": "manifest.json",
                  "eval_manifest": "eval_manifest.json" if eval_manifest else None,
                  "duplicates": sorted(duplicates, key=_canonical_bytes),
                  "conflicts": sorted(conflicts, key=_canonical_bytes)}
        manifest_sha256 = file_sha256(staging / "manifest.json")
        split_fingerprint = _fingerprint(
            {
                "policy": SPLIT_POLICY,
                "rows": [
                    {"audio_sha256": row["audio_sha256"], "split": row["split"]}
                    for row in semantic_rows
                ],
            }
        )
        provenance = {"schema_version": SCHEMA_VERSION, "semantic_fingerprint": semantic_fingerprint,
                      "manifest_sha256": manifest_sha256, "split_fingerprint": split_fingerprint,
                      "vocabulary": {"identity": vocabulary["identity"],
                                     "token_sequence_sha256": vocabulary["token_sequence_sha256"]},
                      "deployment_fingerprint": deployment_fingerprint, "aggregate_context": context,
                      "dedup_policy": DEDUP_POLICY, "conflict_policy": CONFLICT_POLICY,
                      "split_policy": SPLIT_POLICY,
                      "input_sidecar_splits_authoritative": False, "inputs": semantic_inputs,
                      "artifact_hashes": artifacts}
        identity_registry = {"schema_version": "visualnovel-audio-roots-v1",
                             "roots": sorted((item[3]["root"] for item in loaded), key=lambda value: value["audio_root_id"])}
        deployment_registry = {"schema_version": "visualnovel-deployment-roots-v1", "roots": root_paths}
        for name, value in (("aggregate_provenance.json", provenance), ("aggregate_report.json", report),
                            ("audio_roots.json", identity_registry), ("deployment_audio_roots.json", deployment_registry)):
            (staging / name).write_bytes(_canonical_bytes(value))
        ready = {"schema_version": "visualnovel-aggregate-ready-v1", "commit": deployment_fingerprint,
                 "semantic_fingerprint": semantic_fingerprint, "deployment_fingerprint": deployment_fingerprint,
                 "train_manifest_sha256": file_sha256(staging / "manifest.json"),
                 "eval_manifest_sha256": file_sha256(staging / "eval_manifest.json") if eval_manifest else None}
        (staging / "READY").write_bytes(_canonical_bytes(ready))
        os.replace(staging, output)
        return {**report, "semantic_fingerprint": semantic_fingerprint, "deployment_fingerprint": deployment_fingerprint}
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


__all__ = ["AggregateConflictError", "AggregateContext", "AggregateError", "AggregateInput",
           "aggregate_visualnovel_packages"]
