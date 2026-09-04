"""Deterministic kana sidecars for ``visualnovel-package-v1`` metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from f5_tts.infer.ja_frontend import JapaneseFrontendError, frontend_provenance, frontend_v1, result_as_dict
from f5_tts.model.vocab_contract import token_sequence_sha256, validate_vocabulary_contract
from f5_tts.train.datasets.visualnovel_package import source_rejection_reason

SCHEMA_VERSION = "kana-sidecar-v2"
LEGACY_SCHEMA_VERSION = "kana-sidecar-v1"
SOURCE_SCHEMA_VERSIONS = {"visualnovel-package-v1", "visualnovel-package-v2"}
EVAL_SPLIT_SIZE = 5


class SidecarInputError(ValueError):
    pass


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _load_source(source: str | Path | dict) -> tuple[dict, str]:
    if isinstance(source, dict):
        value = source
        metadata_sha256 = canonical_sha256(value)
    else:
        path = Path(source).expanduser().resolve()
        try:
            content = path.read_bytes()
            value = json.loads(content.decode("utf-8"))
            metadata_sha256 = hashlib.sha256(content).hexdigest()
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise SidecarInputError(f"cannot read source metadata: {path}: {error}") from error
    if not isinstance(value, dict):
        raise SidecarInputError("source metadata root must be a JSON object")
    return value, metadata_sha256


def _strict_source(document: dict) -> list[dict]:
    required = {
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
        "speakers",
        "samples",
    }
    missing, extra = sorted(required - set(document)), sorted(set(document) - required)
    if missing or extra:
        raise SidecarInputError(f"invalid metadata keys: missing={missing}, extra={extra}")
    if document["schema_version"] not in SOURCE_SCHEMA_VERSIONS:
        raise SidecarInputError("unsupported VisualNovel package schema")
    for key in ("package_id", "package_sha256", "archive_root", "index_sha256", "index_schema", "source_id"):
        if not isinstance(document[key], str) or not document[key]:
            raise SidecarInputError(f"{key} must be a non-empty string")
    for key in ("archive_bytes", "index_bytes", "record_count"):
        if not isinstance(document[key], int) or document[key] < 0:
            raise SidecarInputError(f"{key} must be a non-negative integer")
    if document["index_schema"] not in {"Voice", "FilePath"}:
        raise SidecarInputError("index_schema must be Voice or FilePath")
    unindexed = document["unindexed_audio"]
    if not isinstance(unindexed, list):
        raise SidecarInputError("unindexed_audio must be an array")
    audit_keys = {"relative_audio_path", "sha256", "bytes"}
    for index, item in enumerate(unindexed):
        if not isinstance(item, dict) or set(item) != audit_keys:
            raise SidecarInputError(f"unindexed_audio[{index}] has invalid keys")
        if not isinstance(item["relative_audio_path"], str) or not item["relative_audio_path"]:
            raise SidecarInputError(f"unindexed_audio[{index}].relative_audio_path must be non-empty")
        if not isinstance(item["sha256"], str) or not item["sha256"]:
            raise SidecarInputError(f"unindexed_audio[{index}].sha256 must be non-empty")
        if not isinstance(item["bytes"], int) or item["bytes"] < 0:
            raise SidecarInputError(f"unindexed_audio[{index}].bytes must be non-negative")
    if not isinstance(document["speakers"], list):
        raise SidecarInputError("speakers must be an array")
    samples = document["samples"]
    if not isinstance(samples, list):
        raise SidecarInputError("samples must be an array")
    if "record_count" in document and document["record_count"] != len(samples):
        raise SidecarInputError("record_count does not match samples length")

    sample_ids = set()
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise SidecarInputError(f"samples[{index}] must be an object")
        sample_keys = set(sample)
        package_v2 = document["schema_version"] == "visualnovel-package-v2"
        required_sample = {
            "sample_id", "row_index", "speaker_id", "speaker", "voice", "text",
            "text_sha256", "relative_audio_path", "audio",
        }
        optional_sample = {"indexed_audio_path"}
        if package_v2:
            required_sample |= {"source_status", "source_rejection_reason"}
        missing, extra = sorted(required_sample - sample_keys), sorted(sample_keys - required_sample - optional_sample)
        if missing or extra:
            raise SidecarInputError(f"samples[{index}] invalid keys: missing={missing}, extra={extra}")
        for key in ("sample_id", "speaker_id", "speaker", "voice", "text_sha256", "relative_audio_path"):
            if not isinstance(sample[key], str) or not sample[key]:
                raise SidecarInputError(f"samples[{index}].{key} must be a non-empty string")
        if not isinstance(sample["text"], str) or (not package_v2 and not sample["text"].strip()):
            raise SidecarInputError(f"samples[{index}].text must be a non-empty string")
        if package_v2 and sample["source_status"] not in {"accepted", "rejected"}:
            raise SidecarInputError(f"samples[{index}].source_status is invalid")
        if sample["sample_id"] in sample_ids:
            raise SidecarInputError(f"duplicate sample_id: {sample['sample_id']}")
        sample_ids.add(sample["sample_id"])
        if not isinstance(sample["row_index"], int) or sample["row_index"] < 0:
            raise SidecarInputError(f"samples[{index}].row_index must be a non-negative integer")
        if not isinstance(sample["audio"], dict):
            raise SidecarInputError(f"samples[{index}].audio must be an object")
        if hashlib.sha256(sample["text"].encode("utf-8")).hexdigest() != sample["text_sha256"]:
            raise SidecarInputError(f"samples[{index}].text_sha256 does not match text")
        if package_v2:
            reason = source_rejection_reason(
                index_schema=document["index_schema"],
                speaker=sample["speaker"],
                indexed_audio_path=sample.get("indexed_audio_path", sample["relative_audio_path"]),
                text=sample["text"],
            )
            status, declared_reason = sample["source_status"], sample["source_rejection_reason"]
            if status == "accepted" and (reason is not None or declared_reason is not None):
                raise SidecarInputError(f"samples[{index}] has inconsistent accepted source status")
            if status == "rejected" and (reason is None or declared_reason != reason):
                raise SidecarInputError(f"samples[{index}] has inconsistent rejected source status")
    return samples


def _select_eval(rows: list[dict]) -> set[str]:
    """Choose five accepted rows by hash while leaving every speaker represented in train."""
    accepted = [row for row in rows if row["status"] == "accepted"]
    by_speaker: dict[str, int] = {}
    for row in accepted:
        by_speaker[row["speaker_id"]] = by_speaker.get(row["speaker_id"], 0) + 1
    ranked = sorted(accepted, key=lambda row: (hashlib.sha256(row["sample_id"].encode()).hexdigest(), row["sample_id"]))
    selected = set()
    remaining = dict(by_speaker)
    for row in ranked:
        if len(selected) == EVAL_SPLIT_SIZE:
            break
        speaker = row["speaker_id"]
        if remaining[speaker] <= 1:
            continue
        selected.add(row["sample_id"])
        remaining[speaker] -= 1
    return selected


def build_sidecar(
    source: str | Path | dict,
    *,
    vocab: set[str],
    vocab_identity: dict,
    schema_version: str = SCHEMA_VERSION,
) -> tuple[dict, list[dict]]:
    """Convert every source sample, retaining structured rejections and cardinality."""
    if schema_version not in {SCHEMA_VERSION, LEGACY_SCHEMA_VERSION}:
        raise ValueError(f"unsupported sidecar schema_version: {schema_version}")
    document, metadata_sha256 = _load_source(source)
    samples = _strict_source(document)
    rows = []
    for source_order, sample in enumerate(samples):
        row = {
            "sample_id": sample["sample_id"],
            "source_order": source_order,
            "row_index": sample["row_index"],
            "speaker_id": sample["speaker_id"],
            "speaker": sample["speaker"],
            "voice": sample["voice"],
            "relative_audio_path": sample["relative_audio_path"],
            "audio": sample["audio"],
            "source_text": sample["text"],
        }
        if sample.get("source_status") == "rejected":
            row.update(status="rejected", rejection_reason=sample.get("source_rejection_reason"), kana_text=None, frontend=None)
        else:
            try:
                result = frontend_v1(sample["text"], vocab=vocab)
            except JapaneseFrontendError as error:
                row.update(status="rejected", rejection_reason=error.to_dict(), kana_text=None, frontend=None)
            else:
                row.update(
                    status="accepted",
                    rejection_reason=None,
                    kana_text=result.kana,
                    frontend=result_as_dict(result),
                )
        if schema_version == LEGACY_SCHEMA_VERSION:
            row["split"] = "rejected" if row["status"] == "rejected" else None
        rows.append(row)

    eval_ids: set[str] = set()
    if schema_version == LEGACY_SCHEMA_VERSION:
        eval_ids = _select_eval(rows)
        for row in rows:
            if row["status"] == "accepted":
                row["split"] = "eval" if row["sample_id"] in eval_ids else "train"

    row_bytes = b"".join(canonical_json_bytes(row) for row in rows)
    accepted_count = sum(row["status"] == "accepted" for row in rows)
    source_identity = {
        key: document[key]
        for key in (
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
    }
    if schema_version == SCHEMA_VERSION:
        source_identity["metadata_sha256"] = metadata_sha256
    metadata = {
        "schema_version": schema_version,
        "source": source_identity,
        "vocabulary": vocab_identity,
        "frontend_provenance": frontend_provenance(),
        "rows": {
            "source_count": len(samples),
            "accepted_count": accepted_count,
            "rejected_count": len(samples) - accepted_count,
            "jsonl_sha256": hashlib.sha256(row_bytes).hexdigest(),
        },
    }
    if schema_version == LEGACY_SCHEMA_VERSION:
        metadata["rows"].update(
            train_count=sum(row["split"] == "train" for row in rows),
            eval_count=len(eval_ids),
            eval_target_count=EVAL_SPLIT_SIZE,
            eval_split_policy="lowest-sha256(sample_id), accepted only, retain >=1 train row per speaker",
            eval_shortfall=EVAL_SPLIT_SIZE - len(eval_ids),
        )
    return metadata, rows


def load_vocab_identity(vocab_path: str | Path, contract_path: str | Path) -> tuple[set[str], dict]:
    tokens, contract = validate_vocabulary_contract(vocab_path, contract_path)
    identity = {
        "identity": contract.identity,
        "provenance": contract.provenance,
        "token_count": contract.token_count,
        "token_sequence_sha256": token_sequence_sha256(tokens),
        "contract_sha256": hashlib.sha256(Path(contract_path).read_bytes()).hexdigest(),
        "vocab_file_sha256": hashlib.sha256(Path(vocab_path).read_bytes()).hexdigest(),
    }
    return set(tokens), identity


def write_sidecar(
    source: str | Path | dict,
    output_dir: str | Path,
    *,
    vocab: set[str],
    vocab_identity: dict,
    schema_version: str = SCHEMA_VERSION,
) -> tuple[Path, Path]:
    metadata, rows = build_sidecar(
        source, vocab=vocab, vocab_identity=vocab_identity, schema_version=schema_version
    )
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    meta_path, jsonl_path = output / "source_kana.meta.json", output / "source_kana.jsonl"
    meta_path.write_bytes(canonical_json_bytes(metadata))
    jsonl_path.write_bytes(b"".join(canonical_json_bytes(row) for row in rows))
    return meta_path, jsonl_path


def cli() -> None:
    parser = argparse.ArgumentParser(description="Generate a deterministic kana sidecar for a VisualNovel package.")
    parser.add_argument("source", help="visualnovel-package-v1 metadata.json")
    parser.add_argument("output_dir")
    parser.add_argument("--vocab", required=True, help="Vocabulary file validated against its contract")
    parser.add_argument("--vocab-contract", help="Defaults to contract.json beside --vocab")
    parser.add_argument(
        "--schema-version",
        choices=(SCHEMA_VERSION, LEGACY_SCHEMA_VERSION),
        default=SCHEMA_VERSION,
        help="Emit v2 by default; v1 is retained for historical reproducibility",
    )
    args = parser.parse_args()
    contract = args.vocab_contract or str(Path(args.vocab).with_name("contract.json"))
    vocab, identity = load_vocab_identity(args.vocab, contract)
    write_sidecar(
        args.source,
        args.output_dir,
        vocab=vocab,
        vocab_identity=identity,
        schema_version=args.schema_version,
    )


if __name__ == "__main__":
    cli()
