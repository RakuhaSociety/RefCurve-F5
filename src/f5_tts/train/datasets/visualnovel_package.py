"""Safely validate and publish encrypted VisualNovel voice packages.

A package is a 7z archive containing exactly one top-level directory.  That
root contains ``index.json`` and the audio files referenced by its records.
Extraction always happens in a sibling quarantine directory; ``READY`` is
written only after a second validation of the extracted filesystem.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

import soundfile as sf


AUDIO_SUFFIX = ".ogg"
INDEX_NAME = "index.json"
VOICE_SCHEMA_KEYS = {"Speaker", "Voice", "Text"}
FILE_PATH_SCHEMA_KEYS = {"FilePath", "Speaker", "Text"}
INDEX_SCHEMAS = {
    frozenset(VOICE_SCHEMA_KEYS): "Voice",
    frozenset(FILE_PATH_SCHEMA_KEYS): "FilePath",
}
METADATA_KEYS = {
    "schema_version", "package_id", "package_sha256", "archive_bytes", "archive_root",
    "index_sha256", "index_bytes", "index_schema", "unindexed_audio", "source_id",
    "record_count", "speakers", "samples",
}
PACKAGE_SCHEMA_VERSION = "visualnovel-package-v2"
PACKAGE_V1_SCHEMA_VERSION = "visualnovel-package-v1"
SAMPLE_SOURCE_KEYS = {"source_status", "source_rejection_reason"}
LEGACY_METADATA_KEYS = METADATA_KEYS - {"index_schema", "unindexed_audio"}
MIGRATION_SCHEMA_VERSION = "visualnovel-package-migration-v1"
PATH_MIGRATION_SCHEMA_VERSION = "visualnovel-package-path-migration-v1"
PATH_MIGRATION_NAME = "metadata.path-migration.json"


class PackageValidationError(ValueError):
    """Raised when an archive is unsafe or violates the package contract."""


@dataclass(frozen=True)
class ResourceLimits:
    max_members: int = 2_000
    max_member_bytes: int = 256 * 1024 * 1024
    max_total_bytes: int = 4 * 1024 * 1024 * 1024
    max_compression_ratio: float = 1_000.0
    max_index_bytes: int = 4 * 1024 * 1024
    max_records_per_package: int = 100_000

    def __post_init__(self) -> None:
        if min(
            self.max_members,
            self.max_member_bytes,
            self.max_total_bytes,
            self.max_index_bytes,
            self.max_records_per_package,
        ) <= 0:
            raise ValueError("resource limits must be positive")
        if self.max_compression_ratio <= 0:
            raise ValueError("max_compression_ratio must be positive")


@dataclass(frozen=True)
class InventoryMember:
    path: str
    size: int
    compressed_size: int | None = None
    is_directory: bool = False
    is_link: bool = False


def _normalized_key(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold()


def _safe_segment(value: object, *, field: str, record_number: int) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or value in {".", ".."}
        or ":" in value
        or "." in value
        or "/" in value
        or "\\" in value
        or any(unicodedata.category(char) == "Cc" for char in value)
    ):
        raise PackageValidationError(f"index record {record_number} has an invalid {field}")
    return value


def _safe_member_path(raw_path: str) -> PurePosixPath:
    if not isinstance(raw_path, str) or not raw_path:
        raise PackageValidationError("archive contains an invalid member path")
    normalized = raw_path.replace("\\", "/")
    path = PurePosixPath(normalized)
    if (
        path.is_absolute()
        or normalized.startswith("//")
        or any(part in {"", ".", ".."} for part in path.parts)
        or any(":" in part or any(unicodedata.category(char) == "Cc" for char in part) for part in path.parts)
    ):
        raise PackageValidationError(f"unsafe archive member path: {raw_path!r}")
    return path


def _safe_file_path(value: object, *, speaker: str, record_number: int) -> tuple[str, str]:
    if not isinstance(value, str) or not value:
        raise PackageValidationError(f"index record {record_number} has an invalid FilePath")
    normalized = value.replace("\\", "/")
    try:
        path = _safe_member_path(normalized)
    except PackageValidationError as error:
        raise PackageValidationError(f"index record {record_number} has an invalid FilePath") from error
    if len(path.parts) != 2 or path.suffix != AUDIO_SUFFIX:
        raise PackageValidationError(f"index record {record_number} FilePath must be exactly Speaker/file.ogg")
    if path.parts[0] != speaker:
        raise PackageValidationError(f"index record {record_number} FilePath speaker does not match Speaker")
    filename = path.parts[1]
    voice = filename[: -len(AUDIO_SUFFIX)]
    _safe_segment(voice, field="FilePath filename", record_number=record_number)
    return path.as_posix(), voice


def source_rejection_reason(*, index_schema: str, speaker: str, indexed_audio_path: str, text: str) -> str | None:
    """Return the package-level semantic rejection reason for a validated source row."""
    if not text.strip():
        return "empty_text"
    if index_schema == "FilePath" and speaker == "　":
        path = PurePosixPath(indexed_audio_path)
        if len(path.parts) == 2 and path.parts[0] == "　":
            return "invalid_speaker"
    return None


def _member_value(member: object, *names: str, default: Any = None) -> Any:
    if isinstance(member, Mapping):
        for name in names:
            if name in member:
                return member[name]
    for name in names:
        if hasattr(member, name):
            return getattr(member, name)
    property_fn = getattr(member, "file_properties", None)
    if callable(property_fn):
        properties = property_fn()
        if isinstance(properties, Mapping):
            for name in names:
                if name in properties:
                    return properties[name]
    return default


def _member_is_link(member: object) -> bool:
    for name in ("is_symlink", "is_hardlink", "is_link"):
        value = _member_value(member, name, default=False)
        if callable(value):
            value = value()
        if bool(value):
            return True
    mode = _member_value(member, "mode", "posix_mode")
    if isinstance(mode, int) and stat.S_ISLNK(mode):
        return True
    attributes = _member_value(member, "attributes", "archivable")
    if isinstance(attributes, int):
        unix_mode = (attributes >> 16) & 0xFFFF
        if stat.S_ISLNK(unix_mode):
            return True
        reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        if attributes & reparse:
            return True
    return False


def validate_inventory(
    members: Iterable[object], limits: ResourceLimits = ResourceLimits()
) -> tuple[str, list[InventoryMember]]:
    """Validate archive inventory objects without extracting them.

    The loose object interface intentionally supports both py7zr ``FileInfo``
    objects and small dictionaries used by security-focused unit tests.
    """
    checked: list[InventoryMember] = []
    seen: set[str] = set()
    roots: set[str] = set()
    total_size = 0

    for raw in members:
        path_value = _member_value(raw, "filename", "path", "name")
        path = _safe_member_path(path_value)
        canonical = path.as_posix()
        folded = _normalized_key(canonical)
        if folded in seen:
            raise PackageValidationError(f"duplicate or Unicode-equivalent archive member path: {canonical!r}")
        seen.add(folded)
        roots.add(path.parts[0])

        is_directory_value = _member_value(raw, "is_directory", "is_dir", default=False)
        if callable(is_directory_value):
            is_directory_value = is_directory_value()
        is_directory = bool(is_directory_value)
        size = int(_member_value(raw, "uncompressed", "uncompressed_size", "size", default=0) or 0)
        compressed_raw = _member_value(raw, "compressed", "compressed_size")
        compressed = None if compressed_raw is None else int(compressed_raw or 0)
        is_link = _member_is_link(raw)
        if size < 0 or (compressed is not None and compressed < 0):
            raise PackageValidationError("archive contains a negative member size")
        if is_link:
            raise PackageValidationError(f"archive links are forbidden: {canonical!r}")
        if not is_directory:
            if size > limits.max_member_bytes:
                raise PackageValidationError(f"archive member exceeds size limit: {canonical!r}")
            total_size += size
            if total_size > limits.max_total_bytes:
                raise PackageValidationError("archive exceeds total uncompressed size limit")
            if compressed == 0 and size > 0:
                raise PackageValidationError(f"archive member has an unsafe compression ratio: {canonical!r}")
            if compressed and size / compressed > limits.max_compression_ratio:
                raise PackageValidationError(f"archive member exceeds compression ratio limit: {canonical!r}")
        checked.append(InventoryMember(canonical, size, compressed, is_directory, is_link))
        if len(checked) > limits.max_members:
            raise PackageValidationError("archive exceeds member count limit")

    if not checked:
        raise PackageValidationError("archive is empty")
    path_types = {_normalized_key(member.path): member.is_directory for member in checked}
    for member in checked:
        path = PurePosixPath(member.path)
        for depth in range(1, len(path.parts)):
            prefix = PurePosixPath(*path.parts[:depth]).as_posix()
            prefix_type = path_types.get(_normalized_key(prefix))
            if prefix_type is False:
                raise PackageValidationError(f"archive has a file/directory prefix conflict: {prefix!r}")
    if len({_normalized_key(root) for root in roots}) != 1:
        raise PackageValidationError("archive must contain exactly one top-level root")
    root_keys = {_normalized_key(root): root for root in roots}
    root = root_keys[next(iter(root_keys))]
    index_key = _normalized_key(f"{root}/{INDEX_NAME}")
    indexes = [member for member in checked if _normalized_key(member.path) == index_key]
    if len(indexes) != 1:
        raise PackageValidationError(f"archive root must contain exactly one {INDEX_NAME}")
    index = indexes[0]
    if index.path != f"{root}/{INDEX_NAME}" or index.is_directory or index.size > limits.max_index_bytes:
        raise PackageValidationError("index.json is not a bounded regular file")
    for member in checked:
        relative = PurePosixPath(member.path).relative_to(root)
        if member.is_directory:
            if relative.parts and len(relative.parts) != 1:
                raise PackageValidationError("archive contains an unexpected directory layout")
            continue
        if member is index:
            continue
        if relative.suffix != AUDIO_SUFFIX or len(relative.parts) != 2:
            raise PackageValidationError("archive may contain only root/index.json and Speaker/Voice.ogg audio")
    return root, checked


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _stable_id(kind: str, *parts: str) -> str:
    digest = hashlib.sha256(_json_bytes([kind, *parts])).hexdigest()[:24]
    return f"{kind}_{digest}"


def _audio_inventory(members: Sequence[str | InventoryMember], root: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for member in members:
        if isinstance(member, InventoryMember) and member.is_directory:
            continue
        member_path = member.path if isinstance(member, InventoryMember) else member
        if _normalized_key(member_path) in {_normalized_key(root), _normalized_key(f"{root}/{INDEX_NAME}")}:
            continue
        relative = PurePosixPath(member_path).relative_to(root).as_posix()
        result[_normalized_key(relative)] = relative
    return result


def validate_index(
    raw_index: object,
    members: Sequence[str | InventoryMember],
    *,
    root: str,
    expected_records: int | None = None,
    max_records_per_package: int | None = None,
    allow_unindexed_audio: bool = False,
) -> list[dict[str, str]]:
    """Validate one strict Voice or FilePath schema while preserving source rows."""
    if not isinstance(raw_index, list):
        raise PackageValidationError("index.json must be a JSON array")
    if expected_records is not None and len(raw_index) != expected_records:
        raise PackageValidationError(f"index.json must contain exactly {expected_records} records")
    if max_records_per_package is not None and len(raw_index) > max_records_per_package:
        raise PackageValidationError(f"index.json exceeds max_records_per_package={max_records_per_package}")
    if not raw_index:
        raise PackageValidationError("index.json must not be empty")

    first = raw_index[0]
    index_schema = INDEX_SCHEMAS.get(frozenset(first) if isinstance(first, dict) else frozenset())
    if index_schema is None:
        raise PackageValidationError(
            "index records must use exactly FilePath, Speaker, Text or exactly Speaker, Voice, Text"
        )

    records: list[dict[str, str]] = []
    audio_paths: set[str] = set()
    for number, raw_record in enumerate(raw_index, start=1):
        if not isinstance(raw_record, dict) or INDEX_SCHEMAS.get(frozenset(raw_record)) != index_schema:
            raise PackageValidationError(f"index record {number} does not match the package {index_schema} schema")
        raw_speaker = raw_record["Speaker"]
        if index_schema == "FilePath" and raw_speaker == "　":
            speaker = raw_speaker
        else:
            speaker = _safe_segment(raw_speaker, field="Speaker", record_number=number)
        text = raw_record["Text"]
        if not isinstance(text, str) or any(
            unicodedata.category(char) == "Cc" and char not in "\t\n\r" for char in text
        ):
            raise PackageValidationError(f"index record {number} has an invalid Text")
        if index_schema == "Voice":
            voice = _safe_segment(raw_record["Voice"], field="Voice", record_number=number)
            indexed_audio_path = f"{speaker}/{voice}{AUDIO_SUFFIX}"
        else:
            indexed_audio_path, voice = _safe_file_path(
                raw_record["FilePath"], speaker=speaker, record_number=number
            )
        audio_paths.add(_normalized_key(indexed_audio_path))
        records.append(
            {
                "Speaker": speaker,
                "Voice": voice,
                "Text": text,
                "indexed_audio_path": indexed_audio_path,
                "relative_audio_path": indexed_audio_path,
                "index_schema": index_schema,
            }
        )

    inventory = _audio_inventory(members, root)
    inventory_paths = set(inventory)
    missing = sorted(audio_paths - inventory_paths)
    extra = sorted(inventory_paths - audio_paths)
    if missing or (extra and not allow_unindexed_audio):
        details = []
        if missing:
            details.append(f"missing referenced audio ({len(missing)})")
        if extra:
            details.append(f"unindexed files ({len(extra)})")
        raise PackageValidationError("audio collection mismatch: " + ", ".join(details))
    for record in records:
        record["relative_audio_path"] = inventory[_normalized_key(record["indexed_audio_path"])]
    return records


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_index_from_archive(archive: object, member_name: str, limit: int) -> tuple[object, str, int]:
    # py7zr intentionally has no stable in-memory read API. Extracting only the
    # already-inventoried index into a private temporary directory keeps parsing bounded.
    with tempfile.TemporaryDirectory(prefix="visualnovel-index-") as temp:
        archive.extract(path=temp, targets=[member_name])
        path = Path(temp).joinpath(*PurePosixPath(member_name).parts)
        if not path.is_file() or path.is_symlink() or path.stat().st_size > limit:
            raise PackageValidationError("index.json did not extract as a bounded regular file")
        try:
            content = path.read_bytes()
            return json.loads(content.decode("utf-8")), hashlib.sha256(content).hexdigest(), len(content)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise PackageValidationError("index.json is not valid UTF-8 JSON") from error


def _py7zr_inventory(archive: object) -> list[object]:
    # ArchiveFile entries retain link/type metadata that FileInfo returned by
    # some py7zr versions omits. Never downgrade to names-only validation.
    files = getattr(archive, "files", None)
    if files is not None:
        return list(files)
    try:
        return list(archive.list())
    except AttributeError as error:
        raise RuntimeError("installed py7zr cannot provide a safe archive inventory") from error


def _open_archive(path: Path, password: str | None):
    try:
        import py7zr
    except ImportError as error:
        raise RuntimeError("py7zr is required to install VisualNovel packages") from error
    try:
        return py7zr.SevenZipFile(path, mode="r", password=password)
    except Exception as error:
        raise PackageValidationError("archive could not be opened; it may be corrupt or require a password") from error


def _is_link_or_reparse(path: Path) -> bool:
    info = path.lstat()
    attrs = getattr(info, "st_file_attributes", 0)
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return stat.S_ISLNK(info.st_mode) or bool(attrs & reparse)


def _audio_diagnostic(
    root_dir: Path, relative_audio_path: str, *, verify_with_torchaudio: bool
) -> dict[str, Any]:
    audio = root_dir.joinpath(*PurePosixPath(relative_audio_path).parts)
    try:
        info = sf.info(audio)
        if info.frames <= 0 or info.samplerate <= 0 or info.channels <= 0:
            raise RuntimeError("empty audio")
        with sf.SoundFile(audio) as handle:
            handle.read(min(handle.frames, handle.samplerate), dtype="float32", always_2d=True)
    except Exception as error:
        raise PackageValidationError(f"audio cannot be decoded: {relative_audio_path!r}") from error
    if verify_with_torchaudio:
        try:
            import torchaudio

            waveform, sample_rate = torchaudio.load(str(audio))
            if waveform.numel() == 0 or sample_rate <= 0:
                raise RuntimeError("empty audio")
        except (ImportError, OSError):
            pass
        except Exception as error:
            raise PackageValidationError(f"audio fails optional torchaudio decoding: {relative_audio_path!r}") from error
    return {
        "relative_audio_path": relative_audio_path,
        "sha256": _sha256_file(audio),
        "bytes": audio.stat().st_size,
        "frames": int(info.frames),
        "sample_rate": int(info.samplerate),
        "channels": int(info.channels),
        "duration_seconds": float(info.duration),
    }


def _verify_extracted_tree(
    root_dir: Path,
    records: Sequence[dict[str, str]],
    *,
    unindexed_audio: Sequence[str] = (),
    inventory_audio: Mapping[str, str] | None = None,
    verify_with_torchaudio: bool = False,
) -> dict[str, dict[str, Any]]:
    if _is_link_or_reparse(root_dir) or not root_dir.is_dir():
        raise PackageValidationError("extracted package root is not a regular directory")
    audio_paths = {record["relative_audio_path"] for record in records} | set(unindexed_audio)
    expected = {INDEX_NAME, *audio_paths}
    expected_keys = {_normalized_key(path) for path in expected}
    actual: set[str] = set()
    base = root_dir.resolve()
    for path in root_dir.rglob("*"):
        if _is_link_or_reparse(path):
            raise PackageValidationError("extracted package contains a link or reparse point")
        try:
            relative = path.relative_to(root_dir).as_posix()
        except ValueError as error:
            raise PackageValidationError("extracted member escaped package root") from error
        if path.is_file():
            if not path.resolve().is_relative_to(base):
                raise PackageValidationError("extracted member escaped package root")
            actual.add(_normalized_key(relative))
        elif not path.is_dir():
            raise PackageValidationError("extracted package contains a non-regular member")
    if actual != expected_keys:
        raise PackageValidationError("extracted file set differs from validated inventory")

    diagnostic_paths = inventory_audio or {_normalized_key(path): path for path in audio_paths}
    diagnostics = {
        relative: _audio_diagnostic(
            root_dir, relative, verify_with_torchaudio=verify_with_torchaudio
        )
        for relative in sorted(diagnostic_paths.values(), key=_normalized_key)
    }
    return {_normalized_key(relative): value for relative, value in diagnostics.items()}


def _metadata(
    package_hash: str,
    archive_bytes: int,
    archive_root: str,
    index_sha256: str,
    index_bytes: int,
    records: Sequence[dict[str, str]],
    audio: Mapping[str, dict[str, Any]],
    unindexed_audio: Sequence[dict[str, Any]] = (),
) -> dict:
    package_id = _stable_id("pkg", package_hash)
    source_id = _stable_id("src", package_id, index_sha256)
    speakers = {name: _stable_id("spk", source_id, name) for name in sorted({r["Speaker"] for r in records})}
    samples = []
    for row_index, record in enumerate(records):
        diagnostic = audio[_normalized_key(record["relative_audio_path"])]
        text_hash = hashlib.sha256(record["Text"].encode("utf-8")).hexdigest()
        rejection_reason = source_rejection_reason(
            index_schema=record["index_schema"],
            speaker=record["Speaker"],
            indexed_audio_path=record.get("indexed_audio_path", record["relative_audio_path"]),
            text=record["Text"],
        )
        sample = {
            "sample_id": _stable_id(
                "smp", package_id, str(row_index), record["Speaker"], record["Voice"], text_hash
            ),
            "row_index": row_index,
            "speaker_id": speakers[record["Speaker"]],
            "speaker": record["Speaker"],
            "voice": record["Voice"],
            "text": record["Text"],
            "text_sha256": text_hash,
            "relative_audio_path": record["relative_audio_path"],
            "audio": diagnostic,
            "source_status": "rejected" if rejection_reason is not None else "accepted",
            "source_rejection_reason": rejection_reason,
        }
        indexed_audio_path = record.get("indexed_audio_path", record["relative_audio_path"])
        if indexed_audio_path != record["relative_audio_path"]:
            sample["indexed_audio_path"] = indexed_audio_path
        samples.append(sample)
    accepted_count = sum(sample["source_status"] == "accepted" for sample in samples)
    if accepted_count == 0:
        raise PackageValidationError("package must contain at least one accepted row")
    return {
        "schema_version": PACKAGE_SCHEMA_VERSION,
        "package_id": package_id,
        "package_sha256": package_hash,
        "archive_bytes": archive_bytes,
        "archive_root": archive_root,
        "index_sha256": index_sha256,
        "index_bytes": index_bytes,
        "index_schema": records[0]["index_schema"],
        "unindexed_audio": list(unindexed_audio),
        "source_id": source_id,
        "record_count": len(records),
        "speakers": [{"speaker_id": speakers[name], "speaker": name} for name in sorted(speakers)],
        "samples": samples,
    }


def _write_bytes_atomic(path: Path, content: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_json_object_bytes(path: Path, *, label: str) -> tuple[dict, bytes]:
    try:
        content = path.read_bytes()
        value = json.loads(content.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PackageValidationError(f"{label} is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise PackageValidationError(f"{label} must be a JSON object")
    return value, content


def _legacy_package_file_set(package_dir: Path, audio_paths: Sequence[str]) -> None:
    expected = {_normalized_key(name) for name in (INDEX_NAME, "metadata.json", "READY", *audio_paths)}
    actual: set[str] = set()
    base = package_dir.resolve()
    if _is_link_or_reparse(package_dir) or not package_dir.is_dir():
        raise PackageValidationError("package directory is not a regular directory")
    for path in package_dir.rglob("*"):
        if _is_link_or_reparse(path):
            raise PackageValidationError("package contains a link or reparse point")
        relative = path.relative_to(package_dir).as_posix()
        if path.is_file():
            if not path.resolve().is_relative_to(base):
                raise PackageValidationError("package member escaped package root")
            actual.add(_normalized_key(relative))
        elif not path.is_dir():
            raise PackageValidationError("package contains a non-regular member")
    if actual != expected:
        raise PackageValidationError("package file set differs from legacy metadata and index")


def _installed_audio_inventory(package_dir: Path, *, allowed_metadata_files: set[str]) -> dict[str, str]:
    """Rescan an installed package and return normalized keys to actual on-disk paths."""
    inventory: dict[str, str] = {}
    allowed_root_files = {INDEX_NAME, "metadata.json", "READY", *allowed_metadata_files}
    base = package_dir.resolve()
    if _is_link_or_reparse(package_dir) or not package_dir.is_dir():
        raise PackageValidationError("package directory is not a regular directory")
    for path in package_dir.rglob("*"):
        if _is_link_or_reparse(path):
            raise PackageValidationError("package contains a link or reparse point")
        relative = path.relative_to(package_dir).as_posix()
        if path.is_dir():
            if len(PurePosixPath(relative).parts) != 1:
                raise PackageValidationError("package contains an unexpected directory layout")
            continue
        if not path.is_file() or not path.resolve().is_relative_to(base):
            raise PackageValidationError("package contains an escaped or non-regular member")
        pure = PurePosixPath(relative)
        if len(pure.parts) == 1:
            if relative not in allowed_root_files:
                raise PackageValidationError(f"package contains an unexpected root file: {relative!r}")
            continue
        if len(pure.parts) != 2 or pure.suffix != AUDIO_SUFFIX:
            raise PackageValidationError(f"package contains an unexpected audio path: {relative!r}")
        key = _normalized_key(relative)
        if key in inventory:
            raise PackageValidationError(f"duplicate or Unicode-equivalent installed audio path: {relative!r}")
        inventory[key] = relative
    return inventory


def migrate_visualnovel_audio_paths(
    package_dir: str | Path,
    *,
    verify_with_torchaudio: bool = False,
) -> dict:
    """Explicitly repair indexed spellings to unique canonical installed audio paths.

    The archive/package identity and every sample ID remain unchanged. Only path
    spellings and their duplicated audio diagnostic path are rewritten, with the
    original index spelling retained as ``indexed_audio_path`` when it differs.
    """
    package = Path(package_dir).expanduser().resolve()
    metadata_path = package / "metadata.json"
    ready_path = package / "READY"
    index_path = package / INDEX_NAME
    provenance_path = package / PATH_MIGRATION_NAME
    if not package.is_dir() or not metadata_path.is_file() or not ready_path.is_file() or not index_path.is_file():
        raise PackageValidationError("package must contain metadata.json, READY, and index.json")
    current, current_bytes = _load_json_object_bytes(metadata_path, label="package metadata")
    if set(current) != METADATA_KEYS or current.get("schema_version") not in {
        PACKAGE_V1_SCHEMA_VERSION,
        PACKAGE_SCHEMA_VERSION,
    }:
        raise PackageValidationError("path migration requires current package metadata")
    package_hash = current.get("package_sha256")
    if ready_path.read_text(encoding="ascii").strip() != package_hash:
        raise PackageValidationError("READY does not bind package_sha256")
    if current.get("package_id") != _stable_id("pkg", package_hash):
        raise PackageValidationError("package_id does not match package_sha256")

    try:
        index_bytes = index_path.read_bytes()
        raw_index = json.loads(index_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PackageValidationError("installed index.json is not valid UTF-8 JSON") from error
    index_hash = hashlib.sha256(index_bytes).hexdigest()
    if current.get("index_sha256") != index_hash or current.get("index_bytes") != len(index_bytes):
        raise PackageValidationError("installed index.json does not match package metadata")

    actual_inventory = _installed_audio_inventory(package, allowed_metadata_files=set())
    archive_root = current.get("archive_root")
    member_paths = [f"{archive_root}/{INDEX_NAME}", *(f"{archive_root}/{path}" for path in actual_inventory.values())]
    records = validate_index(
        raw_index,
        member_paths,
        root=archive_root,
        expected_records=current.get("record_count"),
        allow_unindexed_audio=bool(current.get("unindexed_audio")),
    )
    if records[0]["index_schema"] != current.get("index_schema"):
        raise PackageValidationError("index schema does not match package metadata")
    referenced = {_normalized_key(record["relative_audio_path"]) for record in records}
    unindexed_paths = [path for key, path in sorted(actual_inventory.items()) if key not in referenced]
    if {_normalized_key(item.get("relative_audio_path", "")) for item in current["unindexed_audio"]} != {
        _normalized_key(path) for path in unindexed_paths
    }:
        raise PackageValidationError("unindexed audio set does not match package metadata")
    diagnostics = {
        key: _audio_diagnostic(package, path, verify_with_torchaudio=verify_with_torchaudio)
        for key, path in actual_inventory.items()
    }
    upgraded = _metadata(
        package_hash,
        current["archive_bytes"],
        archive_root,
        index_hash,
        len(index_bytes),
        records,
        diagnostics,
        [
            {"relative_audio_path": path, "sha256": diagnostics[_normalized_key(path)]["sha256"],
             "bytes": diagnostics[_normalized_key(path)]["bytes"]}
            for path in unindexed_paths
        ],
    )
    upgraded["schema_version"] = current["schema_version"]
    if current["schema_version"] == PACKAGE_V1_SCHEMA_VERSION:
        for sample in upgraded["samples"]:
            for key in SAMPLE_SOURCE_KEYS:
                sample.pop(key, None)
    if upgraded["package_id"] != current["package_id"] or upgraded["source_id"] != current["source_id"]:
        raise PackageValidationError("path migration would change package identity")
    if upgraded["speakers"] != current["speakers"] or len(upgraded["samples"]) != len(current["samples"]):
        raise PackageValidationError("path migration would change source rows")
    changed = []
    for old, new in zip(current["samples"], upgraded["samples"]):
        if old.get("sample_id") != new["sample_id"]:
            raise PackageValidationError("path migration would change sample IDs")
        old_without_paths = {key: value for key, value in old.items() if key not in {"relative_audio_path", "indexed_audio_path", "audio"}}
        new_without_paths = {key: value for key, value in new.items() if key not in {"relative_audio_path", "indexed_audio_path", "audio"}}
        old_audio = {key: value for key, value in old.get("audio", {}).items() if key != "relative_audio_path"}
        new_audio = {key: value for key, value in new["audio"].items() if key != "relative_audio_path"}
        if old_without_paths != new_without_paths or old_audio != new_audio:
            raise PackageValidationError(f"sample metadata does not match installed audio: {old.get('sample_id')}")
        if old.get("relative_audio_path") != new["relative_audio_path"]:
            changed.append({
                "sample_id": new["sample_id"],
                "indexed_audio_path": new.get("indexed_audio_path", old["relative_audio_path"]),
                "relative_audio_path": new["relative_audio_path"],
            })
    if not changed:
        raise PackageValidationError("package audio paths are already canonical")

    upgraded_bytes = json.dumps(upgraded, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"
    old_hash = hashlib.sha256(current_bytes).hexdigest()
    new_hash = hashlib.sha256(upgraded_bytes).hexdigest()
    provenance = {
        "schema_version": PATH_MIGRATION_SCHEMA_VERSION,
        "migration_id": _stable_id("mig", current["package_id"], old_hash, new_hash),
        "package_id": current["package_id"],
        "package_sha256": package_hash,
        "old_metadata_sha256": old_hash,
        "new_metadata_sha256": new_hash,
        "normalization": "NFKC+casefold unique inventory mapping",
        "sample_ids_preserved": True,
        "changed_samples": changed,
    }
    if provenance_path.exists():
        raise PackageValidationError("path migration provenance already exists")
    try:
        _write_bytes_atomic(provenance_path, _json_bytes(provenance) + b"\n")
        _write_bytes_atomic(metadata_path, upgraded_bytes)
    except Exception:
        if hashlib.sha256(metadata_path.read_bytes()).hexdigest() == old_hash:
            provenance_path.unlink(missing_ok=True)
        raise
    return {"metadata": upgraded, "provenance": provenance}


def migrate_legacy_visualnovel_package(
    package_dir: str | Path,
    *,
    verify_with_torchaudio: bool = False,
) -> dict:
    """Strictly revalidate and atomically upgrade one legacy installed package.

    Only the historical Voice-index shape missing exactly ``index_schema`` and
    ``unindexed_audio`` is accepted. The normal strict metadata/sidecar readers
    remain unchanged; migration is an explicit trust-boundary operation.
    """
    package = Path(package_dir).expanduser().resolve()
    metadata_path = package / "metadata.json"
    ready_path = package / "READY"
    index_path = package / INDEX_NAME
    if not package.is_dir() or not metadata_path.is_file() or not ready_path.is_file() or not index_path.is_file():
        raise PackageValidationError("legacy package must contain metadata.json, READY, and index.json")
    legacy, legacy_bytes = _load_json_object_bytes(metadata_path, label="legacy metadata")
    if set(legacy) not in (LEGACY_METADATA_KEYS, METADATA_KEYS - {"index_schema", "unindexed_audio"}):
        missing = sorted(LEGACY_METADATA_KEYS - set(legacy))
        extra = sorted(set(legacy) - LEGACY_METADATA_KEYS)
        raise PackageValidationError(f"legacy metadata keys are invalid: missing={missing}, extra={extra}")
    if legacy.get("schema_version") not in {PACKAGE_V1_SCHEMA_VERSION, PACKAGE_SCHEMA_VERSION}:
        raise PackageValidationError("legacy metadata has an unsupported schema_version")

    package_hash = legacy.get("package_sha256")
    if not isinstance(package_hash, str) or len(package_hash) != 64 or any(c not in "0123456789abcdef" for c in package_hash):
        raise PackageValidationError("legacy package_sha256 is invalid")
    try:
        ready_hash = ready_path.read_text(encoding="ascii").strip()
    except (OSError, UnicodeDecodeError) as error:
        raise PackageValidationError("READY is not valid ASCII") from error
    if ready_hash != package_hash:
        raise PackageValidationError("READY does not bind legacy package_sha256")
    if legacy.get("package_id") != _stable_id("pkg", package_hash):
        raise PackageValidationError("legacy package_id does not match package_sha256")

    try:
        index_bytes = index_path.read_bytes()
        raw_index = json.loads(index_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PackageValidationError("installed index.json is not valid UTF-8 JSON") from error
    if legacy.get("index_bytes") != len(index_bytes) or legacy.get("index_sha256") != hashlib.sha256(index_bytes).hexdigest():
        raise PackageValidationError("installed index.json does not match legacy metadata")
    archive_root = legacy.get("archive_root")
    archive_bytes = legacy.get("archive_bytes")
    if not isinstance(archive_root, str) or not archive_root or not isinstance(archive_bytes, int) or archive_bytes < 0:
        raise PackageValidationError("legacy archive metadata is invalid")

    member_paths = [f"{archive_root}/{INDEX_NAME}"]
    samples = legacy.get("samples")
    if not isinstance(samples, list):
        raise PackageValidationError("legacy samples must be an array")
    actual_inventory = _installed_audio_inventory(package, allowed_metadata_files={"metadata.migration.json"})
    legacy_audio_keys = {
        _normalized_key(sample.get("relative_audio_path", ""))
        for sample in samples
        if isinstance(sample, dict) and isinstance(sample.get("relative_audio_path"), str)
    }
    if set(actual_inventory) != legacy_audio_keys:
        raise PackageValidationError("package file set differs from legacy metadata and index")
    member_paths.extend(f"{archive_root}/{path}" for path in actual_inventory.values())
    records = validate_index(
        raw_index,
        member_paths,
        root=archive_root,
        expected_records=legacy.get("record_count") if isinstance(legacy.get("record_count"), int) else None,
        allow_unindexed_audio=False,
    )
    if records[0]["index_schema"] != "Voice":
        raise PackageValidationError("legacy migration accepts only an inferred Voice index")
    referenced_paths = sorted({record["relative_audio_path"] for record in records}, key=_normalized_key)
    _legacy_package_file_set(package, referenced_paths)
    diagnostics = {
        _normalized_key(path): _audio_diagnostic(package, path, verify_with_torchaudio=verify_with_torchaudio)
        for path in referenced_paths
    }
    upgraded = _metadata(
        package_hash,
        archive_bytes,
        archive_root,
        hashlib.sha256(index_bytes).hexdigest(),
        len(index_bytes),
        records,
        diagnostics,
    )
    expected_legacy = {key: value for key, value in upgraded.items() if key in LEGACY_METADATA_KEYS}
    expected_legacy["schema_version"] = legacy["schema_version"]
    if legacy != expected_legacy:
        raise PackageValidationError("legacy metadata does not match the revalidated index and audio")

    upgraded_bytes = json.dumps(upgraded, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"
    old_hash = hashlib.sha256(legacy_bytes).hexdigest()
    new_hash = hashlib.sha256(upgraded_bytes).hexdigest()
    provenance = {
        "schema_version": MIGRATION_SCHEMA_VERSION,
        "migration_id": _stable_id("mig", legacy["package_id"], old_hash, new_hash),
        "package_id": legacy["package_id"],
        "package_sha256": package_hash,
        "old_metadata_sha256": old_hash,
        "new_metadata_sha256": new_hash,
        "added_fields": {"index_schema": "Voice", "unindexed_audio": []},
    }
    provenance_path = package / "metadata.migration.json"
    if provenance_path.exists():
        raise PackageValidationError("migration provenance path already exists")
    # Publish provenance first and metadata last: metadata replacement is the
    # migration commit point. Roll back provenance on ordinary write failures.
    try:
        _write_bytes_atomic(provenance_path, _json_bytes(provenance) + b"\n")
        _write_bytes_atomic(metadata_path, upgraded_bytes)
    except Exception:
        if hashlib.sha256(metadata_path.read_bytes()).hexdigest() == old_hash:
            provenance_path.unlink(missing_ok=True)
        raise
    return {"metadata": upgraded, "provenance": provenance}


def install_visualnovel_package(
    archive_path: str | Path,
    destination: str | Path,
    *,
    password: str | None = None,
    expected_records: int | None = None,
    limits: ResourceLimits = ResourceLimits(),
    verify_with_torchaudio: bool = False,
    allow_unindexed_audio: bool = False,
) -> dict:
    """Validate, extract, and atomically publish a VisualNovel package.

    ``destination`` is the long-lived package directory. A committed destination
    is an idempotent no-op only when its metadata records the same archive hash.
    Passwords are passed only to py7zr and are never included in diagnostics.
    """
    source = Path(archive_path).expanduser().resolve()
    target = Path(destination).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"package archive does not exist: {source}")
    package_hash = _sha256_file(source)

    ready = target / "READY"
    metadata_path = target / "metadata.json"
    if target.exists():
        if ready.is_file() and metadata_path.is_file():
            try:
                existing = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise PackageValidationError("existing package metadata is invalid") from error
            if existing.get("package_sha256") == package_hash:
                return existing
            raise FileExistsError("destination already contains a different committed package")
        raise FileExistsError("destination exists without a valid READY commit marker")

    target.parent.mkdir(parents=True, exist_ok=True)
    quarantine = Path(tempfile.mkdtemp(prefix=f".{target.name}.quarantine-", dir=target.parent))
    partial = quarantine / "package.partial"
    try:
        with _open_archive(source, password) as archive:
            root, inventory = validate_inventory(_py7zr_inventory(archive), limits)
            raw_index, index_sha256, index_bytes = _read_index_from_archive(
                archive, f"{root}/{INDEX_NAME}", limits.max_index_bytes
            )
            records = validate_index(
                raw_index,
                inventory,
                root=root,
                expected_records=expected_records,
                max_records_per_package=limits.max_records_per_package,
                allow_unindexed_audio=allow_unindexed_audio,
            )
            inventory_audio = _audio_inventory(inventory, root)
            referenced = {_normalized_key(record["relative_audio_path"]) for record in records}
            unindexed_paths = [
                path for key, path in sorted(inventory_audio.items()) if key not in referenced
            ]

        # Reopen because extraction consumes archive stream state in some py7zr versions.
        with _open_archive(source, password) as archive:
            partial.mkdir()
            try:
                archive.extractall(path=partial)
            except Exception as error:
                raise PackageValidationError("archive extraction failed") from error

        extracted_root = partial / root
        diagnostics = _verify_extracted_tree(
            extracted_root,
            records,
            unindexed_audio=unindexed_paths,
            inventory_audio=inventory_audio,
            verify_with_torchaudio=verify_with_torchaudio,
        )
        metadata = _metadata(
            package_hash,
            source.stat().st_size,
            root,
            index_sha256,
            index_bytes,
            records,
            diagnostics,
            [
                {
                    "relative_audio_path": path,
                    "sha256": diagnostics[_normalized_key(path)]["sha256"],
                    "bytes": diagnostics[_normalized_key(path)]["bytes"],
                }
                for path in unindexed_paths
            ],
        )
        (extracted_root / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        ready_partial = extracted_root / "READY.partial"
        ready_partial.write_text(package_hash + "\n", encoding="ascii", newline="\n")
        os.replace(ready_partial, extracted_root / "READY")
        os.replace(extracted_root, target)
        return metadata
    except PackageValidationError:
        raise
    except (FileExistsError, FileNotFoundError):
        raise
    except Exception as error:
        raise PackageValidationError("package installation failed") from error
    finally:
        shutil.rmtree(quarantine, ignore_errors=True)


def cli() -> None:
    parser = argparse.ArgumentParser(description="Install or migrate a validated VisualNovel voice package.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    migrate = subparsers.add_parser("migrate-legacy", help="Strictly upgrade legacy installed metadata")
    migrate.add_argument("package_dir", help="Long-lived installed package directory")
    migrate.add_argument("--verify-with-torchaudio", action="store_true")
    migrate_paths = subparsers.add_parser(
        "migrate-paths", help="Repair metadata paths to the unique canonical installed audio spelling"
    )
    migrate_paths.add_argument("package_dir", help="Long-lived installed package directory")
    migrate_paths.add_argument("--verify-with-torchaudio", action="store_true")
    args = parser.parse_args()
    if args.command == "migrate-legacy":
        result = migrate_legacy_visualnovel_package(
            args.package_dir, verify_with_torchaudio=args.verify_with_torchaudio
        )
    else:
        result = migrate_visualnovel_audio_paths(
            args.package_dir, verify_with_torchaudio=args.verify_with_torchaudio
        )
    print(json.dumps(result["provenance"], ensure_ascii=False, sort_keys=True))


__all__ = [
    "InventoryMember",
    "PackageValidationError",
    "ResourceLimits",
    "install_visualnovel_package",
    "migrate_legacy_visualnovel_package",
    "migrate_visualnovel_audio_paths",
    "source_rejection_reason",
    "validate_index",
    "validate_inventory",
]


if __name__ == "__main__":
    cli()
