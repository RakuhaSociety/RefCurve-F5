"""VisualNovel multi-package control plane.

This module owns immutable source selection, SQLite state, leases, download
verification, recovery, and dry-run archive deletion planning.  It deliberately
does not install packages, create sidecars, write Arrow, or start training.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import socket
import sqlite3
import sys
import tempfile
import time
import unicodedata
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

SCHEMA_VERSION = "visualnovel-selection-v1"
SCHEMA_VERSION_V2 = "visualnovel-selection-v2"
RECEIPT_VERSION = "source-receipt-v1"
TRANSFER_RECEIPT_VERSION = "transfer-receipt-v1"
CONTROLLER_VERSION = "visualnovel-control-a0-v3"
DEFAULT_LEASE_TTL_SECONDS = 24 * 60 * 60
MIN_LEASE_TTL_SECONDS = 60
MAX_LEASE_TTL_SECONDS = 7 * 24 * 60 * 60
DEFAULT_SELECTION = Path(__file__).with_name("visualnovel_selection_v1.json")
AGGREGATE_FILES = {
    "aggregate_manifest": "manifest.json",
    "aggregate_eval_manifest": "eval_manifest.json",
    "aggregate_provenance": "aggregate_provenance.json",
    "aggregate_ready": "READY",
}
WAVES = ("A", "B", "C", "D", "E", "F")
V1_WAVES = ("A", "B", "C")
V2_ADDITION_WAVES = ("D", "E", "F")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
BLOB_RE = re.compile(r"^[0-9a-f]{40}$")
ID_RE = re.compile(r"^(sel|item)_[0-9a-f]{24}$")
DURATION_RE = re.compile(r"^\d{2,3}:\d{2}:\d{2}$")

LINEAR_STATES = (
    "SELECTED",
    "TRANSFERRING",
    "DOWNLOADED_UNVERIFIED",
    "VERIFYING",
    "DOWNLOAD_VERIFIED",
    "INSTALLING",
    "PACKAGE_READY",
    "METADATA_EXPORTED",
    "SIDECAR_READY",
    "DATASET_READY",
    "AGGREGATED",
    "ARCHIVE_DELETE_ELIGIBLE",
    "ARCHIVE_DELETED",
)
ERROR_STATES = {
    "RETRYABLE_ERROR",
    "BLOCKED_SCHEMA",
    "BLOCKED_RESOURCE",
    "BLOCKED_INTEGRITY",
    "BLOCKED_DISK_HARD",
    "CANCELLED",
}
ALL_STATES = set(LINEAR_STATES) | ERROR_STATES | {"DOWNLOADING"}
LEGAL_TRANSITIONS = {state: {LINEAR_STATES[index + 1]} for index, state in enumerate(LINEAR_STATES[:-1])}
LEGAL_TRANSITIONS[LINEAR_STATES[-1]] = set()
for state in LINEAR_STATES:
    LEGAL_TRANSITIONS[state] |= ERROR_STATES
for state in ERROR_STATES:
    LEGAL_TRANSITIONS[state] = {"TRANSFERRING", "VERIFYING", "INSTALLING", "CANCELLED"} - {state}
# Legacy v1 download-one remains supported without weakening the new split path.
LEGAL_TRANSITIONS["SELECTED"].add("DOWNLOADING")
LEGAL_TRANSITIONS["DOWNLOADING"] = {"DOWNLOAD_VERIFIED"} | ERROR_STATES

SELECTION_KEYS = {
    "schema_version",
    "selection_id",
    "repo_id",
    "repo_type",
    "revision",
    "catalog",
    "policy",
    "items",
}
SELECTION_V2_KEYS = SELECTION_KEYS | {"base_selection_id", "base_selection_sha256"}
CATALOG_KEYS = {"path", "bytes", "git_blob_id", "content_sha256", "row_count"}
POLICY_KEYS = {"name", "expected_count", "default_wave", "allowed_waves"}
POLICY_V2_KEYS = {
    "name",
    "expected_count",
    "base_count",
    "additions_count",
    "default_wave",
    "ordered_waves",
    "wave_counts",
    "size_strata",
}
V2_SIZE_STRATA = (
    {"name": "small", "min_bytes": 1, "max_bytes": 256 * 1024**2, "count": 8, "wave": "D"},
    {"name": "medium", "min_bytes": 256 * 1024**2, "max_bytes": 1024**3, "count": 8, "wave": "E"},
    {"name": "large", "min_bytes": 1024**3, "max_bytes": 2**63, "count": 8, "wave": "F"},
)
V2_POLICY = {
    "name": "base-plus-size-stratified-v2",
    "expected_count": 36,
    "base_count": 12,
    "additions_count": 24,
    "default_wave": "D",
    "ordered_waves": list(WAVES),
    "wave_counts": {"A": 8, "B": 2, "C": 2, "D": 8, "E": 8, "F": 8},
    "size_strata": [dict(value) for value in V2_SIZE_STRATA],
}
ITEM_KEYS = {
    "selection_order",
    "item_id",
    "wave",
    "catalog_section",
    "catalog_no",
    "company",
    "romaji",
    "duration",
    "total_characters",
    "repo_path",
    "size",
    "git_blob_id",
    "lfs_sha256",
    "expected_record_count",
    "resource_profile",
}


class ControlError(RuntimeError):
    """Base control-plane error."""


class SelectionError(ControlError):
    """The immutable selection or catalog/API join is invalid."""


class StateError(ControlError):
    """A state transition, lease, or artifact invariant failed."""


class IntegrityError(ControlError):
    """Downloaded bytes do not match the frozen source identity."""


def _validated_lease_ttl(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StateError("lease TTL must be numeric")
    ttl = float(value)
    if not MIN_LEASE_TTL_SECONDS <= ttl <= MAX_LEASE_TTL_SECONDS:
        raise StateError(
            f"lease TTL must be between {MIN_LEASE_TTL_SECONDS} and {MAX_LEASE_TTL_SECONDS} seconds"
        )
    return ttl


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Durably replace one evidence file without exposing partial JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        if os.name != "nt":
            directory_descriptor = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def stable_id(prefix: str, value: object) -> str:
    return f"{prefix}_{hashlib.sha256(canonical_bytes(value)).hexdigest()[:24]}"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_exact_keys(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    actual = set(value)
    if actual != keys:
        raise SelectionError(f"{label} keys differ: missing={sorted(keys - actual)}, extra={sorted(actual - keys)}")


def _positive_int(value: object, label: str, *, allow_none: bool = False) -> int | None:
    if allow_none and value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise SelectionError(f"{label} must be a positive integer")
    return value


def _safe_repo_path(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise SelectionError("repo_path must be a non-empty POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or len(path.parts) != 2 or any(part in {"", ".", ".."} for part in path.parts):
        raise SelectionError(f"unsafe repo_path: {value!r}")
    if path.parts[0] != "GalGame" or path.suffix != ".7z" or unicodedata.normalize("NFC", value) != value:
        raise SelectionError(f"invalid archive repo_path: {value!r}")
    return value


def selection_payload(selection: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(selection)
    payload.pop("selection_id", None)
    return payload


def item_payload(item: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(item)
    payload.pop("item_id", None)
    return payload


def validate_selection_v1(selection: object) -> dict[str, Any]:
    if not isinstance(selection, dict):
        raise SelectionError("selection must be a JSON object")
    _require_exact_keys(selection, SELECTION_KEYS, "selection")
    if selection["schema_version"] != SCHEMA_VERSION or selection["repo_type"] != "dataset":
        raise SelectionError("unsupported selection schema or repo type")
    if not isinstance(selection["repo_id"], str) or selection["repo_id"].count("/") != 1:
        raise SelectionError("repo_id must be owner/name")
    if not isinstance(selection["revision"], str) or not COMMIT_RE.fullmatch(selection["revision"]):
        raise SelectionError("revision must be a full lowercase commit SHA")

    catalog = selection["catalog"]
    policy = selection["policy"]
    if not isinstance(catalog, dict) or not isinstance(policy, dict):
        raise SelectionError("catalog and policy must be objects")
    _require_exact_keys(catalog, CATALOG_KEYS, "catalog")
    _require_exact_keys(policy, POLICY_KEYS, "policy")
    if catalog["path"] != "CATALOG.md":
        raise SelectionError("catalog path must be CATALOG.md")
    _positive_int(catalog["bytes"], "catalog.bytes")
    _positive_int(catalog["row_count"], "catalog.row_count")
    if not BLOB_RE.fullmatch(str(catalog["git_blob_id"])) or not SHA256_RE.fullmatch(str(catalog["content_sha256"])):
        raise SelectionError("catalog identities are malformed")
    if policy != {
        "name": "pilot-structure-size-stratified-v1",
        "expected_count": 12,
        "default_wave": "A",
        "allowed_waves": ["A", "B", "C"],
    }:
        raise SelectionError("selection policy is not the approved frozen policy")

    items = selection["items"]
    if not isinstance(items, list) or len(items) != 12:
        raise SelectionError("selection must contain exactly 12 items")
    seen_paths: set[str] = set()
    seen_ids: set[str] = set()
    waves: list[str] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise SelectionError(f"item {index} must be an object")
        _require_exact_keys(item, ITEM_KEYS, f"item {index}")
        if item["selection_order"] != index:
            raise SelectionError("selection_order must be contiguous and match array order")
        if item["wave"] not in {"A", "B", "C"}:
            raise SelectionError("invalid wave")
        waves.append(item["wave"])
        for key in ("catalog_section", "company", "romaji", "resource_profile"):
            if not isinstance(item[key], str) or not item[key]:
                raise SelectionError(f"item {index}.{key} must be non-empty")
        if item["catalog_section"] != "GalGame" or not DURATION_RE.fullmatch(str(item["duration"])):
            raise SelectionError("invalid catalog section or duration")
        _positive_int(item["catalog_no"], "catalog_no")
        _positive_int(item["total_characters"], "total_characters")
        _positive_int(item["size"], "size")
        _positive_int(item["expected_record_count"], "expected_record_count", allow_none=True)
        path = _safe_repo_path(item["repo_path"])
        if path in seen_paths:
            raise SelectionError("duplicate repo_path")
        seen_paths.add(path)
        if not BLOB_RE.fullmatch(str(item["git_blob_id"])) or not SHA256_RE.fullmatch(str(item["lfs_sha256"])):
            raise SelectionError("item identities are malformed")
        expected_item_id = stable_id("item", item_payload(item))
        if item["item_id"] != expected_item_id or not ID_RE.fullmatch(item["item_id"]):
            raise SelectionError(f"item {index} has non-canonical item_id")
        if item["item_id"] in seen_ids:
            raise SelectionError("duplicate item_id")
        seen_ids.add(item["item_id"])
    if waves != ["A"] * 8 + ["B"] * 2 + ["C"] * 2:
        raise SelectionError("selection wave invariant must be A=8, B=2, C=2 in order")
    expected_selection_id = stable_id("sel", selection_payload(selection))
    if selection["selection_id"] != expected_selection_id or not ID_RE.fullmatch(selection["selection_id"]):
        raise SelectionError("selection_id is not canonical")
    return selection


def _validate_v2_policy(policy: object) -> dict[str, Any]:
    if not isinstance(policy, dict):
        raise SelectionError("policy must be an object")
    _require_exact_keys(policy, POLICY_V2_KEYS, "policy")
    if policy != V2_POLICY:
        raise SelectionError("selection policy is not the approved frozen v2 policy")
    return policy


def _normalized_repo_key(path: str) -> str:
    return unicodedata.normalize("NFKC", path).casefold()


def validate_selection_v2(
    selection: object,
    *,
    base_selection: Mapping[str, Any] | None = None,
    base_selection_bytes: bytes | None = None,
) -> dict[str, Any]:
    if not isinstance(selection, dict):
        raise SelectionError("selection must be a JSON object")
    _require_exact_keys(selection, SELECTION_V2_KEYS, "selection")
    if selection["schema_version"] != SCHEMA_VERSION_V2 or selection["repo_type"] != "dataset":
        raise SelectionError("unsupported selection schema or repo type")
    if not isinstance(selection["repo_id"], str) or selection["repo_id"].count("/") != 1:
        raise SelectionError("repo_id must be owner/name")
    if not isinstance(selection["revision"], str) or not COMMIT_RE.fullmatch(selection["revision"]):
        raise SelectionError("revision must be a full lowercase commit SHA")
    catalog = selection["catalog"]
    if not isinstance(catalog, dict):
        raise SelectionError("catalog must be an object")
    _require_exact_keys(catalog, CATALOG_KEYS, "catalog")
    if catalog["path"] != "CATALOG.md":
        raise SelectionError("catalog path must be CATALOG.md")
    _positive_int(catalog["bytes"], "catalog.bytes")
    _positive_int(catalog["row_count"], "catalog.row_count")
    if not BLOB_RE.fullmatch(str(catalog["git_blob_id"])) or not SHA256_RE.fullmatch(str(catalog["content_sha256"])):
        raise SelectionError("catalog identities are malformed")
    _validate_v2_policy(selection["policy"])
    if not ID_RE.fullmatch(str(selection["base_selection_id"])) or not SHA256_RE.fullmatch(
        str(selection["base_selection_sha256"])
    ):
        raise SelectionError("base selection identities are malformed")

    items = selection["items"]
    if not isinstance(items, list) or len(items) != 36:
        raise SelectionError("selection-v2 must contain exactly 36 effective items")
    seen_paths: set[str] = set()
    seen_lfs: set[str] = set()
    seen_ids: set[str] = set()
    expected_waves = ["A"] * 8 + ["B"] * 2 + ["C"] * 2 + ["D"] * 8 + ["E"] * 8 + ["F"] * 8
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise SelectionError(f"item {index} must be an object")
        _require_exact_keys(item, ITEM_KEYS, f"item {index}")
        if item["selection_order"] != index or item["wave"] != expected_waves[index]:
            raise SelectionError("selection-v2 orders and A/B/C/D/E/F wave counts must be exact")
        for key in ("catalog_section", "company", "romaji", "resource_profile"):
            if not isinstance(item[key], str) or not item[key]:
                raise SelectionError(f"item {index}.{key} must be non-empty")
        if item["catalog_section"] != "GalGame" or not DURATION_RE.fullmatch(str(item["duration"])):
            raise SelectionError("invalid catalog section or duration")
        _positive_int(item["catalog_no"], "catalog_no")
        _positive_int(item["total_characters"], "total_characters")
        _positive_int(item["size"], "size")
        _positive_int(item["expected_record_count"], "expected_record_count", allow_none=True)
        path = _safe_repo_path(item["repo_path"])
        path_key = _normalized_repo_key(path)
        lfs_sha = str(item["lfs_sha256"])
        if path_key in seen_paths:
            raise SelectionError("duplicate normalized repo_path")
        if lfs_sha in seen_lfs:
            raise SelectionError("duplicate LFS SHA-256")
        seen_paths.add(path_key)
        seen_lfs.add(lfs_sha)
        if not BLOB_RE.fullmatch(str(item["git_blob_id"])) or not SHA256_RE.fullmatch(lfs_sha):
            raise SelectionError("item identities are malformed")
        if item["item_id"] != stable_id("item", item_payload(item)) or not ID_RE.fullmatch(item["item_id"]):
            raise SelectionError(f"item {index} has non-canonical item_id")
        if item["item_id"] in seen_ids:
            raise SelectionError("duplicate item_id")
        seen_ids.add(item["item_id"])

    if base_selection is None:
        try:
            default_base_bytes = DEFAULT_SELECTION.read_bytes()
            default_base = json.loads(default_base_bytes.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise SelectionError("selection-v2 requires the frozen selection-v1 base") from error
        base_selection = default_base
        base_selection_bytes = default_base_bytes
    if base_selection is not None:
        validate_selection_v1(base_selection)
        expected_base_bytes = canonical_bytes(base_selection) if base_selection_bytes is None else base_selection_bytes
        if selection["base_selection_id"] != base_selection["selection_id"]:
            raise SelectionError("selection-v2 base_selection_id differs from selection-v1")
        if selection["base_selection_sha256"] != hashlib.sha256(expected_base_bytes).hexdigest():
            raise SelectionError("selection-v2 base selection SHA-256 differs from selection-v1")
        if selection["repo_id"] != base_selection["repo_id"] or selection["revision"] != base_selection["revision"]:
            raise SelectionError("selection-v2 repository identity differs from selection-v1")
        if selection["catalog"] != base_selection["catalog"]:
            raise SelectionError("selection-v2 catalog identity differs from selection-v1")
        if items[:12] != base_selection["items"]:
            raise SelectionError("selection-v2 base prefix must exactly equal selection-v1 items")
        base_paths = {_normalized_repo_key(item["repo_path"]) for item in base_selection["items"]}
        base_lfs = {item["lfs_sha256"] for item in base_selection["items"]}
        additions = items[12:]
        if any(_normalized_repo_key(item["repo_path"]) in base_paths for item in additions):
            raise SelectionError("selection-v2 additions overlap base repo paths")
        if any(item["lfs_sha256"] in base_lfs for item in additions):
            raise SelectionError("selection-v2 additions overlap base LFS identities")
    if selection["selection_id"] != stable_id("sel", selection_payload(selection)) or not ID_RE.fullmatch(
        selection["selection_id"]
    ):
        raise SelectionError("selection_id is not canonical")
    return selection


def validate_selection(
    selection: object,
    *,
    base_selection: Mapping[str, Any] | None = None,
    base_selection_bytes: bytes | None = None,
) -> dict[str, Any]:
    if not isinstance(selection, dict):
        raise SelectionError("selection must be a JSON object")
    schema = selection.get("schema_version")
    if schema == SCHEMA_VERSION:
        return validate_selection_v1(selection)
    if schema == SCHEMA_VERSION_V2:
        return validate_selection_v2(
            selection, base_selection=base_selection, base_selection_bytes=base_selection_bytes
        )
    raise SelectionError(f"unsupported selection schema: {schema!r}")


def selection_waves(selection: Mapping[str, Any]) -> tuple[str, ...]:
    validated = validate_selection(selection)
    if validated["schema_version"] == SCHEMA_VERSION:
        return V1_WAVES
    return tuple(validated["policy"]["ordered_waves"])


def run_items(selection: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    validated = validate_selection(selection)
    if validated["schema_version"] == SCHEMA_VERSION_V2:
        return list(validated["items"][validated["policy"]["base_count"] :])
    return list(validated["items"])


def load_selection(path: str | os.PathLike[str] = DEFAULT_SELECTION) -> dict[str, Any]:
    source = Path(path)
    try:
        raw = source.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SelectionError(f"cannot read selection: {source}") from error
    if isinstance(value, dict) and value.get("schema_version") == SCHEMA_VERSION_V2:
        base_path = source.with_name("visualnovel_selection_v1.json")
        try:
            base_raw = base_path.read_bytes()
            base = json.loads(base_raw.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise SelectionError(f"cannot read base selection: {base_path}") from error
        selection = validate_selection(value, base_selection=base, base_selection_bytes=base_raw)
    else:
        selection = validate_selection(value)
    if raw != canonical_bytes(selection):
        raise SelectionError("selection file is not canonical JSON with one trailing newline")
    return selection


def _unescape_markdown(value: str) -> str:
    return re.sub(r"\\([\\`*_{}\[\]()#+\-.!~|])", r"\1", value).strip()


def parse_catalog(text: str) -> list[dict[str, Any]]:
    """Parse strict Markdown table rows from CATALOG.md.

    Section headings select the repo directory. Table rows must have exactly
    five cells: No., Company, Duration, Characters, and Romaji title.
    """
    section: str | None = None
    rows: list[dict[str, Any]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        heading = re.fullmatch(r"#{1,6}\s+(.+?)\s*", line)
        if heading:
            section = _unescape_markdown(heading.group(1))
            continue
        if not line.startswith("|") or re.fullmatch(r"\|?[\s:|-]+\|?", line):
            continue
        cells = [_unescape_markdown(cell) for cell in line.strip("|").split("|")]
        if len(cells) != 5 or not cells[0].isdigit():
            continue
        if section is None:
            raise SelectionError("catalog table appears before a section heading")
        number, company, duration, characters, romaji = cells
        if not company or not romaji or not DURATION_RE.fullmatch(duration) or not characters.isdigit():
            raise SelectionError(f"malformed catalog row {number}")
        repo_path = f"{section}/{company}_{romaji}.7z"
        rows.append(
            {
                "catalog_section": section,
                "catalog_no": int(number),
                "company": company,
                "romaji": romaji,
                "duration": duration,
                "total_characters": int(characters),
                "repo_path": _safe_repo_path(repo_path),
            }
        )
    if not rows:
        raise SelectionError("catalog contains no data rows")
    keys = [unicodedata.normalize("NFKC", row["repo_path"]).casefold() for row in rows]
    if len(keys) != len(set(keys)):
        raise SelectionError("catalog contains duplicate normalized archive keys")
    return rows


def _field(value: object, name: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def normalize_api_file(value: object) -> dict[str, Any]:
    path = _safe_repo_path(_field(value, "path", _field(value, "rfilename")))
    size = _field(value, "size")
    blob = _field(value, "blob_id", _field(value, "blobId", _field(value, "oid")))
    lfs = _field(value, "lfs")
    lfs_size = _field(lfs, "size")
    lfs_sha = _field(lfs, "sha256", _field(lfs, "oid"))
    if not isinstance(size, int) or size <= 0 or size != lfs_size:
        raise SelectionError(f"API size/LFS size mismatch for {path}")
    if not isinstance(blob, str) or not BLOB_RE.fullmatch(blob):
        raise SelectionError(f"missing Git blob identity for {path}")
    if not isinstance(lfs_sha, str) or not SHA256_RE.fullmatch(lfs_sha):
        raise SelectionError(f"missing LFS SHA-256 for {path}")
    return {"repo_path": path, "size": size, "git_blob_id": blob, "lfs_sha256": lfs_sha}


def validate_resolved_join(
    catalog_rows: Sequence[Mapping[str, Any]], api_files: Iterable[object], *, expected_count: int = 597
) -> list[dict[str, Any]]:
    archives = [normalize_api_file(value) for value in api_files if str(_field(value, "path", _field(value, "rfilename", ""))).endswith(".7z")]
    if len(catalog_rows) != expected_count or len(archives) != expected_count:
        raise SelectionError(f"catalog/API count mismatch: {len(catalog_rows)}/{len(archives)}, expected {expected_count}")
    catalog_by_path = {row["repo_path"]: dict(row) for row in catalog_rows}
    api_by_path = {row["repo_path"]: row for row in archives}
    if len(catalog_by_path) != expected_count or len(api_by_path) != expected_count:
        raise SelectionError("duplicate catalog or API paths")
    if set(catalog_by_path) != set(api_by_path):
        missing_api = sorted(set(catalog_by_path) - set(api_by_path))[:3]
        missing_catalog = sorted(set(api_by_path) - set(catalog_by_path))[:3]
        raise SelectionError(f"catalog/API paths differ: missing_api={missing_api}, missing_catalog={missing_catalog}")
    return [{**catalog_by_path[path], **api_by_path[path]} for path in sorted(catalog_by_path)]


def duration_seconds(value: str) -> int:
    if not DURATION_RE.fullmatch(value):
        raise SelectionError(f"invalid duration: {value!r}")
    hours, minutes, seconds = (int(part) for part in value.split(":"))
    if minutes >= 60 or seconds >= 60:
        raise SelectionError(f"invalid duration: {value!r}")
    return hours * 3600 + minutes * 60 + seconds


def _v2_stratum(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    matches = [
        stratum
        for stratum in V2_SIZE_STRATA
        if stratum["min_bytes"] <= candidate["size"] < stratum["max_bytes"]
    ]
    if len(matches) != 1:
        raise SelectionError(f"candidate does not belong to exactly one approved size stratum: {candidate['repo_path']}")
    return matches[0]


def build_selection_v2(
    base_selection: Mapping[str, Any],
    resolved_rows: Sequence[Mapping[str, Any]],
    *,
    base_selection_bytes: bytes | None = None,
) -> dict[str, Any]:
    """Build deterministic selection-v2 from an already pinned, strict catalog/API join."""
    base = validate_selection_v1(dict(base_selection))
    if len(resolved_rows) != base["catalog"]["row_count"]:
        raise SelectionError(
            f"resolved catalog/API row count must be exactly {base['catalog']['row_count']}"
        )
    base_paths = {_normalized_repo_key(item["repo_path"]) for item in base["items"]}
    base_lfs = {item["lfs_sha256"] for item in base["items"]}
    unique_paths: set[str] = set()
    unique_lfs: set[str] = set()
    candidates: list[dict[str, Any]] = []
    for raw in resolved_rows:
        required = {
            "catalog_section", "catalog_no", "company", "romaji", "duration", "total_characters",
            "repo_path", "size", "git_blob_id", "lfs_sha256",
        }
        if not isinstance(raw, Mapping) or not required.issubset(raw):
            raise SelectionError("resolved candidate row is missing required catalog/API fields")
        candidate = {key: raw[key] for key in required}
        path = _safe_repo_path(candidate["repo_path"])
        path_key = _normalized_repo_key(path)
        lfs_sha = str(candidate["lfs_sha256"])
        duration_seconds(str(candidate["duration"]))
        _positive_int(candidate["size"], "candidate.size")
        _positive_int(candidate["catalog_no"], "candidate.catalog_no")
        _positive_int(candidate["total_characters"], "candidate.total_characters")
        if not BLOB_RE.fullmatch(str(candidate["git_blob_id"])) or not SHA256_RE.fullmatch(lfs_sha):
            raise SelectionError("resolved candidate identities are malformed")
        if path_key in unique_paths or lfs_sha in unique_lfs:
            raise SelectionError("resolved rows contain duplicate path or LFS identity")
        unique_paths.add(path_key)
        unique_lfs.add(lfs_sha)
        if path_key in base_paths or lfs_sha in base_lfs:
            continue
        _v2_stratum(candidate)
        candidates.append(candidate)

    additions: list[dict[str, Any]] = []
    for stratum in V2_SIZE_STRATA:
        eligible = [candidate for candidate in candidates if _v2_stratum(candidate)["name"] == stratum["name"]]
        eligible.sort(
            key=lambda value: (
                value["size"],
                duration_seconds(value["duration"]),
                _normalized_repo_key(value["repo_path"]),
                value["git_blob_id"],
                value["lfs_sha256"],
            )
        )
        if len(eligible) < stratum["count"]:
            raise SelectionError(
                f"insufficient candidates in stratum {stratum['name']}: {len(eligible)} < {stratum['count']}"
            )
        for candidate in eligible[: stratum["count"]]:
            item = {
                "selection_order": 12 + len(additions),
                "item_id": "",
                "wave": stratum["wave"],
                **candidate,
                "expected_record_count": None,
                "resource_profile": f"expansion-v2-{stratum['name']}",
            }
            item["item_id"] = stable_id("item", item_payload(item))
            additions.append(item)

    base_bytes = canonical_bytes(base) if base_selection_bytes is None else base_selection_bytes
    selection = {
        "schema_version": SCHEMA_VERSION_V2,
        "selection_id": "",
        "repo_id": base["repo_id"],
        "repo_type": base["repo_type"],
        "revision": base["revision"],
        "catalog": dict(base["catalog"]),
        "base_selection_id": base["selection_id"],
        "base_selection_sha256": hashlib.sha256(base_bytes).hexdigest(),
        "policy": copy.deepcopy(V2_POLICY),
        "items": copy.deepcopy(base["items"]) + additions,
    }
    selection["selection_id"] = stable_id("sel", selection_payload(selection))
    return validate_selection_v2(selection, base_selection=base, base_selection_bytes=base_bytes)


def generate_selection_v2(
    base_selection: Mapping[str, Any],
    catalog_rows: Sequence[Mapping[str, Any]],
    api_files: Iterable[object],
    *,
    base_selection_bytes: bytes | None = None,
) -> dict[str, Any]:
    """Join already-resolved pinned metadata and build v2 without network access."""
    joined = validate_resolved_join(
        catalog_rows, api_files, expected_count=base_selection["catalog"]["row_count"]
    )
    return build_selection_v2(base_selection, joined, base_selection_bytes=base_selection_bytes)


def verify_remote_selection(
    selection: Mapping[str, Any], *, resolve_revision: Callable[..., str], fetch_catalog: Callable[..., bytes], list_files: Callable[..., Iterable[object]]
) -> None:
    """Verify the frozen selection against injectable network adapters."""
    validate_selection(selection)
    resolved = resolve_revision(repo_id=selection["repo_id"], repo_type="dataset", revision=selection["revision"])
    if resolved != selection["revision"]:
        raise SelectionError("revision resolver did not return the frozen commit")
    catalog_bytes = fetch_catalog(
        repo_id=selection["repo_id"], repo_type="dataset", revision=selection["revision"], filename="CATALOG.md"
    )
    catalog = selection["catalog"]
    if len(catalog_bytes) != catalog["bytes"] or hashlib.sha256(catalog_bytes).hexdigest() != catalog["content_sha256"]:
        raise SelectionError("remote CATALOG content differs from frozen identity")
    rows = parse_catalog(catalog_bytes.decode("utf-8"))
    remote_files = list(
        list_files(repo_id=selection["repo_id"], repo_type="dataset", revision=selection["revision"])
    )
    catalog_files = [value for value in remote_files if _field(value, "path", _field(value, "rfilename")) == "CATALOG.md"]
    if len(catalog_files) != 1:
        raise SelectionError("remote API must contain exactly one CATALOG.md object")
    remote_catalog_blob = _field(
        catalog_files[0], "blob_id", _field(catalog_files[0], "blobId", _field(catalog_files[0], "oid"))
    )
    remote_catalog_size = _field(catalog_files[0], "size")
    if remote_catalog_blob != catalog["git_blob_id"] or remote_catalog_size != catalog["bytes"]:
        raise SelectionError("remote CATALOG API identity differs from frozen identity")
    joined = validate_resolved_join(rows, remote_files, expected_count=catalog["row_count"])
    by_path = {row["repo_path"]: row for row in joined}
    for item in selection["items"]:
        remote = by_path.get(item["repo_path"])
        if remote is None:
            raise SelectionError(f"selected path absent remotely: {item['repo_path']}")
        for key in ("size", "git_blob_id", "lfs_sha256", "catalog_no", "company", "romaji", "duration", "total_characters"):
            if remote[key] != item[key]:
                raise SelectionError(f"selected identity drift for {item['repo_path']}: {key}")


@dataclass(frozen=True)
class DiskAdmission:
    level: str
    allowed: bool
    required_bytes: int
    free_bytes: int
    soft_watermark_bytes: int
    hard_watermark_bytes: int
    reason: str


def default_watermarks(total_bytes: int) -> tuple[int, int]:
    gib = 1024**3
    soft = max(500 * gib, int(total_bytes * 0.12))
    hard = max(200 * gib, int(total_bytes * 0.05))
    return soft, hard


def disk_admission(
    *, total_bytes: int, free_bytes: int, archive_size: int, soft_watermark_bytes: int | None = None,
    hard_watermark_bytes: int | None = None, peak_multiplier: int = 8
) -> DiskAdmission:
    default_soft, default_hard = default_watermarks(total_bytes)
    soft = default_soft if soft_watermark_bytes is None else soft_watermark_bytes
    hard = default_hard if hard_watermark_bytes is None else hard_watermark_bytes
    if not (0 < hard < soft <= total_bytes):
        raise ValueError("watermarks must satisfy 0 < hard < soft <= total")
    required = archive_size * peak_multiplier
    if free_bytes <= hard:
        return DiskAdmission("hard", False, required, free_bytes, soft, hard, "free bytes are at/below hard watermark")
    if free_bytes - required <= hard:
        return DiskAdmission("soft", False, required, free_bytes, soft, hard, "conservative peak would cross hard watermark")
    if free_bytes <= soft:
        return DiskAdmission("soft", False, required, free_bytes, soft, hard, "free bytes are at/below soft watermark")
    return DiskAdmission("normal", True, required, free_bytes, soft, hard, "admitted")


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY, selection_id TEXT NOT NULL, created_at REAL NOT NULL,
    controller_version TEXT NOT NULL, host TEXT NOT NULL, status TEXT NOT NULL,
    wave_gate TEXT NOT NULL CHECK (wave_gate IN ('A','B','C','D','E','F')),
    soft_watermark_bytes INTEGER NOT NULL, hard_watermark_bytes INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS items (
    run_id TEXT NOT NULL, item_id TEXT NOT NULL, selection_order INTEGER NOT NULL,
    wave TEXT NOT NULL CHECK (wave IN ('A','B','C','D','E','F')), repo_path TEXT NOT NULL,
    revision TEXT NOT NULL, expected_size INTEGER NOT NULL, git_blob_id TEXT NOT NULL,
    lfs_sha256 TEXT NOT NULL, state TEXT NOT NULL, attempt INTEGER NOT NULL DEFAULT 0,
    lease_owner TEXT, lease_expires_at REAL, last_error_code TEXT, last_error_json TEXT,
    package_id TEXT, PRIMARY KEY (run_id,item_id), UNIQUE (run_id,selection_order),
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL, item_id TEXT NOT NULL,
    kind TEXT NOT NULL, uri TEXT NOT NULL, bytes INTEGER NOT NULL, sha256 TEXT NOT NULL,
    etag TEXT, revision TEXT NOT NULL, committed INTEGER NOT NULL CHECK (committed IN (0,1)),
    created_at REAL NOT NULL, UNIQUE(run_id,item_id,kind),
    FOREIGN KEY (run_id,item_id) REFERENCES items(run_id,item_id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS transitions (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL, item_id TEXT NOT NULL,
    from_state TEXT, to_state TEXT NOT NULL, attempt INTEGER NOT NULL, timestamp REAL NOT NULL,
    details_json TEXT NOT NULL,
    FOREIGN KEY (run_id,item_id) REFERENCES items(run_id,item_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_items_state ON items(run_id,state,selection_order);
CREATE INDEX IF NOT EXISTS idx_items_lease ON items(run_id,lease_expires_at);
"""


class ControlDB:
    def __init__(self, path: str | os.PathLike[str], *, events_path: str | os.PathLike[str] | None = None, busy_timeout_ms: int = 30_000):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.events_path = Path(events_path) if events_path else self.path.with_name("events.jsonl")
        self.connection = sqlite3.connect(self.path, isolation_level=None, timeout=busy_timeout_ms / 1000)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.execute(f"PRAGMA busy_timeout={int(busy_timeout_ms)}")
        self.connection.executescript(SCHEMA_SQL)

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "ControlDB":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            yield self.connection
        except BaseException:
            self.connection.rollback()
            raise
        else:
            self.connection.commit()

    def _append_event(self, event: Mapping[str, Any]) -> None:
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        with self.events_path.open("ab") as output:
            output.write(canonical_bytes(event))
            output.flush()
            os.fsync(output.fileno())

    def init_run(
        self, selection: Mapping[str, Any], *, run_id: str | None = None, wave_gate: str | None = None,
        soft_watermark_bytes: int = 500 * 1024**3, hard_watermark_bytes: int = 200 * 1024**3,
    ) -> str:
        validate_selection(selection)
        waves = selection_waves(selection)
        wave_gate = selection["policy"]["default_wave"] if wave_gate is None else wave_gate
        if wave_gate not in waves:
            raise StateError(f"wave gate must be one of {', '.join(waves)}")
        if hard_watermark_bytes >= soft_watermark_bytes:
            raise StateError("hard watermark must be below soft watermark")
        run_id = run_id or f"run_{uuid.uuid4().hex[:24]}"
        now = time.time()
        with self.transaction() as db:
            db.execute(
                "INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?)",
                (run_id, selection["selection_id"], now, CONTROLLER_VERSION, socket.gethostname(), "ACTIVE", wave_gate,
                 soft_watermark_bytes, hard_watermark_bytes),
            )
            for item in run_items(selection):
                db.execute(
                    "INSERT INTO items(run_id,item_id,selection_order,wave,repo_path,revision,expected_size,git_blob_id,lfs_sha256,state) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (run_id, item["item_id"], item["selection_order"], item["wave"], item["repo_path"], selection["revision"],
                     item["size"], item["git_blob_id"], item["lfs_sha256"], "SELECTED"),
                )
                db.execute(
                    "INSERT INTO transitions(run_id,item_id,from_state,to_state,attempt,timestamp,details_json) VALUES(?,?,?,?,?,?,?)",
                    (run_id, item["item_id"], None, "SELECTED", 0, now, "{}"),
                )
        self._append_event({"run_id": run_id, "event": "RUN_INITIALIZED", "selection_id": selection["selection_id"], "timestamp": now})
        return run_id

    def run(self, run_id: str) -> sqlite3.Row:
        row = self.connection.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if row is None:
            raise StateError(f"unknown run: {run_id}")
        return row

    def item(self, run_id: str, item_id: str) -> sqlite3.Row:
        row = self.connection.execute("SELECT * FROM items WHERE run_id=? AND item_id=?", (run_id, item_id)).fetchone()
        if row is None:
            raise StateError(f"unknown item: {item_id}")
        return row

    def status(self, run_id: str) -> dict[str, Any]:
        run = dict(self.run(run_id))
        rows = self.connection.execute(
            "SELECT state,COUNT(*) AS count FROM items WHERE run_id=? GROUP BY state ORDER BY state", (run_id,)
        ).fetchall()
        run["states"] = {row["state"]: row["count"] for row in rows}
        run["items"] = [dict(row) for row in self.connection.execute(
            "SELECT * FROM items WHERE run_id=? ORDER BY selection_order", (run_id,)
        )]
        return run

    def acquire_lease(self, run_id: str, item_id: str, owner: str, *, ttl_seconds: float = 900, now: float | None = None) -> bool:
        if not owner or ttl_seconds <= 0:
            raise StateError("lease owner and positive TTL are required")
        now = time.time() if now is None else now
        with self.transaction() as db:
            changed = db.execute(
                "UPDATE items SET lease_owner=?,lease_expires_at=? WHERE run_id=? AND item_id=? "
                "AND (lease_owner IS NULL OR lease_expires_at<=? OR lease_owner=?)",
                (owner, now + ttl_seconds, run_id, item_id, now, owner),
            ).rowcount
        return changed == 1

    def release_lease(self, run_id: str, item_id: str, owner: str) -> None:
        with self.transaction() as db:
            changed = db.execute(
                "UPDATE items SET lease_owner=NULL,lease_expires_at=NULL WHERE run_id=? AND item_id=? AND lease_owner=?",
                (run_id, item_id, owner),
            ).rowcount
        if changed != 1:
            raise StateError("lease is not owned by caller")

    def transition(
        self, run_id: str, item_id: str, expected_state: str, to_state: str, *, details: Mapping[str, Any] | None = None,
        owner: str | None = None, artifact: Mapping[str, Any] | None = None,
    ) -> int:
        if expected_state not in ALL_STATES or to_state not in LEGAL_TRANSITIONS.get(expected_state, set()):
            raise StateError(f"illegal transition: {expected_state} -> {to_state}")
        now = time.time()
        details_json = canonical_bytes(dict(details or {})).decode().strip()
        with self.transaction() as db:
            row = db.execute("SELECT * FROM items WHERE run_id=? AND item_id=?", (run_id, item_id)).fetchone()
            if row is None or row["state"] != expected_state:
                actual = None if row is None else row["state"]
                raise StateError(f"expected state {expected_state}, found {actual}")
            if owner is not None and (row["lease_owner"] != owner or (row["lease_expires_at"] or 0) <= now):
                raise StateError("valid caller-owned lease required")
            attempt = row["attempt"] + (1 if to_state in {"DOWNLOADING", "TRANSFERRING"} else 0)
            error_code = details.get("error_code") if details and to_state in ERROR_STATES else None
            db.execute(
                "UPDATE items SET state=?,attempt=?,last_error_code=?,last_error_json=? WHERE run_id=? AND item_id=? AND state=?",
                (to_state, attempt, error_code, details_json if error_code else None, run_id, item_id, expected_state),
            )
            if artifact is not None:
                required = {"kind", "uri", "bytes", "sha256", "etag", "revision", "committed"}
                if set(artifact) != required or not SHA256_RE.fullmatch(str(artifact["sha256"])):
                    raise StateError("artifact schema is invalid")
                db.execute(
                    "INSERT INTO artifacts(run_id,item_id,kind,uri,bytes,sha256,etag,revision,committed,created_at) VALUES(?,?,?,?,?,?,?,?,?,?) "
                    "ON CONFLICT(run_id,item_id,kind) DO UPDATE SET uri=excluded.uri,bytes=excluded.bytes,sha256=excluded.sha256,"
                    "etag=excluded.etag,revision=excluded.revision,committed=excluded.committed,created_at=excluded.created_at",
                    (run_id, item_id, artifact["kind"], artifact["uri"], artifact["bytes"], artifact["sha256"], artifact["etag"],
                     artifact["revision"], int(artifact["committed"]), now),
                )
            cursor = db.execute(
                "INSERT INTO transitions(run_id,item_id,from_state,to_state,attempt,timestamp,details_json) VALUES(?,?,?,?,?,?,?)",
                (run_id, item_id, expected_state, to_state, attempt, now, details_json),
            )
            event_id = int(cursor.lastrowid)
        self._append_event({"event_id": event_id, "run_id": run_id, "item_id": item_id, "from_state": expected_state,
                            "to_state": to_state, "attempt": attempt, "timestamp": now, "details": dict(details or {})})
        return event_id

    def next_selected(self, run_id: str) -> sqlite3.Row | None:
        gate = self.run(run_id)["wave_gate"]
        permitted = WAVES[: WAVES.index(gate) + 1]
        placeholders = ",".join("?" for _ in permitted)
        return self.connection.execute(
            f"SELECT * FROM items WHERE run_id=? AND state='SELECTED' AND wave IN ({placeholders}) ORDER BY selection_order LIMIT 1",
            (run_id, *permitted),
        ).fetchone()

    def register_aggregate(
        self,
        run_id: str,
        selection: Mapping[str, Any],
        aggregate_dir: str | os.PathLike[str],
        *,
        wave: str | None = None,
        expected_fingerprint: str | None = None,
        promote_wave: bool = False,
    ) -> dict[str, Any]:
        """Audit and atomically register an aggregate before advancing item state."""
        evidence = inspect_aggregate(
            selection,
            aggregate_dir,
            wave=wave or self.run(run_id)["wave_gate"],
            expected_fingerprint=expected_fingerprint,
        )
        target_wave = evidence["wave"]
        now = time.time()
        events: list[dict[str, Any]] = []
        with self.transaction() as db:
            run = db.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if run is None:
                raise StateError(f"unknown run: {run_id}")
            if run["selection_id"] != selection["selection_id"]:
                raise StateError("run selection does not match aggregate selection")
            rows = db.execute(
                "SELECT * FROM items WHERE run_id=? AND wave=? ORDER BY selection_order", (run_id, target_wave)
            ).fetchall()
            expected_items = [item for item in selection["items"] if item["wave"] == target_wave]
            if [row["item_id"] for row in rows] != [item["item_id"] for item in expected_items]:
                raise StateError("database wave items do not match frozen selection")
            for row in rows:
                if row["lease_owner"] is not None or row["lease_expires_at"] is not None:
                    raise StateError(f"aggregate target item has a lease: {row['item_id']}")
                if row["state"] not in {"DATASET_READY", "AGGREGATED"}:
                    raise StateError(f"aggregate target item is not DATASET_READY: {row['item_id']} ({row['state']})")
                chain = [event["to_state"] for event in db.execute(
                    "SELECT to_state FROM transitions WHERE run_id=? AND item_id=? ORDER BY event_id",
                    (run_id, row["item_id"]),
                ).fetchall()]
                try:
                    package_index = chain.index("PACKAGE_READY")
                    dataset_index = chain.index("DATASET_READY", package_index + 1)
                except ValueError as error:
                    raise StateError(f"item lacks PACKAGE_READY -> DATASET_READY chain: {row['item_id']}") from error
                if row["state"] == "AGGREGATED" and "AGGREGATED" not in chain[dataset_index + 1:]:
                    raise StateError(f"AGGREGATED item lacks transition evidence: {row['item_id']}")

            existing = db.execute(
                "SELECT a.item_id,a.kind,a.uri,a.bytes,a.sha256,a.revision,a.committed FROM artifacts a "
                "JOIN items i ON i.run_id=a.run_id AND i.item_id=a.item_id "
                "WHERE a.run_id=? AND i.wave=? AND a.kind IN "
                "('aggregate_manifest','aggregate_eval_manifest','aggregate_provenance','aggregate_ready')",
                (run_id, target_wave),
            ).fetchall()
            expected_artifacts = {
                (item["item_id"], artifact["kind"]): (
                    artifact["uri"], artifact["bytes"], artifact["sha256"], selection["revision"], 1
                )
                for item in expected_items
                for artifact in evidence["artifacts"]
            }
            for artifact in existing:
                key = (artifact["item_id"], artifact["kind"])
                actual = (artifact["uri"], artifact["bytes"], artifact["sha256"], artifact["revision"], artifact["committed"])
                if key not in expected_artifacts or actual != expected_artifacts[key]:
                    raise StateError("existing aggregate artifact does not match audited aggregate")

            details = {
                "aggregate_dir": evidence["aggregate_dir"],
                "deployment_fingerprint": evidence["deployment_fingerprint"],
                "semantic_fingerprint": evidence["semantic_fingerprint"],
                "wave": target_wave,
            }
            details_json = canonical_bytes(details).decode().strip()
            for row in rows:
                for artifact in evidence["artifacts"]:
                    db.execute(
                        "INSERT INTO artifacts(run_id,item_id,kind,uri,bytes,sha256,etag,revision,committed,created_at) "
                        "VALUES(?,?,?,?,?,?,?,?,?,?) ON CONFLICT(run_id,item_id,kind) DO NOTHING",
                        (run_id, row["item_id"], artifact["kind"], artifact["uri"], artifact["bytes"],
                         artifact["sha256"], None, selection["revision"], 1, now),
                    )
                if row["state"] == "AGGREGATED":
                    continue
                cursor = db.execute(
                    "UPDATE items SET state='AGGREGATED' WHERE run_id=? AND item_id=? AND state=?",
                    (run_id, row["item_id"], row["state"]),
                )
                if cursor.rowcount != 1:
                    raise StateError(f"aggregate item changed concurrently: {row['item_id']}")
                transition = db.execute(
                    "INSERT INTO transitions(run_id,item_id,from_state,to_state,attempt,timestamp,details_json) "
                    "VALUES(?,?,?,?,?,?,?)",
                    (run_id, row["item_id"], row["state"], "AGGREGATED", row["attempt"], now, details_json),
                )
                events.append({"event_id": int(transition.lastrowid), "run_id": run_id, "item_id": row["item_id"],
                               "from_state": row["state"], "to_state": "AGGREGATED", "attempt": row["attempt"],
                               "timestamp": now, "details": details})
            promoted = self._promote_wave_in_transaction(db, run, target_wave, now) if promote_wave else None

        for event in events:
            self._append_event(event)
        if promoted is not None:
            self._append_event(promoted)
        return {**evidence, "registered_items": len(rows), "changed_items": len(events),
                "wave_gate": promoted["to_wave"] if promoted else self.run(run_id)["wave_gate"]}

    @staticmethod
    def _promote_wave_in_transaction(
        db: sqlite3.Connection, run: sqlite3.Row, completed_wave: str, now: float
    ) -> dict[str, Any] | None:
        current = run["wave_gate"]
        waves = tuple(
            row["wave"]
            for row in db.execute(
                "SELECT wave FROM items WHERE run_id=? GROUP BY wave ORDER BY MIN(selection_order)",
                (run["run_id"],),
            ).fetchall()
        )
        if completed_wave not in waves:
            raise StateError(f"wave must be one of {', '.join(waves)}")
        index = waves.index(current)
        if completed_wave != current:
            if waves.index(completed_wave) < index:
                return None
            raise StateError(f"cannot promote wave gate {current} from aggregate wave {completed_wave}")
        if index == len(waves) - 1:
            return None
        incomplete = db.execute(
            "SELECT item_id,state FROM items WHERE run_id=? AND wave=? AND state!='AGGREGATED' ORDER BY selection_order",
            (run["run_id"], current),
        ).fetchall()
        if incomplete:
            raise StateError(f"cannot promote wave {current}: items are not AGGREGATED")
        target = waves[index + 1]
        changed = db.execute(
            "UPDATE runs SET wave_gate=? WHERE run_id=? AND wave_gate=?", (target, run["run_id"], current)
        ).rowcount
        if changed != 1:
            raise StateError("wave gate changed concurrently")
        return {"run_id": run["run_id"], "event": "WAVE_PROMOTED", "from_wave": current,
                "to_wave": target, "timestamp": now}

    def promote_wave(self, run_id: str, *, completed_wave: str | None = None) -> str:
        """Advance the gate by one only after the current wave is fully aggregated."""
        now = time.time()
        with self.transaction() as db:
            run = db.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if run is None:
                raise StateError(f"unknown run: {run_id}")
            event = self._promote_wave_in_transaction(db, run, completed_wave or run["wave_gate"], now)
        if event is not None:
            self._append_event(event)
            return event["to_wave"]
        return self.run(run_id)["wave_gate"]


def aggregate_catalog_identity(selection: Mapping[str, Any]) -> str:
    validate_selection(selection)
    return selection["catalog"]["content_sha256"]


def aggregate_selection_fingerprint(selection: Mapping[str, Any]) -> str:
    validate_selection(selection)
    return hashlib.sha256(canonical_bytes(selection_payload(selection))).hexdigest()


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise StateError(f"cannot read aggregate {label}: {path}") from error
    if not isinstance(value, dict):
        raise StateError(f"aggregate {label} must be a JSON object")
    return value


def inspect_aggregate(
    selection: Mapping[str, Any],
    aggregate_dir: str | os.PathLike[str],
    *,
    wave: str,
    expected_fingerprint: str | None = None,
) -> dict[str, Any]:
    """Validate committed aggregate evidence without mutating the control database."""
    validate_selection(selection)
    if wave not in selection_waves(selection):
        raise StateError(f"wave must be one of {', '.join(selection_waves(selection))}")
    if expected_fingerprint is not None and not SHA256_RE.fullmatch(expected_fingerprint):
        raise StateError("expected aggregate fingerprint must be a lowercase SHA-256")
    root = Path(aggregate_dir).expanduser().resolve()
    if not root.is_dir() or root.is_symlink():
        raise StateError(f"aggregate directory is missing or unsafe: {root}")
    required = {name: root / name for name in ("READY", "aggregate_provenance.json", "manifest.json", "eval_manifest.json")}
    missing = [name for name, path in required.items() if not path.is_file() or path.is_symlink()]
    if missing:
        raise StateError(f"aggregate evidence is incomplete: {missing}")
    ready = _read_json_object(required["READY"], "READY")
    provenance = _read_json_object(required["aggregate_provenance.json"], "provenance")
    for key in ("commit", "semantic_fingerprint", "deployment_fingerprint", "train_manifest_sha256", "eval_manifest_sha256"):
        if not SHA256_RE.fullmatch(str(ready.get(key))):
            raise StateError(f"aggregate READY.{key} is missing or malformed")
    for key in ("semantic_fingerprint", "deployment_fingerprint", "manifest_sha256"):
        if not SHA256_RE.fullmatch(str(provenance.get(key))):
            raise StateError(f"aggregate provenance.{key} is missing or malformed")
    if ready["commit"] != ready["deployment_fingerprint"] or ready["deployment_fingerprint"] != provenance["deployment_fingerprint"]:
        raise StateError("aggregate deployment fingerprint mismatch")
    if ready["semantic_fingerprint"] != provenance["semantic_fingerprint"]:
        raise StateError("aggregate semantic fingerprint mismatch")
    if expected_fingerprint is not None and expected_fingerprint not in {
        ready["semantic_fingerprint"], ready["deployment_fingerprint"]
    }:
        raise StateError("aggregate fingerprint does not match expected fingerprint")
    manifest_sha = sha256_file(required["manifest.json"])
    eval_manifest_sha = sha256_file(required["eval_manifest.json"])
    if ready["train_manifest_sha256"] != manifest_sha or provenance["manifest_sha256"] != manifest_sha:
        raise StateError("aggregate train manifest SHA-256 mismatch")
    if ready["eval_manifest_sha256"] != eval_manifest_sha:
        raise StateError("aggregate eval manifest SHA-256 mismatch")
    context = provenance.get("aggregate_context")
    expected_context = {
        "source_revision": selection["revision"],
        "catalog_identity": aggregate_catalog_identity(selection),
        "selection_fingerprint": aggregate_selection_fingerprint(selection),
    }
    if context != expected_context:
        raise StateError("aggregate context does not match frozen revision/catalog/selection")
    inputs = provenance.get("inputs")
    waves = selection_waves(selection)
    target_index = waves.index(wave)
    selected = [item for item in selection["items"] if waves.index(item["wave"]) <= target_index]
    if not isinstance(inputs, list) or len(inputs) != len(selected):
        raise StateError("aggregate inputs do not cover exactly the cumulative selection through target wave")
    expected_positions = {(item["wave"], item["selection_order"]) for item in selected}
    actual_positions = {
        (value.get("wave"), value.get("selection_order")) for value in inputs if isinstance(value, dict)
    }
    if actual_positions != expected_positions:
        raise StateError("aggregate inputs do not match cumulative selection positions through target wave")
    artifacts = []
    for kind, filename in AGGREGATE_FILES.items():
        path = root / filename
        artifacts.append({"kind": kind, "uri": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return {
        "aggregate_dir": str(root),
        "wave": wave,
        "semantic_fingerprint": ready["semantic_fingerprint"],
        "deployment_fingerprint": ready["deployment_fingerprint"],
        "artifacts": artifacts,
    }


class OperationLock:
    """Atomic filesystem lock with explicit conflict classes."""

    CONFLICTS = {
        "transfer": {"transfer"},
        "training": {"heavy"},
        "heavy": {"training"},
    }

    def __init__(self, root: str | os.PathLike[str], operation: str, owner: str):
        if operation not in self.CONFLICTS or not owner:
            raise StateError("operation lock requires transfer/training/heavy and a non-empty owner")
        self.root = Path(root)
        self.operation = operation
        self.owner = owner
        self.token = uuid.uuid4().hex
        self.path = self.root / f"{operation}.lock"
        self._held = False

    def acquire(self) -> "OperationLock":
        self.root.mkdir(parents=True, exist_ok=True)
        conflicts = self.CONFLICTS[self.operation]
        present = sorted(operation for operation in conflicts if (self.root / f"{operation}.lock").exists())
        if present:
            raise StateError(f"operation lock conflict: {self.operation} vs {present}")
        payload = canonical_bytes(
            {"operation": self.operation, "owner": self.owner, "pid": os.getpid(), "token": self.token}
        )
        try:
            descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError as error:
            raise StateError(f"operation lock already held: {self.operation}") from error
        try:
            os.write(descriptor, payload)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        # Close the check/create race for cross-class locks by rechecking and backing out.
        present = sorted(
            operation
            for operation in conflicts
            if operation != self.operation and (self.root / f"{operation}.lock").exists()
        )
        if present:
            self.path.unlink(missing_ok=True)
            raise StateError(f"operation lock conflict: {self.operation} vs {present}")
        self._held = True
        return self

    def release(self) -> None:
        if not self._held:
            return
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise StateError(f"cannot verify operation lock ownership: {self.path}") from error
        if value.get("token") != self.token:
            raise StateError("operation lock is not owned by caller")
        self.path.unlink()
        self._held = False

    def __enter__(self) -> "OperationLock":
        return self.acquire()

    def __exit__(self, *_: object) -> None:
        self.release()


def operation_lock(root: str | os.PathLike[str], operation: str, owner: str) -> OperationLock:
    return OperationLock(root, operation, owner)


def assert_operation_allowed(root: str | os.PathLike[str], operation: str) -> None:
    """Fail closed without acquiring, for wrappers that own a longer-lived lock."""
    if operation not in OperationLock.CONFLICTS:
        raise StateError(f"unknown operation class: {operation}")
    path = Path(root)
    conflicts = sorted(
        value for value in OperationLock.CONFLICTS[operation] if (path / f"{value}.lock").exists()
    )
    if conflicts:
        raise StateError(f"operation lock conflict: {operation} vs {conflicts}")


def _selected_item(selection: Mapping[str, Any], item_id: str) -> Mapping[str, Any]:
    validate_selection(selection)
    selected = {item["item_id"]: item for item in run_items(selection)}
    if item_id not in selected:
        raise SelectionError("item is not in this run's frozen selection")
    return selected[item_id]


def _check_wave_gate(db: ControlDB, run_id: str, wave: str) -> None:
    gate = db.run(run_id)["wave_gate"]
    if WAVES.index(wave) > WAVES.index(gate):
        raise StateError(f"wave gate {gate} blocks wave {wave}")


def fetch_one(
    *,
    db: ControlDB,
    run_id: str,
    selection: Mapping[str, Any],
    item_id: str,
    spool_root: Path,
    owner: str,
    authorized_paths: Iterable[str],
    downloader: Callable[..., str | os.PathLike[str]] | None = None,
    disk_usage: Callable[[Path], Any] = shutil.disk_usage,
    lock_root: Path | None = None,
    lease_ttl_seconds: float = DEFAULT_LEASE_TTL_SECONDS,
) -> Path:
    """Transfer one exact pinned file and check expected size only; never hash it."""
    lease_ttl_seconds = _validated_lease_ttl(lease_ttl_seconds)
    item = _selected_item(selection, item_id)
    allowed = {_safe_repo_path(path) for path in authorized_paths}
    if item["repo_path"] not in allowed:
        raise StateError("selected repo_path is not present in the exact transfer authorization allowlist")
    row = db.item(run_id, item_id)
    if row["state"] != "SELECTED":
        raise StateError("fetch-one requires SELECTED state")
    spool_root.mkdir(parents=True, exist_ok=True)
    usage = disk_usage(spool_root)
    run = db.run(run_id)
    admission = disk_admission(
        total_bytes=usage.total,
        free_bytes=usage.free,
        archive_size=item["size"],
        soft_watermark_bytes=run["soft_watermark_bytes"],
        hard_watermark_bytes=run["hard_watermark_bytes"],
    )
    if not admission.allowed:
        raise StateError(f"disk admission denied ({admission.level}): {admission.reason}")
    locks = lock_root or db.path.parent / "operation-locks"
    with operation_lock(locks, "transfer", owner):
        if not db.acquire_lease(run_id, item_id, owner, ttl_seconds=lease_ttl_seconds):
            raise StateError("item lease is held by another worker")
        try:
            db.transition(run_id, item_id, "SELECTED", "TRANSFERRING", owner=owner)
            if downloader is None:
                try:
                    from huggingface_hub import hf_hub_download
                except ImportError as error:
                    raise ControlError("huggingface_hub is required for downloads") from error
                downloader = hf_hub_download
            downloaded = Path(
                downloader(
                    repo_id=selection["repo_id"],
                    repo_type="dataset",
                    revision=selection["revision"],
                    filename=item["repo_path"],
                    cache_dir=str(spool_root / ".hf-cache"),
                    local_dir=str(spool_root / "files"),
                    token=True,
                )
            )
            if not db.acquire_lease(run_id, item_id, owner, ttl_seconds=lease_ttl_seconds):
                raise StateError("could not renew caller-owned lease after transfer")
            expected_path = spool_root / "files" / PurePosixPath(item["repo_path"])
            if (
                not downloaded.is_file()
                or downloaded.is_symlink()
                or downloaded.resolve() != expected_path.resolve()
            ):
                raise IntegrityError("download adapter returned an unexpected or unsafe path")
            observed_bytes = downloaded.stat().st_size
            if observed_bytes != item["size"]:
                raise IntegrityError("transferred archive size differs from frozen expected size")
            transfer_receipt = {
                "schema_version": TRANSFER_RECEIPT_VERSION,
                "repo_id": selection["repo_id"],
                "repo_type": selection["repo_type"],
                "revision": selection["revision"],
                "repo_path": item["repo_path"],
                "expected_size": item["size"],
                "observed_bytes": observed_bytes,
            }
            receipt_path = downloaded.with_name(downloaded.name + ".transfer-receipt-v1.json")
            atomic_write_bytes(receipt_path, canonical_bytes(transfer_receipt))
            db.transition(
                run_id,
                item_id,
                "TRANSFERRING",
                "DOWNLOADED_UNVERIFIED",
                owner=owner,
                details={"transfer_receipt_path": str(receipt_path), "observed_bytes": observed_bytes},
            )
            return downloaded
        except Exception as error:
            current = db.item(run_id, item_id)["state"]
            if current == "TRANSFERRING":
                target = "BLOCKED_INTEGRITY" if isinstance(error, IntegrityError) else "RETRYABLE_ERROR"
                db.transition(
                    run_id,
                    item_id,
                    current,
                    target,
                    details={"error_code": type(error).__name__, "message": str(error)},
                    owner=owner,
                )
            raise
        finally:
            try:
                db.release_lease(run_id, item_id, owner)
            except StateError:
                pass


def verify_one(
    *,
    db: ControlDB,
    run_id: str,
    selection: Mapping[str, Any],
    item_id: str,
    spool_root: Path,
    owner: str,
    lock_root: Path | None = None,
    lease_ttl_seconds: float = DEFAULT_LEASE_TTL_SECONDS,
) -> Path:
    """Hash a transferred file, verify LFS identity, and write the authoritative receipt."""
    lease_ttl_seconds = _validated_lease_ttl(lease_ttl_seconds)
    item = _selected_item(selection, item_id)
    row = db.item(run_id, item_id)
    if row["state"] != "DOWNLOADED_UNVERIFIED":
        raise StateError("verify-one requires DOWNLOADED_UNVERIFIED state")
    archive = spool_root / "files" / PurePosixPath(item["repo_path"])
    transfer_receipt_path = archive.with_name(archive.name + ".transfer-receipt-v1.json")
    locks = lock_root or db.path.parent / "operation-locks"
    with operation_lock(locks, "heavy", owner):
        if not db.acquire_lease(run_id, item_id, owner, ttl_seconds=lease_ttl_seconds):
            raise StateError("item lease is held by another worker")
        try:
            db.transition(run_id, item_id, "DOWNLOADED_UNVERIFIED", "VERIFYING", owner=owner)
            if not archive.is_file() or archive.is_symlink():
                raise IntegrityError("transferred archive is missing or unsafe")
            transfer_receipt = _read_json_object(transfer_receipt_path, "transfer receipt")
            expected_transfer = {
                "schema_version": TRANSFER_RECEIPT_VERSION,
                "repo_id": selection["repo_id"],
                "repo_type": selection["repo_type"],
                "revision": selection["revision"],
                "repo_path": item["repo_path"],
                "expected_size": item["size"],
                "observed_bytes": item["size"],
            }
            if transfer_receipt != expected_transfer:
                raise IntegrityError("transfer receipt does not match the frozen item identity")
            receipt = make_source_receipt(selection=selection, item=item, archive_path=archive)
            if not db.acquire_lease(run_id, item_id, owner, ttl_seconds=lease_ttl_seconds):
                raise StateError("could not renew caller-owned lease after verification")
            receipt_path = archive.with_name(archive.name + ".source-receipt-v1.json")
            atomic_write_bytes(receipt_path, canonical_bytes(receipt))
            db.transition(
                run_id,
                item_id,
                "VERIFYING",
                "DOWNLOAD_VERIFIED",
                owner=owner,
                details={"receipt_path": str(receipt_path)},
                artifact={
                    "kind": "archive",
                    "uri": str(archive.resolve()),
                    "bytes": receipt["observed_bytes"],
                    "sha256": receipt["observed_sha256"],
                    "etag": receipt["etag"],
                    "revision": selection["revision"],
                    "committed": True,
                },
            )
            return archive
        except Exception as error:
            current = db.item(run_id, item_id)["state"]
            if current == "VERIFYING":
                target = "BLOCKED_INTEGRITY" if isinstance(error, IntegrityError) else "RETRYABLE_ERROR"
                db.transition(
                    run_id,
                    item_id,
                    current,
                    target,
                    details={"error_code": type(error).__name__, "message": str(error)},
                    owner=owner,
                )
            raise
        finally:
            try:
                db.release_lease(run_id, item_id, owner)
            except StateError:
                pass


def make_source_receipt(
    *, selection: Mapping[str, Any], item: Mapping[str, Any], archive_path: Path, etag: str | None = None
) -> dict[str, Any]:
    actual_size = archive_path.stat().st_size
    actual_sha = sha256_file(archive_path)
    if actual_size != item["size"] or actual_sha != item["lfs_sha256"]:
        raise IntegrityError("archive size or SHA-256 differs from frozen LFS identity")
    return {
        "schema_version": RECEIPT_VERSION,
        "repo_id": selection["repo_id"],
        "repo_type": selection["repo_type"],
        "revision": selection["revision"],
        "repo_path": item["repo_path"],
        "expected_size": item["size"],
        "git_blob_id": item["git_blob_id"],
        "lfs_sha256": item["lfs_sha256"],
        "etag": etag,
        "observed_bytes": actual_size,
        "observed_sha256": actual_sha,
    }


def download_one(
    *, db: ControlDB, run_id: str, selection: Mapping[str, Any], item_id: str, spool_root: Path, owner: str,
    downloader: Callable[..., str | os.PathLike[str]] | None = None, disk_usage: Callable[[Path], Any] = shutil.disk_usage,
    lease_ttl_seconds: float = DEFAULT_LEASE_TTL_SECONDS,
) -> Path:
    """Download exactly one selected item at a pinned revision and verify it."""
    lease_ttl_seconds = _validated_lease_ttl(lease_ttl_seconds)
    selected = {item["item_id"]: item for item in selection["items"]}
    if item_id not in selected:
        raise SelectionError("item is not in the frozen selection")
    item = selected[item_id]
    row = db.item(run_id, item_id)
    if row["wave"] != "A" and db.run(run_id)["wave_gate"] == "A":
        raise StateError("default Wave A gate blocks this item")
    if row["state"] != "SELECTED":
        raise StateError("download-one requires SELECTED state")
    spool_root.mkdir(parents=True, exist_ok=True)
    usage = disk_usage(spool_root)
    run = db.run(run_id)
    admission = disk_admission(
        total_bytes=usage.total, free_bytes=usage.free, archive_size=item["size"],
        soft_watermark_bytes=run["soft_watermark_bytes"], hard_watermark_bytes=run["hard_watermark_bytes"],
    )
    if not admission.allowed:
        raise StateError(f"disk admission denied ({admission.level}): {admission.reason}")
    if not db.acquire_lease(run_id, item_id, owner, ttl_seconds=lease_ttl_seconds):
        raise StateError("item lease is held by another worker")
    try:
        db.transition(run_id, item_id, "SELECTED", "DOWNLOADING", owner=owner)
        if downloader is None:
            try:
                from huggingface_hub import hf_hub_download
            except ImportError as error:
                raise ControlError("huggingface_hub is required for downloads") from error
            downloader = hf_hub_download
        spool_root.mkdir(parents=True, exist_ok=True)
        downloaded = Path(
            downloader(
                repo_id=selection["repo_id"], repo_type="dataset", revision=selection["revision"], filename=item["repo_path"],
                cache_dir=str(spool_root / ".hf-cache"), local_dir=str(spool_root / "files"), token=True,
            )
        )
        if not downloaded.is_file() or downloaded.is_symlink():
            raise IntegrityError("download adapter did not return a regular file")
        receipt = make_source_receipt(selection=selection, item=item, archive_path=downloaded)
        if not db.acquire_lease(run_id, item_id, owner, ttl_seconds=lease_ttl_seconds):
            raise StateError("could not renew caller-owned lease after download verification")
        receipt_path = downloaded.with_name(downloaded.name + ".source-receipt-v1.json")
        atomic_write_bytes(receipt_path, canonical_bytes(receipt))
        db.transition(
            run_id, item_id, "DOWNLOADING", "DOWNLOAD_VERIFIED", owner=owner,
            details={"receipt_path": str(receipt_path)},
            artifact={"kind": "archive", "uri": str(downloaded.resolve()), "bytes": receipt["observed_bytes"],
                      "sha256": receipt["observed_sha256"], "etag": receipt["etag"], "revision": selection["revision"],
                      "committed": True},
        )
        return downloaded
    except Exception as error:
        current = db.item(run_id, item_id)["state"]
        if current == "DOWNLOADING":
            target = "BLOCKED_INTEGRITY" if isinstance(error, IntegrityError) else "RETRYABLE_ERROR"
            db.transition(run_id, item_id, current, target, details={"error_code": type(error).__name__, "message": str(error)}, owner=owner)
        raise
    finally:
        try:
            db.release_lease(run_id, item_id, owner)
        except StateError:
            pass


def reconcile(db: ControlDB, run_id: str, *, spool_root: Path, package_root: Path | None = None) -> list[dict[str, Any]]:
    """Repair only states supported by complete, hash-verified filesystem evidence."""
    actions: list[dict[str, Any]] = []
    for row in db.status(run_id)["items"]:
        if row["state"] == "DOWNLOADING":
            archive = spool_root / "files" / PurePosixPath(row["repo_path"])
            if archive.is_file() and not archive.is_symlink() and archive.stat().st_size == row["expected_size"]:
                actual_sha = sha256_file(archive)
                if actual_sha == row["lfs_sha256"]:
                    db.transition(
                        run_id, row["item_id"], "DOWNLOADING", "DOWNLOAD_VERIFIED",
                        details={"reconciled": "verified archive after interrupted DB commit"},
                        artifact={"kind": "archive", "uri": str(archive.resolve()), "bytes": archive.stat().st_size,
                                  "sha256": actual_sha, "etag": None, "revision": row["revision"], "committed": True},
                    )
                    actions.append({"item_id": row["item_id"], "to_state": "DOWNLOAD_VERIFIED"})
        elif row["state"] == "INSTALLING" and package_root is not None:
            candidates = list(package_root.glob("pkg_*/READY"))
            matching = [ready for ready in candidates if ready.parent.joinpath("metadata.json").is_file()]
            for ready in matching:
                metadata = json.loads(ready.parent.joinpath("metadata.json").read_text(encoding="utf-8"))
                if metadata.get("package_sha256") == row["lfs_sha256"]:
                    with db.transaction() as conn:
                        conn.execute("UPDATE items SET package_id=? WHERE run_id=? AND item_id=?", (metadata["package_id"], run_id, row["item_id"]))
                    db.transition(run_id, row["item_id"], "INSTALLING", "PACKAGE_READY", details={"reconciled": str(ready)})
                    actions.append({"item_id": row["item_id"], "to_state": "PACKAGE_READY"})
                    break
    return actions


def plan_archive_deletions(db: ControlDB, run_id: str, *, spool_root: Path) -> list[dict[str, Any]]:
    """Return a dry-run-only exact-path deletion plan.  This function never deletes."""
    root = spool_root.resolve()
    rows = db.connection.execute(
        "SELECT i.*,a.uri,a.bytes AS artifact_bytes,a.sha256 AS artifact_sha256 FROM items i "
        "JOIN artifacts a ON a.run_id=i.run_id AND a.item_id=i.item_id AND a.kind='archive' AND a.committed=1 "
        "WHERE i.run_id=? AND i.state='ARCHIVE_DELETE_ELIGIBLE' ORDER BY i.selection_order", (run_id,)
    ).fetchall()
    plan: list[dict[str, Any]] = []
    for row in rows:
        path = Path(row["uri"])
        resolved = path.resolve(strict=True)
        if path.is_symlink() or not resolved.is_file() or root not in resolved.parents:
            raise StateError(f"archive deletion path is unsafe: {path}")
        actual_bytes = resolved.stat().st_size
        actual_sha = sha256_file(resolved)
        if actual_bytes != row["expected_size"] or actual_bytes != row["artifact_bytes"]:
            raise IntegrityError(f"archive size changed before deletion planning: {resolved}")
        if actual_sha != row["lfs_sha256"] or actual_sha != row["artifact_sha256"]:
            raise IntegrityError(f"archive hash changed before deletion planning: {resolved}")
        plan.append({"dry_run": True, "run_id": run_id, "item_id": row["item_id"], "path": str(resolved),
                     "bytes": actual_bytes, "sha256": actual_sha, "recovery": {"repo_path": row["repo_path"],
                     "revision": row["revision"], "lfs_sha256": row["lfs_sha256"]}})
    return plan


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VisualNovel Phase A0 control plane")
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init")
    init.add_argument("--run-id")
    init.add_argument("--wave-gate", choices=WAVES)
    status = commands.add_parser("status")
    status.add_argument("run_id")
    rec = commands.add_parser("reconcile")
    rec.add_argument("run_id")
    rec.add_argument("--spool-root", type=Path, required=True)
    rec.add_argument("--package-root", type=Path)
    fetch = commands.add_parser("fetch-one")
    fetch.add_argument("run_id")
    fetch.add_argument("item_id")
    fetch.add_argument("--spool-root", type=Path, required=True)
    fetch.add_argument("--owner", default=f"{socket.gethostname()}:{os.getpid()}")
    fetch.add_argument("--authorize-path", action="append", required=True)
    fetch.add_argument("--lock-root", type=Path)
    fetch.add_argument("--lease-ttl-seconds", type=float, default=DEFAULT_LEASE_TTL_SECONDS)
    verify = commands.add_parser("verify-one")
    verify.add_argument("run_id")
    verify.add_argument("item_id")
    verify.add_argument("--spool-root", type=Path, required=True)
    verify.add_argument("--owner", default=f"{socket.gethostname()}:{os.getpid()}")
    verify.add_argument("--lock-root", type=Path)
    verify.add_argument("--lease-ttl-seconds", type=float, default=DEFAULT_LEASE_TTL_SECONDS)
    download = commands.add_parser("download-one")
    download.add_argument("run_id")
    download.add_argument("item_id")
    download.add_argument("--spool-root", type=Path, required=True)
    download.add_argument("--owner", default=f"{socket.gethostname()}:{os.getpid()}")
    download.add_argument("--lease-ttl-seconds", type=float, default=DEFAULT_LEASE_TTL_SECONDS)
    register = commands.add_parser("register-aggregate")
    register.add_argument("run_id")
    register.add_argument("aggregate_dir", type=Path)
    register.add_argument("--wave", choices=WAVES)
    register.add_argument("--expected-fingerprint")
    promote = commands.add_parser("promote-wave")
    promote.add_argument("run_id")
    promote.add_argument("--aggregate-dir", type=Path)
    promote.add_argument("--wave", choices=WAVES)
    promote.add_argument("--expected-fingerprint")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    selection = load_selection(args.selection)
    with ControlDB(args.db) as db:
        if args.command == "init":
            print(db.init_run(selection, run_id=args.run_id, wave_gate=args.wave_gate))
        elif args.command == "status":
            print(json.dumps(db.status(args.run_id), ensure_ascii=False, indent=2, sort_keys=True))
        elif args.command == "reconcile":
            print(json.dumps(reconcile(db, args.run_id, spool_root=args.spool_root, package_root=args.package_root), indent=2))
        elif args.command == "fetch-one":
            print(fetch_one(
                db=db, run_id=args.run_id, selection=selection, item_id=args.item_id,
                spool_root=args.spool_root, owner=args.owner, authorized_paths=args.authorize_path,
                lock_root=args.lock_root, lease_ttl_seconds=args.lease_ttl_seconds,
            ))
        elif args.command == "verify-one":
            print(verify_one(
                db=db, run_id=args.run_id, selection=selection, item_id=args.item_id,
                spool_root=args.spool_root, owner=args.owner, lock_root=args.lock_root,
                lease_ttl_seconds=args.lease_ttl_seconds,
            ))
        elif args.command == "download-one":
            print(download_one(db=db, run_id=args.run_id, selection=selection, item_id=args.item_id,
                               spool_root=args.spool_root, owner=args.owner,
                               lease_ttl_seconds=args.lease_ttl_seconds))
        elif args.command == "register-aggregate":
            print(json.dumps(db.register_aggregate(
                args.run_id, selection, args.aggregate_dir, wave=args.wave,
                expected_fingerprint=args.expected_fingerprint,
            ), ensure_ascii=False, indent=2, sort_keys=True))
        elif args.command == "promote-wave":
            if args.aggregate_dir is not None:
                result = db.register_aggregate(
                    args.run_id, selection, args.aggregate_dir, wave=args.wave,
                    expected_fingerprint=args.expected_fingerprint, promote_wave=True,
                )
                print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
            else:
                if args.expected_fingerprint is not None:
                    raise StateError("--expected-fingerprint requires --aggregate-dir")
                print(db.promote_wave(args.run_id, completed_wave=args.wave))
    return 0


if __name__ == "__main__":
    sys.exit(main())
