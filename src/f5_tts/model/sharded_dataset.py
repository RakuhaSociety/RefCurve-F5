"""Memory-bounded dataset primitives for sharded, map-style training data.

This module is intentionally independent from the current training data path.  It
specifies manifest v1, reads independent Arrow files through a small LRU cache,
and stores dynamic batches as two contiguous NumPy arrays.
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterator, Literal, Mapping

import numpy as np
from datasets import Dataset

_MANIFEST_KEYS = {"version", "total_rows", "shards", "frame_index", "audio", "vocabulary"}
_SHARD_KEYS = {"path", "rows", "sha256"}
_FRAME_INDEX_KEYS = {"path", "count", "dtype", "sha256"}
_AUDIO_KEYS = {"sample_rate", "hop_length"}
_VOCABULARY_KEYS = {"identity", "token_sequence_sha256"}
_AGGREGATE_PROVENANCE_KEYS = {
    "schema_version",
    "semantic_fingerprint",
    "manifest_sha256",
    "split_fingerprint",
    "vocabulary",
    "deployment_fingerprint",
    "aggregate_context",
    "dedup_policy",
    "conflict_policy",
    "split_policy",
    "input_sidecar_splits_authoritative",
    "inputs",
    "artifact_hashes",
}
_AUDIO_ROOTS_KEYS = {"schema_version", "roots"}
_AUDIO_ROOT_IDENTITY_KEYS = {"audio_root_id", "package_id", "package_sha256", "ready_sha256"}
_DEPLOYMENT_ROOT_KEYS = {"audio_root_id", "package_dir"}
_SHA256_LENGTH = 64


def _strict_keys(value: Mapping, expected: set[str], context: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{context} schema mismatch: missing={missing}, extra={extra}")


def _validate_lower_hex(value: object, length: int, context: str) -> str:
    if not isinstance(value, str) or len(value) != length:
        raise ValueError(f"{context} must be a {length}-character lowercase hexadecimal digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{context} is not hexadecimal") from error
    if value != value.lower():
        raise ValueError(f"{context} must use lowercase hexadecimal")
    return value


def _validate_sha256(value: object, context: str) -> str:
    return _validate_lower_hex(value, _SHA256_LENGTH, context)


def _validate_commit_sha(value: object, context: str) -> str:
    return _validate_lower_hex(value, 40, context)


def _safe_relative_path(root: Path, value: object, context: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{context} must be a non-empty POSIX relative path")
    relative = Path(value)
    if relative.is_absolute() or any(part in ("", ".", "..") for part in relative.parts):
        raise ValueError(f"{context} must not be absolute or contain '.'/'..'")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{context} escapes the manifest directory: {value}") from error
    return resolved


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ShardSpec:
    path: Path
    rows: int
    sha256: str


@dataclass(frozen=True)
class FrameIndexSpec:
    path: Path
    count: int
    sha256: str


@dataclass(frozen=True)
class ManifestV1:
    root: Path
    total_rows: int
    shards: tuple[ShardSpec, ...]
    frame_index: FrameIndexSpec
    sample_rate: int
    hop_length: int
    vocabulary_identity: str
    vocabulary_sha256: str


def load_manifest(path: str | Path, *, verify_hashes: bool = True) -> ManifestV1:
    """Load and strictly validate a manifest v1 and all referenced files."""
    manifest_path = Path(path).expanduser().resolve()
    root = manifest_path.parent
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("manifest must be a JSON object")
    _strict_keys(raw, _MANIFEST_KEYS, "manifest")
    if type(raw["version"]) is not int or raw["version"] != 1:
        raise ValueError("manifest.version must be integer 1")
    if type(raw["total_rows"]) is not int or raw["total_rows"] <= 0:
        raise ValueError("manifest.total_rows must be a positive integer")
    if not isinstance(raw["shards"], list) or not raw["shards"]:
        raise ValueError("manifest.shards must be a non-empty array")

    shards = []
    for index, item in enumerate(raw["shards"]):
        if not isinstance(item, dict):
            raise ValueError(f"manifest.shards[{index}] must be an object")
        _strict_keys(item, _SHARD_KEYS, f"manifest.shards[{index}]")
        if type(item["rows"]) is not int or item["rows"] <= 0:
            raise ValueError(f"manifest.shards[{index}].rows must be a positive integer")
        shard_path = _safe_relative_path(root, item["path"], f"manifest.shards[{index}].path")
        digest = _validate_sha256(item["sha256"], f"manifest.shards[{index}].sha256")
        shards.append(ShardSpec(shard_path, item["rows"], digest))

    if sum(shard.rows for shard in shards) != raw["total_rows"]:
        raise ValueError("manifest.total_rows does not equal the sum of shard row counts")

    item = raw["frame_index"]
    if not isinstance(item, dict):
        raise ValueError("manifest.frame_index must be an object")
    _strict_keys(item, _FRAME_INDEX_KEYS, "manifest.frame_index")
    if item["dtype"] != "uint32":
        raise ValueError("manifest.frame_index.dtype must be 'uint32'")
    if type(item["count"]) is not int or item["count"] != raw["total_rows"]:
        raise ValueError("manifest.frame_index.count must equal manifest.total_rows")
    frame_path = _safe_relative_path(root, item["path"], "manifest.frame_index.path")
    frame_digest = _validate_sha256(item["sha256"], "manifest.frame_index.sha256")

    audio = raw["audio"]
    if not isinstance(audio, dict):
        raise ValueError("manifest.audio must be an object")
    _strict_keys(audio, _AUDIO_KEYS, "manifest.audio")
    for key in _AUDIO_KEYS:
        if type(audio[key]) is not int or audio[key] <= 0:
            raise ValueError(f"manifest.audio.{key} must be a positive integer")

    vocabulary = raw["vocabulary"]
    if not isinstance(vocabulary, dict):
        raise ValueError("manifest.vocabulary must be an object")
    _strict_keys(vocabulary, _VOCABULARY_KEYS, "manifest.vocabulary")
    if not isinstance(vocabulary["identity"], str) or not vocabulary["identity"]:
        raise ValueError("manifest.vocabulary.identity must be a non-empty string")
    vocabulary_digest = _validate_sha256(
        vocabulary["token_sequence_sha256"], "manifest.vocabulary.token_sequence_sha256"
    )

    referenced = [*shards, FrameIndexSpec(frame_path, item["count"], frame_digest)]
    for spec in referenced:
        if not spec.path.is_file():
            raise FileNotFoundError(spec.path)
        if verify_hashes and file_sha256(spec.path) != spec.sha256:
            raise ValueError(f"SHA-256 mismatch: {spec.path}")

    expected_bytes = item["count"] * np.dtype("<u4").itemsize
    if frame_path.stat().st_size != expected_bytes:
        raise ValueError(f"frame index size mismatch: expected {expected_bytes} bytes")
    return ManifestV1(
        root,
        raw["total_rows"],
        tuple(shards),
        referenced[-1],
        audio["sample_rate"],
        audio["hop_length"],
        vocabulary["identity"],
        vocabulary_digest,
    )


def load_audio_root_registry(
    identity_path: str | Path, deployment_path: str | Path
) -> dict[str, dict[str, str]]:
    """Load separate immutable identities and runtime paths, verifying package READY."""
    identities = json.loads(Path(identity_path).read_text(encoding="utf-8"))
    deployments = json.loads(Path(deployment_path).read_text(encoding="utf-8"))
    for value, schema, label in (
        (identities, "visualnovel-audio-roots-v1", "audio root identity registry"),
        (deployments, "visualnovel-deployment-roots-v1", "deployment root registry"),
    ):
        if not isinstance(value, dict):
            raise ValueError(f"{label} must be a JSON object")
        _strict_keys(value, _AUDIO_ROOTS_KEYS, label)
        if value["schema_version"] != schema or not isinstance(value["roots"], list):
            raise ValueError(f"unsupported {label} schema")
    identity_by_id: dict[str, dict] = {}
    for index, item in enumerate(identities["roots"]):
        if not isinstance(item, dict):
            raise ValueError(f"audio root identity {index} must be an object")
        _strict_keys(item, _AUDIO_ROOT_IDENTITY_KEYS, f"audio root identity {index}")
        root_id = item["audio_root_id"]
        if not isinstance(root_id, str) or not root_id or root_id in identity_by_id:
            raise ValueError("audio root identity IDs must be unique non-empty strings")
        for key in ("package_sha256", "ready_sha256"):
            _validate_sha256(item[key], f"audio root identity {index}.{key}")
        identity_by_id[root_id] = item
    result: dict[str, dict[str, str]] = {}
    for index, item in enumerate(deployments["roots"]):
        if not isinstance(item, dict):
            raise ValueError(f"deployment root {index} must be an object")
        _strict_keys(item, _DEPLOYMENT_ROOT_KEYS, f"deployment root {index}")
        root_id = item["audio_root_id"]
        if root_id not in identity_by_id or root_id in result:
            raise ValueError("deployment roots must match unique identity roots")
        root_value = item["package_dir"]
        if not isinstance(root_value, str) or not root_value or not Path(root_value).is_absolute():
            raise ValueError(f"deployment root {index}.package_dir must be an absolute path")
        root = Path(root_value).expanduser().resolve()
        ready = root / "READY"
        if not root.is_dir() or not ready.is_file():
            raise ValueError(f"registered audio root must be a directory containing READY: {root_id!r}")
        identity = identity_by_id[root_id]
        if file_sha256(ready) != identity["ready_sha256"] or ready.read_text(encoding="ascii").strip() != identity["package_sha256"]:
            raise ValueError(f"registered audio root READY identity mismatch: {root_id!r}")
        metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
        if metadata.get("package_id") != identity["package_id"] or metadata.get("package_sha256") != identity["package_sha256"]:
            raise ValueError(f"registered audio root package identity mismatch: {root_id!r}")
        result[root_id] = {**identity, "package_dir": root.as_posix()}
    if set(result) != set(identity_by_id):
        raise ValueError("deployment roots do not cover every identity root")
    return result


def resolve_audio_path(row: Mapping, audio_roots: Mapping[str, str | Path | Mapping] | None = None) -> Path:
    """Resolve either a legacy absolute path or a registry-backed contained path."""
    root_id = row.get("audio_root_id")
    relative = row.get("relative_audio_path")
    if root_id is None and relative is None:
        legacy = row.get("audio_path")
        if not isinstance(legacy, str) or not legacy or not Path(legacy).is_absolute():
            raise ValueError("row must contain an absolute audio_path or audio_root_id/relative_audio_path")
        return Path(legacy)
    if not isinstance(root_id, str) or not root_id or not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError("audio_root_id and relative_audio_path must be non-empty strings using POSIX paths")
    if audio_roots is None or root_id not in audio_roots:
        raise KeyError(f"audio root is not registered: {root_id!r}")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or any(part in ("", ".", "..") for part in pure.parts):
        raise ValueError("relative_audio_path must be a contained POSIX relative path")
    registered = audio_roots[root_id]
    if isinstance(registered, Mapping):
        root_value = registered.get("package_dir")
        if row.get("package_id") != registered.get("package_id") or row.get("package_sha256") != registered.get("package_sha256"):
            raise ValueError(f"row package identity does not match registered audio root: {root_id!r}")
    else:
        root_value = registered
    root = Path(root_value).expanduser().resolve()
    if not root.is_dir() or not (root / "READY").is_file():
        raise ValueError(f"registered audio root must be a directory containing READY: {root_id!r}")
    candidate = root.joinpath(*pure.parts).resolve()
    if not candidate.is_relative_to(root) or not candidate.is_file() or candidate.is_symlink():
        raise ValueError(f"audio path is not a contained regular file: {root_id!r}/{relative}")
    return candidate


def load_aggregate_dataset_identity(manifest: ManifestV1) -> dict[str, Any] | None:
    """Return the frozen aggregate identity when aggregate provenance is present."""
    provenance_path = manifest.root / "aggregate_provenance.json"
    if not provenance_path.is_file():
        return None
    raw = json.loads(provenance_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("aggregate provenance must be a JSON object")
    _strict_keys(raw, _AGGREGATE_PROVENANCE_KEYS, "aggregate provenance")
    if not isinstance(raw["schema_version"], str) or not raw["schema_version"]:
        raise ValueError("aggregate provenance.schema_version must be a non-empty string")
    semantic = _validate_sha256(raw["semantic_fingerprint"], "aggregate provenance.semantic_fingerprint")
    manifest_sha = _validate_sha256(raw["manifest_sha256"], "aggregate provenance.manifest_sha256")
    split = _validate_sha256(raw["split_fingerprint"], "aggregate provenance.split_fingerprint")
    deployment = _validate_sha256(raw["deployment_fingerprint"], "aggregate provenance.deployment_fingerprint")
    if file_sha256(manifest.root / "manifest.json") != manifest_sha:
        raise ValueError("aggregate provenance manifest SHA-256 does not match manifest.json")
    vocabulary = raw["vocabulary"]
    if not isinstance(vocabulary, dict):
        raise ValueError("aggregate provenance.vocabulary must be an object")
    _strict_keys(vocabulary, _VOCABULARY_KEYS, "aggregate provenance.vocabulary")
    vocab_sha = _validate_sha256(
        vocabulary["token_sequence_sha256"], "aggregate provenance.vocabulary.token_sequence_sha256"
    )
    if vocabulary["identity"] != manifest.vocabulary_identity or vocab_sha != manifest.vocabulary_sha256:
        raise ValueError("aggregate provenance vocabulary does not match manifest.json")
    context = raw["aggregate_context"]
    if not isinstance(context, dict) or set(context) != {"source_revision", "catalog_identity", "selection_fingerprint"}:
        raise ValueError("aggregate provenance.aggregate_context schema mismatch")
    _validate_commit_sha(context["source_revision"], "aggregate provenance source revision")
    _validate_sha256(context["selection_fingerprint"], "aggregate provenance selection fingerprint")
    if not isinstance(context["catalog_identity"], str) or not context["catalog_identity"]:
        raise ValueError("aggregate provenance catalog identity must be non-empty")
    if not isinstance(raw["artifact_hashes"], list) or not raw["artifact_hashes"]:
        raise ValueError("aggregate provenance.artifact_hashes must be non-empty")
    return {
        "contract_version": 2,
        "kind": "visualnovel-aggregate",
        "semantic_fingerprint": semantic,
        "manifest_sha256": manifest_sha,
        "split_fingerprint": split,
        "deployment_fingerprint": deployment,
        "aggregate_context": context,
        "vocabulary": {"identity": vocabulary["identity"], "token_sequence_sha256": vocab_sha},
    }


class ShardedArrowDataset:
    """Map-style view over independent Arrow shards with a bounded LRU cache."""

    def __init__(
        self,
        manifest: str | Path | ManifestV1,
        *,
        cache_size: int = 2,
        verify_hashes: bool = True,
        audio_roots: Mapping[str, str | Path | Mapping] | None = None,
    ):
        if type(cache_size) is not int or cache_size <= 0:
            raise ValueError("cache_size must be a positive integer")
        self.manifest = load_manifest(manifest, verify_hashes=verify_hashes) if not isinstance(manifest, ManifestV1) else manifest
        self.cache_size = cache_size
        self.audio_roots = dict(audio_roots or {})
        self.dataset_identity = load_aggregate_dataset_identity(self.manifest)
        self._ends = np.cumsum([shard.rows for shard in self.manifest.shards], dtype=np.uint64)
        self._cache: OrderedDict[int, Dataset] = OrderedDict()

    def __len__(self) -> int:
        return self.manifest.total_rows

    def _load_shard(self, shard_index: int) -> Dataset:
        if shard_index in self._cache:
            return self._cache.pop(shard_index)
        spec = self.manifest.shards[shard_index]
        shard = Dataset.from_file(spec.path.as_posix())
        if len(shard) != spec.rows:
            raise ValueError(f"Arrow row count mismatch for {spec.path}: expected {spec.rows}, got {len(shard)}")
        return shard

    def __getitem__(self, index: int) -> dict:
        if not isinstance(index, (int, np.integer)):
            raise TypeError("dataset index must be an integer")
        index = int(index)
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        shard_index = int(np.searchsorted(self._ends, index, side="right"))
        start = 0 if shard_index == 0 else int(self._ends[shard_index - 1])
        shard = self._load_shard(shard_index)
        self._cache[shard_index] = shard
        self._cache.move_to_end(shard_index)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        row = dict(shard[index - start])
        if "audio_root_id" in row or "relative_audio_path" in row:
            row["audio_path"] = resolve_audio_path(row, self.audio_roots).as_posix()
        return row


def open_frame_index(manifest: str | Path | ManifestV1, *, verify_hashes: bool = True) -> np.memmap:
    """Open the validated, contiguous little-endian uint32 frame index read-only."""
    loaded = load_manifest(manifest, verify_hashes=verify_hashes) if not isinstance(manifest, ManifestV1) else manifest
    spec = loaded.frame_index
    if spec.path.stat().st_size != spec.count * 4:
        raise ValueError("frame index byte count changed after manifest validation")
    return np.memmap(spec.path, mode="r", dtype="<u4", shape=(spec.count,))


@dataclass(frozen=True)
class BatchPlanStats:
    sample_count: int
    batch_count: int
    oversized_count: int
    oversized_max_frames: int


class DynamicBatchPlan:
    """Dynamic batches represented by flat indices and offsets, never nested lists."""

    def __init__(self, indices: np.ndarray, offsets: np.ndarray, stats: BatchPlanStats, *, seed: int = 0):
        self.indices = np.ascontiguousarray(indices)
        self.offsets = np.ascontiguousarray(offsets)
        self.stats = stats
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[np.ndarray]:
        order = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch])).permutation(len(self))
        for batch_index in order:
            yield self.indices[self.offsets[batch_index] : self.offsets[batch_index + 1]]


def build_dynamic_batch_plan(
    frame_lengths: np.ndarray,
    *,
    padded_frame_budget: int,
    max_samples: int = 0,
    oversized: Literal["error", "drop"] = "error",
    seed: int = 0,
) -> DynamicBatchPlan:
    """Pack length-sorted samples under ``max_length * batch_size`` budget."""
    frames = np.asarray(frame_lengths)
    if frames.ndim != 1 or frames.dtype.kind not in "ui" or np.any(frames == 0):
        raise ValueError("frame_lengths must be a one-dimensional positive integer array")
    if type(padded_frame_budget) is not int or padded_frame_budget <= 0:
        raise ValueError("padded_frame_budget must be a positive integer")
    if type(max_samples) is not int or max_samples < 0:
        raise ValueError("max_samples must be a non-negative integer")
    if oversized not in ("error", "drop"):
        raise ValueError("oversized must be 'error' or 'drop'")

    oversized_mask = frames > padded_frame_budget
    oversized_count = int(np.count_nonzero(oversized_mask))
    oversized_max = int(frames[oversized_mask].max()) if oversized_count else 0
    if oversized_count and oversized == "error":
        raise ValueError(
            f"{oversized_count} samples exceed padded_frame_budget={padded_frame_budget}; "
            f"maximum is {oversized_max} frames"
        )

    valid = np.flatnonzero(~oversized_mask)
    order = valid[np.argsort(frames[valid], kind="stable")]
    flat = np.empty(len(order), dtype=np.uint64)
    offsets = np.empty(len(order) + 1, dtype=np.uint64)
    offsets[0] = 0
    write_at = batch_count = batch_start = 0
    batch_max = 0
    for raw_index in order:
        frame_count = int(frames[raw_index])
        next_size = write_at - batch_start + 1
        exceeds = batch_start < write_at and (
            frame_count * next_size > padded_frame_budget or (max_samples and next_size > max_samples)
        )
        if exceeds:
            batch_count += 1
            offsets[batch_count] = write_at
            batch_start = write_at
            batch_max = 0
        flat[write_at] = raw_index
        write_at += 1
        batch_max = max(batch_max, frame_count)
    if write_at > batch_start:
        batch_count += 1
        offsets[batch_count] = write_at

    stats = BatchPlanStats(write_at, batch_count, oversized_count, oversized_max)
    return DynamicBatchPlan(flat[:write_at], offsets[: batch_count + 1], stats, seed=seed)
