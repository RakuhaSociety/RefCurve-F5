"""Immutable, canonical training-run identity manifests."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "f5-tts-training-run-v1"


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(repo_root: Path, *args: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout


def collect_code_identity(repo_root: str | Path) -> dict[str, Any]:
    """Bind the checked-out commit and all non-ignored working-tree changes."""
    root = Path(repo_root).resolve()
    try:
        commit = _git(root, "rev-parse", "HEAD").decode("ascii").strip()
        diff = _git(root, "diff", "--binary", "HEAD", "--")
        untracked = _git(root, "ls-files", "--others", "--exclude-standard", "-z").split(b"\0")
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise RuntimeError(f"cannot determine git code identity for {root}") from error

    dirty = hashlib.sha256()
    dirty.update(b"tracked-diff\0")
    dirty.update(diff)
    untracked_paths = []
    code_roots = {"src", "tools", "tests"}
    root_code_suffixes = {".py", ".toml", ".yaml", ".yml"}
    for raw_path in sorted(path for path in untracked if path):
        relative = raw_path.decode("utf-8", errors="surrogateescape")
        relative_path = Path(relative)
        if relative_path.parts[0] not in code_roots and not (
            len(relative_path.parts) == 1 and relative_path.suffix in root_code_suffixes
        ):
            continue
        path = root / relative
        if not path.is_file():
            continue
        untracked_paths.append(relative)
        dirty.update(b"\0untracked\0")
        dirty.update(raw_path)
        dirty.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                dirty.update(chunk)
    return {
        "commit": commit,
        "dirty": bool(diff or untracked_paths),
        "dirty_sha256": dirty.hexdigest(),
    }


def collect_environment_identity() -> dict[str, Any]:
    import accelerate
    import torch

    visible_gpus = []
    for index in range(torch.cuda.device_count()):
        capability = torch.cuda.get_device_capability(index)
        visible_gpus.append(
            {
                "name": torch.cuda.get_device_name(index),
                "capability": [int(capability[0]), int(capability[1])],
                "bf16_supported": bool(torch.cuda.is_bf16_supported(index)),
            }
        )
    return {
        "hostname": platform.node(),
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "accelerate": accelerate.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
        "mixed_precision": os.environ.get("ACCELERATE_MIXED_PRECISION", "no"),
        "visible_gpu_count": len(visible_gpus),
        "visible_gpus": visible_gpus,
    }


def file_identity(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"identity source file not found: {resolved}")
    return {"path": str(resolved), "size": resolved.stat().st_size, "sha256": sha256_file(resolved)}


def files_identity(paths: dict[str, str | Path]) -> dict[str, dict[str, Any]]:
    return {name: file_identity(path) for name, path in sorted(paths.items())}


def build_run_manifest(
    *,
    resolved_config: dict[str, Any],
    code_identity: dict[str, Any],
    dataset_identity: dict[str, Any] | None,
    source_checkpoint: str | Path | None,
    environment_identity: dict[str, Any],
    evidence_paths: dict[str, str | Path] | None = None,
) -> dict[str, Any]:
    if dataset_identity is None:
        raise ValueError("run manifest requires a dataset identity")
    payload = {
        "resolved_config": resolved_config,
        "code": code_identity,
        "dataset": dataset_identity,
        "source_checkpoint": file_identity(source_checkpoint),
        "environment": environment_identity,
        "evidence": files_identity(evidence_paths or {}),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "run_identity": hashlib.sha256(canonical_json(payload)).hexdigest(),
        "identity": payload,
    }


def validate_run_manifest(manifest: Any) -> dict[str, Any]:
    if not isinstance(manifest, dict) or set(manifest) != {"schema_version", "run_identity", "identity"}:
        raise ValueError("malformed run manifest")
    if manifest["schema_version"] != SCHEMA_VERSION or not isinstance(manifest["identity"], dict):
        raise ValueError("unsupported or malformed run manifest")
    expected = hashlib.sha256(canonical_json(manifest["identity"])).hexdigest()
    if manifest["run_identity"] != expected:
        raise ValueError("run manifest identity hash does not match its canonical payload")
    return manifest


def load_run_manifest(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        return validate_run_manifest(json.loads(source.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise RuntimeError(f"run manifest is invalid: {source}") from error


def wait_for_run_manifest(
    path: str | Path,
    *,
    timeout: float = 120.0,
    poll_interval: float = 0.1,
) -> dict[str, Any]:
    """Wait boundedly for rank zero to atomically publish a valid manifest."""
    source = Path(path)
    deadline = time.monotonic() + timeout
    while not source.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for rank zero to publish run manifest: {source}")
        time.sleep(poll_interval)
    return load_run_manifest(source)


def ensure_run_manifest(path: str | Path, manifest: dict[str, Any], *, lock_timeout: float = 30.0) -> dict[str, Any]:
    """Atomically create a manifest, or accept only an identical existing one."""
    expected = validate_run_manifest(manifest)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock = destination.with_name(f".{destination.name}.lock")
    deadline = time.monotonic() + lock_timeout
    while True:
        try:
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.close(descriptor)
            break
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"timed out waiting for run manifest lock: {lock}")
            time.sleep(0.05)

    try:
        if destination.exists():
            try:
                existing = validate_run_manifest(json.loads(destination.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError, ValueError) as error:
                raise RuntimeError(f"existing run manifest is invalid: {destination}") from error
            if existing != expected:
                raise RuntimeError(
                    f"run identity mismatch for immutable manifest {destination}: "
                    f"existing={existing['run_identity']}, current={expected['run_identity']}"
                )
            return existing

        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(canonical_json(expected) + b"\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return expected
    finally:
        lock.unlink(missing_ok=True)
