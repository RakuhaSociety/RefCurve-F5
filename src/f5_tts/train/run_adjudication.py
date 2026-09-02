"""Create-only adjudication sidecars: retire a training run without mutating it.

A run manifest is immutable by contract, so a run that later turns out to be
invalid cannot be corrected in place — and must not be, because the invalid
artifacts *are* the evidence. This module writes a separate
`run_adjudication.json` beside the manifest that records the verdict, the
reason and the supporting evidence, and binds itself to the manifest so that
later tampering with either file is detectable.

Two deliberate design choices:

* **Bound by content, not by path.** The sidecar records the run's
  `run_identity` and the sha256 of its `run_manifest.json` bytes, never the
  absolute run directory. The file's own location already says which run it
  judges, and content-binding means the verdict can be verified anywhere.
* **No wall-clock timestamp.** The verdict is a pure function of the facts it
  asserts, so re-running is a byte-for-byte no-op and two people adjudicating
  the same run independently produce identical files. When the judgment was
  made is recorded by the filesystem and by version control, not by a field
  that would defeat idempotency.

Nothing here ever opens a `run_manifest.json` for writing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from f5_tts.train.run_manifest import canonical_json, load_run_manifest, sha256_file


SCHEMA_VERSION = "f5-tts-run-adjudication-v1"

MANIFEST_FILENAME = "run_manifest.json"
ADJUDICATION_FILENAME = "run_adjudication.json"

# A verdict is one of these; anything else is a typo, not a new policy.
STATUSES = frozenset(
    {
        "invalid_for_calibration",  # run executed, but its training dynamics are not trustworthy
        "invalid_for_all_purposes",  # run is unusable for any conclusion
        "superseded",  # valid, but a corrected run replaces it
        "valid",  # explicitly cleared after review
    }
)

# What downstream consumers must refuse to do with this run.
PROHIBITIONS = frozenset(
    {
        "resume",  # do not continue training from its checkpoints
        "model_selection",  # do not let its checkpoints win a comparison
        "deployment",  # do not ship its weights
        "quality_conclusions",  # do not cite its evals as evidence about the model or data
    }
)


def _reject_non_finite(value: Any, path: str = "$") -> None:
    """Reject NaN/Infinity, which json.dumps emits as invalid JSON literals."""
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite float is not representable in JSON at {path}: {value!r}")
    elif isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"JSON object keys must be strings at {path}: {key!r}")
            _reject_non_finite(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_non_finite(item, f"{path}[{index}]")


def _assert_round_trips(payload: Any) -> bytes:
    """Guarantee the payload survives serialise -> parse -> serialise unchanged.

    Byte-idempotency is the whole contract here; a value that re-serialises
    differently would make a second adjudication of the same facts look like a
    conflicting verdict.
    """
    _reject_non_finite(payload)
    encoded = canonical_json(payload)
    if canonical_json(json.loads(encoded.decode("utf-8"))) != encoded:
        raise ValueError("adjudication payload does not round-trip through canonical JSON")
    return encoded


def manifest_binding(run_dir: str | Path) -> dict[str, Any]:
    """Read the run's manifest and derive the content binding for the sidecar.

    The manifest is opened read-only and validated; its identity hash must
    match its own payload before we are willing to cite it.
    """
    manifest_path = Path(run_dir) / MANIFEST_FILENAME
    manifest = load_run_manifest(manifest_path)
    return {
        "run_identity": manifest["run_identity"],
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_schema_version": manifest["schema_version"],
    }


def checkpoint_binding(checkpoint_paths: list[str | Path]) -> list[dict[str, Any]]:
    """Bind checkpoints by name and content, so the verdict names its artifacts.

    Names are basenames on purpose: the sidecar lives in the run directory, and
    absolute paths would make an otherwise portable record machine-specific.
    """
    entries = []
    for raw in checkpoint_paths:
        path = Path(raw)
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint not found: {path}")
        entries.append({"name": path.name, "size": path.stat().st_size, "sha256": sha256_file(path)})
    names = [entry["name"] for entry in entries]
    if len(set(names)) != len(names):
        raise ValueError(f"checkpoint basenames must be unique within a run: {sorted(names)}")
    return sorted(entries, key=lambda entry: entry["name"])


def build_run_adjudication(
    *,
    run_binding: dict[str, Any],
    status: str,
    reason_code: str,
    summary: str,
    evidence: dict[str, Any],
    prohibit: list[str],
    retain: bool = True,
    forensic_only: bool = True,
    checkpoints: list[dict[str, Any]] | None = None,
    supersedes_with: str | None = None,
) -> dict[str, Any]:
    if status not in STATUSES:
        raise ValueError(f"unsupported adjudication status {status!r}; expected one of {sorted(STATUSES)}")
    if not reason_code or not reason_code.replace("_", "").isalnum():
        raise ValueError(f"reason_code must be a non-empty snake_case token, got {reason_code!r}")
    if not summary.strip():
        raise ValueError("summary must be a non-empty human-readable explanation")
    unknown = sorted(set(prohibit) - PROHIBITIONS)
    if unknown:
        raise ValueError(f"unknown prohibitions {unknown}; expected a subset of {sorted(PROHIBITIONS)}")
    if not retain:
        raise ValueError("adjudication may not authorise deletion; evidence is retained by contract")
    required_binding = {"run_identity", "manifest_sha256", "manifest_schema_version"}
    if set(run_binding) != required_binding:
        raise ValueError(f"run binding must have exactly {sorted(required_binding)}, got {sorted(run_binding)}")

    payload = {
        "run": dict(run_binding),
        "verdict": {
            "status": status,
            "reason_code": reason_code,
            "summary": summary.strip(),
            "superseded_by": supersedes_with,
        },
        "disposition": {
            "retain": True,
            "forensic_only": bool(forensic_only),
            "prohibit": sorted(set(prohibit)),
        },
        "evidence": evidence,
        "checkpoints": list(checkpoints or []),
    }
    _assert_round_trips(payload)
    return {
        "schema_version": SCHEMA_VERSION,
        "adjudication_identity": hashlib.sha256(canonical_json(payload)).hexdigest(),
        "identity": payload,
    }


def validate_run_adjudication(document: Any) -> dict[str, Any]:
    if not isinstance(document, dict) or set(document) != {"schema_version", "adjudication_identity", "identity"}:
        raise ValueError("malformed run adjudication")
    if document["schema_version"] != SCHEMA_VERSION or not isinstance(document["identity"], dict):
        raise ValueError("unsupported or malformed run adjudication")
    expected = hashlib.sha256(canonical_json(document["identity"])).hexdigest()
    if document["adjudication_identity"] != expected:
        raise ValueError("adjudication identity hash does not match its canonical payload")
    payload = document["identity"]
    if set(payload) != {"run", "verdict", "disposition", "evidence", "checkpoints"}:
        raise ValueError(f"unexpected adjudication payload sections: {sorted(payload)}")
    if payload["verdict"]["status"] not in STATUSES:
        raise ValueError(f"unsupported adjudication status: {payload['verdict']['status']!r}")
    if payload["disposition"]["retain"] is not True:
        raise ValueError("adjudication must retain evidence")
    return document


def load_run_adjudication(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        return validate_run_adjudication(json.loads(source.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise RuntimeError(f"run adjudication is invalid: {source}") from error


def verify_binding(document: dict[str, Any], run_dir: str | Path) -> dict[str, Any]:
    """Re-derive the manifest binding and confirm the verdict still applies.

    A mismatch means the manifest changed after the verdict was recorded, which
    is exactly the tampering this sidecar exists to make visible.
    """
    recorded = validate_run_adjudication(document)["identity"]["run"]
    current = manifest_binding(run_dir)
    if current != recorded:
        differing = sorted(key for key in current if current[key] != recorded.get(key))
        raise RuntimeError(
            f"run manifest no longer matches the adjudication binding in {run_dir}; differing fields: {differing}"
        )
    return current


def ensure_run_adjudication(
    path: str | Path,
    document: dict[str, Any],
    *,
    lock_timeout: float = 30.0,
) -> dict[str, Any]:
    """Atomically create the sidecar, or accept only a byte-identical existing one."""
    import time

    expected = validate_run_adjudication(document)
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
                raise TimeoutError(f"timed out waiting for run adjudication lock: {lock}")
            time.sleep(0.05)

    try:
        if destination.exists():
            try:
                existing = validate_run_adjudication(json.loads(destination.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError, ValueError) as error:
                raise RuntimeError(f"existing run adjudication is invalid: {destination}") from error
            if existing != expected:
                raise RuntimeError(
                    f"conflicting adjudication already exists at {destination}: "
                    f"existing={existing['adjudication_identity']}, current={expected['adjudication_identity']}"
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


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True, help="training run directory containing run_manifest.json")
    parser.add_argument("--status", required=True, choices=sorted(STATUSES))
    parser.add_argument("--reason-code", required=True, help="stable snake_case token naming the defect")
    parser.add_argument("--summary", required=True, help="human-readable explanation of the verdict")
    parser.add_argument("--evidence-json", required=True, help="path to a JSON file holding the evidence object")
    parser.add_argument(
        "--prohibit",
        action="append",
        default=[],
        choices=sorted(PROHIBITIONS),
        help="repeatable; downstream uses that must refuse this run",
    )
    parser.add_argument("--checkpoint", action="append", default=[], help="repeatable; checkpoint file to bind")
    parser.add_argument("--superseded-by", default=None, help="run identity of the corrected replacement, if known")
    parser.add_argument(
        "--write",
        action="store_true",
        help="actually create the sidecar; without this the command only prints the bytes it would write",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    run_dir = Path(args.run_dir)

    evidence = json.loads(Path(args.evidence_json).read_text(encoding="utf-8"))
    if not isinstance(evidence, dict):
        raise ValueError("evidence JSON must be an object")

    document = build_run_adjudication(
        run_binding=manifest_binding(run_dir),
        status=args.status,
        reason_code=args.reason_code,
        summary=args.summary,
        evidence=evidence,
        prohibit=args.prohibit,
        checkpoints=checkpoint_binding(args.checkpoint),
        supersedes_with=args.superseded_by,
    )
    payload = canonical_json(document) + b"\n"
    destination = run_dir / ADJUDICATION_FILENAME

    if not args.write:
        sys.stdout.write(payload.decode("utf-8"))
        sys.stderr.write(
            f"\nDRY RUN: would write {len(payload)} bytes to {destination}\n"
            f"adjudication_identity={document['adjudication_identity']}\n"
            f"sha256(file bytes)={hashlib.sha256(payload).hexdigest()}\n"
            "Pass --write to create it.\n"
        )
        return 0

    result = ensure_run_adjudication(destination, document)
    sys.stderr.write(f"adjudication in place at {destination}: {result['adjudication_identity']}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
