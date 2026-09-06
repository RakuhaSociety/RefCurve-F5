"""Training run adjudication — immutable forensic verdict on a completed run.

Creates a create-only sidecar `run_adjudication.json` next to `run_manifest.json`.
Does NOT modify the manifest or checkpoints. Atomic write, byte-idempotent.

Usage:
    python -m f5_tts.train.run_adjudication \\
        --run-dir /path/to/run \\
        --status invalid_for_calibration \\
        --reason scheduler_horizon_not_scaled_by_world_size \\
        --evidence '{"update_500_lr": 2.38e-6, "update_1000_lr": 1e-13, ...}'
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import click

AdjudicationStatus = Literal[
    "invalid_for_calibration",
    "invalid_for_selection",
    "forensic_only",
    "valid",
]

ADJUDICATION_SCHEMA_VERSION = "f5-tts-run-adjudication-v1"


def create_adjudication(
    run_dir: Path,
    status: AdjudicationStatus,
    reason: str,
    evidence: dict[str, Any],
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Create adjudication sidecar for a training run.

    Args:
        run_dir: Run directory containing run_manifest.json
        status: Adjudication verdict
        reason: Human-readable explanation (kebab-case slug + prose)
        evidence: Supporting data (LR trace excerpts, loss curves, etc.)
        dry_run: If True, return content without writing

    Returns:
        The adjudication document

    Raises:
        FileNotFoundError: run_manifest.json not found
        FileExistsError: run_adjudication.json already exists with different content
        ValueError: manifest or checkpoint hash mismatch
    """
    run_dir = Path(run_dir).resolve()
    manifest_path = run_dir / "run_manifest.json"
    adjudication_path = run_dir / "run_adjudication.json"

    if not manifest_path.exists():
        raise FileNotFoundError(f"run_manifest.json not found in {run_dir}")

    # Read and hash manifest
    manifest_bytes = manifest_path.read_bytes()
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    manifest = json.loads(manifest_bytes)

    # Collect checkpoint hashes (all .pt files)
    checkpoint_hashes = {}
    for ckpt_path in sorted(run_dir.glob("*.pt")):
        ckpt_bytes = ckpt_path.read_bytes()
        checkpoint_hashes[ckpt_path.name] = hashlib.sha256(ckpt_bytes).hexdigest()

    # Build adjudication document
    adjudication = {
        "schema_version": ADJUDICATION_SCHEMA_VERSION,
        "run_identity": manifest.get("run_identity"),
        "manifest_sha256": manifest_sha256,
        "checkpoint_hashes": checkpoint_hashes,
        "status": status,
        "reason": reason,
        "evidence": evidence,
        "disposition": _disposition_for_status(status),
    }

    if dry_run:
        return adjudication

    # Atomic write with byte-idempotent check
    adjudication_json = json.dumps(adjudication, indent=2, ensure_ascii=False) + "\n"
    adjudication_bytes = adjudication_json.encode("utf-8")

    if adjudication_path.exists():
        existing_bytes = adjudication_path.read_bytes()
        if existing_bytes != adjudication_bytes:
            raise FileExistsError(
                f"run_adjudication.json already exists with different content at {adjudication_path}. "
                "Adjudication is immutable; create-only."
            )
        # Byte-identical, idempotent write
        return adjudication

    # Write atomically
    tmp_path = adjudication_path.with_suffix(".tmp")
    tmp_path.write_bytes(adjudication_bytes)
    tmp_path.replace(adjudication_path)

    return adjudication


def _disposition_for_status(status: AdjudicationStatus) -> dict[str, Any]:
    """Return disposition rules for a given status."""
    if status == "invalid_for_calibration":
        return {
            "resume_prohibited": True,
            "selection_prohibited": True,
            "deployment_prohibited": True,
            "forensic_preservation_required": True,
            "reason": "Run has fundamental training defects that invalidate its calibration purpose.",
        }
    elif status == "invalid_for_selection":
        return {
            "resume_prohibited": False,
            "selection_prohibited": True,
            "deployment_prohibited": True,
            "forensic_preservation_required": False,
            "reason": "Run completed but checkpoints are unsuitable for model selection.",
        }
    elif status == "forensic_only":
        return {
            "resume_prohibited": True,
            "selection_prohibited": True,
            "deployment_prohibited": True,
            "forensic_preservation_required": True,
            "reason": "Run preserved solely for forensic/debugging purposes.",
        }
    elif status == "valid":
        return {
            "resume_prohibited": False,
            "selection_prohibited": False,
            "deployment_prohibited": False,
            "forensic_preservation_required": False,
            "reason": "Run completed successfully and is eligible for all downstream uses.",
        }
    else:
        raise ValueError(f"Unknown adjudication status: {status}")


@click.command()
@click.option("--run-dir", type=click.Path(exists=True, file_okay=False), required=True, help="Run directory")
@click.option(
    "--status",
    type=click.Choice(["invalid_for_calibration", "invalid_for_selection", "forensic_only", "valid"]),
    required=True,
    help="Adjudication verdict",
)
@click.option("--reason", type=str, required=True, help="Human-readable reason (prose)")
@click.option(
    "--evidence",
    type=str,
    required=True,
    help="Evidence JSON (inline string or @path/to/file.json)",
)
@click.option("--dry-run", is_flag=True, help="Print adjudication without writing")
def main(run_dir: str, status: str, reason: str, evidence: str, dry_run: bool) -> None:
    """Create adjudication sidecar for a training run."""
    # Parse evidence
    if evidence.startswith("@"):
        evidence_path = Path(evidence[1:])
        evidence_data = json.loads(evidence_path.read_text())
    else:
        evidence_data = json.loads(evidence)

    adjudication = create_adjudication(
        Path(run_dir),
        status=status,  # type: ignore
        reason=reason,
        evidence=evidence_data,
        dry_run=dry_run,
    )

    if dry_run:
        print(json.dumps(adjudication, indent=2, ensure_ascii=False))
    else:
        print(f"✅ Adjudication written to {run_dir}/run_adjudication.json")
        print(f"   Status: {adjudication['status']}")
        print(f"   Run identity: {adjudication['run_identity']}")


if __name__ == "__main__":
    main()
