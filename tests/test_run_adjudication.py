"""Create-only run adjudication sidecars: idempotency, binding and tamper detection.

The sidecar exists because run manifests are immutable: an invalid run cannot be
corrected in place, so the verdict has to live beside it without touching it.
These tests pin the three properties that make that safe — the manifest is never
written, re-adjudicating the same facts is a byte-level no-op, and a manifest
changed after the fact is detected rather than silently accepted.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from f5_tts.train import run_adjudication
from f5_tts.train.run_adjudication import (
    ADJUDICATION_FILENAME,
    MANIFEST_FILENAME,
    build_run_adjudication,
    checkpoint_binding,
    ensure_run_adjudication,
    load_run_adjudication,
    manifest_binding,
    validate_run_adjudication,
    verify_binding,
)
from f5_tts.train.run_manifest import SCHEMA_VERSION as MANIFEST_SCHEMA_VERSION
from f5_tts.train.run_manifest import canonical_json


EVIDENCE = {
    "observed_learning_rate": [
        {"global_update": 500, "lr": 2.380952457142951e-06},
        {"global_update": 1000, "lr": 9.99999994581827e-14},
    ],
    "approx_floor_global_update": 625,
    "notes": ["warmup scaled by world size, horizon not"],
}


class AdjudicationTestCase(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="adjudication-test-"))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.run_dir = self.root / "run"
        self.run_dir.mkdir()
        self.manifest_payload = {"resolved_config": {"optim": {"max_updates": 5000}}, "code": {"commit": "abc123"}}
        self.write_manifest(self.manifest_payload)

    def write_manifest(self, payload):
        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "run_identity": hashlib.sha256(canonical_json(payload)).hexdigest(),
            "identity": payload,
        }
        (self.run_dir / MANIFEST_FILENAME).write_bytes(canonical_json(manifest) + b"\n")
        return manifest

    def build(self, **overrides):
        kwargs = {
            "run_binding": manifest_binding(self.run_dir),
            "status": "invalid_for_calibration",
            "reason_code": "scheduler_horizon_not_scaled_by_world_size",
            "summary": "LR reached floor at ~625 of 5000 global updates.",
            "evidence": EVIDENCE,
            "prohibit": ["resume", "model_selection", "deployment", "quality_conclusions"],
        }
        kwargs.update(overrides)
        return build_run_adjudication(**kwargs)


class BuildAndValidateTests(AdjudicationTestCase):
    def test_build_produces_self_consistent_identity(self):
        document = self.build()
        self.assertEqual(document["schema_version"], run_adjudication.SCHEMA_VERSION)
        expected = hashlib.sha256(canonical_json(document["identity"])).hexdigest()
        self.assertEqual(document["adjudication_identity"], expected)
        validate_run_adjudication(document)

    def test_binds_run_by_content_not_by_path(self):
        # Path is deliberately absent: the sidecar's own location identifies the
        # run, and a recorded absolute path would not survive being moved.
        binding = self.build()["identity"]["run"]
        self.assertEqual(set(binding), {"run_identity", "manifest_sha256", "manifest_schema_version"})
        serialised = canonical_json(self.build()).decode("utf-8")
        self.assertNotIn(str(self.run_dir), serialised)

    def test_rejects_unknown_status(self):
        with self.assertRaises(ValueError):
            self.build(status="probably_fine")

    def test_rejects_unknown_prohibition(self):
        with self.assertRaises(ValueError):
            self.build(prohibit=["resume", "delete_everything"])

    def test_rejects_empty_summary(self):
        with self.assertRaises(ValueError):
            self.build(summary="   ")

    def test_rejects_non_snake_case_reason(self):
        with self.assertRaises(ValueError):
            self.build(reason_code="not a token!")

    def test_refuses_to_authorise_deletion(self):
        with self.assertRaises(ValueError):
            self.build(retain=False)

    def test_disposition_always_retains(self):
        self.assertIs(self.build()["identity"]["disposition"]["retain"], True)

    def test_rejects_non_finite_evidence(self):
        with self.assertRaises(ValueError):
            self.build(evidence={"lr": float("nan")})
        with self.assertRaises(ValueError):
            self.build(evidence={"lr": float("inf")})

    def test_validate_rejects_tampered_identity_hash(self):
        document = self.build()
        document["adjudication_identity"] = "0" * 64
        with self.assertRaises(ValueError):
            validate_run_adjudication(document)

    def test_validate_rejects_tampered_payload(self):
        document = self.build()
        document["identity"]["verdict"]["status"] = "valid"
        with self.assertRaises(ValueError):
            validate_run_adjudication(document)


class IdempotencyTests(AdjudicationTestCase):
    def test_same_facts_produce_identical_bytes(self):
        # No wall-clock field, so re-adjudicating is a no-op rather than a conflict.
        self.assertEqual(canonical_json(self.build()), canonical_json(self.build()))

    def test_prohibition_order_does_not_change_bytes(self):
        forward = self.build(prohibit=["resume", "deployment", "model_selection"])
        reverse = self.build(prohibit=["model_selection", "deployment", "resume"])
        self.assertEqual(canonical_json(forward), canonical_json(reverse))

    def test_duplicate_prohibitions_collapse(self):
        document = self.build(prohibit=["resume", "resume", "deployment"])
        self.assertEqual(document["identity"]["disposition"]["prohibit"], ["deployment", "resume"])

    def test_ensure_creates_then_accepts_identical(self):
        destination = self.run_dir / ADJUDICATION_FILENAME
        first = ensure_run_adjudication(destination, self.build())
        first_bytes = destination.read_bytes()
        second = ensure_run_adjudication(destination, self.build())
        self.assertEqual(first, second)
        self.assertEqual(destination.read_bytes(), first_bytes)

    def test_ensure_rejects_conflicting_verdict(self):
        destination = self.run_dir / ADJUDICATION_FILENAME
        ensure_run_adjudication(destination, self.build())
        before = destination.read_bytes()
        with self.assertRaises(RuntimeError):
            ensure_run_adjudication(destination, self.build(status="valid"))
        self.assertEqual(destination.read_bytes(), before)

    def test_ensure_rejects_corrupt_existing_file(self):
        destination = self.run_dir / ADJUDICATION_FILENAME
        destination.write_text("{not json", encoding="utf-8")
        with self.assertRaises(RuntimeError):
            ensure_run_adjudication(destination, self.build())

    def test_ensure_leaves_no_lock_or_temp_files(self):
        destination = self.run_dir / ADJUDICATION_FILENAME
        ensure_run_adjudication(destination, self.build())
        leftovers = sorted(p.name for p in self.run_dir.iterdir() if p.name.startswith("."))
        self.assertEqual(leftovers, [])

    def test_written_file_reloads(self):
        destination = self.run_dir / ADJUDICATION_FILENAME
        written = ensure_run_adjudication(destination, self.build())
        self.assertEqual(load_run_adjudication(destination), written)


class ManifestImmutabilityTests(AdjudicationTestCase):
    def test_manifest_bytes_untouched_by_adjudication(self):
        manifest_path = self.run_dir / MANIFEST_FILENAME
        before = manifest_path.read_bytes()
        ensure_run_adjudication(self.run_dir / ADJUDICATION_FILENAME, self.build())
        self.assertEqual(manifest_path.read_bytes(), before)

    def test_verify_binding_accepts_untouched_manifest(self):
        document = self.build()
        self.assertEqual(verify_binding(document, self.run_dir), document["identity"]["run"])

    def test_verify_binding_detects_manifest_tampering(self):
        document = self.build()
        self.write_manifest({**self.manifest_payload, "code": {"commit": "tampered"}})
        with self.assertRaises(RuntimeError) as caught:
            verify_binding(document, self.run_dir)
        self.assertIn("no longer matches", str(caught.exception))

    def test_verify_binding_detects_byte_level_manifest_edit(self):
        # Re-serialising the same payload with different whitespace keeps
        # run_identity intact but changes the bytes; that must still be caught.
        document = self.build()
        manifest_path = self.run_dir / MANIFEST_FILENAME
        reserialised = json.dumps(json.loads(manifest_path.read_text(encoding="utf-8")), indent=2)
        manifest_path.write_text(reserialised, encoding="utf-8")
        with self.assertRaises(RuntimeError):
            verify_binding(document, self.run_dir)

    def test_manifest_binding_rejects_invalid_manifest(self):
        (self.run_dir / MANIFEST_FILENAME).write_text('{"schema_version": "wrong"}', encoding="utf-8")
        with self.assertRaises(RuntimeError):
            manifest_binding(self.run_dir)

    def test_manifest_binding_requires_a_manifest(self):
        (self.run_dir / MANIFEST_FILENAME).unlink()
        with self.assertRaises(RuntimeError):
            manifest_binding(self.run_dir)


class CheckpointBindingTests(AdjudicationTestCase):
    def make_checkpoint(self, name, content=b"weights"):
        path = self.run_dir / name
        path.write_bytes(content)
        return path

    def test_binds_name_size_and_hash(self):
        path = self.make_checkpoint("model_5000.pt", b"abc")
        (entry,) = checkpoint_binding([path])
        self.assertEqual(entry["name"], "model_5000.pt")
        self.assertEqual(entry["size"], 3)
        self.assertEqual(entry["sha256"], hashlib.sha256(b"abc").hexdigest())

    def test_entries_are_sorted_for_stable_bytes(self):
        paths = [self.make_checkpoint("model_1000.pt"), self.make_checkpoint("model_500.pt")]
        names = [entry["name"] for entry in checkpoint_binding(list(reversed(paths)))]
        self.assertEqual(names, ["model_1000.pt", "model_500.pt"])

    def test_rejects_missing_checkpoint(self):
        with self.assertRaises(FileNotFoundError):
            checkpoint_binding([self.run_dir / "absent.pt"])

    def test_rejects_duplicate_basenames(self):
        nested = self.run_dir / "nested"
        nested.mkdir()
        first = self.make_checkpoint("model_500.pt")
        second = nested / "model_500.pt"
        second.write_bytes(b"other")
        with self.assertRaises(ValueError):
            checkpoint_binding([first, second])

    def test_checkpoints_participate_in_identity(self):
        path = self.make_checkpoint("model_5000.pt", b"abc")
        without = self.build()
        with_checkpoint = self.build(checkpoints=checkpoint_binding([path]))
        self.assertNotEqual(without["adjudication_identity"], with_checkpoint["adjudication_identity"])


class CommandLineTests(AdjudicationTestCase):
    def write_evidence(self):
        path = self.root / "evidence.json"
        path.write_text(json.dumps(EVIDENCE), encoding="utf-8")
        return path

    def base_argv(self):
        return [
            "--run-dir",
            str(self.run_dir),
            "--status",
            "invalid_for_calibration",
            "--reason-code",
            "scheduler_horizon_not_scaled_by_world_size",
            "--summary",
            "LR reached floor early.",
            "--evidence-json",
            str(self.write_evidence()),
            "--prohibit",
            "resume",
        ]

    def test_dry_run_has_no_filesystem_effect(self):
        before = sorted(p.name for p in self.run_dir.iterdir())
        self.assertEqual(run_adjudication.main(self.base_argv()), 0)
        self.assertEqual(sorted(p.name for p in self.run_dir.iterdir()), before)
        self.assertFalse((self.run_dir / ADJUDICATION_FILENAME).exists())

    def test_write_creates_sidecar(self):
        self.assertEqual(run_adjudication.main(self.base_argv() + ["--write"]), 0)
        document = load_run_adjudication(self.run_dir / ADJUDICATION_FILENAME)
        self.assertEqual(document["identity"]["verdict"]["reason_code"], "scheduler_horizon_not_scaled_by_world_size")

    def test_write_is_idempotent(self):
        run_adjudication.main(self.base_argv() + ["--write"])
        before = (self.run_dir / ADJUDICATION_FILENAME).read_bytes()
        run_adjudication.main(self.base_argv() + ["--write"])
        self.assertEqual((self.run_dir / ADJUDICATION_FILENAME).read_bytes(), before)

    def test_dry_run_bytes_match_what_write_produces(self):
        # The preview must be exactly the artifact, or reviewing it proves nothing.
        import contextlib
        import io

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(io.StringIO()):
            run_adjudication.main(self.base_argv())
        run_adjudication.main(self.base_argv() + ["--write"])
        self.assertEqual(buffer.getvalue().encode("utf-8"), (self.run_dir / ADJUDICATION_FILENAME).read_bytes())


if __name__ == "__main__":
    unittest.main()
