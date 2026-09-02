from __future__ import annotations

import json
import sys
import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from f5_tts.train.run_manifest import (
    SCHEMA_VERSION,
    build_run_manifest,
    canonical_json,
    collect_environment_identity,
    ensure_run_manifest,
    sha256_file,
    validate_run_manifest,
    wait_for_run_manifest,
)


class TrainingRunManifestTests(unittest.TestCase):
    def build(
        self,
        root: Path,
        *,
        learning_rate=1e-5,
        source_content=b"source checkpoint",
        evidence_content='{"dataset":"frozen"}',
    ):
        source = root / "source.pt"
        evidence = root / "manifest.json"
        source.write_bytes(source_content)
        evidence.write_text(evidence_content, encoding="utf-8")
        return build_run_manifest(
            resolved_config={"optim": {"learning_rate": learning_rate}, "datasets": {"name": "/data/frozen"}},
            code_identity={"commit": "a" * 40, "dirty": True, "dirty_sha256": "b" * 64},
            dataset_identity={"contract_version": 2, "semantic_fingerprint": "c" * 64},
            source_checkpoint=source,
            environment_identity={
                "python": "3.10.0",
                "implementation": "CPython",
                "platform": "Linux",
                "torch": "2.4.0",
                "cuda": "12.4",
                "cudnn": 90100,
                "world_size": 8,
                "mixed_precision": "bf16",
            },
            evidence_paths={"train_manifest": evidence},
        )

    def test_environment_identity_has_planned_deterministic_hardware_fields(self):
        identity = collect_environment_identity()
        self.assertIn("hostname", identity)
        self.assertIn("accelerate", identity)
        self.assertEqual(identity["visible_gpu_count"], len(identity["visible_gpus"]))
        for gpu in identity["visible_gpus"]:
            self.assertEqual(set(gpu), {"name", "capability", "bf16_supported"})
        self.assertNotIn("rank", identity)
        self.assertNotIn("local_rank", identity)

    def test_manifest_binds_all_identity_inputs_canonically(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.build(root)
            self.assertEqual(manifest["schema_version"], SCHEMA_VERSION)
            self.assertEqual(len(manifest["run_identity"]), 64)
            self.assertEqual(manifest["identity"]["source_checkpoint"]["sha256"], sha256_file(root / "source.pt"))
            self.assertEqual(
                manifest["identity"]["evidence"]["train_manifest"]["sha256"],
                sha256_file(root / "manifest.json"),
            )
            reordered = {
                "identity": manifest["identity"],
                "run_identity": manifest["run_identity"],
                "schema_version": manifest["schema_version"],
            }
            self.assertEqual(canonical_json(manifest), canonical_json(reordered))
            self.assertEqual(validate_run_manifest(reordered), reordered)

    def test_identity_changes_with_config_or_source_checkpoint(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = self.build(root)
            changed_config = self.build(root, learning_rate=2e-5)
            changed_source = self.build(root, source_content=b"different")
            changed_evidence = self.build(root, evidence_content='{"dataset":"different"}')
            self.assertNotEqual(first["run_identity"], changed_config["run_identity"])
            self.assertNotEqual(first["run_identity"], changed_source["run_identity"])
            self.assertNotEqual(first["run_identity"], changed_evidence["run_identity"])

    def test_manifest_requires_dataset_identity(self):
        with TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.pt"
            source.write_bytes(b"checkpoint")
            with self.assertRaisesRegex(ValueError, "dataset identity"):
                build_run_manifest(
                    resolved_config={},
                    code_identity={},
                    dataset_identity=None,
                    source_checkpoint=source,
                    environment_identity={},
                )

    def test_follower_waits_for_rank_zero_publication(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            destination = root / "run" / "run_manifest.json"
            manifest = self.build(root)

            def publish():
                time.sleep(0.05)
                ensure_run_manifest(destination, manifest)

            thread = threading.Thread(target=publish)
            thread.start()
            try:
                loaded = wait_for_run_manifest(destination, timeout=1.0, poll_interval=0.01)
            finally:
                thread.join()
            self.assertEqual(loaded["run_identity"], manifest["run_identity"])

    def test_follower_timeout_is_clear_and_bounded(self):
        with TemporaryDirectory() as tmp:
            destination = Path(tmp) / "missing" / "run_manifest.json"
            started = time.monotonic()
            with self.assertRaisesRegex(TimeoutError, "rank zero.*run manifest"):
                wait_for_run_manifest(destination, timeout=0.05, poll_interval=0.01)
            self.assertLess(time.monotonic() - started, 0.5)

    def test_follower_rejects_invalid_published_manifest(self):
        with TemporaryDirectory() as tmp:
            destination = Path(tmp) / "run_manifest.json"
            destination.write_text('{"not":"valid"}', encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "run manifest is invalid"):
                wait_for_run_manifest(destination, timeout=0.1)

    def test_atomic_write_is_idempotent_and_immutable(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            destination = root / "run" / "run_manifest.json"
            manifest = self.build(root)
            ensure_run_manifest(destination, manifest)
            first_bytes = destination.read_bytes()
            ensure_run_manifest(destination, manifest)
            self.assertEqual(destination.read_bytes(), first_bytes)

            other = self.build(root, learning_rate=2e-5)
            with self.assertRaisesRegex(RuntimeError, "run identity mismatch"):
                ensure_run_manifest(destination, other)
            self.assertEqual(destination.read_bytes(), first_bytes)

    def test_replace_failure_does_not_publish_partial_manifest(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            destination = root / "run" / "run_manifest.json"
            with patch("f5_tts.train.run_manifest.os.replace", side_effect=OSError("boom")):
                with self.assertRaisesRegex(OSError, "boom"):
                    ensure_run_manifest(destination, self.build(root))
            self.assertFalse(destination.exists())
            self.assertEqual(list(destination.parent.glob(".run_manifest.json.*")), [])

    def test_tampering_is_detected(self):
        with TemporaryDirectory() as tmp:
            manifest = self.build(Path(tmp))
            manifest["identity"]["environment"]["world_size"] = 1
            with self.assertRaisesRegex(ValueError, "identity hash"):
                validate_run_manifest(json.loads(json.dumps(manifest)))


class CalibrationLauncherTests(unittest.TestCase):
    def test_launcher_pins_protocol_evidence_and_safe_resume(self):
        launcher = Path(__file__).resolve().parents[1] / "tools" / "train_visualnovel_calibration_ja.sh"
        text = launcher.read_text(encoding="utf-8")
        for fixed in (
            "--mixed_precision bf16",
            "datasets.num_workers=1",
            "optim.max_updates=5000",
            "ckpts.save_per_updates=500",
            "ckpts.last_per_updates=100",
            "ckpts.keep_last_n_checkpoints=10",
        ):
            self.assertIn(fixed, text)
        # World size is now a variable so that smoke runs can use the same launch
        # path; the 8-process protocol is pinned by the default plus the guard
        # below rather than by a literal, and both are asserted behaviourally in
        # CalibrationLauncherGuardTests.
        self.assertIn("VISUALNOVEL_CALIBRATION_NUM_PROCESSES:-8", text)
        self.assertIn("--multi_gpu --num_processes $NUM_PROCESSES", text)
        for evidence in (
            "visualnovel_selection_v1.json",
            "READY",
            "aggregate_provenance.json",
            "aggregate_report.json",
            "manifest.json",
            "eval_manifest.json",
            "audio_roots.json",
            "deployment_audio_roots.json",
        ):
            self.assertIn(evidence, text)
        self.assertIn("EXPLICIT_RUN_NAME", text)
        self.assertIn("run_manifest.json", text)
        self.assertIn("model_last.pt", text)
        self.assertIn("f5_tts.train.operation_lock_run", text)
        self.assertIn("--operation training", text)


class CalibrationLauncherGuardTests(unittest.TestCase):
    """Execute the launcher's argument guards rather than grepping for them.

    Every case here is rejected before the launcher reaches its required-file
    checks, so none of them can start training. The accepted case is asserted by
    the *reason* it stops: a missing dataset rather than a protocol violation.
    """

    def run_launcher(self, **env_overrides):
        import os
        import shutil
        import subprocess

        bash = shutil.which("bash")
        if bash is None:
            self.skipTest("bash is not available on this platform")
        launcher = Path(__file__).resolve().parents[1] / "tools" / "train_visualnovel_calibration_ja.sh"
        env = {key: value for key, value in os.environ.items() if not key.startswith("VISUALNOVEL_")}
        env.update(env_overrides)
        return subprocess.run(
            [bash, str(launcher)], capture_output=True, text=True, env=env, timeout=120
        )

    def test_non_smoke_run_may_not_change_world_size(self):
        result = self.run_launcher(VISUALNOVEL_CALIBRATION_NUM_PROCESSES="4")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("requires 8 processes", result.stderr)

    def test_smoke_run_may_change_world_size(self):
        # Passes the protocol guard, then stops on the absent dataset — which is
        # how we know it got past the guard without ever reaching a launch.
        result = self.run_launcher(
            VISUALNOVEL_CALIBRATION_NUM_PROCESSES="1",
            VISUALNOVEL_CALIBRATION_SMOKE_UPDATES="120",
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("requires 8 processes", result.stderr)
        self.assertIn("Required file not found", result.stderr)

    def test_rejects_non_integer_world_size(self):
        for bad in ("0", "-1", "abc", "8x"):
            with self.subTest(num_processes=bad):
                result = self.run_launcher(VISUALNOVEL_CALIBRATION_NUM_PROCESSES=bad)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("must be a positive integer", result.stderr)

    def test_rejects_non_integer_smoke_updates(self):
        for bad in ("0", "-5", "12x"):
            with self.subTest(smoke_updates=bad):
                result = self.run_launcher(VISUALNOVEL_CALIBRATION_SMOKE_UPDATES=bad)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("must be a positive integer", result.stderr)


class OperationLockRunnerTests(unittest.TestCase):
    def test_lock_is_held_for_child_lifetime_and_released(self):
        from f5_tts.train.operation_lock_run import run_locked

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            probe = root / "probe.py"
            observed = root / "observed.txt"
            probe.write_text(
                "import pathlib, sys\n"
                "lock = pathlib.Path(sys.argv[1]) / 'training.lock'\n"
                "pathlib.Path(sys.argv[2]).write_text(str(lock.is_file()), encoding='utf-8')\n",
                encoding="utf-8",
            )
            result = run_locked(root, "training", "unit-test", [sys.executable, str(probe), str(root), str(observed)])
            self.assertEqual(result, 0)
            self.assertEqual(observed.read_text(encoding="utf-8"), "True")
            self.assertFalse((root / "training.lock").exists())


if __name__ == "__main__":
    unittest.main()
