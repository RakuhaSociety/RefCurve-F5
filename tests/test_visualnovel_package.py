from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import py7zr
import soundfile as sf


SCRIPT = Path(__file__).resolve().parents[1] / "src" / "f5_tts" / "train" / "datasets" / "visualnovel_package.py"
SPEC = importlib.util.spec_from_file_location("visualnovel_package", SCRIPT)
package = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = package
SPEC.loader.exec_module(package)


def member(path: str, size: int = 1, compressed: int = 1, **values):
    return {"filename": path, "uncompressed": size, "compressed": compressed, **values}


class InventoryValidationTests(unittest.TestCase):
    def test_real_layout_and_extensionless_voice(self):
        inventory = [member("pilot/index.json", 100), member("pilot/Alice/0010001.ogg", 200)]
        root, checked = package.validate_inventory(inventory)
        records = package.validate_index(
            [{"Speaker": "Alice", "Voice": "0010001", "Text": "hello"}], checked, root=root, expected_records=1
        )
        self.assertEqual(records[0]["relative_audio_path"], "Alice/0010001.ogg")

    def test_nfkc_casefold_match_uses_actual_inventory_path(self):
        inventory = [member("root/index.json"), member("root/冒険者ａ/m_adva0001.ogg")]
        root, checked = package.validate_inventory(inventory)
        records = package.validate_index(
            [{"Speaker": "冒険者Ａ", "Voice": "m_adva0001", "Text": "hello"}],
            checked,
            root=root,
            expected_records=1,
        )
        self.assertEqual(records[0]["indexed_audio_path"], "冒険者Ａ/m_adva0001.ogg")
        self.assertEqual(records[0]["relative_audio_path"], "冒険者ａ/m_adva0001.ogg")

    def test_file_path_nfkc_casefold_match_uses_actual_inventory_path(self):
        inventory = [member("root/index.json"), member("root/冒険者ａ/m_adva0001.ogg")]
        root, checked = package.validate_inventory(inventory)
        records = package.validate_index(
            [{"FilePath": "冒険者Ａ\\m_adva0001.ogg", "Speaker": "冒険者Ａ", "Text": "hello"}],
            checked,
            root=root,
            expected_records=1,
        )
        self.assertEqual(records[0]["indexed_audio_path"], "冒険者Ａ/m_adva0001.ogg")
        self.assertEqual(records[0]["relative_audio_path"], "冒険者ａ/m_adva0001.ogg")

    def test_path_ads_control_traversal_and_multi_root_rejected(self):
        bad_paths = ["../escape.ogg", "/absolute.ogg", "C:/escape.ogg", "root/../escape.ogg", "root/a:ads.ogg", "root/a\x01.ogg"]
        for path in bad_paths:
            with self.subTest(path=path), self.assertRaises(package.PackageValidationError):
                package.validate_inventory([member("root/index.json"), member(path)])
        with self.assertRaisesRegex(package.PackageValidationError, "one top-level"):
            package.validate_inventory([member("one/index.json"), member("two/A/a.ogg")])

    def test_nfkc_duplicate_prefix_link_and_resource_limits(self):
        with self.assertRaisesRegex(package.PackageValidationError, "Unicode-equivalent"):
            package.validate_inventory([member("root/index.json"), member("root/A/Ａ.ogg"), member("root/A/A.ogg")])
        with self.assertRaisesRegex(package.PackageValidationError, "prefix conflict"):
            package.validate_inventory([member("root/index.json"), member("root/A"), member("root/A/a.ogg")])
        with self.assertRaisesRegex(package.PackageValidationError, "links are forbidden"):
            package.validate_inventory([member("root/index.json"), member("root/A/a.ogg", is_symlink=True)])
        with self.assertRaisesRegex(package.PackageValidationError, "member exceeds"):
            package.validate_inventory(
                [member("root/index.json"), member("root/A/a.ogg", size=11)], package.ResourceLimits(max_member_bytes=10)
            )
        with self.assertRaisesRegex(package.PackageValidationError, "compression ratio"):
            package.validate_inventory(
                [member("root/index.json"), member("root/A/a.ogg", size=101, compressed=1)],
                package.ResourceLimits(max_compression_ratio=100),
            )

    def test_only_index_and_two_segment_ogg_allowed(self):
        for bad in ("root/readme.txt", "root/a.wav", "root/a.ogg", "root/A/deep/a.ogg"):
            with self.subTest(bad=bad), self.assertRaisesRegex(package.PackageValidationError, "may contain only"):
                package.validate_inventory([member("root/index.json"), member(bad)])

    def test_strict_schema_segments_count_and_audio_set(self):
        paths = ["root/index.json", "root/A/a.ogg"]
        with self.assertRaisesRegex(package.PackageValidationError, "must not be empty"):
            package.validate_index([], paths, root="root")
        with self.assertRaisesRegex(package.PackageValidationError, "exactly 107"):
            package.validate_index([], paths, root="root", expected_records=107)
        with self.assertRaisesRegex(package.PackageValidationError, "must use exactly"):
            package.validate_index(
                [{"Speaker": "A", "Voice": "a", "Text": "x", "Extra": "bad"}], paths, root="root", expected_records=1
            )
        for field, value in (("Speaker", "A/B"), ("Voice", "a.ogg"), ("Voice", "a:ads")):
            record = {"Speaker": "A", "Voice": "a", "Text": "x"}
            record[field] = value
            with self.subTest(field=field, value=value), self.assertRaisesRegex(package.PackageValidationError, "invalid"):
                package.validate_index([record], paths, root="root", expected_records=1)
        with self.assertRaisesRegex(package.PackageValidationError, "missing referenced"):
            package.validate_index(
                [{"Speaker": "A", "Voice": "missing", "Text": "x"}], paths, root="root", expected_records=1
            )

    def test_dynamic_count_and_max_records_policy(self):
        paths = ["root/index.json", "root/A/a.ogg"]
        records = package.validate_index(
            [{"Speaker": "A", "Voice": "a", "Text": "x"}], paths, root="root"
        )
        self.assertEqual(len(records), 1)
        with self.assertRaisesRegex(package.PackageValidationError, "max_records_per_package=1"):
            package.validate_index(
                [
                    {"Speaker": "A", "Voice": "a", "Text": "x"},
                    {"Speaker": "B", "Voice": "b", "Text": "y"},
                ],
                ["root/index.json", "root/A/a.ogg", "root/B/b.ogg"],
                root="root",
                max_records_per_package=1,
            )

    def test_file_path_schema_duplicates_and_raw_text_are_supported(self):
        paths = ["root/index.json", "root/A/a.ogg"]
        raw = [
            {"FilePath": "A\\a.ogg", "Speaker": "A", "Text": "same  \n"},
            {"FilePath": "A\\a.ogg", "Speaker": "A", "Text": "different"},
        ]
        records = package.validate_index(raw, paths, root="root", expected_records=2)
        self.assertEqual([record["relative_audio_path"] for record in records], ["A/a.ogg", "A/a.ogg"])
        self.assertEqual(records[0]["Text"], "same  \n")
        self.assertEqual(records[0]["index_schema"], "FilePath")

    def test_file_path_schema_is_strict_and_cannot_mix(self):
        paths = ["root/index.json", "root/A/a.ogg"]
        invalid = [
            {"FilePath": "a\\a.ogg", "Speaker": "A", "Text": "x"},
            {"FilePath": "A\\deep\\a.ogg", "Speaker": "A", "Text": "x"},
            {"FilePath": "A\\a.wav", "Speaker": "A", "Text": "x"},
            {"FilePath": "A\\a.ogg", "Speaker": "A", "Text": "x", "Voice": "a"},
        ]
        for record in invalid:
            with self.subTest(record=record), self.assertRaises(package.PackageValidationError):
                package.validate_index([record], paths, root="root")
        with self.assertRaisesRegex(package.PackageValidationError, "does not match"):
            package.validate_index(
                [
                    {"Speaker": "A", "Voice": "a", "Text": "x"},
                    {"FilePath": "A\\a.ogg", "Speaker": "A", "Text": "x"},
                ],
                paths,
                root="root",
            )

    def test_file_path_schema_allows_only_exact_ideographic_space_speaker_exception(self):
        paths = ["root/index.json", "root/　/a.ogg"]
        records = package.validate_index(
            [{"FilePath": "　\\a.ogg", "Speaker": "　", "Text": "spoken"}],
            paths,
            root="root",
            expected_records=1,
        )
        self.assertEqual(records[0]["Speaker"], "　")
        self.assertEqual(records[0]["relative_audio_path"], "　/a.ogg")
        diagnostic = {"relative_audio_path": "　/a.ogg", "sha256": "c" * 64, "bytes": 1}
        accepted = {
            "Speaker": "A", "Voice": "b", "Text": "accepted", "indexed_audio_path": "A/b.ogg",
            "relative_audio_path": "A/b.ogg", "index_schema": "FilePath",
        }
        accepted_diagnostic = {"relative_audio_path": "A/b.ogg", "sha256": "d" * 64, "bytes": 2}
        metadata = package._metadata(
            "a" * 64, 1, "root", "b" * 64, 1, [records[0], accepted],
            {
                package._normalized_key("　/a.ogg"): diagnostic,
                package._normalized_key("A/b.ogg"): accepted_diagnostic,
            },
        )
        sample = metadata["samples"][0]
        self.assertEqual(sample["source_status"], "rejected")
        self.assertEqual(sample["source_rejection_reason"], "invalid_speaker")
        self.assertEqual(sample["text"], "spoken")
        self.assertEqual(sample["audio"], diagnostic)

        spoof_or_mismatch = [
            {"FilePath": " /a.ogg", "Speaker": " ", "Text": "spoken"},
            {"FilePath": "　/a.ogg", "Speaker": " ", "Text": "spoken"},
            {"FilePath": "A/a.ogg", "Speaker": "　", "Text": "spoken"},
        ]
        for record in spoof_or_mismatch:
            with self.subTest(record=record), self.assertRaises(package.PackageValidationError):
                package.validate_index([record], ["root/index.json", "root/　/a.ogg", "root/A/a.ogg"], root="root")
        with self.assertRaises(package.PackageValidationError):
            package.validate_index(
                [{"Speaker": "　", "Voice": "a", "Text": "spoken"}], paths, root="root"
            )

    def test_unindexed_audio_requires_explicit_policy(self):
        paths = ["root/index.json", "root/A/a.ogg", "root/A/extra.ogg"]
        row = [{"Speaker": "A", "Voice": "a", "Text": "x"}]
        with self.assertRaisesRegex(package.PackageValidationError, "unindexed files"):
            package.validate_index(row, paths, root="root")
        records = package.validate_index(row, paths, root="root", allow_unindexed_audio=True)
        self.assertEqual(len(records), 1)

    def test_metadata_ids_and_provenance(self):
        records = [{"Speaker": "Alice", "Voice": "a", "Text": "hello", "relative_audio_path": "Alice/a.ogg", "index_schema": "Voice"}]
        diagnostic = {"relative_audio_path": "Alice/a.ogg", "sha256": "c" * 64, "bytes": 9, "frames": 24,
                      "sample_rate": 24000, "channels": 1, "duration_seconds": 0.001}
        audio = {package._normalized_key("Alice/a.ogg"): diagnostic}
        first = package._metadata("a" * 64, 99, "pilot", "b" * 64, 42, records, audio)
        second = package._metadata("a" * 64, 99, "pilot", "b" * 64, 42, records, audio)
        self.assertEqual(first, second)
        self.assertEqual(first["archive_bytes"], 99)
        self.assertEqual(first["archive_root"], "pilot")
        self.assertEqual(first["index_sha256"], "b" * 64)
        self.assertEqual(first["index_schema"], "Voice")
        self.assertEqual(first["unindexed_audio"], [])
        sample = first["samples"][0]
        self.assertEqual(sample["row_index"], 0)
        self.assertEqual(sample["relative_audio_path"], "Alice/a.ogg")
        self.assertEqual(sample["audio"]["bytes"], 9)


class SevenZipIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def _fixture(
        self,
        name="fixture.7z",
        password=None,
        *,
        extra=False,
        file_path_schema=False,
        duplicate=False,
        indexed_speaker="Alice",
        actual_speaker="Alice",
    ):
        source = self.root / f"source-{name}"
        voice_root = source / "pilot"
        (voice_root / actual_speaker).mkdir(parents=True)
        sf.write(voice_root / actual_speaker / "0010001.ogg", np.zeros(1200, dtype=np.float32), 24_000)
        rows = [{"Speaker": indexed_speaker, "Voice": "0010001", "Text": "hello"}]
        if file_path_schema:
            rows = [{"FilePath": f"{indexed_speaker}\\0010001.ogg", "Speaker": indexed_speaker, "Text": "hello  \n"}]
        if duplicate:
            rows.append(dict(rows[0], Text="different"))
        index_bytes = json.dumps(rows, ensure_ascii=False).encode("utf-8")
        (voice_root / "index.json").write_bytes(index_bytes)
        if extra:
            (voice_root / actual_speaker / "extra.ogg").write_bytes(
                (voice_root / actual_speaker / "0010001.ogg").read_bytes()
            )
        archive = self.root / name
        with py7zr.SevenZipFile(archive, "w", password=password) as handle:
            handle.writeall(voice_root, arcname="pilot")
        return archive, index_bytes

    def test_py7zr_113_inventory_and_atomic_idempotent_install(self):
        self.assertEqual(py7zr.__version__, "1.1.3")
        archive, index_bytes = self._fixture()
        with py7zr.SevenZipFile(archive, "r") as handle:
            inventory = package._py7zr_inventory(handle)
            self.assertTrue(inventory)
            root, _ = package.validate_inventory(inventory)
            self.assertEqual(root, "pilot")
        target = self.root / "installed"
        first = package.install_visualnovel_package(archive, target, expected_records=1)
        second = package.install_visualnovel_package(archive, target, expected_records=1)
        self.assertEqual(first, second)
        self.assertEqual(first["archive_bytes"], archive.stat().st_size)
        self.assertEqual(first["index_sha256"], hashlib.sha256(index_bytes).hexdigest())
        sample = first["samples"][0]
        audio = target / sample["relative_audio_path"]
        self.assertEqual(sample["audio"]["sha256"], hashlib.sha256(audio.read_bytes()).hexdigest())
        self.assertEqual(sample["audio"]["bytes"], audio.stat().st_size)
        self.assertTrue((target / "READY").is_file())
        self.assertFalse(list(self.root.glob(".installed.quarantine-*")))

    def test_nfkc_casefold_install_writes_canonical_path_and_preserves_index_identity(self):
        archive, _ = self._fixture(
            name="casefold.7z", indexed_speaker="冒険者Ａ", actual_speaker="冒険者ａ"
        )
        target = self.root / "installed-casefold"
        metadata = package.install_visualnovel_package(archive, target, expected_records=1)
        sample = metadata["samples"][0]
        expected_id = package._stable_id(
            "smp",
            metadata["package_id"],
            "0",
            "冒険者Ａ",
            "0010001",
            hashlib.sha256(b"hello").hexdigest(),
        )
        self.assertEqual(sample["speaker"], "冒険者Ａ")
        self.assertEqual(sample["sample_id"], expected_id)
        self.assertEqual(sample["indexed_audio_path"], "冒険者Ａ/0010001.ogg")
        self.assertEqual(sample["relative_audio_path"], "冒険者ａ/0010001.ogg")
        self.assertEqual(sample["audio"]["relative_audio_path"], sample["relative_audio_path"])
        self.assertTrue((target / "冒険者ａ" / "0010001.ogg").is_file())

    def test_file_path_install_preserves_duplicate_rows_and_shared_audio(self):
        archive, _ = self._fixture(name="filepath.7z", file_path_schema=True, duplicate=True)
        target = self.root / "installed"
        metadata = package.install_visualnovel_package(archive, target, expected_records=2)
        self.assertEqual(metadata["index_schema"], "FilePath")
        self.assertEqual(len(metadata["samples"]), 2)
        self.assertNotEqual(metadata["samples"][0]["sample_id"], metadata["samples"][1]["sample_id"])
        self.assertEqual(metadata["samples"][0]["audio"], metadata["samples"][1]["audio"])
        self.assertEqual(metadata["samples"][0]["text"], "hello  \n")

    def test_extra_audio_rejected_without_publication(self):
        archive, _ = self._fixture(name="extra.7z", extra=True)
        target = self.root / "installed"
        with self.assertRaisesRegex(package.PackageValidationError, "unindexed files"):
            package.install_visualnovel_package(archive, target, expected_records=1)
        self.assertFalse(target.exists())

    def test_extra_audio_allowed_and_audited_with_explicit_policy(self):
        archive, _ = self._fixture(name="extra-allowed.7z", extra=True)
        target = self.root / "installed"
        metadata = package.install_visualnovel_package(
            archive, target, expected_records=1, allow_unindexed_audio=True
        )
        self.assertEqual(len(metadata["unindexed_audio"]), 1)
        audit = metadata["unindexed_audio"][0]
        self.assertEqual(audit["relative_audio_path"], "Alice/extra.ogg")
        extra = target / "Alice" / "extra.ogg"
        self.assertEqual(audit["sha256"], hashlib.sha256(extra.read_bytes()).hexdigest())
        self.assertEqual(audit["bytes"], extra.stat().st_size)

    def _legacy_install(self, *, name="legacy.7z"):
        archive, _ = self._fixture(name=name)
        target = self.root / f"installed-{name}"
        current = package.install_visualnovel_package(archive, target, expected_records=1)
        legacy = {key: value for key, value in current.items() if key not in {"index_schema", "unindexed_audio"}}
        (target / "metadata.json").write_text(
            json.dumps(legacy, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        return target, legacy

    def test_legacy_metadata_migration_revalidates_and_preserves_ids(self):
        target, legacy = self._legacy_install()
        old_bytes = (target / "metadata.json").read_bytes()
        result = package.migrate_legacy_visualnovel_package(target)
        upgraded = json.loads((target / "metadata.json").read_text(encoding="utf-8"))
        provenance = json.loads((target / "metadata.migration.json").read_text(encoding="utf-8"))
        self.assertEqual(upgraded["index_schema"], "Voice")
        self.assertEqual(upgraded["unindexed_audio"], [])
        self.assertEqual(upgraded["package_id"], legacy["package_id"])
        self.assertEqual(
            [sample["sample_id"] for sample in upgraded["samples"]],
            [sample["sample_id"] for sample in legacy["samples"]],
        )
        self.assertEqual(result["metadata"], upgraded)
        self.assertEqual(result["provenance"], provenance)
        self.assertEqual(provenance["old_metadata_sha256"], hashlib.sha256(old_bytes).hexdigest())
        self.assertEqual(
            provenance["new_metadata_sha256"], hashlib.sha256((target / "metadata.json").read_bytes()).hexdigest()
        )

    def test_legacy_migration_rejects_tampered_audio_without_writes(self):
        target, _ = self._legacy_install(name="tampered.7z")
        metadata_before = (target / "metadata.json").read_bytes()
        audio = next(target.glob("*/*.ogg"))
        audio.write_bytes(audio.read_bytes() + b"tamper")
        with self.assertRaisesRegex(package.PackageValidationError, "does not match"):
            package.migrate_legacy_visualnovel_package(target)
        self.assertEqual((target / "metadata.json").read_bytes(), metadata_before)
        self.assertFalse((target / "metadata.migration.json").exists())

    def test_legacy_migration_rejects_unindexed_file_without_writes(self):
        target, _ = self._legacy_install(name="extra-after-install.7z")
        metadata_before = (target / "metadata.json").read_bytes()
        audio = next(target.glob("*/*.ogg"))
        (audio.parent / "extra.ogg").write_bytes(audio.read_bytes())
        with self.assertRaisesRegex(package.PackageValidationError, "file set differs"):
            package.migrate_legacy_visualnovel_package(target)
        self.assertEqual((target / "metadata.json").read_bytes(), metadata_before)
        self.assertFalse((target / "metadata.migration.json").exists())

    def test_legacy_migration_rejects_nonlegacy_metadata(self):
        archive, _ = self._fixture(name="current.7z")
        target = self.root / "installed-current"
        package.install_visualnovel_package(archive, target, expected_records=1)
        with self.assertRaisesRegex(package.PackageValidationError, "legacy metadata keys"):
            package.migrate_legacy_visualnovel_package(target)

    def test_path_migration_repairs_metadata_and_preserves_ids(self):
        archive, _ = self._fixture(
            name="migration-casefold.7z", indexed_speaker="冒険者Ａ", actual_speaker="冒険者ａ"
        )
        target = self.root / "installed-migration-casefold"
        canonical = package.install_visualnovel_package(archive, target, expected_records=1)
        broken = json.loads(json.dumps(canonical, ensure_ascii=False))
        broken["schema_version"] = package.PACKAGE_V1_SCHEMA_VERSION
        for old_sample in broken["samples"]:
            old_sample.pop("source_status")
            old_sample.pop("source_rejection_reason")
        sample = broken["samples"][0]
        old_id = sample["sample_id"]
        sample.pop("indexed_audio_path")
        sample["relative_audio_path"] = "冒険者Ａ/0010001.ogg"
        (target / "metadata.json").write_text(
            json.dumps(broken, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
        )

        result = package.migrate_visualnovel_audio_paths(target)
        upgraded = result["metadata"]
        repaired = upgraded["samples"][0]
        self.assertEqual(repaired["sample_id"], old_id)
        self.assertEqual(upgraded["schema_version"], package.PACKAGE_V1_SCHEMA_VERSION)
        self.assertNotIn("source_status", repaired)
        self.assertNotIn("source_rejection_reason", repaired)
        self.assertEqual(repaired["indexed_audio_path"], "冒険者Ａ/0010001.ogg")
        self.assertEqual(repaired["relative_audio_path"], "冒険者ａ/0010001.ogg")
        self.assertEqual(repaired["audio"]["relative_audio_path"], repaired["relative_audio_path"])
        self.assertTrue(result["provenance"]["sample_ids_preserved"])
        self.assertEqual(len(result["provenance"]["changed_samples"]), 1)
        self.assertTrue((target / package.PATH_MIGRATION_NAME).is_file())

    def test_path_migration_rejects_tampered_audio_without_writes(self):
        archive, _ = self._fixture(
            name="migration-tampered.7z", indexed_speaker="冒険者Ａ", actual_speaker="冒険者ａ"
        )
        target = self.root / "installed-migration-tampered"
        canonical = package.install_visualnovel_package(archive, target, expected_records=1)
        broken = json.loads(json.dumps(canonical, ensure_ascii=False))
        broken["samples"][0].pop("indexed_audio_path")
        broken["samples"][0]["relative_audio_path"] = "冒険者Ａ/0010001.ogg"
        (target / "metadata.json").write_text(
            json.dumps(broken, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        metadata_before = (target / "metadata.json").read_bytes()
        audio = target / "冒険者ａ" / "0010001.ogg"
        audio.write_bytes(audio.read_bytes() + b"tamper")
        with self.assertRaisesRegex(package.PackageValidationError, "does not match installed audio"):
            package.migrate_visualnovel_audio_paths(target)
        self.assertEqual((target / "metadata.json").read_bytes(), metadata_before)
        self.assertFalse((target / package.PATH_MIGRATION_NAME).exists())

    def test_missing_or_wrong_password_does_not_leak_secret(self):
        archive, _ = self._fixture(name="secret.7z", password="top-secret")
        for number, supplied in enumerate((None, "wrong-secret")):
            target = self.root / f"installed-{number}"
            with self.assertRaises(Exception) as caught:
                package.install_visualnovel_package(archive, target, password=supplied, expected_records=1)
            message = str(caught.exception)
            self.assertNotIn("top-secret", message)
            self.assertNotIn("wrong-secret", message)
            self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
