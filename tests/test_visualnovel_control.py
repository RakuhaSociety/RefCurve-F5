from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from f5_tts.train.datasets import visualnovel_control as control


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.selection = control.load_selection()

    def test_frozen_selection_is_canonical_and_exact(self):
        self.assertEqual(len(self.selection["items"]), 12)
        self.assertEqual([item["wave"] for item in self.selection["items"]], ["A"] * 8 + ["B"] * 2 + ["C"] * 2)
        self.assertEqual(sum(item["size"] for item in self.selection["items"]), 7_891_385_875)
        self.assertEqual(self.selection["revision"], "f69468368df70700f7abb8d73c957d4f8cf0274d")
        self.assertEqual(self.selection["catalog"]["content_sha256"], "0f26549c49c2adfd902271a63182b021aad8a1cd58a202d5c7bb84fab8b6f77a")

    def test_schema_and_canonical_ids_fail_closed(self):
        changed = copy.deepcopy(self.selection)
        changed["items"][0]["size"] += 1
        with self.assertRaisesRegex(control.SelectionError, "item_id"):
            control.validate_selection(changed)
        changed = copy.deepcopy(self.selection)
        changed["extra"] = True
        with self.assertRaisesRegex(control.SelectionError, "keys differ"):
            control.validate_selection(changed)
        changed = copy.deepcopy(self.selection)
        changed["revision"] = "main"
        with self.assertRaisesRegex(control.SelectionError, "full lowercase commit"):
            control.validate_selection(changed)

    def test_catalog_api_resolved_join_and_network_injection(self):
        catalog = "# GalGame\n| No | Company | Total Duration | Total Characters | Romaji |\n|---:|---|---|---:|---|\n| 1 | Acme | A \\~ B | 2 | Title |\n"
        # malformed column placement remains rejected
        with self.assertRaises(control.SelectionError):
            control.parse_catalog(catalog)
        catalog = "# GalGame\n| No | Company | Duration | Characters | Romaji |\n|---:|---|---|---:|---|\n| 1 | Acme | 00:01:02 | 2 | A \\~ B |\n"
        rows = control.parse_catalog(catalog)
        self.assertEqual(rows[0]["repo_path"], "GalGame/Acme_A ~ B.7z")
        api = [{"path": rows[0]["repo_path"], "size": 3, "blob_id": "a" * 40, "lfs": {"size": 3, "sha256": "b" * 64}}]
        joined = control.validate_resolved_join(rows, api, expected_count=1)
        self.assertEqual(joined[0]["lfs_sha256"], "b" * 64)
        with self.assertRaisesRegex(control.SelectionError, "paths differ"):
            control.validate_resolved_join(rows, [{**api[0], "path": "GalGame/Acme_Other.7z"}], expected_count=1)


class SelectionV2Tests(unittest.TestCase):
    def setUp(self):
        self.base = control.load_selection()
        self.base_bytes = control.canonical_bytes(self.base)

    def _rows_and_api(self):
        rows = []
        api = []
        sizes = [64 * 1024**2 + index for index in range(8)]
        sizes += [512 * 1024**2 + index for index in range(8)]
        sizes += [2 * 1024**3 + index for index in range(8)]
        for item in self.base["items"]:
            rows.append({key: item[key] for key in (
                "catalog_section", "catalog_no", "company", "romaji", "duration",
                "total_characters", "repo_path",
            )})
            api.append({"path": item["repo_path"], "size": item["size"], "blob_id": item["git_blob_id"],
                        "lfs": {"size": item["size"], "sha256": item["lfs_sha256"]}})
        for index, size in enumerate(sizes):
            path = f"GalGame/Expansion_{index:02d}.7z"
            rows.append({"catalog_section": "GalGame", "catalog_no": 1000 + index,
                         "company": f"Company {index:02d}", "romaji": f"Expansion {index:02d}",
                         "duration": f"{index + 1:02d}:00:00", "total_characters": 10_000 + index,
                         "repo_path": path})
            api.append({"path": path, "size": size, "blob_id": f"{index + 1:040x}",
                        "lfs": {"size": size, "sha256": f"{index + 1:064x}"}})
        # The production catalog count is frozen at 597. Fill with valid, non-selected small rows.
        for index in range(len(rows), self.base["catalog"]["row_count"]):
            path = f"GalGame/Filler_{index:03d}.7z"
            size = 128 * 1024**2 + index
            rows.append({"catalog_section": "GalGame", "catalog_no": 2000 + index,
                         "company": f"Filler {index:03d}", "romaji": f"Filler {index:03d}",
                         "duration": "01:00:00", "total_characters": 20_000 + index, "repo_path": path})
            api.append({"path": path, "size": size, "blob_id": f"{index + 1000:040x}",
                        "lfs": {"size": size, "sha256": f"{index + 1000:064x}"}})
        return rows, api

    def test_schema_dispatch_exact_base_and_24_additions(self):
        rows, api = self._rows_and_api()
        selection = control.generate_selection_v2(self.base, rows, api, base_selection_bytes=self.base_bytes)
        self.assertEqual(selection["schema_version"], control.SCHEMA_VERSION_V2)
        self.assertEqual(selection["items"][:12], self.base["items"])
        self.assertEqual([item["wave"] for item in selection["items"]],
                         ["A"] * 8 + ["B"] * 2 + ["C"] * 2 + ["D"] * 8 + ["E"] * 8 + ["F"] * 8)
        self.assertEqual([item["selection_order"] for item in selection["items"]], list(range(36)))
        with self.assertRaisesRegex(control.SelectionError, "keys differ"):
            control.validate_selection_v1(selection)

    def test_disjoint_path_and_lfs_checks(self):
        rows, api = self._rows_and_api()
        selection = control.generate_selection_v2(self.base, rows, api, base_selection_bytes=self.base_bytes)
        changed = copy.deepcopy(selection)
        changed["items"][12]["repo_path"] = self.base["items"][0]["repo_path"]
        changed["items"][12]["item_id"] = control.stable_id("item", control.item_payload(changed["items"][12]))
        changed["selection_id"] = control.stable_id("sel", control.selection_payload(changed))
        with self.assertRaisesRegex(control.SelectionError, "normalized repo_path|overlap"):
            control.validate_selection_v2(changed, base_selection=self.base, base_selection_bytes=self.base_bytes)
        changed = copy.deepcopy(selection)
        changed["items"][12]["lfs_sha256"] = self.base["items"][0]["lfs_sha256"]
        changed["items"][12]["item_id"] = control.stable_id("item", control.item_payload(changed["items"][12]))
        changed["selection_id"] = control.stable_id("sel", control.selection_payload(changed))
        with self.assertRaisesRegex(control.SelectionError, "LFS"):
            control.validate_selection_v2(changed, base_selection=self.base, base_selection_bytes=self.base_bytes)

    def test_api_order_invariance_and_strict_insufficient_stratum(self):
        rows, api = self._rows_and_api()
        first = control.generate_selection_v2(self.base, rows, api, base_selection_bytes=self.base_bytes)
        second = control.generate_selection_v2(self.base, rows, reversed(api), base_selection_bytes=self.base_bytes)
        self.assertEqual(control.canonical_bytes(first), control.canonical_bytes(second))
        insufficient = copy.deepcopy(api)
        for value in insufficient:
            if 256 * 1024**2 <= value["size"] < 1024**3:
                value["size"] = 128 * 1024**2
                value["lfs"]["size"] = value["size"]
        with self.assertRaisesRegex(control.SelectionError, "insufficient candidates in stratum medium"):
            control.generate_selection_v2(self.base, rows, insufficient, base_selection_bytes=self.base_bytes)

    def test_v2_run_contains_only_independent_additions(self):
        rows, api = self._rows_and_api()
        selection = control.generate_selection_v2(self.base, rows, api, base_selection_bytes=self.base_bytes)
        with tempfile.TemporaryDirectory() as directory, control.ControlDB(Path(directory) / "v2.sqlite") as db:
            run = db.init_run(selection, run_id="v2")
            status = db.status(run)
            self.assertEqual(len(status["items"]), 24)
            self.assertEqual(status["wave_gate"], "D")
            self.assertEqual([item["wave"] for item in status["items"]], ["D"] * 8 + ["E"] * 8 + ["F"] * 8)
    def test_future_wave_exact_authorized_prefetch_does_not_advance_processing_gate(self):
        rows, api = self._rows_and_api()
        selection = control.generate_selection_v2(self.base, rows, api, base_selection_bytes=self.base_bytes)
        item = next(value for value in selection["items"] if value["wave"] == "F")
        payload = b"f" * item["size"] if item["size"] < 1024 else b"future-wave"
        # Keep the fixture small while retaining a canonical frozen identity.
        item["size"] = len(payload)
        item["lfs_sha256"] = hashlib.sha256(payload).hexdigest()
        item["item_id"] = control.stable_id("item", control.item_payload(item))
        selection["selection_id"] = control.stable_id("sel", control.selection_payload(selection))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with control.ControlDB(root / "control.sqlite") as db:
                run = db.init_run(selection, run_id="future", soft_watermark_bytes=100, hard_watermark_bytes=10)
                self.assertEqual(db.run(run)["wave_gate"], "D")

                def downloader(**kwargs):
                    path = Path(kwargs["local_dir"]) / item["repo_path"]
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(payload)
                    return path

                usage = lambda path: SimpleNamespace(total=10_000, free=9_000)
                with self.assertRaisesRegex(control.StateError, "authorization allowlist"):
                    control.fetch_one(
                        db=db, run_id=run, selection=selection, item_id=item["item_id"],
                        spool_root=root / "spool", owner="worker", authorized_paths=[],
                        downloader=downloader, disk_usage=usage, lock_root=root / "locks",
                    )
                archive = control.fetch_one(
                    db=db, run_id=run, selection=selection, item_id=item["item_id"],
                    spool_root=root / "spool", owner="worker", authorized_paths=[item["repo_path"]],
                    downloader=downloader, disk_usage=usage, lock_root=root / "locks",
                )
                self.assertTrue(archive.is_file())
                self.assertEqual(db.run(run)["wave_gate"], "D")
                self.assertEqual(db.item(run, item["item_id"])["state"], "DOWNLOADED_UNVERIFIED")
                self.assertFalse(archive.with_name(archive.name + ".source-receipt-v1.json").exists())
                self.assertEqual(db.connection.execute(
                    "SELECT COUNT(*) FROM artifacts WHERE run_id=? AND item_id=?",
                    (run, item["item_id"]),
                ).fetchone()[0], 0)


class ControlStateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.selection = control.load_selection()
        self.db = control.ControlDB(self.root / "control.sqlite")
        self.run_id = self.db.init_run(
            self.selection,
            run_id="run_test",
            soft_watermark_bytes=1_000_000,
            hard_watermark_bytes=500_000,
        )
        self.item = self.selection["items"][0]

    def tearDown(self):
        self.db.close()
        self.temp.cleanup()

    def test_sqlite_pragmas_schema_wave_gate_and_illegal_transition(self):
        self.assertEqual(self.db.connection.execute("PRAGMA journal_mode").fetchone()[0], "wal")
        self.assertEqual(self.db.connection.execute("PRAGMA foreign_keys").fetchone()[0], 1)
        self.assertEqual(len(self.db.status(self.run_id)["items"]), 12)
        self.assertEqual(self.db.next_selected(self.run_id)["wave"], "A")
        with self.assertRaisesRegex(control.StateError, "illegal transition"):
            self.db.transition(self.run_id, self.item["item_id"], "SELECTED", "PACKAGE_READY")

    def test_lease_competition_and_transactional_artifact_event(self):
        item_id = self.item["item_id"]
        self.assertTrue(self.db.acquire_lease(self.run_id, item_id, "one", ttl_seconds=10, now=100))
        self.assertFalse(self.db.acquire_lease(self.run_id, item_id, "two", ttl_seconds=10, now=101))
        self.assertTrue(self.db.acquire_lease(self.run_id, item_id, "two", ttl_seconds=10, now=111))
        # Use a current, valid lease for transition.
        self.db.release_lease(self.run_id, item_id, "two")
        self.assertTrue(self.db.acquire_lease(self.run_id, item_id, "one"))
        self.db.transition(self.run_id, item_id, "SELECTED", "DOWNLOADING", owner="one")
        artifact = {"kind": "archive", "uri": "x", "bytes": self.item["size"], "sha256": self.item["lfs_sha256"],
                    "etag": None, "revision": self.selection["revision"], "committed": True}
        self.db.transition(self.run_id, item_id, "DOWNLOADING", "DOWNLOAD_VERIFIED", owner="one", artifact=artifact)
        self.assertEqual(self.db.item(self.run_id, item_id)["state"], "DOWNLOAD_VERIFIED")
        self.assertEqual(self.db.connection.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0], 1)
        self.assertEqual(self.db.connection.execute("SELECT COUNT(*) FROM transitions WHERE item_id=?", (item_id,)).fetchone()[0], 3)

    def _mark_wave_dataset_ready(self, wave="A"):
        now = 10.0
        for item in (value for value in self.selection["items"] if value["wave"] == wave):
            for state in ("DOWNLOADING", "DOWNLOAD_VERIFIED", "INSTALLING", "PACKAGE_READY", "METADATA_EXPORTED",
                          "SIDECAR_READY", "DATASET_READY"):
                previous = self.db.item(self.run_id, item["item_id"])["state"]
                self.db.connection.execute(
                    "UPDATE items SET state=? WHERE run_id=? AND item_id=?",
                    (state, self.run_id, item["item_id"]),
                )
                self.db.connection.execute(
                    "INSERT INTO transitions(run_id,item_id,from_state,to_state,attempt,timestamp,details_json) "
                    "VALUES(?,?,?,?,?,?,?)",
                    (self.run_id, item["item_id"], previous, state, 0, now, "{}"),
                )
                now += 1

    def _make_aggregate(self, wave="A"):
        aggregate = self.root / f"aggregate-{wave.lower()}"
        aggregate.mkdir()
        manifest = b'{"fixture":"train"}\n'
        eval_manifest = b'{"fixture":"eval"}\n'
        (aggregate / "manifest.json").write_bytes(manifest)
        (aggregate / "eval_manifest.json").write_bytes(eval_manifest)
        semantic, deployment = "a" * 64, "b" * 64
        inputs = [{"wave": item["wave"], "selection_order": item["selection_order"]}
                  for item in self.selection["items"] if control.WAVES.index(item["wave"]) <= control.WAVES.index(wave)]
        provenance = {
            "semantic_fingerprint": semantic,
            "deployment_fingerprint": deployment,
            "manifest_sha256": hashlib.sha256(manifest).hexdigest(),
            "aggregate_context": {
                "source_revision": self.selection["revision"],
                "catalog_identity": control.aggregate_catalog_identity(self.selection),
                "selection_fingerprint": control.aggregate_selection_fingerprint(self.selection),
            },
            "inputs": inputs,
        }
        (aggregate / "aggregate_provenance.json").write_bytes(control.canonical_bytes(provenance))
        ready = {
            "commit": deployment,
            "semantic_fingerprint": semantic,
            "deployment_fingerprint": deployment,
            "train_manifest_sha256": hashlib.sha256(manifest).hexdigest(),
            "eval_manifest_sha256": hashlib.sha256(eval_manifest).hexdigest(),
        }
        (aggregate / "READY").write_bytes(control.canonical_bytes(ready))
        return aggregate, deployment

    def test_register_aggregate_promotes_items_records_artifacts_and_events(self):
        self.assertEqual(self.db.run(self.run_id)["wave_gate"], "A")
        self._mark_wave_dataset_ready()
        aggregate, fingerprint = self._make_aggregate()
        result = self.db.register_aggregate(
            self.run_id, self.selection, aggregate, expected_fingerprint=fingerprint, promote_wave=True
        )
        self.assertEqual((result["registered_items"], result["changed_items"], result["wave_gate"]), (8, 8, "B"))
        states = self.db.connection.execute(
            "SELECT DISTINCT state FROM items WHERE run_id=? AND wave='A'", (self.run_id,)
        ).fetchall()
        self.assertEqual([row[0] for row in states], ["AGGREGATED"])
        self.assertEqual(self.db.connection.execute(
            "SELECT COUNT(*) FROM artifacts WHERE run_id=? AND kind LIKE 'aggregate_%'", (self.run_id,)
        ).fetchone()[0], 32)
        events = [json.loads(line) for line in self.db.events_path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(sum(event.get("to_state") == "AGGREGATED" for event in events), 8)
        self.assertEqual(events[-1]["event"], "WAVE_PROMOTED")

    def test_register_wave_b_requires_cumulative_inputs_and_only_changes_b(self):
        self._mark_wave_dataset_ready("A")
        aggregate_a, fingerprint_a = self._make_aggregate("A")
        self.db.register_aggregate(
            self.run_id, self.selection, aggregate_a, expected_fingerprint=fingerprint_a, promote_wave=True
        )
        a_transition_count = self.db.connection.execute(
            "SELECT COUNT(*) FROM transitions t JOIN items i ON i.run_id=t.run_id AND i.item_id=t.item_id "
            "WHERE t.run_id=? AND i.wave='A' AND t.to_state='AGGREGATED'",
            (self.run_id,),
        ).fetchone()[0]
        a_artifacts = self.db.connection.execute(
            "SELECT item_id,kind,uri,sha256 FROM artifacts WHERE run_id=? AND item_id IN "
            "(SELECT item_id FROM items WHERE run_id=? AND wave='A') ORDER BY item_id,kind",
            (self.run_id, self.run_id),
        ).fetchall()

        self._mark_wave_dataset_ready("B")
        aggregate_b, fingerprint_b = self._make_aggregate("B")
        provenance_path = aggregate_b / "aggregate_provenance.json"
        valid_provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        cumulative_inputs = valid_provenance["inputs"]
        self.assertEqual([value["wave"] for value in cumulative_inputs], ["A"] * 8 + ["B"] * 2)

        invalid_inputs = {
            "only target wave": [value for value in cumulative_inputs if value["wave"] == "B"],
            "missing prior-wave position": cumulative_inputs[1:],
            "extra later-wave position": cumulative_inputs + [{"wave": "C", "selection_order": 10}],
        }
        for label, inputs in invalid_inputs.items():
            with self.subTest(label=label):
                provenance = copy.deepcopy(valid_provenance)
                provenance["inputs"] = inputs
                provenance_path.write_bytes(control.canonical_bytes(provenance))
                with self.assertRaisesRegex(control.StateError, "cumulative selection"):
                    self.db.register_aggregate(
                        self.run_id, self.selection, aggregate_b, expected_fingerprint=fingerprint_b
                    )

        provenance_path.write_bytes(control.canonical_bytes(valid_provenance))
        with self.assertRaisesRegex(control.StateError, "cannot promote wave B"):
            self.db.promote_wave(self.run_id)
        result = self.db.register_aggregate(
            self.run_id, self.selection, aggregate_b, expected_fingerprint=fingerprint_b, promote_wave=True
        )
        self.assertEqual((result["registered_items"], result["changed_items"], result["wave_gate"]), (2, 2, "C"))
        self.assertEqual(
            self.db.connection.execute(
                "SELECT DISTINCT state FROM items WHERE run_id=? AND wave='A'", (self.run_id,)
            ).fetchall()[0][0],
            "AGGREGATED",
        )
        self.assertEqual(
            self.db.connection.execute(
                "SELECT COUNT(*) FROM transitions t JOIN items i ON i.run_id=t.run_id AND i.item_id=t.item_id "
                "WHERE t.run_id=? AND i.wave='A' AND t.to_state='AGGREGATED'",
                (self.run_id,),
            ).fetchone()[0],
            a_transition_count,
        )
        self.assertEqual(
            self.db.connection.execute(
                "SELECT item_id,kind,uri,sha256 FROM artifacts WHERE run_id=? AND item_id IN "
                "(SELECT item_id FROM items WHERE run_id=? AND wave='A') ORDER BY item_id,kind",
                (self.run_id, self.run_id),
            ).fetchall(),
            a_artifacts,
        )

    def test_inspect_wave_c_accepts_all_cumulative_positions(self):
        aggregate, fingerprint = self._make_aggregate("C")
        result = control.inspect_aggregate(
            self.selection, aggregate, wave="C", expected_fingerprint=fingerprint
        )
        self.assertEqual(result["wave"], "C")

    def test_register_aggregate_fail_closed_and_repeat_is_idempotent(self):
        aggregate, fingerprint = self._make_aggregate()
        with self.assertRaisesRegex(control.StateError, "cannot promote wave A"):
            self.db.promote_wave(self.run_id)
        with self.assertRaisesRegex(control.StateError, "not DATASET_READY"):
            self.db.register_aggregate(self.run_id, self.selection, aggregate, expected_fingerprint=fingerprint)
        self._mark_wave_dataset_ready()
        with self.assertRaisesRegex(control.StateError, "fingerprint"):
            self.db.register_aggregate(self.run_id, self.selection, aggregate, expected_fingerprint="c" * 64)
        self.db.register_aggregate(self.run_id, self.selection, aggregate, expected_fingerprint=fingerprint)
        transition_count = self.db.connection.execute(
            "SELECT COUNT(*) FROM transitions WHERE run_id=? AND to_state='AGGREGATED'", (self.run_id,)
        ).fetchone()[0]
        repeated = self.db.register_aggregate(self.run_id, self.selection, aggregate, expected_fingerprint=fingerprint)
        self.assertEqual(repeated["changed_items"], 0)
        self.assertEqual(self.db.connection.execute(
            "SELECT COUNT(*) FROM transitions WHERE run_id=? AND to_state='AGGREGATED'", (self.run_id,)
        ).fetchone()[0], transition_count)
        self.assertEqual(self.db.promote_wave(self.run_id, completed_wave="A"), "B")
        self.assertEqual(self.db.promote_wave(self.run_id, completed_wave="A"), "B")
        with self.assertRaisesRegex(control.StateError, "aggregate wave C"):
            self.db.promote_wave(self.run_id, completed_wave="C")

    def test_register_aggregate_rejects_context_mismatch_and_lease(self):
        self._mark_wave_dataset_ready()
        aggregate, fingerprint = self._make_aggregate()
        provenance_path = aggregate / "aggregate_provenance.json"
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        provenance["aggregate_context"]["selection_fingerprint"] = "f" * 64
        provenance_path.write_bytes(control.canonical_bytes(provenance))
        with self.assertRaisesRegex(control.StateError, "context"):
            self.db.register_aggregate(self.run_id, self.selection, aggregate, expected_fingerprint=fingerprint)
        provenance["aggregate_context"]["selection_fingerprint"] = control.aggregate_selection_fingerprint(self.selection)
        provenance_path.write_bytes(control.canonical_bytes(provenance))
        item_id = self.selection["items"][0]["item_id"]
        self.assertTrue(self.db.acquire_lease(self.run_id, item_id, "worker"))
        with self.assertRaisesRegex(control.StateError, "has a lease"):
            self.db.register_aggregate(self.run_id, self.selection, aggregate, expected_fingerprint=fingerprint)

    def test_crash_reconcile_hash_after_db_lag(self):
        item_id = self.item["item_id"]
        self.assertTrue(self.db.acquire_lease(self.run_id, item_id, "worker"))
        self.db.transition(self.run_id, item_id, "SELECTED", "DOWNLOADING", owner="worker")
        self.db.release_lease(self.run_id, item_id, "worker")
        archive = self.root / "spool" / "files" / self.item["repo_path"]
        archive.parent.mkdir(parents=True)
        payload = b"reconciled"
        archive.write_bytes(payload)
        self.db.connection.execute(
            "UPDATE items SET expected_size=?,lfs_sha256=? WHERE run_id=? AND item_id=?",
            (len(payload), hashlib.sha256(payload).hexdigest(), self.run_id, item_id),
        )
        actions = control.reconcile(self.db, self.run_id, spool_root=self.root / "spool")
        self.assertEqual(actions, [{"item_id": item_id, "to_state": "DOWNLOAD_VERIFIED"}])
        self.assertEqual(self.db.item(self.run_id, item_id)["state"], "DOWNLOAD_VERIFIED")


class DownloadDiskAndDeletionTests(unittest.TestCase):
    def test_watermarks(self):
        gib = 1024**3
        total = 4 * 1024**4
        soft, hard = control.default_watermarks(total)
        self.assertEqual(soft, max(500 * gib, int(total * 0.12)))
        self.assertEqual(hard, max(200 * gib, int(total * 0.05)))
        self.assertFalse(control.disk_admission(total_bytes=total, free_bytes=hard, archive_size=1).allowed)
        self.assertFalse(control.disk_admission(total_bytes=total, free_bytes=soft, archive_size=1).allowed)
        self.assertTrue(control.disk_admission(total_bytes=total, free_bytes=soft + 10 * gib, archive_size=1).allowed)

    def test_mock_download_is_pinned_and_hash_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selection = copy.deepcopy(control.load_selection())
            payload = b"mock archive"
            item = selection["items"][0]
            item["size"] = len(payload)
            item["lfs_sha256"] = hashlib.sha256(payload).hexdigest()
            item["item_id"] = control.stable_id("item", control.item_payload(item))
            selection["selection_id"] = control.stable_id("sel", control.selection_payload(selection))
            control.validate_selection(selection)
            calls = []

            def downloader(**kwargs):
                calls.append(kwargs)
                path = Path(kwargs["local_dir"]) / item["repo_path"]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)
                return path

            with control.ControlDB(root / "control.sqlite") as db:
                run_id = db.init_run(selection, run_id="download", soft_watermark_bytes=100, hard_watermark_bytes=10)
                usage = lambda path: (
                    self.assertTrue(Path(path).is_dir())
                    or SimpleNamespace(total=10_000, free=9_000)
                )
                downloaded = control.download_one(db=db, run_id=run_id, selection=selection, item_id=item["item_id"],
                                                  spool_root=root / "spool", owner="worker", downloader=downloader,
                                                  disk_usage=usage)
                self.assertTrue(downloaded.is_file())
                self.assertEqual(db.item(run_id, item["item_id"])["state"], "DOWNLOAD_VERIFIED")
                self.assertEqual(calls[0]["revision"], selection["revision"])
                self.assertEqual(calls[0]["filename"], item["repo_path"])
                self.assertTrue(calls[0]["token"])
                receipt = json.loads(downloaded.with_name(downloaded.name + ".source-receipt-v1.json").read_text())
                self.assertEqual(receipt["observed_sha256"], item["lfs_sha256"])

    def test_fetch_only_size_then_verify_hash_and_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selection = copy.deepcopy(control.load_selection())
            payload = b"transfer only archive"
            item = selection["items"][0]
            item["size"] = len(payload)
            item["lfs_sha256"] = hashlib.sha256(payload).hexdigest()
            item["item_id"] = control.stable_id("item", control.item_payload(item))
            selection["selection_id"] = control.stable_id("sel", control.selection_payload(selection))
            control.validate_selection(selection)

            def downloader(**kwargs):
                path = Path(kwargs["local_dir"]) / item["repo_path"]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)
                return path

            with control.ControlDB(root / "control.sqlite") as db:
                run = db.init_run(selection, run_id="split", soft_watermark_bytes=100, hard_watermark_bytes=10)
                usage = lambda path: SimpleNamespace(total=10_000, free=9_000)
                with mock.patch.object(control, "sha256_file", side_effect=AssertionError("fetch hashed archive")), \
                     mock.patch.object(control, "make_source_receipt", side_effect=AssertionError("fetch verified source")):
                    archive = control.fetch_one(
                        db=db, run_id=run, selection=selection, item_id=item["item_id"],
                        spool_root=root / "spool", owner="worker", authorized_paths=[item["repo_path"]],
                        downloader=downloader, disk_usage=usage, lock_root=root / "locks",
                    )
                self.assertEqual(db.item(run, item["item_id"])["state"], "DOWNLOADED_UNVERIFIED")
                self.assertFalse(archive.with_name(archive.name + ".source-receipt-v1.json").exists())
                transfer = json.loads(archive.with_name(archive.name + ".transfer-receipt-v1.json").read_text())
                self.assertNotIn("observed_sha256", transfer)
                control.verify_one(db=db, run_id=run, selection=selection, item_id=item["item_id"],
                                   spool_root=root / "spool", owner="worker", lock_root=root / "locks")
                self.assertEqual(db.item(run, item["item_id"])["state"], "DOWNLOAD_VERIFIED")
                receipt = json.loads(archive.with_name(archive.name + ".source-receipt-v1.json").read_text())
                self.assertEqual(receipt["observed_sha256"], item["lfs_sha256"])

    def test_fetch_allowlist_and_verify_integrity_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selection = copy.deepcopy(control.load_selection())
            payload = b"expected"
            item = selection["items"][0]
            item["size"] = len(payload)
            item["lfs_sha256"] = hashlib.sha256(payload).hexdigest()
            item["item_id"] = control.stable_id("item", control.item_payload(item))
            selection["selection_id"] = control.stable_id("sel", control.selection_payload(selection))

            def downloader(**kwargs):
                path = Path(kwargs["local_dir"]) / item["repo_path"]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)
                return path

            with control.ControlDB(root / "control.sqlite") as db:
                run = db.init_run(selection, run_id="integrity", soft_watermark_bytes=100, hard_watermark_bytes=10)
                usage = lambda path: SimpleNamespace(total=10_000, free=9_000)
                with self.assertRaisesRegex(control.StateError, "authorization allowlist"):
                    control.fetch_one(db=db, run_id=run, selection=selection, item_id=item["item_id"],
                                      spool_root=root / "spool", owner="worker", authorized_paths=[],
                                      downloader=downloader, disk_usage=usage, lock_root=root / "locks")
                archive = control.fetch_one(db=db, run_id=run, selection=selection, item_id=item["item_id"],
                                            spool_root=root / "spool", owner="worker",
                                            authorized_paths=[item["repo_path"]], downloader=downloader,
                                            disk_usage=usage, lock_root=root / "locks")
                archive.write_bytes(b"tampered")
                with self.assertRaises(control.IntegrityError):
                    control.verify_one(db=db, run_id=run, selection=selection, item_id=item["item_id"],
                                       spool_root=root / "spool", owner="worker", lock_root=root / "locks")
                self.assertEqual(db.item(run, item["item_id"])["state"], "BLOCKED_INTEGRITY")

    def test_transfer_lease_ttl_is_long_bounded_and_receipt_atomic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selection = copy.deepcopy(control.load_selection())
            payload = b"long transfer"
            item = selection["items"][0]
            item["size"] = len(payload)
            item["lfs_sha256"] = hashlib.sha256(payload).hexdigest()
            item["item_id"] = control.stable_id("item", control.item_payload(item))
            selection["selection_id"] = control.stable_id("sel", control.selection_payload(selection))
            with control.ControlDB(root / "control.sqlite") as db:
                run = db.init_run(selection, run_id="ttl", soft_watermark_bytes=100, hard_watermark_bytes=10)

                def downloader(**kwargs):
                    row = db.item(run, item["item_id"])
                    self.assertGreater(row["lease_expires_at"] - __import__("time").time(), 23 * 60 * 60)
                    path = Path(kwargs["local_dir"]) / item["repo_path"]
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(payload)
                    return path

                usage = lambda path: SimpleNamespace(total=10_000, free=9_000)
                archive = control.fetch_one(
                    db=db, run_id=run, selection=selection, item_id=item["item_id"],
                    spool_root=root / "spool", owner="worker", authorized_paths=[item["repo_path"]],
                    downloader=downloader, disk_usage=usage, lock_root=root / "locks",
                )
                self.assertTrue(archive.with_name(archive.name + ".transfer-receipt-v1.json").is_file())
                self.assertEqual(list(archive.parent.glob(".*.tmp")), [])
            with self.assertRaisesRegex(control.StateError, "between"):
                control._validated_lease_ttl(59)
            with self.assertRaisesRegex(control.StateError, "between"):
                control._validated_lease_ttl(8 * 24 * 60 * 60)

    def test_global_operation_locks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with control.operation_lock(root, "transfer", "one"):
                with self.assertRaisesRegex(control.StateError, "already held|conflict"):
                    with control.operation_lock(root, "transfer", "two"):
                        pass
                # Training and transfer may overlap.
                with control.operation_lock(root, "training", "trainer"):
                    self.assertTrue((root / "transfer.lock").exists())
                    with self.assertRaisesRegex(control.StateError, "conflict"):
                        control.assert_operation_allowed(root, "heavy")
            with control.operation_lock(root, "heavy", "prep"):
                with self.assertRaisesRegex(control.StateError, "conflict"):
                    control.assert_operation_allowed(root, "training")

    def test_archive_deletion_planner_is_dry_run_and_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selection = copy.deepcopy(control.load_selection())
            payload = b"archive"
            item = selection["items"][0]
            item["size"] = len(payload)
            item["lfs_sha256"] = hashlib.sha256(payload).hexdigest()
            item["item_id"] = control.stable_id("item", control.item_payload(item))
            selection["selection_id"] = control.stable_id("sel", control.selection_payload(selection))
            spool = root / "spool"
            archive = spool / "files" / item["repo_path"]
            archive.parent.mkdir(parents=True)
            archive.write_bytes(payload)
            with control.ControlDB(root / "control.sqlite") as db:
                run = db.init_run(selection, run_id="delete")
                db.connection.execute("UPDATE items SET state='ARCHIVE_DELETE_ELIGIBLE' WHERE run_id=? AND item_id=?", (run, item["item_id"]))
                db.connection.execute(
                    "INSERT INTO artifacts(run_id,item_id,kind,uri,bytes,sha256,etag,revision,committed,created_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (run, item["item_id"], "archive", str(archive.resolve()), len(payload), item["lfs_sha256"], None,
                     selection["revision"], 1, 1.0),
                )
                plan = control.plan_archive_deletions(db, run, spool_root=spool)
                self.assertEqual(plan[0]["path"], str(archive.resolve()))
                self.assertTrue(plan[0]["dry_run"])
                self.assertTrue(archive.exists(), "planner must never delete")


if __name__ == "__main__":
    unittest.main()
