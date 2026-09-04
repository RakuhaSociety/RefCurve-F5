from __future__ import annotations

import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch

SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "eval_jmica_matrix.py"
SPEC = importlib.util.spec_from_file_location("eval_jmica_matrix", SCRIPT)
matrix = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(matrix)


def valid_manifest(*, strict: bool = False) -> dict:
    system_contract = {
        "config_path": "config.yaml",
        "vocab_path": "vocab.txt",
        "vocab_contract_path": "contract.json",
        "model_family": "legacy_jmica",
        "expected_vocab_identity": "test-vocab",
        "expected_token_sequence_sha256": "0" * 64,
        "expected_embedding_rows": 3,
        "expected_embedding_width": 4,
    }
    systems = [
        {"id": "baseline", "checkpoint": "base.pt", "checkpoint_step": 1, "weight_source": "ema", "baseline": True, "checkpoint_sha256": None},
        {"id": "candidate", "checkpoint": "candidate.pt", "checkpoint_step": 2, "weight_source": "online", "baseline": False, "checkpoint_sha256": None},
    ]
    if strict:
        for system in systems:
            system.update(system_contract)
    return {
        "schema_version": matrix.SCHEMA if strict else matrix.LEGACY_SCHEMA,
        "inference": {"vocoder_path": "vocoder", **({} if strict else {"vocab_path": "vocab.txt"})},
        "generation": {"primary_seed": 777, "device": "auto"},
        "systems": systems,
        "references": [{"id": "ref", "audio": "ref.wav", "audio_sha256": None, "text_original": "参照", "text_kana": None, "emotion_label": "neutral", "data_relation": None}],
        "texts": [
            {"id": "t1", "split": "core", "category": "narration", "text_original": "試験一", "text_kana": None, "text_kana_sha256": None, "reference_id": "ref", "leakage_status": None},
            {"id": "t2", "split": "robustness", "category": "long", "text_original": "試験二", "text_kana": None, "text_kana_sha256": None, "reference_id": "ref", "leakage_status": None},
        ],
        "gates": {
            "hard": {"total_clips_required": 4, "format": {"sample_rate": 24000, "channels": 1}, "max_clipping_fraction": 0.001, "min_duration_seconds": 0.5, "max_duration_seconds": 20.0, "allow_nan_inf": False, "max_silence_fraction": 0.6},
            "objective": {"max_core_median_cer_delta_vs_baseline": 0.03, "max_core_p90_cer_delta_vs_baseline": 0.08, "max_catastrophic_cer_per_text": 0.5, "min_speaker_similarity_delta_vs_baseline_median": -0.03, "min_speaker_similarity_delta_vs_baseline_p10": -0.05, "max_utmos_delta_vs_baseline_mean": -0.15},
        },
        "leakage_check": {"training_metadata_path": "metadata.csv", "max_kana_ngram_jaccard_flag": 0.6, "min_ngram_n": 3},
    }


def prepared_manifest() -> dict:
    manifest = valid_manifest()
    manifest["references"][0].update(audio_sha256="abc", text_kana="サンショウ", data_relation="held-out")
    for text, kana in zip(manifest["texts"], ("シケンイチ", "シケンニ")):
        text.update(text_kana=kana, text_kana_sha256=matrix.canonical_hash(kana), leakage_status="clean")
    manifest["cases"] = [
        {"id": f"{system['id']}__{text['id']}", "system_id": system["id"], "text_id": text["id"], "reference_id": "ref", "seed": 777, "wav": f"audio/{system['id']}/{text['id']}.wav"}
        for system in manifest["systems"] for text in manifest["texts"]
    ]
    return manifest


class ManifestTests(unittest.TestCase):
    def test_text_ids_references_and_case_count_are_validated(self):
        manifest = prepared_manifest()
        self.assertIs(matrix.validate_manifest(manifest, prepared=True), manifest)
        self.assertEqual(len(matrix.jobs(manifest)), 4)
        manifest["texts"][1]["id"] = "t1"
        with self.assertRaisesRegex(matrix.ManifestError, "unique"):
            matrix.validate_manifest(manifest)

    def test_prepared_manifest_requires_frozen_reference(self):
        manifest = prepared_manifest()
        manifest["references"][0]["data_relation"] = None
        with self.assertRaisesRegex(matrix.ManifestError, "not frozen"):
            matrix.validate_manifest(manifest, prepared=True)

    def test_leakage_jaccard_and_blinding_are_deterministic(self):
        self.assertEqual(matrix.ngram_jaccard("アイウエ", "アイウオ", 3), 1 / 3)
        self.assertEqual(matrix.blind_code("secret", "job"), matrix.blind_code("secret", "job"))
        self.assertNotEqual(matrix.blind_code("secret", "job"), matrix.blind_code("other", "job"))

    def test_generation_parser_accepts_explicit_system_subset(self):
        args = matrix.parser().parse_args(["generate", "manifest.json", "new-run", "--systems", "a,b"])
        self.assertEqual(args.systems, "a,b")

        with TemporaryDirectory() as tmp:
            log = Path(tmp) / "provenance.jsonl"
            matrix.append_jsonl(log, {"stage": "generate", "status": "ok", "job_id": "a"})
            matrix.append_jsonl(log, {"stage": "generate", "status": "error", "job_id": "b"})
            self.assertEqual(matrix.completed_ids(log, "generate"), {"a"})

    def test_hard_audio_gates(self):
        spec = valid_manifest()["gates"]["hard"]
        spec.update(
            max_prewrite_overshoot_fraction=0.001,
            min_normalization_gain=0.5,
            max_stored_clipping_fraction=0.001,
            allow_prewrite_nan_inf=False,
        )
        diag = {"sample_rate": 24000, "channels": 1, "clipping_fraction": 0.0, "duration_seconds": 1.0, "silence_fraction": 0.1, "has_nan_inf": False, "prewrite_overshoot_fraction": 0.0, "normalization_gain": 1.0, "stored_clipping_fraction": 0.0, "prewrite_has_nan_inf": False}
        self.assertEqual(matrix._hard_gate(diag, spec), [])
        diag["prewrite_has_nan_inf"] = True
        diag["normalization_gain"] = 0.4
        self.assertEqual(matrix._hard_gate(diag, spec), ["normalization_gain", "prewrite_nan_inf"])

    def test_new_schema_requires_explicit_system_contract_and_forbids_global_vocab(self):
        manifest = valid_manifest(strict=True)
        self.assertIs(matrix.validate_manifest(manifest), manifest)
        manifest["inference"]["vocab_path"] = "global.txt"
        with self.assertRaisesRegex(matrix.ManifestError, "forbidden"):
            matrix.validate_manifest(manifest)
        manifest["inference"].pop("vocab_path")
        manifest["systems"][0].pop("model_family")
        with self.assertRaisesRegex(matrix.ManifestError, "compatibility fields"):
            matrix.validate_manifest(manifest)

    def test_objective_verdict_is_provisional_when_required_metric_unavailable(self):
        summary = {"core_cer": {"median": 0.1, "p90": 0.2}, "cer": {"max": 0.3}}
        verdict, checks = matrix._objective_verdict(summary, summary, valid_manifest()["gates"]["objective"])
        self.assertIsNone(verdict)
        self.assertTrue(any(check["passed"] is None for check in checks))

    def test_score_reuses_identical_audio_by_sha(self):
        import numpy as np
        import soundfile as sf

        manifest = prepared_manifest()
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec_path = root / "prepared.json"
            spec_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
            reference = root / "ref.wav"
            sf.write(reference, np.zeros(24000, dtype=np.float32), 24000)
            manifest["references"][0]["audio"] = str(reference)
            spec_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
            for job in matrix.jobs(manifest):
                wav = matrix._output_wav(root, job)
                wav.parent.mkdir(parents=True, exist_ok=True)
                sf.write(wav, np.zeros(24000, dtype=np.float32), 24000)
            calls = []
            protocol = types.ModuleType("tools.eval_protocol")
            protocol.score_audio = lambda *args, **kwargs: calls.append(args[0]) or {"sample_rate": 24000, "channels": 1, "clipping_fraction": 0.0, "duration_seconds": 1.0, "silence_fraction": 0.1, "has_nan_inf": False}
            with patch.dict(sys.modules, {"tools.eval_protocol": protocol}):
                args = types.SimpleNamespace(manifest=str(spec_path), run_dir=str(root), metrics="", keep_going=False)
                self.assertEqual(matrix.cmd_score(args), 0)
            self.assertEqual(len(calls), 2)  # one score per distinct reference text, not per system
            rows = [row for row in matrix.read_jsonl(root / "provenance.jsonl") if row["stage"] == "score"]
            self.assertEqual(len({row["wav_sha256"] for row in rows}), 1)

        manifest = prepared_manifest()
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec_path = root / "prepared.json"
            spec_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
            log = root / "provenance.jsonl"
            values = {"baseline": (0.10, 0.90), "candidate": (0.11, 0.92)}
            for job in matrix.jobs(manifest):
                cer, speaker = values[job["system_id"]]
                matrix.append_jsonl(log, {"stage": "score", "status": "ok", "job_id": job["job_id"], "system_id": job["system_id"], "text_id": job["text_id"], "split": job["text"]["split"], "cer": cer, "speaker_sim": speaker, "hard_gate_failures": []})
            args = types.SimpleNamespace(manifest=str(spec_path), run_dir=str(root), limit=3, quiet=True)
            self.assertEqual(matrix.cmd_select(args), 0)
            selection = json.loads((root / "selection.json").read_text(encoding="utf-8"))
            candidate = next(row for row in selection["ranking"] if row["system_id"] == "candidate")
            self.assertEqual(candidate["result_status"], "incomplete/provisional")
            self.assertFalse(candidate["passed"])


class CheckpointCompatibilityTests(unittest.TestCase):
    def test_requested_state_and_embedding_metadata_are_strict(self):
        from f5_tts.model.checkpoint_init import validate_checkpoint_architecture

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.pt"
            expected = {
                "transformer.text_embed.text_embed.weight": torch.empty(3, 4),
                "transformer.block.weight": torch.empty(2, 2),
            }
            torch.save(
                {
                    "ema_model_state_dict": {
                        "ema_model.transformer.text_embed.text_embed.weight": torch.ones(3, 4),
                        "ema_model.transformer.block.weight": torch.ones(2, 2),
                    }
                },
                path,
            )
            details = validate_checkpoint_architecture(
                path, "ema", expected, expected_embedding_rows=3, expected_embedding_width=4
            )
            self.assertEqual(details["embedding_rows"], 3)
            with self.assertRaisesRegex(KeyError, "model_state_dict"):
                validate_checkpoint_architecture(
                    path, "online", expected, expected_embedding_rows=3, expected_embedding_width=4
                )
            with self.assertRaisesRegex(ValueError, "embedding metadata"):
                validate_checkpoint_architecture(
                    path, "ema", expected, expected_embedding_rows=4, expected_embedding_width=4
                )

    def test_dtype_aware_loaded_state_verification_covers_buffers_and_detects_changes(self):
        from f5_tts.model.checkpoint_init import (
            checkpoint_state_cast_to_model,
            state_tensor_sha256,
            verify_checkpoint_state_loaded,
        )

        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(2, dtype=torch.float16))
                self.register_buffer("scale", torch.zeros(1, dtype=torch.float16))

        checkpoint = {
            "weight": torch.tensor([1.25, -2.5], dtype=torch.float32),
            "scale": torch.tensor([0.75], dtype=torch.float32),
        }
        model = Tiny()
        model.load_state_dict(checkpoint)
        digest = verify_checkpoint_state_loaded(checkpoint, model)
        casted = checkpoint_state_cast_to_model(checkpoint, model.state_dict())
        self.assertEqual(digest, state_tensor_sha256(casted))
        altered = dict(checkpoint)
        altered["weight"] = checkpoint["weight"] + 1.0
        with self.assertRaisesRegex(RuntimeError, "digest mismatch"):
            verify_checkpoint_state_loaded(altered, model)
        with self.assertRaisesRegex(KeyError, "state-key mismatch"):
            verify_checkpoint_state_loaded({"weight": checkpoint["weight"]}, model)

    def test_distinct_online_checkpoint_states_load_distinct_model_parameters(self):
        from f5_tts.infer.utils_infer import load_checkpoint
        from f5_tts.model.checkpoint_init import checkpoint_state_for_source, model_parameter_sha256, state_tensor_sha256, verify_checkpoint_state_loaded

        with TemporaryDirectory() as tmp:
            first = Path(tmp) / "first.pt"
            second = Path(tmp) / "second.pt"
            torch.save({"model_state_dict": {"weight": torch.tensor([[1.0, 2.0]])}}, first)
            torch.save({"model_state_dict": {"weight": torch.tensor([[3.0, 4.0]])}}, second)
            model_a = torch.nn.Linear(2, 1, bias=False)
            model_b = torch.nn.Linear(2, 1, bias=False)
            load_checkpoint(model_a, str(first), "cpu", dtype=torch.float32, use_ema=False)
            load_checkpoint(model_b, str(second), "cpu", dtype=torch.float32, use_ema=False)
            checkpoint_a = checkpoint_state_for_source(first, "online")
            checkpoint_b = checkpoint_state_for_source(second, "online")
            digest_a = state_tensor_sha256(checkpoint_a)
            digest_b = state_tensor_sha256(checkpoint_b)
            self.assertNotEqual(digest_a, digest_b)
            self.assertEqual(verify_checkpoint_state_loaded(checkpoint_a, model_a), model_parameter_sha256(model_a))
            self.assertEqual(verify_checkpoint_state_loaded(checkpoint_b, model_b), model_parameter_sha256(model_b))
            self.assertNotEqual(model_parameter_sha256(model_a), model_parameter_sha256(model_b))

    def test_family_backbone_and_mel_contract_are_enforced(self):
        root = SCRIPT.parents[1]
        system = {
            "config_path": str(root / "src/f5_tts/configs/F5TTS_v1_JA_Base.yaml"),
            "vocab_path": str(root / "src/f5_tts/configs/vocab/F5TTS_v1_JA_Base/vocab.txt"),
            "model_family": "legacy_jmica",
        }
        with self.assertRaisesRegex(ValueError, "declared family"):
            matrix._expected_architecture_state(system)

        from f5_tts.model.checkpoint_init import validate_checkpoint_architecture

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.pt"
            torch.save(
                {
                    "model_state_dict": {
                        "transformer.text_embed.text_embed.weight": torch.ones(3, 4),
                        "unexpected": torch.ones(1),
                    }
                },
                path,
            )
            expected = {"transformer.text_embed.text_embed.weight": torch.empty(3, 4)}
            with self.assertRaisesRegex(KeyError, "state-key mismatch"):
                validate_checkpoint_architecture(
                    path, "online", expected, expected_embedding_rows=3, expected_embedding_width=4
                )


class BaselineMaterializationTests(unittest.TestCase):
    def test_materializer_freezes_hashes_and_is_deterministic(self):
        script = SCRIPT.parents[1] / "tools" / "materialize_v1_ja_eval_baseline.py"
        spec = importlib.util.spec_from_file_location("materialize_v1_ja_eval_baseline", script)
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        root = SCRIPT.parents[1]
        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "baseline.safetensors"
            provenance = Path(tmp) / "provenance.json"
            args = types.SimpleNamespace(
                source_checkpoint=str(root / "ckpts/F5TTS_v1_Base/model_1250000.safetensors"),
                source_vocab=str(root / "src/f5_tts/configs/vocab/F5TTS_v1_Base/vocab.txt"),
                source_contract=str(root / "src/f5_tts/configs/vocab/F5TTS_v1_Base/contract.json"),
                target_config=str(root / "src/f5_tts/configs/F5TTS_v1_JA_Base.yaml"),
                target_vocab=str(root / "src/f5_tts/configs/vocab/F5TTS_v1_JA_Base/vocab.txt"),
                target_contract=str(root / "src/f5_tts/configs/vocab/F5TTS_v1_JA_Base/contract.json"),
                output=str(output),
                provenance=str(provenance),
                seed=666,
            )
            self.assertEqual(helper.materialize(args), 0)
            first = helper.file_digest(output)
            self.assertEqual(helper.materialize(args), 0)
            self.assertEqual(helper.file_digest(output), first)
            payload = json.loads(provenance.read_text(encoding="utf-8"))
            self.assertEqual(payload["artifact"]["checkpoint_sha256"], first)
            self.assertEqual(payload["source"]["contract_identity"], "f5tts-v1-base-official")
            self.assertEqual(payload["target"]["contract_identity"], "refcurve-f5tts-v1-ja-base-v1")
            self.assertEqual(payload["seed"], 666)
            self.assertEqual(payload["environment"]["torch"], torch.__version__)
            self.assertIn("python", payload["environment"])
            self.assertIn("platform", payload["environment"])


class CalibrationSpecTests(unittest.TestCase):
    def test_calibration_spec_is_ema_only_and_reuses_frozen_matrix_content(self):
        root = SCRIPT.parents[1]
        calibration = json.loads((root / "eval_specs" / "calibration_checkpoint_eval_v1.json").read_text(encoding="utf-8-sig"))
        historical = json.loads((root / "eval_specs" / "jmica_checkpoint_eval_v1.json").read_text(encoding="utf-8"))
        self.assertEqual(calibration["schema_version"], matrix.SCHEMA)
        self.assertNotIn("vocab_path", calibration["inference"])
        self.assertEqual(len(calibration["systems"]), 12)
        self.assertEqual(calibration["systems"][0]["id"], "v1_ja_update0_ema")
        self.assertEqual(calibration["systems"][0]["expected_embedding_rows"], 2977)
        self.assertNotIn("official_v1_base_ema", {system["id"] for system in calibration["systems"]})
        self.assertTrue(
            all(
                "ckpts/visualnovel_calibration_ja/abc-v1-calibration-5000-20260829/" in system["checkpoint"]
                for system in calibration["systems"][2:]
            )
        )
        self.assertTrue(all(system["weight_source"] == "ema" for system in calibration["systems"]))
        self.assertEqual(calibration["texts"], historical["texts"])
        self.assertEqual(calibration["references"], historical["references"])
        self.assertEqual(calibration["gates"]["hard"]["total_clips_required"], 192)
        matrix.validate_manifest(calibration)


class JaModelCacheTests(unittest.TestCase):
    def test_cache_separates_weights_and_can_be_cleared(self):
        original = {name: sys.modules.get(name) for name in ("torch", "omegaconf", "f5_tts.infer.utils_infer", "f5_tts.model")}
        torch = types.ModuleType("torch")
        torch.cuda = types.SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None)
        omega = types.ModuleType("omegaconf")
        mel = types.SimpleNamespace(target_sample_rate=24000, n_mel_channels=100, hop_length=256, win_length=1024, n_fft=1024, mel_spec_type="vocos")
        omega.OmegaConf = types.SimpleNamespace(load=lambda path: types.SimpleNamespace(model=types.SimpleNamespace(arch={}, mel_spec=mel)))
        utils = types.ModuleType("f5_tts.infer.utils_infer")
        calls = []
        utils.load_vocoder = lambda **kwargs: object()
        utils.load_model = lambda **kwargs: calls.append(kwargs) or object()
        model = types.ModuleType("f5_tts.model"); model.DiT = object
        sys.modules.update({"torch": torch, "omegaconf": omega, "f5_tts.infer.utils_infer": utils, "f5_tts.model": model})
        try:
            path = SCRIPT.parents[1] / "src" / "f5_tts" / "infer" / "ja_model.py"
            spec = importlib.util.spec_from_file_location("ja_model_isolated", path)
            module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
            with TemporaryDirectory() as tmp, patch.object(Path, "exists", autospec=True, return_value=True):
                root = Path(tmp); ckpt = root / "model.pt"; vocab = root / "vocab"; vocoder = root / "vocoder"
                config_a = root / "legacy.yaml"; config_a.touch()
                config_b = root / "v1.yaml"; config_b.touch()
                contract = root / "contract.json"; contract.touch()
                module.load_ja_model(str(ckpt), str(vocab), str(vocoder), use_ema=True, device="cpu", config_path=str(config_a), vocab_contract_path=str(contract), model_family="legacy")
                module.load_ja_model(str(ckpt), str(vocab), str(vocoder), use_ema=False, device="cpu", config_path=str(config_a), vocab_contract_path=str(contract), model_family="legacy")
                module.load_ja_model(str(ckpt), str(vocab), str(vocoder), use_ema=True, device="cpu", config_path=str(config_b), vocab_contract_path=str(contract), model_family="v1")
                self.assertEqual(len(calls), 3)
                module.clear_ja_model_cache()
                module.load_ja_model(str(ckpt), str(vocab), str(vocoder), use_ema=True, device="cpu", config_path=str(config_a), vocab_contract_path=str(contract), model_family="legacy")
                self.assertEqual(len(calls), 4)
        finally:
            for name, value in original.items():
                if value is None: sys.modules.pop(name, None)
                else: sys.modules[name] = value


if __name__ == "__main__":
    unittest.main()
