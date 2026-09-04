from __future__ import annotations

import json
import sys
import types
import unittest
from importlib.machinery import ModuleSpec
from pathlib import Path
from tempfile import TemporaryDirectory

import torch


# trainer imports torchaudio/wandb at module import time; these tests exercise checkpoint
# semantics only. Prefer the real torchaudio and, if unavailable, remove the temporary
# stub immediately after importing Trainer so later test modules are not polluted.
_torchaudio_stubbed = False
try:
    import torchaudio  # noqa: F401
except (ImportError, OSError):
    sys.modules["torchaudio"] = types.ModuleType("torchaudio")
    _torchaudio_stubbed = True
if "wandb" not in sys.modules:
    wandb = types.ModuleType("wandb")
    wandb.__spec__ = ModuleSpec("wandb", loader=None)
    wandb.api = types.SimpleNamespace(api_key=None)
    sys.modules["wandb"] = wandb

from f5_tts.model.checkpoint_init import EMBEDDING_KEY, remap_ema_state_by_token
from f5_tts.model.trainer import Trainer
from f5_tts.model.vocab_contract import token_sequence_sha256, validate_vocabulary_contract

if _torchaudio_stubbed:
    sys.modules.pop("torchaudio", None)


class _Accelerator:
    @staticmethod
    def unwrap_model(model):
        return model

    @staticmethod
    def wait_for_everyone():
        return None


class _Ema:
    def __init__(self):
        self.loaded = None

    def load_state_dict(self, state_dict):
        self.loaded = state_dict


class _Stateful:
    def __init__(self):
        self.loaded = None

    def load_state_dict(self, state_dict):
        self.loaded = state_dict


class TrainerInitializationTests(unittest.TestCase):
    def make_trainer(
        self,
        checkpoint_path,
        pretrained_init,
        dataset_identity=None,
        run_identity=None,
        require_run_identity=False,
    ):
        trainer = Trainer.__new__(Trainer)
        trainer.accelerator = _Accelerator()
        trainer.model = torch.nn.Linear(2, 2)
        trainer.ema_model = _Ema()
        trainer.optimizer = _Stateful()
        trainer.scheduler = _Stateful()
        trainer.checkpoint_path = str(checkpoint_path)
        trainer.pretrained_init = str(pretrained_init)
        trainer.dataset_identity = dataset_identity
        trainer.run_identity = run_identity
        trainer.require_run_identity = require_run_identity
        trainer.grad_accumulation_steps = 1
        trainer.accelerator.is_main_process = True
        return trainer

    @staticmethod
    def ema_state(weight, bias, step=99):
        return {
            "initted": torch.tensor(True),
            "step": torch.tensor(step),
            "ema_model.weight": weight.clone(),
            "ema_model.bias": bias.clone(),
        }

    @staticmethod
    def resume_checkpoint(weight, bias, *, identity_marker=..., run_identity=...):
        checkpoint = {
            "model_state_dict": {"weight": weight, "bias": bias},
            "ema_model_state_dict": TrainerInitializationTests.ema_state(weight, bias, step=12),
            "optimizer_state_dict": {"optimizer": "resume"},
            "scheduler_state_dict": {"scheduler": "resume"},
            "update": 12,
        }
        if run_identity is not ...:
            checkpoint["resume_contract"] = {
                "version": 2,
                "dataset_identity": None if identity_marker is ... else identity_marker,
                "run_identity": run_identity,
            }
        elif identity_marker is not ...:
            checkpoint["resume_contract"] = {"version": 1, "dataset_identity": identity_marker}
        return checkpoint

    def test_pretrained_ema_initializes_online_and_resets_trainer_ema_step(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            pretrained = root / "jmica.pt"
            weight = torch.full((2, 2), 3.0)
            bias = torch.full((2,), 4.0)
            torch.save({"ema_model_state_dict": self.ema_state(weight, bias)}, pretrained)
            trainer = self.make_trainer(root / "run", pretrained)

            update = trainer.load_checkpoint()

            self.assertEqual(update, 0)
            torch.testing.assert_close(trainer.model.weight, weight)
            torch.testing.assert_close(trainer.model.bias, bias)
            self.assertEqual(trainer.ema_model.loaded["step"].item(), 0)
            torch.testing.assert_close(trainer.ema_model.loaded["ema_model.weight"], weight)
            self.assertIsNone(trainer.optimizer.loaded)
            self.assertIsNone(trainer.scheduler.loaded)

    def test_resume_checkpoint_wins_over_pretrained_init(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "run"
            run.mkdir()
            pretrained = root / "jmica.pt"
            torch.save(
                {"ema_model_state_dict": self.ema_state(torch.full((2, 2), 3.0), torch.full((2,), 4.0))},
                pretrained,
            )
            resume_weight = torch.full((2, 2), 7.0)
            resume_bias = torch.full((2,), 8.0)
            torch.save(
                self.resume_checkpoint(resume_weight, resume_bias),
                run / "model_last.pt",
            )
            trainer = self.make_trainer(run, pretrained)

            update = trainer.load_checkpoint()

            self.assertEqual(update, 12)
            torch.testing.assert_close(trainer.model.weight, resume_weight)
            self.assertEqual(trainer.ema_model.loaded["step"].item(), 12)
            self.assertEqual(trainer.optimizer.loaded, {"optimizer": "resume"})
            self.assertEqual(trainer.scheduler.loaded, {"scheduler": "resume"})

    def test_resume_with_matching_dataset_identity_succeeds(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "run"
            run.mkdir()
            pretrained = root / "jmica.pt"
            identity = {"contract_version": 1, "semantic_fingerprint": "a" * 64}
            weight = torch.full((2, 2), 7.0)
            bias = torch.full((2,), 8.0)
            torch.save(self.resume_checkpoint(weight, bias, identity_marker=identity), run / "model_last.pt")
            trainer = self.make_trainer(run, pretrained, dataset_identity=identity)

            self.assertEqual(trainer.load_checkpoint(), 12)
            self.assertEqual(trainer.optimizer.loaded, {"optimizer": "resume"})

    def test_resume_with_different_dataset_identity_fails_before_state_load(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "run"
            run.mkdir()
            pretrained = root / "jmica.pt"
            saved = {"contract_version": 1, "semantic_fingerprint": "a" * 64}
            current = {"contract_version": 1, "semantic_fingerprint": "b" * 64}
            weight = torch.full((2, 2), 7.0)
            bias = torch.full((2,), 8.0)
            torch.save(self.resume_checkpoint(weight, bias, identity_marker=saved), run / "model_last.pt")
            trainer = self.make_trainer(run, pretrained, dataset_identity=current)
            original_weight = trainer.model.weight.detach().clone()

            with self.assertRaisesRegex(RuntimeError, "dataset identity differs"):
                trainer.load_checkpoint()
            torch.testing.assert_close(trainer.model.weight, original_weight)
            self.assertIsNone(trainer.ema_model.loaded)
            self.assertIsNone(trainer.optimizer.loaded)
            self.assertIsNone(trainer.scheduler.loaded)

    def test_identity_required_run_rejects_legacy_checkpoint(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "run"
            run.mkdir()
            pretrained = root / "jmica.pt"
            identity = {"contract_version": 1, "semantic_fingerprint": "a" * 64}
            weight = torch.full((2, 2), 7.0)
            bias = torch.full((2,), 8.0)
            torch.save(self.resume_checkpoint(weight, bias), run / "model_last.pt")
            trainer = self.make_trainer(run, pretrained, dataset_identity=identity)

            with self.assertRaisesRegex(RuntimeError, "no resume_contract"):
                trainer.load_checkpoint()
            self.assertIsNone(trainer.optimizer.loaded)
    def test_required_run_identity_rejects_version1_before_state_load(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "run"
            run.mkdir()
            pretrained = root / "jmica.pt"
            identity = {"contract_version": 1, "semantic_fingerprint": "a" * 64}
            weight = torch.full((2, 2), 7.0)
            bias = torch.full((2,), 8.0)
            torch.save(self.resume_checkpoint(weight, bias, identity_marker=identity), run / "model_last.pt")
            trainer = self.make_trainer(
                run,
                pretrained,
                dataset_identity=identity,
                run_identity="r" * 64,
                require_run_identity=True,
            )
            original_weight = trainer.model.weight.detach().clone()

            with self.assertRaisesRegex(RuntimeError, "version 1 resume_contract has no run identity"):
                trainer.load_checkpoint()
            torch.testing.assert_close(trainer.model.weight, original_weight)
            self.assertIsNone(trainer.ema_model.loaded)
            self.assertIsNone(trainer.optimizer.loaded)
            self.assertIsNone(trainer.scheduler.loaded)

    def test_matching_version2_run_identity_resumes(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "run"
            run.mkdir()
            pretrained = root / "jmica.pt"
            identity = {"contract_version": 1, "semantic_fingerprint": "a" * 64}
            run_identity = "r" * 64
            weight = torch.full((2, 2), 7.0)
            bias = torch.full((2,), 8.0)
            torch.save(
                self.resume_checkpoint(
                    weight,
                    bias,
                    identity_marker=identity,
                    run_identity=run_identity,
                ),
                run / "model_last.pt",
            )
            trainer = self.make_trainer(
                run,
                pretrained,
                dataset_identity=identity,
                run_identity=run_identity,
                require_run_identity=True,
            )

            self.assertEqual(trainer.load_checkpoint(), 12)
            self.assertEqual(trainer.optimizer.loaded, {"optimizer": "resume"})

    def test_run_identity_mismatch_fails_before_any_state_load(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "run"
            run.mkdir()
            pretrained = root / "jmica.pt"
            identity = {"contract_version": 1, "semantic_fingerprint": "a" * 64}
            weight = torch.full((2, 2), 7.0)
            bias = torch.full((2,), 8.0)
            torch.save(
                self.resume_checkpoint(weight, bias, identity_marker=identity, run_identity="a" * 64),
                run / "model_last.pt",
            )
            trainer = self.make_trainer(
                run,
                pretrained,
                dataset_identity=identity,
                run_identity="b" * 64,
                require_run_identity=True,
            )
            original_weight = trainer.model.weight.detach().clone()

            with self.assertRaisesRegex(RuntimeError, "run identity differs"):
                trainer.load_checkpoint()
            torch.testing.assert_close(trainer.model.weight, original_weight)
            self.assertIsNone(trainer.ema_model.loaded)
            self.assertIsNone(trainer.optimizer.loaded)
            self.assertIsNone(trainer.scheduler.loaded)


class VocabularyAndRemapTests(unittest.TestCase):
    def write_contract(self, vocab, contract):
        tokens = vocab.read_text(encoding="utf-8").splitlines()
        contract.write_text(
            json.dumps(
                {
                    "identity": "test-vocab-v1",
                    "provenance": "unit test",
                    "token_sequence_sha256": token_sequence_sha256(tokens),
                    "token_count": len(tokens),
                    "embedding_rows": len(tokens) + 1,
                    "token_row_offset": 1,
                    "padding_row": 0,
                }
            ),
            encoding="utf-8",
        )

    def test_contract_detects_modified_vocabulary(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            vocab = root / "vocab.txt"
            contract = root / "contract.json"
            vocab.write_text(" \na\nb\n", encoding="utf-8")
            self.write_contract(vocab, contract)
            validate_vocabulary_contract(vocab, contract)
            vocab.write_text(" \nb\na\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "token-sequence sha256 mismatch"):
                validate_vocabulary_contract(vocab, contract)

    def test_canonical_hash_and_validation_are_line_ending_independent(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            lf_vocab = root / "lf.txt"
            crlf_vocab = root / "crlf.txt"
            contract = root / "contract.json"
            lf_vocab.write_bytes(b" \na\nb\n")
            crlf_vocab.write_bytes(b" \r\na\r\nb\r\n")
            self.assertEqual(
                token_sequence_sha256(lf_vocab.read_text().splitlines()),
                token_sequence_sha256(crlf_vocab.read_text().splitlines()),
            )
            self.write_contract(lf_vocab, contract)
            validate_vocabulary_contract(crlf_vocab, contract)

    def test_contract_rejects_bom_empty_token_and_non_space_index0(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            contract = root / "contract.json"
            for name, content, message in [
                ("bom.txt", b"\xef\xbb\xbf \na\n", "must not contain a UTF-8 BOM"),
                ("empty.txt", b" \n\na\n", "empty token"),
                ("index0.txt", b"x\na\n", "index 0 must be one ASCII space"),
            ]:
                vocab = root / name
                vocab.write_bytes(content)
                contract.write_text("{}", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    from f5_tts.model.vocab_contract import read_vocabulary

                    read_vocabulary(vocab)

    def test_token_remap_copies_row0_and_shared_tokens_deterministically(self):
        source_embedding = torch.tensor([[99.0, 98.0], [1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        source = {
            "step": torch.tensor(10),
            EMBEDDING_KEY: source_embedding,
            "ema_model.transformer.proj.weight": torch.arange(4, dtype=torch.float32).reshape(2, 2),
        }
        target = {
            "initted": torch.tensor(False),
            "step": torch.tensor(0),
            EMBEDDING_KEY: torch.empty(5, 2),
            "ema_model.transformer.proj.weight": torch.empty(2, 2),
        }
        first = remap_ema_state_by_token(source, target, ["a", "b", "c"], ["c", "new", "a", "new2"], seed=7)
        second = remap_ema_state_by_token(source, target, ["a", "b", "c"], ["c", "new", "a", "new2"], seed=7)

        embedding = first[EMBEDDING_KEY]
        torch.testing.assert_close(embedding[0], source_embedding[0])
        torch.testing.assert_close(embedding[1], source_embedding[3])
        torch.testing.assert_close(embedding[3], source_embedding[1])
        torch.testing.assert_close(embedding[2], second[EMBEDDING_KEY][2])
        torch.testing.assert_close(embedding[4], second[EMBEDDING_KEY][4])
        self.assertFalse(torch.equal(embedding[2], embedding[4]))

    def test_token_remap_rejects_other_shape_mismatch(self):
        source = {
            EMBEDDING_KEY: torch.empty(4, 2),
            "ema_model.transformer.proj.weight": torch.empty(2, 2),
        }
        target = {
            EMBEDDING_KEY: torch.empty(5, 2),
            "ema_model.transformer.proj.weight": torch.empty(3, 2),
        }
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            remap_ema_state_by_token(source, target, ["a", "b", "c"], ["a", "b", "c", "d"], seed=1)


class TrainerMaxUpdatesTests(unittest.TestCase):
    def make_trainer(self, *, epochs=10, grad_accumulation_steps=2, max_updates=None):
        trainer = Trainer.__new__(Trainer)
        trainer.epochs = epochs
        trainer.grad_accumulation_steps = grad_accumulation_steps
        trainer.max_updates = max_updates
        return trainer

    def test_update_horizon_is_capped_without_off_by_one(self):
        trainer = self.make_trainer(max_updates=7)
        self.assertEqual(trainer._training_update_horizon(5), 7)
        self.assertFalse(trainer._reached_max_updates(6))
        self.assertTrue(trainer._reached_max_updates(7))

    def test_null_max_updates_uses_epoch_horizon(self):
        trainer = self.make_trainer(max_updates=None)
        self.assertEqual(trainer._training_update_horizon(5), 30)
        self.assertFalse(trainer._reached_max_updates(100))

    def test_resume_at_or_beyond_limit_is_finished(self):
        trainer = self.make_trainer(max_updates=12)
        self.assertTrue(trainer._reached_max_updates(12))
        self.assertTrue(trainer._reached_max_updates(13))


if __name__ == "__main__":
    unittest.main()
