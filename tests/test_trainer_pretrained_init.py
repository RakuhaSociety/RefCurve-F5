from __future__ import annotations

import sys
import types
import unittest
from importlib.machinery import ModuleSpec
from pathlib import Path
from tempfile import TemporaryDirectory

import torch


# trainer imports torchaudio/wandb at module import time; these tests exercise checkpoint
# semantics only and do not require either optional runtime dependency.
if "torchaudio" not in sys.modules:
    sys.modules["torchaudio"] = types.ModuleType("torchaudio")
if "wandb" not in sys.modules:
    wandb = types.ModuleType("wandb")
    wandb.__spec__ = ModuleSpec("wandb", loader=None)
    wandb.api = types.SimpleNamespace(api_key=None)
    sys.modules["wandb"] = wandb

from f5_tts.model.trainer import Trainer


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
    def make_trainer(self, checkpoint_path, pretrained_init):
        trainer = Trainer.__new__(Trainer)
        trainer.accelerator = _Accelerator()
        trainer.model = torch.nn.Linear(2, 2)
        trainer.ema_model = _Ema()
        trainer.optimizer = _Stateful()
        trainer.scheduler = _Stateful()
        trainer.checkpoint_path = str(checkpoint_path)
        trainer.pretrained_init = str(pretrained_init)
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
                {
                    "model_state_dict": {"weight": resume_weight, "bias": resume_bias},
                    "ema_model_state_dict": self.ema_state(resume_weight, resume_bias, step=12),
                    "optimizer_state_dict": {"optimizer": "resume"},
                    "scheduler_state_dict": {"scheduler": "resume"},
                    "update": 12,
                },
                run / "model_last.pt",
            )
            trainer = self.make_trainer(run, pretrained)

            update = trainer.load_checkpoint()

            self.assertEqual(update, 12)
            torch.testing.assert_close(trainer.model.weight, resume_weight)
            self.assertEqual(trainer.ema_model.loaded["step"].item(), 12)
            self.assertEqual(trainer.optimizer.loaded, {"optimizer": "resume"})
            self.assertEqual(trainer.scheduler.loaded, {"scheduler": "resume"})


if __name__ == "__main__":
    unittest.main()
