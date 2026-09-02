from __future__ import annotations

import gc
import json
import math
import os
import re

import torch
import torchaudio
import wandb
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from ema_pytorch import EMA
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from tqdm import tqdm

from f5_tts.model import CFM
from f5_tts.model.checkpoint_init import build_pretrained_ema_init, online_state_from_ema
from f5_tts.model.dataset import DynamicBatchSampler, collate_fn
from f5_tts.model.utils import default, exists


# trainer


# ✅ scheduler 单位契约（world-size 不变性）
#
# Accelerate 的 AcceleratedScheduler 在 split_batches=False 时，每个**成功的**
# global optimizer update 会把 inner scheduler 推进 num_processes 次
# （accelerate/scheduler.py: step() 里 for _ in range(num_processes)）。
#
# 因此 warmup 与 total horizon 必须**同时**换算到 inner scheduler units。
# 历史 bug：warmup 乘了 world size、total 没乘，8 卡下 LR 在约 total/8 个
# global update 就跌到 1e-8 floor（5000 步的 run 实际只有约 625 步在学习）。
def resolve_scheduler_contract(
    *,
    global_warmup_updates: int,
    global_total_updates: int,
    num_processes: int,
    split_batches: bool,
) -> dict:
    """把 global optimizer update 语义换算成 Accelerate inner scheduler units。

    返回的 warmup/total 均为 inner units；global_* 字段保留原始语义用于取证。
    """
    if num_processes < 1:
        raise ValueError(f"num_processes must be >= 1, got {num_processes}")
    if global_warmup_updates < 0:
        raise ValueError(f"global_warmup_updates must be >= 0, got {global_warmup_updates}")
    if global_total_updates < 0:
        raise ValueError("Training update horizon cannot be negative.")

    multiplier = 1 if split_batches else num_processes

    # clamp 必须在 global 单位下完成，再统一乘 multiplier，否则两边单位不一致。
    clamped_global_warmup = min(global_warmup_updates, global_total_updates)

    return {
        "num_processes": int(num_processes),
        "split_batches": bool(split_batches),
        "scheduler_step_multiplier": int(multiplier),
        "global_warmup_updates": int(clamped_global_warmup),
        "global_warmup_updates_requested": int(global_warmup_updates),
        "global_total_updates": int(global_total_updates),
        "warmup_updates": int(clamped_global_warmup * multiplier),
        "total_updates": int(global_total_updates * multiplier),
    }


# ✅ split_batches 直接决定 scheduler step multiplier——正是历史 bug 的所在量。
# 静默默认成 False 在当前配置下"碰巧"正确，却会把同一类单位错误埋回去：
# 一旦 accelerate 改了属性位置，我们会再次得到一条错的 LR 曲线且毫无提示。
# 因此找不到就抛错，不猜。
def resolve_split_batches(accelerator) -> bool:
    dataloader_config = getattr(accelerator, "dataloader_config", None)
    if dataloader_config is not None and hasattr(dataloader_config, "split_batches"):
        return bool(dataloader_config.split_batches)
    if hasattr(accelerator, "split_batches"):
        return bool(accelerator.split_batches)
    raise AttributeError(
        "Cannot determine Accelerate split_batches; the scheduler step multiplier would be a guess. "
        "Checked accelerator.dataloader_config.split_batches and accelerator.split_batches."
    )


LR_FLOOR_FACTOR = 1e-8


# ✅ 抽成函数是为了让 scheduler smoke 验证**生产代码本身**，而不是一份复制品。
# 复制品会漂移，而"两份本该一致的东西悄悄不一致"正是本次事故的形状。
# 参数单位是 Accelerate inner scheduler units，不是 global optimizer updates。
def build_scheduler(optimizer, *, warmup_updates: int, total_updates: int):
    if total_updates == 0:
        return LinearLR(optimizer, start_factor=1.0, end_factor=1.0, total_iters=1)
    if warmup_updates == 0:
        return LinearLR(optimizer, start_factor=1.0, end_factor=LR_FLOOR_FACTOR, total_iters=total_updates)
    if warmup_updates >= total_updates:
        return LinearLR(optimizer, start_factor=LR_FLOOR_FACTOR, end_factor=1.0, total_iters=warmup_updates)
    warmup_scheduler = LinearLR(optimizer, start_factor=LR_FLOOR_FACTOR, end_factor=1.0, total_iters=warmup_updates)
    decay_scheduler = LinearLR(
        optimizer, start_factor=1.0, end_factor=LR_FLOOR_FACTOR, total_iters=total_updates - warmup_updates
    )
    return SequentialLR(optimizer, schedulers=[warmup_scheduler, decay_scheduler], milestones=[warmup_updates])


class Trainer:
    def __init__(
        self,
        model: CFM,
        epochs,
        learning_rate,
        num_warmup_updates=20000,
        max_updates: int | None = None,
        stop_after_updates: int | None = None,  # ✅ smoke 用：限制实际步数但不改 scheduler horizon
        lr_trace_path: str | None = None,  # ✅ rank 0 逐 update 记录 LR，默认关闭
        save_per_updates=1000,
        keep_last_n_checkpoints: int = -1,  # -1 to keep all, 0 to not save intermediate, > 0 to keep last N checkpoints
        checkpoint_path=None,
        pretrained_init=None,
        pretrained_init_options=None,
        batch_size_per_gpu=32,
        batch_size_type: str = "sample",
        max_samples=32,
        grad_accumulation_steps=1,
        max_grad_norm=1.0,
        noise_scheduler: str | None = None,
        duration_predictor: torch.nn.Module | None = None,
        logger: str | None = "wandb",  # "wandb" | "tensorboard" | None
        wandb_project="test_f5-tts",
        wandb_run_name="test_run",
        wandb_resume_id: str = None,
        log_samples: bool = False,
        last_per_updates=None,
        accelerate_kwargs: dict = dict(),
        ema_kwargs: dict = dict(),
        bnb_optimizer: bool = False,
        mel_spec_type: str = "vocos",  # "vocos" | "bigvgan"
        is_local_vocoder: bool = False,  # use local path vocoder
        local_vocoder_path: str = "",  # local vocoder path
        model_cfg_dict: dict = dict(),  # training config
        dataset_identity: dict | None = None,
        run_identity: str | None = None,
        require_run_identity: bool = False,
    ):
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

        if logger == "wandb" and not wandb.api.api_key:
            logger = None
        self.log_samples = log_samples

        self.accelerator = Accelerator(
            log_with=logger if logger == "wandb" else None,
            kwargs_handlers=[ddp_kwargs],
            gradient_accumulation_steps=grad_accumulation_steps,
            **accelerate_kwargs,
        )

        self.logger = logger
        if self.logger == "wandb":
            if exists(wandb_resume_id):
                init_kwargs = {"wandb": {"resume": "allow", "name": wandb_run_name, "id": wandb_resume_id}}
            else:
                init_kwargs = {"wandb": {"resume": "allow", "name": wandb_run_name}}

            if not model_cfg_dict:
                model_cfg_dict = {
                    "epochs": epochs,
                    "learning_rate": learning_rate,
                    "num_warmup_updates": num_warmup_updates,
                    "batch_size_per_gpu": batch_size_per_gpu,
                    "batch_size_type": batch_size_type,
                    "max_samples": max_samples,
                    "grad_accumulation_steps": grad_accumulation_steps,
                    "max_grad_norm": max_grad_norm,
                    "noise_scheduler": noise_scheduler,
                    "bnb_optimizer": bnb_optimizer,
                }
            model_cfg_dict["gpus"] = self.accelerator.num_processes
            self.accelerator.init_trackers(
                project_name=wandb_project,
                init_kwargs=init_kwargs,
                config=model_cfg_dict,
            )

        elif self.logger == "tensorboard":
            from torch.utils.tensorboard import SummaryWriter

            self.writer = None
            if self.accelerator.is_main_process:
                self.writer = SummaryWriter(log_dir=f"runs/{wandb_run_name}")

        self.model = model

        if self.is_main:
            self.ema_model = EMA(model, include_online_model=False, **ema_kwargs)
            self.ema_model.to(self.accelerator.device)

            print(f"Using logger: {logger}")
            if grad_accumulation_steps > 1:
                print(
                    "Gradient accumulation checkpointing with per_updates now, old logic per_steps used with before f992c4e"
                )

        self.epochs = epochs
        self.num_warmup_updates = num_warmup_updates
        if exists(max_updates) and max_updates < 0:
            raise ValueError("max_updates must be non-negative or null")
        self.max_updates = max_updates
        # ✅ stop_after_updates 只提前停训练循环，不参与 _training_update_horizon()，
        # 因此 scheduler 仍按完整 horizon 规划——smoke 才能验证真实 LR 轨迹而非压缩版。
        if exists(stop_after_updates) and stop_after_updates < 1:
            raise ValueError("stop_after_updates must be >= 1 or null")
        self.stop_after_updates = stop_after_updates
        # ✅ 这次 scheduler 事故能存活 5000 步不被发现，根本原因是没有任何东西记录
        # 过 LR。默认关闭以免影响正式 run 的产物，但一旦打开就是 rank 0 的逐 update
        # 事实记录，可直接跨 world size 比对。
        self.lr_trace_path = lr_trace_path
        self.save_per_updates = save_per_updates
        self.keep_last_n_checkpoints = keep_last_n_checkpoints
        self.last_per_updates = default(last_per_updates, save_per_updates)
        self.checkpoint_path = default(checkpoint_path, "ckpts/test_f5-tts")
        self.pretrained_init = pretrained_init
        self.pretrained_init_options = default(pretrained_init_options, {})
        self.dataset_identity = dataset_identity
        self.run_identity = run_identity
        self.require_run_identity = require_run_identity
        if self.require_run_identity and not self.run_identity:
            raise ValueError("require_run_identity=True requires a run identity")

        self.batch_size_per_gpu = batch_size_per_gpu
        self.batch_size_type = batch_size_type
        self.max_samples = max_samples
        self.grad_accumulation_steps = grad_accumulation_steps
        self.max_grad_norm = max_grad_norm

        # mel vocoder config
        self.vocoder_name = mel_spec_type
        self.is_local_vocoder = is_local_vocoder
        self.local_vocoder_path = local_vocoder_path

        self.noise_scheduler = noise_scheduler

        self.duration_predictor = duration_predictor

        if bnb_optimizer:
            import bitsandbytes as bnb

            self.optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=learning_rate)
        else:
            self.optimizer = AdamW(model.parameters(), lr=learning_rate, fused=True)
        self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    def save_checkpoint(self, update, last=False):
        self.accelerator.wait_for_everyone()
        if self.is_main:
            checkpoint = dict(
                model_state_dict=self.accelerator.unwrap_model(self.model).state_dict(),
                optimizer_state_dict=self.optimizer.state_dict(),
                ema_model_state_dict=self.ema_model.state_dict(),
                scheduler_state_dict=self.scheduler.state_dict(),
                update=update,
                resume_contract=(
                    {
                        "version": 2,
                        "dataset_identity": self.dataset_identity,
                        "run_identity": self.run_identity,
                    }
                    if self.run_identity is not None
                    else {"version": 1, "dataset_identity": self.dataset_identity}
                ),
            )
            if not os.path.exists(self.checkpoint_path):
                os.makedirs(self.checkpoint_path)
            if last:
                self.accelerator.save(checkpoint, f"{self.checkpoint_path}/model_last.pt")
                print(f"Saved last checkpoint at update {update}")
            else:
                if self.keep_last_n_checkpoints == 0:
                    return
                self.accelerator.save(checkpoint, f"{self.checkpoint_path}/model_{update}.pt")
                if self.keep_last_n_checkpoints > 0:
                    # Updated logic to exclude pretrained model from rotation
                    checkpoints = [
                        f
                        for f in os.listdir(self.checkpoint_path)
                        if f.startswith("model_")
                        and not f.startswith("pretrained_")  # Exclude pretrained models
                        and f.endswith(".pt")
                        and f != "model_last.pt"
                    ]
                    checkpoints.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
                    while len(checkpoints) > self.keep_last_n_checkpoints:
                        oldest_checkpoint = checkpoints.pop(0)
                        os.remove(os.path.join(self.checkpoint_path, oldest_checkpoint))
                        print(f"Removed old checkpoint: {oldest_checkpoint}")

    @staticmethod
    def _remove_legacy_mel_keys(state_dict, prefix=""):
        for key in [
            f"{prefix}mel_spec.mel_stft.mel_scale.fb",
            f"{prefix}mel_spec.mel_stft.spectrogram.window",
        ]:
            state_dict.pop(key, None)

    @classmethod
    def _online_state_from_ema(cls, ema_state_dict):
        state_dict = {
            key.removeprefix("ema_model."): value
            for key, value in ema_state_dict.items()
            if key.startswith("ema_model.")
        }
        cls._remove_legacy_mel_keys(state_dict)
        return state_dict

    @staticmethod
    def _load_checkpoint_file(checkpoint_path):
        if checkpoint_path.endswith(".safetensors"):
            from safetensors.torch import load_file

            return {"ema_model_state_dict": load_file(checkpoint_path, device="cpu")}
        return torch.load(checkpoint_path, weights_only=True, map_location="cpu")

    def _validate_resume_contract(self, checkpoint, checkpoint_path):
        current_identity = getattr(self, "dataset_identity", None)
        current_run_identity = getattr(self, "run_identity", None)
        run_identity_required = getattr(self, "require_run_identity", False)
        contract = checkpoint.get("resume_contract")
        if contract is None:
            if run_identity_required:
                raise RuntimeError(
                    f"Cannot resume {checkpoint_path}: checkpoint has no resume_contract, "
                    "but this run requires run identity validation."
                )
            if current_identity is not None:
                raise RuntimeError(
                    f"Cannot resume {checkpoint_path}: checkpoint has no resume_contract, "
                    "but the current dataset requires identity validation."
                )
            return
        if not isinstance(contract, dict) or "version" not in contract:
            raise RuntimeError(f"Cannot resume {checkpoint_path}: unsupported or malformed resume_contract.")

        version = contract["version"]
        if version == 1:
            if set(contract) != {"version", "dataset_identity"}:
                raise RuntimeError(f"Cannot resume {checkpoint_path}: unsupported or malformed resume_contract.")
            if run_identity_required:
                raise RuntimeError(
                    f"Cannot resume {checkpoint_path}: version 1 resume_contract has no run identity, "
                    "but this run requires it."
                )
        elif version == 2:
            if set(contract) != {"version", "dataset_identity", "run_identity"}:
                raise RuntimeError(f"Cannot resume {checkpoint_path}: unsupported or malformed resume_contract.")
            saved_run_identity = contract["run_identity"]
            if current_run_identity is None:
                raise RuntimeError(
                    f"Cannot resume {checkpoint_path}: checkpoint requires run identity {saved_run_identity!r}, "
                    "but the current run has none."
                )
            if saved_run_identity != current_run_identity:
                raise RuntimeError(
                    f"Cannot resume {checkpoint_path}: run identity differs from the checkpoint "
                    f"(checkpoint={saved_run_identity!r}, current={current_run_identity!r})."
                )
        else:
            raise RuntimeError(f"Cannot resume {checkpoint_path}: unsupported or malformed resume_contract.")

        saved_identity = contract["dataset_identity"]
        if current_identity is None:
            if saved_identity is not None:
                raise RuntimeError(
                    f"Cannot resume {checkpoint_path}: checkpoint requires a dataset identity, "
                    "but the current run has none."
                )
            return
        if saved_identity is None:
            raise RuntimeError(
                f"Cannot resume {checkpoint_path}: checkpoint resume_contract has no dataset identity."
            )
        if saved_identity != current_identity:
            raise RuntimeError(
                f"Cannot resume {checkpoint_path}: dataset identity differs from the checkpoint "
                f"(checkpoint={saved_identity!r}, current={current_identity!r})."
            )

    def load_pretrained_init(self):
        if not exists(self.pretrained_init):
            return
        if not os.path.isfile(self.pretrained_init):
            raise FileNotFoundError(f"pretrained_init checkpoint not found: {self.pretrained_init}")

        pretrained_init_options = getattr(self, "pretrained_init_options", {})
        if pretrained_init_options:
            target_ema_state = self.ema_model.state_dict() if self.is_main else {
                f"ema_model.{key}": value
                for key, value in self.accelerator.unwrap_model(self.model).state_dict().items()
            }
            ema_state_dict = build_pretrained_ema_init(
                self.pretrained_init,
                target_ema_state,
                **pretrained_init_options,
            )
        else:
            checkpoint = self._load_checkpoint_file(self.pretrained_init)
            ema_state_dict = checkpoint.get("ema_model_state_dict")
            if ema_state_dict is None:
                raise KeyError(f"pretrained_init checkpoint has no ema_model_state_dict: {self.pretrained_init}")

        online_state_dict = online_state_from_ema(ema_state_dict)
        self._remove_legacy_mel_keys(online_state_dict)
        self.accelerator.unwrap_model(self.model).load_state_dict(online_state_dict)

        if self.is_main:
            ema_init_state = {
                key: value
                for key, value in ema_state_dict.items()
                if key not in ["initted", "step", "update"]
            }
            self._remove_legacy_mel_keys(ema_init_state, prefix="ema_model.")
            ema_init_state["initted"] = torch.tensor(True)
            ema_init_state["step"] = torch.tensor(0, dtype=torch.long)
            self.ema_model.load_state_dict(ema_init_state)
            print(f"Initializing weights from pretrained EMA: {self.pretrained_init}")
            print("Online model initialized: yes")
            print("Trainer EMA initialized: yes, EMA step reset to 0")
            print("Optimizer restored: no; scheduler restored: no; starting update: 0")

        self.accelerator.wait_for_everyone()
        gc.collect()

    def _find_resume_checkpoint(self):
        if not exists(self.checkpoint_path) or not os.path.isdir(self.checkpoint_path):
            return None
        filenames = os.listdir(self.checkpoint_path)
        if "model_last.pt" in filenames:
            return "model_last.pt"
        training_checkpoints = [
            filename
            for filename in filenames
            if re.fullmatch(r"model_\d+\.pt", filename)
        ]
        if training_checkpoints:
            return max(
                training_checkpoints,
                key=lambda filename: int(filename.removeprefix("model_").removesuffix(".pt")),
            )

        # 兼容旧 finetune_cli：它会把初始化权重复制为 pretrained_* 放进输出目录。
        # 显式 pretrained_init 存在时不走这条旧路径，避免把外部 Jmica checkpoint
        # 误当作可恢复训练状态；没有显式配置时则保持原有行为。
        if not exists(self.pretrained_init):
            pretrained_checkpoints = [
                filename
                for filename in filenames
                if filename.startswith("pretrained_") and filename.endswith((".pt", ".safetensors"))
            ]
            if pretrained_checkpoints:
                return sorted(pretrained_checkpoints)[0]
        return None

    def load_checkpoint(self):
        latest_checkpoint = self._find_resume_checkpoint()
        if latest_checkpoint is None:
            self.load_pretrained_init()
            return 0

        self.accelerator.wait_for_everyone()
        checkpoint_path = os.path.join(self.checkpoint_path, latest_checkpoint)
        checkpoint = self._load_checkpoint_file(checkpoint_path)
        self._validate_resume_contract(checkpoint, checkpoint_path)

        # patch for backward compatibility, 305e3ea
        self._remove_legacy_mel_keys(checkpoint["ema_model_state_dict"], prefix="ema_model.")

        if self.is_main:
            self.ema_model.load_state_dict(checkpoint["ema_model_state_dict"])

        if "update" in checkpoint or "step" in checkpoint:
            # patch for backward compatibility, with before f992c4e
            if "step" in checkpoint:
                checkpoint["update"] = checkpoint["step"] // self.grad_accumulation_steps
                if self.grad_accumulation_steps > 1 and self.is_main:
                    print(
                        "F5-TTS WARNING: Loading checkpoint saved with per_steps logic (before f992c4e), will convert to per_updates according to grad_accumulation_steps setting, may have unexpected behaviour."
                    )
            # patch for backward compatibility, 305e3ea
            self._remove_legacy_mel_keys(checkpoint["model_state_dict"])

            self.accelerator.unwrap_model(self.model).load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if self.scheduler:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            update = checkpoint["update"]
        else:
            checkpoint["model_state_dict"] = self._online_state_from_ema(checkpoint["ema_model_state_dict"])
            self.accelerator.unwrap_model(self.model).load_state_dict(checkpoint["model_state_dict"])
            update = 0

        del checkpoint
        gc.collect()
        return update

    def _training_update_horizon(self, dataloader_length):
        epoch_horizon = math.ceil(dataloader_length / self.grad_accumulation_steps) * self.epochs
        return min(epoch_horizon, self.max_updates) if exists(self.max_updates) else epoch_horizon

    def _reached_max_updates(self, update):
        return exists(self.max_updates) and update >= self.max_updates

    def _should_stop_training(self, update):
        if self._reached_max_updates(update):
            return True
        return exists(self.stop_after_updates) and update >= self.stop_after_updates

    def _write_lr_trace(self, record):
        """Append one JSONL record to the LR trace; rank 0 only, no-op when disabled.

        Creates the parent directory because the first record is written while the
        scheduler is being built, long before the first checkpoint save makes it.
        Opened per write so a killed run still leaves a complete, readable trace.
        """
        if not self.lr_trace_path or not self.is_main:
            return
        parent = os.path.dirname(os.path.abspath(self.lr_trace_path))
        os.makedirs(parent, exist_ok=True)
        with open(self.lr_trace_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def _record_lr(self, global_update):
        if not self.lr_trace_path or not self.is_main:
            return
        self._write_lr_trace({"global_update": int(global_update), "lr": float(self.scheduler.get_last_lr()[0])})

    def train(self, train_dataset: Dataset, num_workers=16, resumable_with_seed: int = None):
        if self.log_samples:
            from f5_tts.infer.utils_infer import cfg_strength, load_vocoder, nfe_step, sway_sampling_coef

            vocoder = load_vocoder(
                vocoder_name=self.vocoder_name, is_local=self.is_local_vocoder, local_path=self.local_vocoder_path
            )
            target_sample_rate = self.accelerator.unwrap_model(self.model).mel_spec.target_sample_rate
            log_samples_path = f"{self.checkpoint_path}/samples"
            os.makedirs(log_samples_path, exist_ok=True)

        if exists(resumable_with_seed):
            generator = torch.Generator()
            generator.manual_seed(resumable_with_seed)
        else:
            generator = None

        if self.batch_size_type == "sample":
            train_dataloader = DataLoader(
                train_dataset,
                collate_fn=collate_fn,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=num_workers > 0,
                batch_size=self.batch_size_per_gpu,
                shuffle=True,
                generator=generator,
            )
        elif self.batch_size_type == "frame":
            self.accelerator.even_batches = False
            sampler = SequentialSampler(train_dataset)
            batch_sampler = DynamicBatchSampler(
                sampler,
                self.batch_size_per_gpu,
                max_samples=self.max_samples,
                random_seed=resumable_with_seed,  # This enables reproducible shuffling
                drop_residual=False,
            )
            train_dataloader = DataLoader(
                train_dataset,
                collate_fn=collate_fn,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=num_workers > 0,
                batch_sampler=batch_sampler,
            )
        else:
            raise ValueError(f"batch_size_type must be either 'sample' or 'frame', but received {self.batch_size_type}")

        # ✅ scheduler 单位统一：warmup 与 total horizon 必须同时换算到 Accelerate
        # inner scheduler units（见 resolve_scheduler_contract 的说明）。
        # 历史 bug 只乘了 warmup，8 卡下 LR 会在 total/8 步就到 floor。
        global_total_updates = self._training_update_horizon(len(train_dataloader))
        if global_total_updates == 0 and self.max_updates != 0:
            raise ValueError("Training dataloader has no optimizer updates. Check the dataset and frame batch limit.")

        self.scheduler_contract = resolve_scheduler_contract(
            global_warmup_updates=self.num_warmup_updates,
            global_total_updates=global_total_updates,
            num_processes=self.accelerator.num_processes,
            split_batches=resolve_split_batches(self.accelerator),
        )
        # 小数据 smoke / 微调可能总步数还短于 warmup；clamp 已在 global 单位下完成。
        warmup_updates = self.scheduler_contract["warmup_updates"]
        total_updates = self.scheduler_contract["total_updates"]
        if self.is_main:
            self.accelerator.print(f"scheduler contract: {self.scheduler_contract}")
        self._write_lr_trace({"record": "scheduler_contract", **self.scheduler_contract})
        self.scheduler = build_scheduler(
            self.optimizer, warmup_updates=warmup_updates, total_updates=total_updates
        )
        train_dataloader, self.scheduler = self.accelerator.prepare(
            train_dataloader, self.scheduler
        )  # actual multi_gpu updates = single_gpu updates / gpu nums
        start_update = self.load_checkpoint()
        global_update = start_update
        if self._reached_max_updates(start_update):
            if self.is_main:
                print(f"Checkpoint update {start_update} already reached max_updates={self.max_updates}; nothing to do.")
            self.accelerator.end_training()
            return

        if exists(resumable_with_seed):
            orig_epoch_step = len(train_dataloader)
            start_step = start_update * self.grad_accumulation_steps
            skipped_epoch = int(start_step // orig_epoch_step)
            skipped_batch = start_step % orig_epoch_step
            skipped_dataloader = self.accelerator.skip_first_batches(train_dataloader, num_batches=skipped_batch)
        else:
            skipped_epoch = 0

        for epoch in range(skipped_epoch, self.epochs):
            self.model.train()
            if exists(resumable_with_seed) and epoch == skipped_epoch:
                progress_bar_initial = math.ceil(skipped_batch / self.grad_accumulation_steps)
                current_dataloader = skipped_dataloader
            else:
                progress_bar_initial = 0
                current_dataloader = train_dataloader

            # Set epoch for the batch sampler if it exists
            if hasattr(train_dataloader, "batch_sampler") and hasattr(train_dataloader.batch_sampler, "set_epoch"):
                train_dataloader.batch_sampler.set_epoch(epoch)

            progress_bar = tqdm(
                range(math.ceil(len(train_dataloader) / self.grad_accumulation_steps)),
                desc=f"Epoch {epoch + 1}/{self.epochs}",
                unit="update",
                disable=not self.accelerator.is_local_main_process,
                initial=progress_bar_initial,
            )

            for batch in current_dataloader:
                step_was_skipped = False
                with self.accelerator.accumulate(self.model):
                    text_inputs = batch["text"]
                    mel_spec = batch["mel"].permute(0, 2, 1)
                    mel_lengths = batch["mel_lengths"]

                    # TODO. add duration predictor training
                    if self.duration_predictor is not None and self.accelerator.is_local_main_process:
                        dur_loss = self.duration_predictor(mel_spec, lens=batch.get("durations"))
                        self.accelerator.log({"duration loss": dur_loss.item()}, step=global_update)

                    loss, cond, pred = self.model(
                        mel_spec, text=text_inputs, lens=mel_lengths, noise_scheduler=self.noise_scheduler
                    )
                    self.accelerator.backward(loss)

                    if self.max_grad_norm > 0 and self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                    self.optimizer.step()
                    # ✅ AcceleratedScheduler 在 optimizer step 被跳过时不会推进 LR
                    # （accelerate/scheduler.py step(): step_was_skipped -> return），
                    # 所以训练身份也必须同步跳过，否则 global_update / EMA 会领先 LR 计划。
                    step_was_skipped = bool(getattr(self.optimizer, "step_was_skipped", False))
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                if self.accelerator.sync_gradients and not step_was_skipped:
                    if self.is_main:
                        self.ema_model.update()

                    global_update += 1
                    progress_bar.update(1)
                    progress_bar.set_postfix(update=str(global_update), loss=loss.item())
                    self._record_lr(global_update)

                if self.accelerator.is_local_main_process:
                    self.accelerator.log(
                        {"loss": loss.item(), "lr": self.scheduler.get_last_lr()[0]}, step=global_update
                    )
                if self.logger == "tensorboard" and self.accelerator.is_main_process:
                    self.writer.add_scalar("loss", loss.item(), global_update)
                    self.writer.add_scalar("lr", self.scheduler.get_last_lr()[0], global_update)

                if global_update % self.last_per_updates == 0 and self.accelerator.sync_gradients:
                    self.save_checkpoint(global_update, last=True)

                if global_update % self.save_per_updates == 0 and self.accelerator.sync_gradients:
                    self.save_checkpoint(global_update)

                    if self.log_samples and self.accelerator.is_local_main_process:
                        ref_audio_len = mel_lengths[0]
                        infer_text = [
                            text_inputs[0] + ([" "] if isinstance(text_inputs[0], list) else " ") + text_inputs[0]
                        ]
                        with torch.inference_mode(), self.accelerator.autocast():
                            generated, _ = self.accelerator.unwrap_model(self.model).sample(
                                cond=mel_spec[0][:ref_audio_len].unsqueeze(0),
                                text=infer_text,
                                duration=ref_audio_len * 2,
                                steps=nfe_step,
                                cfg_strength=cfg_strength,
                                sway_sampling_coef=sway_sampling_coef,
                            )
                            generated = generated.to(torch.float32)
                            gen_mel_spec = generated[:, ref_audio_len:, :].permute(0, 2, 1).to(self.accelerator.device)
                            ref_mel_spec = batch["mel"][0, :, :ref_audio_len].unsqueeze(0)
                            if self.vocoder_name == "vocos":
                                gen_audio = vocoder.decode(gen_mel_spec).cpu()
                                ref_audio = vocoder.decode(ref_mel_spec).cpu()
                            elif self.vocoder_name == "bigvgan":
                                gen_audio = vocoder(gen_mel_spec).squeeze(0).cpu()
                                ref_audio = vocoder(ref_mel_spec).squeeze(0).cpu()

                        torchaudio.save(
                            f"{log_samples_path}/update_{global_update}_gen.wav", gen_audio, target_sample_rate
                        )
                        torchaudio.save(
                            f"{log_samples_path}/update_{global_update}_ref.wav", ref_audio, target_sample_rate
                        )
                        self.model.train()

                if self._should_stop_training(global_update):
                    break

            if self._should_stop_training(global_update):
                break

        self.save_checkpoint(global_update, last=True)

        self.accelerator.end_training()
