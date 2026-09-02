# training script.

import os
from importlib.resources import files
from pathlib import Path

import hydra
from omegaconf import OmegaConf

from f5_tts.model import CFM, Trainer
from f5_tts.model.dataset import load_dataset
from f5_tts.model.sharded_dataset import load_audio_root_registry
from f5_tts.model.utils import get_tokenizer
from f5_tts.model.vocab_contract import validate_vocabulary_contract
from f5_tts.train.run_manifest import (
    build_run_manifest,
    collect_code_identity,
    collect_environment_identity,
    ensure_run_manifest,
    wait_for_run_manifest,
)


os.chdir(str(files("f5_tts").joinpath("../..")))  # change working directory to root of project (local editable)
REPO_ROOT = Path.cwd()


def _resolve_repo_path(path):
    if path is None:
        return None
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = REPO_ROOT / resolved
    return resolved.resolve()


def _resolve_existing_file(path, label):
    resolved = _resolve_repo_path(path)
    if resolved is None or not resolved.is_file():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return str(resolved)


def _parse_pretrained_init(model_cfg, tokenizer_path):
    pretrained_cfg = model_cfg.ckpts.get("pretrained_init", None)
    if pretrained_cfg is None:
        return None, None
    if isinstance(pretrained_cfg, str):
        return _resolve_existing_file(pretrained_cfg, "pretrained_init checkpoint"), None

    checkpoint_path = _resolve_existing_file(pretrained_cfg.checkpoint_path, "pretrained_init checkpoint")
    options = {
        "source_vocab_path": _resolve_existing_file(pretrained_cfg.source_vocab_path, "source vocabulary"),
        "source_vocab_contract_path": _resolve_existing_file(
            pretrained_cfg.source_vocab_contract_path, "source vocabulary contract"
        ),
        "target_vocab_path": tokenizer_path,
        "target_vocab_contract_path": _resolve_existing_file(
            model_cfg.model.tokenizer_contract_path, "target vocabulary contract"
        ),
        "remap_strategy": pretrained_cfg.get("remap_strategy", "token"),
        "seed": int(pretrained_cfg.get("seed", 666)),
    }
    return checkpoint_path, options


def _load_audio_roots(dataset_dir: Path, configured):
    identity_path = dataset_dir / "audio_roots.json"
    if configured is None:
        deployment_path = dataset_dir / "deployment_audio_roots.json"
        if not identity_path.is_file() and not deployment_path.is_file():
            return None
    elif isinstance(configured, str):
        deployment_path = _resolve_repo_path(configured)
    else:
        identity_path = _resolve_repo_path(configured.get("identity", identity_path))
        deployment_path = _resolve_repo_path(configured.get("deployment", dataset_dir / "deployment_audio_roots.json"))
    if not identity_path.is_file():
        raise FileNotFoundError(f"audio root identity registry not found: {identity_path}")
    if not deployment_path.is_file():
        raise FileNotFoundError(f"audio root deployment registry not found: {deployment_path}")
    return load_audio_root_registry(identity_path, deployment_path)


@hydra.main(version_base="1.3", config_path=str(files("f5_tts").joinpath("configs")), config_name=None)
def main(model_cfg):
    model_cls = hydra.utils.get_class(f"f5_tts.model.{model_cfg.model.backbone}")
    model_arc = model_cfg.model.arch
    tokenizer = model_cfg.model.tokenizer
    mel_spec_type = model_cfg.model.mel_spec.mel_spec_type

    wandb_project = model_cfg.ckpts.get("wandb_project", "CFM-TTS")
    wandb_run_name = model_cfg.ckpts.get(
        "wandb_run_name",
        f"{model_cfg.model.name}_{mel_spec_type}_{model_cfg.model.tokenizer}_{model_cfg.datasets.name}",
    )
    wandb_resume_id = model_cfg.ckpts.get("wandb_resume_id", None)

    # set text tokenizer
    if tokenizer != "custom":
        tokenizer_path = model_cfg.datasets.name
    else:
        tokenizer_path = _resolve_repo_path(model_cfg.model.tokenizer_path)
        if not tokenizer_path.is_file():
            raise FileNotFoundError(f"custom tokenizer not found: {tokenizer_path}")
        tokenizer_path = str(tokenizer_path)
    vocab_char_map, vocab_size = get_tokenizer(tokenizer_path, tokenizer)
    tokenizer_contract_path = model_cfg.model.get("tokenizer_contract_path", None)
    if tokenizer_contract_path is not None:
        validate_vocabulary_contract(
            tokenizer_path,
            _resolve_existing_file(tokenizer_contract_path, "target vocabulary contract"),
        )

    pretrained_init, pretrained_init_options = _parse_pretrained_init(model_cfg, tokenizer_path)

    dataset_type = model_cfg.datasets.get("dataset_type", "CustomDataset")
    dataset_name = model_cfg.datasets.name
    audio_roots = None
    if dataset_type in ("CustomDatasetPath", "ShardedArrowDataset"):
        dataset_name = _resolve_repo_path(dataset_name)
        expected_kind = "file or directory" if dataset_type == "ShardedArrowDataset" else "directory"
        exists = dataset_name.exists() if dataset_type == "ShardedArrowDataset" else dataset_name.is_dir()
        if not exists:
            raise FileNotFoundError(f"dataset {expected_kind} not found: {dataset_name}")
        if dataset_type == "ShardedArrowDataset":
            dataset_dir = dataset_name if dataset_name.is_dir() else dataset_name.parent
            if dataset_name.is_dir():
                manifest_path = dataset_name / "manifest.json"
                if not manifest_path.is_file():
                    raise FileNotFoundError(f"sharded dataset manifest not found: {manifest_path}")
            audio_roots = _load_audio_roots(dataset_dir, model_cfg.datasets.get("audio_roots", None))
        dataset_name = str(dataset_name)

    # set model
    model = CFM(
        transformer=model_cls(**model_arc, text_num_embeds=vocab_size, mel_dim=model_cfg.model.mel_spec.n_mel_channels),
        mel_spec_kwargs=model_cfg.model.mel_spec,
        vocab_char_map=vocab_char_map,
    )

    train_dataset = load_dataset(
        dataset_name,
        tokenizer,
        dataset_type=dataset_type,
        mel_spec_kwargs=model_cfg.model.mel_spec,
        audio_roots=audio_roots,
    )

    resolved_config = OmegaConf.to_container(model_cfg, resolve=True)
    run_manifest_cfg = model_cfg.ckpts.get("run_manifest", None)
    run_identity = None
    require_run_identity = False
    if run_manifest_cfg is not None and bool(run_manifest_cfg.get("enabled", False)):
        require_run_identity = bool(run_manifest_cfg.get("required", False))
        checkpoint_dir = _resolve_repo_path(model_cfg.ckpts.save_dir)
        manifest_path = checkpoint_dir / run_manifest_cfg.get("filename", "run_manifest.json")
        global_rank = int(os.environ.get("RANK", "0"))
        if global_rank == 0:
            source_checkpoint = pretrained_init
            evidence_paths = {}
            for name, path in run_manifest_cfg.get("evidence_paths", {}).items():
                evidence_paths[str(name)] = _resolve_existing_file(path, f"run manifest evidence {name}")
            tokenizer_evidence_path = _resolve_repo_path(tokenizer_path)
            if tokenizer_evidence_path.is_file():
                evidence_paths["tokenizer_vocab"] = str(tokenizer_evidence_path)
            elif tokenizer == "custom":
                raise FileNotFoundError(f"tokenizer vocabulary not found: {tokenizer_evidence_path}")
            if tokenizer_contract_path is not None:
                evidence_paths["tokenizer_contract"] = _resolve_existing_file(
                    tokenizer_contract_path, "tokenizer contract"
                )
            manifest = build_run_manifest(
                resolved_config=resolved_config,
                code_identity=collect_code_identity(REPO_ROOT),
                dataset_identity=getattr(train_dataset, "dataset_identity", None),
                source_checkpoint=source_checkpoint,
                environment_identity=collect_environment_identity(),
                evidence_paths=evidence_paths,
            )
            manifest = ensure_run_manifest(manifest_path, manifest)
        else:
            manifest = wait_for_run_manifest(
                manifest_path,
                timeout=float(run_manifest_cfg.get("wait_timeout_seconds", 120.0)),
            )
        run_identity = manifest["run_identity"]
    elif run_manifest_cfg is not None and bool(run_manifest_cfg.get("required", False)):
        raise ValueError("ckpts.run_manifest.required=true requires enabled=true")

    # init trainer
    trainer = Trainer(
        model,
        epochs=model_cfg.optim.epochs,
        learning_rate=model_cfg.optim.learning_rate,
        num_warmup_updates=model_cfg.optim.num_warmup_updates,
        max_updates=model_cfg.optim.get("max_updates", None),
        # ✅ smoke 专用：只提前停步，不缩短 scheduler horizon
        stop_after_updates=model_cfg.optim.get("stop_after_updates", None),
        # ✅ 可选的逐 update LR 事实记录；不设置则完全不产生副作用
        lr_trace_path=model_cfg.ckpts.get("lr_trace_path", None),
        save_per_updates=model_cfg.ckpts.save_per_updates,
        keep_last_n_checkpoints=model_cfg.ckpts.keep_last_n_checkpoints,
        checkpoint_path=str(_resolve_repo_path(model_cfg.ckpts.save_dir)),
        pretrained_init=pretrained_init,
        pretrained_init_options=pretrained_init_options,
        batch_size_per_gpu=model_cfg.datasets.batch_size_per_gpu,
        batch_size_type=model_cfg.datasets.batch_size_type,
        max_samples=model_cfg.datasets.max_samples,
        grad_accumulation_steps=model_cfg.optim.grad_accumulation_steps,
        max_grad_norm=model_cfg.optim.max_grad_norm,
        logger=model_cfg.ckpts.logger,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
        wandb_resume_id=wandb_resume_id,
        last_per_updates=model_cfg.ckpts.last_per_updates,
        log_samples=model_cfg.ckpts.log_samples,
        bnb_optimizer=model_cfg.optim.bnb_optimizer,
        mel_spec_type=mel_spec_type,
        is_local_vocoder=model_cfg.model.vocoder.is_local,
        local_vocoder_path=model_cfg.model.vocoder.local_path,
        model_cfg_dict=resolved_config,
        dataset_identity=getattr(train_dataset, "dataset_identity", None),
        run_identity=run_identity,
        require_run_identity=require_run_identity,
    )

    trainer.train(
        train_dataset,
        num_workers=model_cfg.datasets.num_workers,
        resumable_with_seed=666,  # seed for shuffling dataset
    )


if __name__ == "__main__":
    main()
