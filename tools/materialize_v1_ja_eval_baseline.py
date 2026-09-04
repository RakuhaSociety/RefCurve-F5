#!/usr/bin/env python3
"""Materialize the deterministic update-0 v1 Japanese EMA evaluation baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def build_target_ema_state(config_path: Path, vocab_path: Path):
    import torch
    from omegaconf import OmegaConf

    from f5_tts.infer.utils_infer import get_tokenizer
    from f5_tts.model import CFM, DiT

    cfg = OmegaConf.load(config_path)
    if str(cfg.model.backbone) != "DiT":
        raise ValueError(f"unsupported target backbone: {cfg.model.backbone}")
    _, vocab_size = get_tokenizer(str(vocab_path), "custom")
    mel = cfg.model.mel_spec
    with torch.device("meta"):
        model = CFM(
            transformer=DiT(**cfg.model.arch, text_num_embeds=vocab_size, mel_dim=int(mel.n_mel_channels)),
            mel_spec_kwargs={
                "n_fft": int(mel.n_fft),
                "hop_length": int(mel.hop_length),
                "win_length": int(mel.win_length),
                "n_mel_channels": int(mel.n_mel_channels),
                "target_sample_rate": int(mel.target_sample_rate),
                "mel_spec_type": str(mel.mel_spec_type),
            },
            odeint_kwargs={"method": "euler"},
            vocab_char_map={},
        )
    state = {f"ema_model.{key}": value for key, value in model.state_dict().items()}
    embedding_key = "ema_model.transformer.text_embed.text_embed.weight"
    state[embedding_key] = torch.empty(tuple(state[embedding_key].shape), dtype=torch.float32, device="cpu")
    return state


def materialize(args: argparse.Namespace) -> int:
    import torch
    from safetensors.torch import save_file

    from f5_tts.model.checkpoint_init import build_pretrained_ema_init
    from f5_tts.model.vocab_contract import validate_vocabulary_contract

    paths = {
        name: Path(getattr(args, name)).resolve()
        for name in (
            "source_checkpoint",
            "source_vocab",
            "source_contract",
            "target_config",
            "target_vocab",
            "target_contract",
            "output",
            "provenance",
        )
    }
    for name in ("source_checkpoint", "source_vocab", "source_contract", "target_config", "target_vocab", "target_contract"):
        if not paths[name].is_file():
            raise FileNotFoundError(f"{name} missing: {paths[name]}")
    source_tokens, source_contract = validate_vocabulary_contract(paths["source_vocab"], paths["source_contract"])
    target_tokens, target_contract = validate_vocabulary_contract(paths["target_vocab"], paths["target_contract"])
    target_state = build_target_ema_state(paths["target_config"], paths["target_vocab"])
    remapped = build_pretrained_ema_init(
        paths["source_checkpoint"],
        target_state,
        source_vocab_path=paths["source_vocab"],
        source_vocab_contract_path=paths["source_contract"],
        target_vocab_path=paths["target_vocab"],
        target_vocab_contract_path=paths["target_contract"],
        remap_strategy="token",
        seed=args.seed,
    )
    remapped = dict(sorted(remapped.items()))
    paths["output"].parent.mkdir(parents=True, exist_ok=True)
    if paths["output"].exists():
        from safetensors.torch import load_file

        existing = load_file(str(paths["output"]), device="cpu")
        same = set(existing) == set(remapped) and all(
            existing[key].shape == remapped[key].shape and existing[key].dtype == remapped[key].dtype
            and existing[key].equal(remapped[key])
            for key in remapped
        )
        if not same:
            raise FileExistsError(f"refusing to overwrite different frozen baseline: {paths['output']}")
    else:
        save_file(remapped, str(paths["output"]), metadata={"format": "pt", "purpose": "f5tts-v1-ja-update0-eval"})
    output_sha = file_digest(paths["output"])
    provenance = {
        "schema_version": "f5tts-v1-ja-update0-eval-baseline-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "remap_strategy": "token",
        "seed": args.seed,
        "environment": {
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        },
        "source": {
            "checkpoint": portable_path(paths["source_checkpoint"]),
            "checkpoint_sha256": file_digest(paths["source_checkpoint"]),
            "vocab": portable_path(paths["source_vocab"]),
            "vocab_sha256": file_digest(paths["source_vocab"]),
            "contract": portable_path(paths["source_contract"]),
            "contract_sha256": file_digest(paths["source_contract"]),
            "contract_identity": source_contract.identity,
            "token_sequence_sha256": source_contract.token_sequence_sha256,
            "token_count": len(source_tokens),
        },
        "target": {
            "config": portable_path(paths["target_config"]),
            "config_sha256": file_digest(paths["target_config"]),
            "vocab": portable_path(paths["target_vocab"]),
            "vocab_sha256": file_digest(paths["target_vocab"]),
            "contract": portable_path(paths["target_contract"]),
            "contract_sha256": file_digest(paths["target_contract"]),
            "contract_identity": target_contract.identity,
            "token_sequence_sha256": target_contract.token_sequence_sha256,
            "token_count": len(target_tokens),
            "embedding_rows": target_contract.embedding_rows,
        },
        "artifact": {"checkpoint": portable_path(paths["output"]), "checkpoint_sha256": output_sha},
    }
    paths["provenance"].parent.mkdir(parents=True, exist_ok=True)
    paths["provenance"].write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"materialized {paths['output']} sha256={output_sha}")
    print(f"provenance {paths['provenance']}")
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-checkpoint", default=ROOT / "ckpts/F5TTS_v1_Base/model_1250000.safetensors")
    p.add_argument("--source-vocab", default=ROOT / "src/f5_tts/configs/vocab/F5TTS_v1_Base/vocab.txt")
    p.add_argument("--source-contract", default=ROOT / "src/f5_tts/configs/vocab/F5TTS_v1_Base/contract.json")
    p.add_argument("--target-config", default=ROOT / "src/f5_tts/configs/F5TTS_v1_JA_Base.yaml")
    p.add_argument("--target-vocab", default=ROOT / "src/f5_tts/configs/vocab/F5TTS_v1_JA_Base/vocab.txt")
    p.add_argument("--target-contract", default=ROOT / "src/f5_tts/configs/vocab/F5TTS_v1_JA_Base/contract.json")
    p.add_argument("--output", default=ROOT / "ckpts/F5TTS_v1_JA_Update0_Eval/model_update0_ema.safetensors")
    p.add_argument("--provenance", default=ROOT / "ckpts/F5TTS_v1_JA_Update0_Eval/provenance.json")
    p.add_argument("--seed", type=int, default=666)
    return p


if __name__ == "__main__":
    raise SystemExit(materialize(parser().parse_args()))
