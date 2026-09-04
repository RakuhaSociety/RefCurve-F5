from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path

import torch

from f5_tts.model.vocab_contract import validate_vocabulary_contract


EMA_PREFIX = "ema_model."
EMBEDDING_KEY = "ema_model.transformer.text_embed.text_embed.weight"
MODEL_EMBEDDING_KEY = "transformer.text_embed.text_embed.weight"
_METADATA_KEYS = {"initted", "step", "update"}
_LEGACY_MEL_KEYS = {
    "ema_model.mel_spec.mel_stft.mel_scale.fb",
    "ema_model.mel_spec.mel_stft.spectrogram.window",
}


def state_tensor_sha256(state: Mapping[str, torch.Tensor]) -> str:
    """Canonical digest of tensor keys, metadata and bytes, independent of checkpoint container."""
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        key_bytes = key.encode("utf-8")
        dtype_bytes = str(value.dtype).encode("ascii")
        digest.update(len(key_bytes).to_bytes(8, "big"))
        digest.update(key_bytes)
        digest.update(len(dtype_bytes).to_bytes(4, "big"))
        digest.update(dtype_bytes)
        digest.update(len(value.shape).to_bytes(4, "big"))
        for dimension in value.shape:
            digest.update(int(dimension).to_bytes(8, "big", signed=False))
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def checkpoint_state_cast_to_model(
    checkpoint_state: Mapping[str, torch.Tensor],
    model_state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Cast checkpoint tensors exactly as loaded, retaining strict keys, shapes and buffer coverage."""
    checkpoint_keys = set(checkpoint_state)
    model_keys = set(model_state)
    if checkpoint_keys != model_keys:
        raise KeyError(
            f"loaded model state-key mismatch: missing={sorted(model_keys - checkpoint_keys)}, "
            f"unexpected={sorted(checkpoint_keys - model_keys)}"
        )
    casted: dict[str, torch.Tensor] = {}
    for key in sorted(model_keys):
        source = checkpoint_state[key]
        target = model_state[key]
        if source.shape != target.shape:
            raise ValueError(
                f"loaded model tensor shape mismatch for {key}: checkpoint={tuple(source.shape)}, "
                f"loaded={tuple(target.shape)}"
            )
        casted[key] = source.detach().to(device="cpu", dtype=target.dtype).contiguous()
    return casted


def verify_checkpoint_state_loaded(
    checkpoint_state: Mapping[str, torch.Tensor],
    model: torch.nn.Module,
) -> str:
    """Verify weights and buffers after applying the loader's model dtype conversion."""
    loaded_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    expected_state = checkpoint_state_cast_to_model(checkpoint_state, loaded_state)
    checkpoint_digest = state_tensor_sha256(expected_state)
    loaded_digest = state_tensor_sha256(loaded_state)
    if checkpoint_digest != loaded_digest:
        raise RuntimeError(
            f"loaded model state digest mismatch: checkpoint_casted={checkpoint_digest}, loaded={loaded_digest}"
        )
    return loaded_digest


def model_parameter_sha256(model: torch.nn.Module) -> str:
    return state_tensor_sha256(model.state_dict())


def checkpoint_state_for_source(
    checkpoint_path: str | Path,
    weight_source: str,
) -> dict[str, torch.Tensor]:
    """Load one explicitly requested inference state without implicit fallback."""
    path = Path(checkpoint_path)
    if weight_source not in {"online", "ema"}:
        raise ValueError(f"unsupported weight source: {weight_source!r}")
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        raw = dict(load_file(str(path), device="cpu"))
        if weight_source != "ema":
            raise KeyError(f"flat safetensors checkpoint is EMA-only for strict evaluation: {path}")
        state = {
            key.removeprefix(EMA_PREFIX): value
            for key, value in raw.items()
            if key.startswith(EMA_PREFIX) and key.removeprefix(EMA_PREFIX) not in _METADATA_KEYS
        }
    else:
        checkpoint = torch.load(path, weights_only=True, map_location="cpu")
        mapping_key = "ema_model_state_dict" if weight_source == "ema" else "model_state_dict"
        raw = checkpoint.get(mapping_key)
        if not isinstance(raw, Mapping):
            raise KeyError(f"checkpoint has no requested {mapping_key}: {path}")
        if weight_source == "ema":
            state = {
                key.removeprefix(EMA_PREFIX): value
                for key, value in raw.items()
                if key.startswith(EMA_PREFIX) and key.removeprefix(EMA_PREFIX) not in _METADATA_KEYS
            }
        else:
            state = dict(raw)
    for key in tuple(state):
        if key in {value.removeprefix(EMA_PREFIX) for value in _LEGACY_MEL_KEYS}:
            del state[key]
    if not state:
        raise ValueError(f"requested {weight_source} state is empty or has no model tensors: {path}")
    return state


def validate_checkpoint_architecture(
    checkpoint_path: str | Path,
    weight_source: str,
    expected_state: Mapping[str, torch.Tensor],
    *,
    expected_embedding_rows: int,
    expected_embedding_width: int,
) -> dict[str, int]:
    """Strictly compare checkpoint keys/shapes and declared embedding metadata."""
    state = checkpoint_state_for_source(checkpoint_path, weight_source)
    expected_keys = set(expected_state)
    actual_keys = set(state)
    if actual_keys != expected_keys:
        raise KeyError(
            f"architecture state-key mismatch: missing={sorted(expected_keys - actual_keys)}, "
            f"unexpected={sorted(actual_keys - expected_keys)}"
        )
    mismatched = [
        (key, tuple(state[key].shape), tuple(expected_state[key].shape))
        for key in sorted(expected_keys)
        if state[key].shape != expected_state[key].shape
    ]
    if mismatched:
        key, actual, expected = mismatched[0]
        raise ValueError(f"architecture tensor shape mismatch for {key}: checkpoint={actual}, expected={expected}")
    embedding = state.get(MODEL_EMBEDDING_KEY)
    if embedding is None or embedding.ndim != 2:
        raise KeyError(f"checkpoint has no rank-2 text embedding: {MODEL_EMBEDDING_KEY}")
    rows, width = map(int, embedding.shape)
    if rows != expected_embedding_rows or width != expected_embedding_width:
        raise ValueError(
            f"embedding metadata mismatch: checkpoint=({rows}, {width}), "
            f"declared=({expected_embedding_rows}, {expected_embedding_width})"
        )
    return {"state_keys": len(state), "embedding_rows": rows, "embedding_width": width}


def load_ema_checkpoint(checkpoint_path: str | Path) -> dict[str, torch.Tensor]:
    path = Path(checkpoint_path)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state = load_file(str(path), device="cpu")
    else:
        checkpoint = torch.load(path, weights_only=True, map_location="cpu")
        state = checkpoint.get("ema_model_state_dict")
        if state is None:
            raise KeyError(f"checkpoint has no ema_model_state_dict: {path}")
    if not isinstance(state, Mapping):
        raise TypeError(f"EMA state must be a mapping: {path}")
    return dict(state)


def _initialise_new_rows(source_embedding: torch.Tensor, count: int, seed: int) -> torch.Tensor:
    if count == 0:
        return source_embedding.new_empty((0, source_embedding.shape[1]))
    nonpadding = source_embedding[1:].to(torch.float32)
    if nonpadding.numel() == 0:
        raise ValueError("source embedding has no non-padding rows")
    mean = nonpadding.mean()
    std = nonpadding.std(unbiased=False)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    rows = torch.randn((count, source_embedding.shape[1]), generator=generator, dtype=torch.float32)
    rows = rows.mul(std).add(mean)
    return rows.to(dtype=source_embedding.dtype)


def remap_ema_state_by_token(
    source_ema_state: Mapping[str, torch.Tensor],
    target_ema_state: Mapping[str, torch.Tensor],
    source_tokens: list[str],
    target_tokens: list[str],
    *,
    seed: int,
) -> dict[str, torch.Tensor]:
    source_keys = set(source_ema_state) - _METADATA_KEYS - _LEGACY_MEL_KEYS
    target_keys = set(target_ema_state) - _METADATA_KEYS - _LEGACY_MEL_KEYS
    if source_keys != target_keys:
        missing = sorted(target_keys - source_keys)
        unexpected = sorted(source_keys - target_keys)
        raise KeyError(f"EMA key mismatch: missing={missing}, unexpected={unexpected}")
    if EMBEDDING_KEY not in source_keys:
        raise KeyError(f"EMA state has no text embedding key: {EMBEDDING_KEY}")

    source_embedding = source_ema_state[EMBEDDING_KEY]
    target_embedding = target_ema_state[EMBEDDING_KEY]
    if source_embedding.ndim != 2 or target_embedding.ndim != 2:
        raise ValueError("text embeddings must be rank-2 tensors")
    if source_embedding.shape[0] != len(source_tokens) + 1:
        raise ValueError(
            f"source embedding rows {source_embedding.shape[0]} do not match source vocabulary {len(source_tokens)} + row0"
        )
    if target_embedding.shape[0] != len(target_tokens) + 1:
        raise ValueError(
            f"target embedding rows {target_embedding.shape[0]} do not match target vocabulary {len(target_tokens)} + row0"
        )
    if source_embedding.shape[1] != target_embedding.shape[1]:
        raise ValueError(
            f"embedding width mismatch: source={source_embedding.shape[1]}, target={target_embedding.shape[1]}"
        )

    remapped = {}
    for key in sorted(target_keys):
        source_value = source_ema_state[key]
        target_value = target_ema_state[key]
        if key != EMBEDDING_KEY and source_value.shape != target_value.shape:
            raise ValueError(
                f"EMA tensor shape mismatch for {key}: source={tuple(source_value.shape)}, target={tuple(target_value.shape)}"
            )
        remapped[key] = source_value.clone()

    source_index = {token: index + 1 for index, token in enumerate(source_tokens)}
    target_index = {token: index + 1 for index, token in enumerate(target_tokens)}
    new_tokens = [token for token in target_tokens if token not in source_index]
    new_rows = _initialise_new_rows(source_embedding, len(new_tokens), seed)

    embedding = target_embedding.new_empty(target_embedding.shape)
    embedding[0] = source_embedding[0]
    new_row_index = 0
    for token, row in target_index.items():
        source_row = source_index.get(token)
        if source_row is None:
            embedding[row] = new_rows[new_row_index]
            new_row_index += 1
        else:
            embedding[row] = source_embedding[source_row]
    remapped[EMBEDDING_KEY] = embedding
    return remapped


def build_pretrained_ema_init(
    checkpoint_path: str | Path,
    target_ema_state: Mapping[str, torch.Tensor],
    *,
    source_vocab_path: str | Path,
    source_vocab_contract_path: str | Path,
    target_vocab_path: str | Path,
    target_vocab_contract_path: str | Path,
    remap_strategy: str = "token",
    seed: int = 666,
) -> dict[str, torch.Tensor]:
    if remap_strategy != "token":
        raise ValueError(f"unsupported pretrained vocabulary remap strategy: {remap_strategy!r}")

    source_tokens, _ = validate_vocabulary_contract(source_vocab_path, source_vocab_contract_path)
    target_tokens, _ = validate_vocabulary_contract(target_vocab_path, target_vocab_contract_path)
    source_ema_state = load_ema_checkpoint(checkpoint_path)
    return remap_ema_state_by_token(source_ema_state, target_ema_state, source_tokens, target_tokens, seed=seed)


def online_state_from_ema(ema_state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        key.removeprefix(EMA_PREFIX): value
        for key, value in ema_state.items()
        if key.startswith(EMA_PREFIX)
    }
