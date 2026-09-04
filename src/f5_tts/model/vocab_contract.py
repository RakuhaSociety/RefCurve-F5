from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VocabularyContract:
    identity: str
    provenance: str
    token_sequence_sha256: str
    token_count: int
    embedding_rows: int
    token_row_offset: int = 1
    padding_row: int = 0


def read_vocabulary(vocab_path: str | Path) -> list[str]:
    path = Path(vocab_path)
    data = path.read_bytes()
    if data.startswith(b"\xef\xbb\xbf"):
        raise ValueError(f"vocabulary must not contain a UTF-8 BOM: {path}")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"vocabulary must be UTF-8: {path}") from exc

    tokens = text.splitlines()
    if not tokens:
        raise ValueError(f"vocabulary is empty: {path}")
    if any(token == "" for token in tokens):
        raise ValueError(f"vocabulary contains an empty token: {path}")
    if tokens[0] != " ":
        raise ValueError(f"vocabulary token at index 0 must be one ASCII space: {path}")
    if len(tokens) != len(set(tokens)):
        seen = set()
        duplicate = next(token for token in tokens if token in seen or seen.add(token))
        raise ValueError(f"vocabulary contains duplicate token {duplicate!r}: {path}")
    return tokens


def token_sequence_sha256(tokens: list[str]) -> str:
    digest = hashlib.sha256()
    for token in tokens:
        encoded = token.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def load_vocabulary_contract(contract_path: str | Path) -> VocabularyContract:
    path = Path(contract_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_keys = {
        "identity",
        "provenance",
        "token_sequence_sha256",
        "token_count",
        "embedding_rows",
        "token_row_offset",
        "padding_row",
    }
    actual_keys = set(payload)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        raise ValueError(f"invalid vocabulary contract keys in {path}: missing={missing}, extra={extra}")

    contract = VocabularyContract(**payload)
    if not contract.identity or not isinstance(contract.identity, str):
        raise ValueError(f"vocabulary contract identity must be a non-empty string: {path}")
    if not contract.provenance or not isinstance(contract.provenance, str):
        raise ValueError(f"vocabulary contract provenance must be a non-empty string: {path}")
    if contract.token_count <= 0:
        raise ValueError(f"vocabulary contract token_count must be positive: {path}")
    if contract.token_row_offset != 1 or contract.padding_row != 0:
        raise ValueError(f"unsupported embedding row layout in vocabulary contract: {path}")
    if contract.embedding_rows != contract.token_count + contract.token_row_offset:
        raise ValueError(f"vocabulary contract embedding_rows does not match its token layout: {path}")
    return contract


def validate_vocabulary_contract(
    vocab_path: str | Path,
    contract_path: str | Path,
    *,
    expected_identity: str | None = None,
) -> tuple[list[str], VocabularyContract]:
    vocab_path = Path(vocab_path)
    contract = load_vocabulary_contract(contract_path)
    tokens = read_vocabulary(vocab_path)

    if expected_identity is not None and contract.identity != expected_identity:
        raise ValueError(
            f"vocabulary identity mismatch for {vocab_path}: expected {expected_identity!r}, got {contract.identity!r}"
        )
    if len(tokens) != contract.token_count:
        raise ValueError(
            f"vocabulary token count mismatch for {vocab_path}: expected {contract.token_count}, got {len(tokens)}"
        )
    canonical_digest = token_sequence_sha256(tokens)
    if canonical_digest != contract.token_sequence_sha256:
        raise ValueError(
            f"vocabulary token-sequence sha256 mismatch for {vocab_path}: "
            f"expected {contract.token_sequence_sha256}, got {canonical_digest}"
        )
    return tokens, contract
