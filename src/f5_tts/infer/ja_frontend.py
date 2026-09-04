"""Versioned Japanese text frontend for kana-only F5-TTS models."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

FRONTEND_VERSION = "ja-kana-v1"
EXPECTED_PYOPENJTALK_VERSION = "0.4.1"
_DICT_READY = False

# These tables are deliberately explicit and versioned. Changing one changes frontend behavior.
_WAVE_MAP = {"〜": "ー", "～": "ー", "〰": "ー", "~": "ー"}
_QUOTE_MAP = {
    '"': "",
    "'": "",
    "`": "",
    "「": "",
    "」": "",
    "『": "",
    "』": "",
    "“": "",
    "”": "",
    "‘": "",
    "’": "",
}
_PUNCT_MAP = {
    ",": "、",
    "，": "、",
    "､": "、",
    ".": "。",
    "｡": "。",
    "!": "！",
    "?": "？",
    "…": "…",
    "・": "・",
    "：": "：",
    ":": "：",
    "；": "；",
    ";": "；",
    "（": "（",
    "）": "）",
    "(": "（",
    ")": "）",
    "、": "、",
    "。": "。",
    "！": "！",
    "？": "？",
}
_BOUNDARIES = frozenset(_PUNCT_MAP.values())


@dataclass(frozen=True)
class MappingEvent:
    stage: str
    input_start: int
    input_text: str
    output_text: str
    kind: str


@dataclass(frozen=True)
class FrontendResult:
    source_text: str
    normalized_text: str
    kana: str
    mapping_events: tuple[MappingEvent, ...]
    frontend_version: str = FRONTEND_VERSION


class JapaneseFrontendError(ValueError):
    """Structured rejection raised when frontend output is unsafe for training."""

    def __init__(self, code: str, message: str, *, details: dict | None = None):
        self.code = code
        self.details = details or {}
        super().__init__(message)

    def to_dict(self) -> dict:
        return {"code": self.code, "message": str(self), "details": self.details}


def _ensure_dict():
    """Import pyopenjtalk once and provide an actionable dependency error."""
    global _DICT_READY
    if _DICT_READY:
        return
    try:
        import pyopenjtalk  # noqa: F401
    except ImportError as error:
        raise RuntimeError("缺少 pyopenjtalk。日文推理需要它做汉字→假名：pip install pyopenjtalk") from error
    _DICT_READY = True


def _mapping_kind(char: str) -> tuple[str, str] | None:
    if char in _WAVE_MAP:
        return "wave", _WAVE_MAP[char]
    if char in _QUOTE_MAP:
        return "quote", _QUOTE_MAP[char]
    if char in _PUNCT_MAP:
        return "punct", _PUNCT_MAP[char]
    if char.isspace():
        return "whitespace", " "
    return None


def normalize_japanese(text: str) -> tuple[str, tuple[MappingEvent, ...]]:
    """Apply the frozen v1 NFKC and character mapping policy."""
    if not isinstance(text, str):
        raise TypeError("text must be str")
    if "�" in text:
        raise JapaneseFrontendError("replacement_character", "input contains U+FFFD", details={"positions": [i for i, c in enumerate(text) if c == "�"]})

    nfkc = unicodedata.normalize("NFKC", text)
    events: list[MappingEvent] = []
    if nfkc != text:
        events.append(MappingEvent("normalize", 0, text, nfkc, "nfkc"))

    output: list[str] = []
    for index, char in enumerate(nfkc):
        mapped = _mapping_kind(char)
        if mapped is None:
            output.append(char)
            continue
        kind, replacement = mapped
        # Collapse all whitespace runs to one ASCII space.
        if kind == "whitespace" and output and output[-1] == " ":
            replacement = ""
        output.append(replacement)
        if replacement != char or kind == "whitespace":
            events.append(MappingEvent("map", index, char, replacement, kind))
    mapped = "".join(output)
    normalized = mapped.strip()
    if normalized != mapped:
        leading = len(mapped) - len(mapped.lstrip())
        trailing = len(mapped) - len(mapped.rstrip())
        if leading:
            events.append(MappingEvent("trim", 0, mapped[:leading], "", "whitespace"))
        if trailing:
            events.append(MappingEvent("trim", len(mapped) - trailing, mapped[-trailing:], "", "whitespace"))
    return normalized, tuple(events)


def _is_han(char: str) -> bool:
    codepoint = ord(char)
    return (
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
        or 0x20000 <= codepoint <= 0x323AF
    )


def _validate_kana(kana: str, vocab: set[str] | None) -> None:
    failures: dict[str, list[dict]] = {}
    for index, char in enumerate(kana):
        if _is_han(char):
            failures.setdefault("han", []).append({"index": index, "char": char, "codepoint": f"U+{ord(char):04X}"})
        if char == "�":
            failures.setdefault("replacement_character", []).append({"index": index})
        category = unicodedata.category(char)
        if category in {"Cc", "Cs"}:
            failures.setdefault("control", []).append(
                {"index": index, "char": repr(char), "codepoint": f"U+{ord(char):04X}", "category": category}
            )
    if vocab is not None:
        oov = sorted({char for char in kana if char not in vocab})
        if oov:
            failures["oov"] = [{"char": char, "codepoint": f"U+{ord(char):04X}"} for char in oov]
    if failures:
        raise JapaneseFrontendError("invalid_g2p_output", "G2P output failed strict validation", details=failures)


def frontend_v1(text: str, *, vocab: Iterable[str] | None = None) -> FrontendResult:
    """Convert Japanese to kana and reject residual Han, controls, U+FFFD, or OOV."""
    _ensure_dict()
    import pyopenjtalk

    normalized, events = normalize_japanese(text)
    pieces: list[str] = []
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            segment = "".join(buffer)
            pieces.append(pyopenjtalk.g2p(segment, kana=True))
            buffer.clear()

    for char in normalized:
        if char in _BOUNDARIES or char == " ":
            flush()
            pieces.append(char)
        else:
            buffer.append(char)
    flush()
    kana = "".join(pieces)
    _validate_kana(kana, set(vocab) if vocab is not None else None)
    return FrontendResult(text, normalized, kana, events)


def ja_to_kana(text: str) -> str:
    """Backward-compatible string-returning wrapper around :func:`frontend_v1`."""
    return frontend_v1(text).kana


def check_vocab_coverage(text: str, vocab_path: str) -> list[str]:
    """Return characters absent from a vocabulary file, preserving legacy behavior."""
    with open(vocab_path, encoding="utf-8") as handle:
        vocab = set(handle.read().splitlines())
    return sorted({char for char in text if char not in vocab and char != " "})


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted((item for item in root.rglob("*") if item.is_file()), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def frontend_provenance(*, hash_dictionary: bool = True) -> dict:
    """Return reproducibility metadata, optionally hashing the Open JTalk dictionary tree."""
    _ensure_dict()
    import pyopenjtalk

    version = importlib.metadata.version("pyopenjtalk")
    dictionary_value = (
        pyopenjtalk.OPEN_JTALK_DICT_DIR.decode()
        if isinstance(pyopenjtalk.OPEN_JTALK_DICT_DIR, bytes)
        else pyopenjtalk.OPEN_JTALK_DICT_DIR
    )
    dictionary = Path(dictionary_value).resolve()
    dictionary_version = dictionary.name.removeprefix("open_jtalk_dic_utf_8-")
    return {
        "frontend_version": FRONTEND_VERSION,
        "pyopenjtalk_version": version,
        "pyopenjtalk_version_expected": EXPECTED_PYOPENJTALK_VERSION,
        "unicode_version": unicodedata.unidata_version,
        "dictionary_version": dictionary_version,
        "dictionary_tree_sha256": _tree_sha256(dictionary) if hash_dictionary else None,
        "mapping_policy_sha256": hashlib.sha256(
            json.dumps(
                {"wave": _WAVE_MAP, "quotes": _QUOTE_MAP, "punct": _PUNCT_MAP, "whitespace": "collapse-to-ascii-space"},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
    }


def result_as_dict(result: FrontendResult) -> dict:
    """Serialize a result without dataclass-specific objects."""
    return asdict(result)
