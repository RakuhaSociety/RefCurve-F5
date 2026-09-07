#!/usr/bin/env python3
"""Prepare, run, score and review the frozen Yanase/Jmica checkpoint matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import random
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
LEGACY_SCHEMA = "jmica-checkpoint-eval-v1"
SCHEMA = "calibration-checkpoint-eval-v1"
SCHEMA_V2 = "calibration-checkpoint-eval-v2"
ACCEPTED_SCHEMAS = {SCHEMA, SCHEMA_V2, LEGACY_SCHEMA}
SYSTEM_CONTRACT_FIELDS = (
    "config_path",
    "vocab_path",
    "vocab_contract_path",
    "model_family",
    "expected_vocab_identity",
    "expected_token_sequence_sha256",
    "expected_embedding_rows",
    "expected_embedding_width",
    "checkpoint",
    "weight_source",
)
SUPPORTED_MODEL_FAMILIES = {
    "legacy_jmica": {"backbone": "DiT", "text_mask_padding": False, "pe_attn_head": 1},
    "f5tts_v1_ja": {"backbone": "DiT", "text_mask_padding": True, "pe_attn_head": None},
}
SUPPORTED_MEL_CONTRACT = {
    "target_sample_rate": 24000,
    "n_mel_channels": 100,
    "hop_length": 256,
    "win_length": 1024,
    "n_fft": 1024,
    "mel_spec_type": "vocos",
}


class ManifestError(ValueError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_path(spec_path: Path, value: str) -> Path:
    """Resolve repository specs from repo root; prepared absolute paths stay fixed."""
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _portable_id(value: Any, where: str) -> str:
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
    if not isinstance(value, str) or not value or any(ch not in allowed for ch in value):
        raise ManifestError(f"{where} must be a non-empty portable id")
    return value


def _unique_ids(items: list[dict[str, Any]], where: str) -> set[str]:
    ids = [_portable_id(item.get("id"), f"{where}[].id") for item in items]
    if len(ids) != len(set(ids)):
        raise ManifestError(f"{where} ids must be unique")
    return set(ids)


def validate_manifest(spec: Any, *, prepared: bool = False) -> dict[str, Any]:
    if not isinstance(spec, dict) or spec.get("schema_version") not in ACCEPTED_SCHEMAS:
        raise ManifestError(f"schema_version must be one of {sorted(ACCEPTED_SCHEMAS)}")
    strict_contracts = spec["schema_version"] in {SCHEMA, SCHEMA_V2}
    for key in ("inference", "generation", "systems", "references", "texts", "gates", "leakage_check"):
        if key not in spec:
            raise ManifestError(f"missing top-level key: {key}")
    systems, references, texts = spec["systems"], spec["references"], spec["texts"]
    if not all(isinstance(items, list) and items for items in (systems, references, texts)):
        raise ManifestError("systems, references and texts must be non-empty arrays")
    system_ids = _unique_ids(systems, "systems")
    reference_ids = _unique_ids(references, "references")
    text_ids = _unique_ids(texts, "texts")
    if len(system_ids) != len(systems) or len(text_ids) != len(texts):  # defensive clarity
        raise ManifestError("duplicate IDs")
    baselines = [x for x in systems if x.get("baseline")]
    if len(baselines) != 1:
        raise ManifestError("exactly one system must be baseline")
    for i, system in enumerate(systems):
        if system.get("weight_source") not in {"online", "ema"}:
            raise ManifestError(f"systems[{i}].weight_source must be online or ema")
        if not isinstance(system.get("checkpoint"), str) or not system["checkpoint"]:
            raise ManifestError(f"systems[{i}].checkpoint must be a path")
        if strict_contracts:
            missing = [field for field in SYSTEM_CONTRACT_FIELDS if system.get(field) in (None, "")]
            if missing:
                raise ManifestError(f"systems[{i}] missing explicit compatibility fields: {missing}")
            if not isinstance(system["expected_embedding_rows"], int) or system["expected_embedding_rows"] <= 0:
                raise ManifestError(f"systems[{i}].expected_embedding_rows must be positive")
            if not isinstance(system["expected_embedding_width"], int) or system["expected_embedding_width"] <= 0:
                raise ManifestError(f"systems[{i}].expected_embedding_width must be positive")
            if system["model_family"] not in SUPPORTED_MODEL_FAMILIES:
                raise ManifestError(f"systems[{i}].model_family is unsupported: {system['model_family']!r}")
            if prepared:
                for hash_field in ("config_sha256", "vocab_sha256", "vocab_contract_sha256", "checkpoint_sha256"):
                    if not isinstance(system.get(hash_field), str) or not system[hash_field]:
                        raise ManifestError(f"systems[{i}].{hash_field} must be frozen")
    if strict_contracts and "vocab_path" in spec["inference"]:
        raise ManifestError("inference.vocab_path is forbidden; declare vocab_path per system")
    for i, reference in enumerate(references):
        if prepared and not all(isinstance(reference.get(k), str) and reference[k] for k in ("audio", "text_original", "text_kana", "audio_sha256", "data_relation")):
            raise ManifestError(f"references[{i}] is not frozen; fill audio/text/data_relation before prepare")
    for i, text in enumerate(texts):
        if not isinstance(text.get("text_original"), str) or not text["text_original"]:
            raise ManifestError(f"texts[{i}].text_original must be non-empty")
        if text.get("reference_id") not in reference_ids:
            raise ManifestError(f"texts[{i}] references unknown reference_id {text.get('reference_id')!r}")
        if prepared and not all(isinstance(text.get(k), str) and text[k] for k in ("text_kana", "text_kana_sha256", "leakage_status")):
            raise ManifestError(f"texts[{i}] is not prepared")
    if prepared:
        cases = spec.get("cases")
        expected = len(systems) * len(texts)
        if not isinstance(cases, list) or len(cases) != expected:
            raise ManifestError(f"prepared cases must contain exactly {expected} entries")
        if len({_portable_id(x.get("id"), "cases[].id") for x in cases}) != expected:
            raise ManifestError("case IDs must be unique")
    return spec


def load_manifest(path: Path, *, prepared: bool = False) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"cannot read spec {path}: {exc}") from exc
    return validate_manifest(data, prepared=prepared)


def jobs(spec: dict[str, Any]) -> list[dict[str, Any]]:
    systems = {x["id"]: x for x in spec["systems"]}
    texts = {x["id"]: x for x in spec["texts"]}
    references = {x["id"]: x for x in spec["references"]}
    if spec.get("cases"):
        return [{**case, "job_id": case["id"], "system": systems[case["system_id"]], "text": texts[case["text_id"]], "reference": references[case["reference_id"]]} for case in spec["cases"]]
    seed = int(spec["generation"]["primary_seed"])
    return [{"id": f"{system['id']}__{text['id']}", "job_id": f"{system['id']}__{text['id']}", "system_id": system["id"], "text_id": text["id"], "reference_id": text["reference_id"], "seed": seed, "system": system, "text": text, "reference": references[text["reference_id"]]} for system in spec["systems"] for text in spec["texts"]]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ManifestError(f"invalid JSONL {path}:{line_no}: {exc}") from exc
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def completed_ids(path: Path, stage: str) -> set[str]:
    rows = read_jsonl(path)
    # JSONL 允许同一 job 重试。只看每个 stage/job 的最后一条记录，避免历史成功
    # 或失败把新一轮补评分挡住。
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("stage") == stage and row.get("job_id"):
            latest[row["job_id"]] = row
    return {job_id for job_id, row in latest.items() if row.get("status") == "ok"}


def _normalize_leakage(text: str) -> str:
    return "".join(ch for ch in text.casefold() if ch.isalnum())


def _ngrams(text: str, n: int) -> set[str]:
    return {text[i : i + n] for i in range(max(0, len(text) - n + 1))}


def ngram_jaccard(left: str, right: str, n: int) -> float:
    a, b = _ngrams(_normalize_leakage(left), n), _ngrams(_normalize_leakage(right), n)
    return len(a & b) / len(a | b) if a or b else 1.0


def _read_metadata(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters="|,\t")
        reader = csv.DictReader(handle, dialect=dialect)
        field = next((x for x in ("text", "text_original", "transcript") if x in (reader.fieldnames or [])), None)
        if not field:
            raise ManifestError(f"metadata has no text column: {path}")
        return [row[field] for row in reader if row.get(field)]


def _reference_overrides(spec: dict[str, Any], values: list[str]) -> None:
    refs = {x["id"]: x for x in spec["references"]}
    for raw in values:
        parts = raw.split("::", 3)
        if len(parts) != 4 or parts[0] not in refs:
            raise ManifestError("--reference must be ID::AUDIO::TEXT::DATA_RELATION")
        refs[parts[0]].update(audio=parts[1], text_original=parts[2], data_relation=parts[3])


def cmd_prepare(args: argparse.Namespace) -> int:
    source = Path(args.manifest).resolve()
    spec = load_manifest(source)
    _reference_overrides(spec, args.reference or [])
    from f5_tts.infer.ja_frontend import check_vocab_coverage, ja_to_kana

    strict_contracts = spec["schema_version"] in {SCHEMA, SCHEMA_V2}
    metadata = resolve_path(source, spec["leakage_check"]["training_metadata_path"])
    if not metadata.is_file():
        raise ManifestError(f"metadata missing: {metadata}")
    if strict_contracts:
        vocab_paths = {system["id"]: resolve_path(source, system["vocab_path"]) for system in spec["systems"]}
    else:
        vocab = resolve_path(source, spec["inference"]["vocab_path"])
        vocab_paths = {system["id"]: vocab for system in spec["systems"]}
    missing_vocabs = [str(path) for path in vocab_paths.values() if not path.is_file()]
    if missing_vocabs:
        raise ManifestError(f"vocab missing: {sorted(set(missing_vocabs))}")
    training_original = _read_metadata(metadata)
    training_kana = [ja_to_kana(x) for x in training_original]
    originals = {_normalize_leakage(x) for x in training_original}
    kanas = {_normalize_leakage(x) for x in training_kana}
    n = int(spec["leakage_check"]["min_ngram_n"])
    threshold = float(spec["leakage_check"]["max_kana_ngram_jaccard_flag"])

    for reference in spec["references"]:
        if not all(isinstance(reference.get(k), str) and reference[k] for k in ("audio", "text_original", "data_relation")):
            raise ManifestError(f"reference {reference['id']} is incomplete; use --reference")
        audio = resolve_path(source, reference["audio"])
        if not audio.is_file():
            raise ManifestError(f"reference audio missing: {audio}")
        reference["audio"] = str(audio)
        reference["audio_sha256"] = file_digest(audio)
        reference["text_kana"] = ja_to_kana(reference["text_original"])
        for system_id, vocab_path in vocab_paths.items():
            oov = check_vocab_coverage(reference["text_kana"], str(vocab_path))
            if oov:
                raise ManifestError(f"reference {reference['id']} has OOV characters for {system_id}: {oov}")

    for text in spec["texts"]:
        kana = ja_to_kana(text["text_original"])
        for system_id, vocab_path in vocab_paths.items():
            oov = check_vocab_coverage(kana, str(vocab_path))
            if oov:
                raise ManifestError(f"text {text['id']} has OOV characters for {system_id}: {oov}")
        exact_original = _normalize_leakage(text["text_original"]) in originals
        exact_kana = _normalize_leakage(kana) in kanas
        max_sim = max((ngram_jaccard(kana, candidate, n) for candidate in training_kana), default=0.0)
        text["text_kana"] = kana
        text["text_kana_sha256"] = hashlib.sha256(kana.encode()).hexdigest()
        text["leakage_status"] = "flagged" if exact_original or exact_kana or max_sim >= threshold else "clean"
        text["leakage"] = {"exact_original": exact_original, "exact_kana": exact_kana, "max_kana_ngram_jaccard": round(max_sim, 6)}
    flagged = [x["id"] for x in spec["texts"] if x["leakage_status"] != "clean"]
    if flagged and not args.allow_leakage:
        raise ManifestError(f"training leakage flagged for text IDs: {flagged}")

    if strict_contracts:
        for system in spec["systems"]:
            for field in ("config_path", "vocab_path", "vocab_contract_path", "checkpoint"):
                path = resolve_path(source, system[field])
                if not path.is_file():
                    raise ManifestError(f"system {system['id']} {field} missing: {path}")
                system[field] = str(path)
                system[field.removesuffix("_path") + "_sha256" if field != "checkpoint" else "checkpoint_sha256"] = file_digest(path)
            if system.get("checkpoint_provenance_path"):
                provenance = resolve_path(source, system["checkpoint_provenance_path"])
                if not provenance.is_file():
                    raise ManifestError(f"system {system['id']} checkpoint provenance missing: {provenance}")
                system["checkpoint_provenance_path"] = str(provenance)
                system["checkpoint_provenance_sha256"] = file_digest(provenance)
    else:
        vocab = next(iter(vocab_paths.values()))
        spec["inference"]["vocab_path"] = str(vocab)
        spec["inference"]["vocab_sha256"] = file_digest(vocab)
        for system in spec["systems"]:
            system["checkpoint"] = str(resolve_path(source, system["checkpoint"]))
            checkpoint = Path(system["checkpoint"])
            system["checkpoint_sha256"] = file_digest(checkpoint) if checkpoint.is_file() else None
    spec["inference"]["vocoder_path"] = str(resolve_path(source, spec["inference"]["vocoder_path"]))
    spec["leakage_check"]["training_metadata_path"] = str(metadata)
    spec["leakage_check"]["training_metadata_sha256"] = file_digest(metadata)
    spec["cases"] = [{"id": f"{system['id']}__{text['id']}", "system_id": system["id"], "text_id": text["id"], "reference_id": text["reference_id"], "seed": int(spec["generation"]["primary_seed"]), "wav": f"audio/{system['id']}/{text['id']}.wav"} for system in spec["systems"] for text in spec["texts"]]
    spec["status"] = "prepared"
    spec["prepared_at"] = utc_now()
    validate_manifest(spec, prepared=True)
    target = Path(args.output).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(spec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"prepared {target}: {len(spec['cases'])} cases, sha256={canonical_hash(spec)}")
    return 0


def _expected_architecture_state(system: dict[str, Any]) -> dict[str, Any]:
    """Build the declared architecture on meta so preflight does not allocate model weights."""
    import torch
    from omegaconf import OmegaConf

    from f5_tts.infer.utils_infer import get_tokenizer
    from f5_tts.model import CFM, DiT

    cfg = OmegaConf.load(system["config_path"])
    family = SUPPORTED_MODEL_FAMILIES[system["model_family"]]
    if str(cfg.model.backbone) != family["backbone"]:
        raise ValueError(f"unsupported backbone for {system['model_family']}: {cfg.model.backbone}")
    arch = cfg.model.arch
    if bool(arch.text_mask_padding) != family["text_mask_padding"] or arch.get("pe_attn_head") != family["pe_attn_head"]:
        raise ValueError(f"config architecture does not match declared family {system['model_family']}")
    mel_contract = {key: getattr(cfg.model.mel_spec, key) for key in SUPPORTED_MEL_CONTRACT}
    if mel_contract != SUPPORTED_MEL_CONTRACT:
        raise ValueError(f"unsupported mel contract: {mel_contract}")
    _, vocab_size = get_tokenizer(system["vocab_path"], "custom")
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
    return model.state_dict()


def _strict_system_diagnostics(system: dict[str, Any], checks: list[dict[str, str]]) -> None:
    from f5_tts.model.checkpoint_init import validate_checkpoint_architecture
    from f5_tts.model.vocab_contract import validate_vocabulary_contract

    system_id = system["id"]
    assets = (
        ("config", "config_path", "config_sha256"),
        ("vocab", "vocab_path", "vocab_sha256"),
        ("vocab-contract", "vocab_contract_path", "vocab_contract_sha256"),
        ("checkpoint", "checkpoint", "checkpoint_sha256"),
    )
    asset_ok = True
    for label, path_field, hash_field in assets:
        path = Path(system[path_field])
        ok = path.is_file()
        detail = str(path)
        if ok and file_digest(path) != system[hash_field]:
            ok, detail = False, f"{path} hash changed after prepare"
        asset_ok &= ok
        checks.append({"name": f"{label}:{system_id}", "status": "ok" if ok else "error", "detail": detail})
    provenance_path = system.get("checkpoint_provenance_path")
    if provenance_path:
        provenance = Path(provenance_path)
        ok = provenance.is_file()
        detail = str(provenance)
        if ok:
            try:
                if system.get("checkpoint_provenance_sha256") and file_digest(provenance) != system["checkpoint_provenance_sha256"]:
                    raise ValueError("provenance hash changed after prepare")
                payload = json.loads(provenance.read_text(encoding="utf-8"))
                artifact = payload["artifact"]
                ok = artifact["checkpoint_sha256"] == system["checkpoint_sha256"]
                detail = f"{provenance}; artifact hash {'matches' if ok else 'mismatch'}"
            except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
                ok, detail = False, f"invalid provenance: {exc}"
        asset_ok &= ok
        checks.append({"name": f"checkpoint-provenance:{system_id}", "status": "ok" if ok else "error", "detail": detail})
    if not asset_ok:
        return
    try:
        tokens, contract = validate_vocabulary_contract(
            system["vocab_path"],
            system["vocab_contract_path"],
            expected_identity=system["expected_vocab_identity"],
        )
        if contract.token_sequence_sha256 != system["expected_token_sequence_sha256"]:
            raise ValueError(
                f"contract token hash={contract.token_sequence_sha256}, "
                f"declared={system['expected_token_sequence_sha256']}"
            )
        if contract.embedding_rows != system["expected_embedding_rows"]:
            raise ValueError(
                f"contract embedding_rows={contract.embedding_rows}, declared={system['expected_embedding_rows']}"
            )
        checks.append({"name": f"contract:{system_id}", "status": "ok", "detail": f"{contract.identity}; {len(tokens)} tokens"})
        details = validate_checkpoint_architecture(
            system["checkpoint"],
            system["weight_source"],
            _expected_architecture_state(system),
            expected_embedding_rows=system["expected_embedding_rows"],
            expected_embedding_width=system["expected_embedding_width"],
        )
        checks.append({"name": f"architecture:{system_id}", "status": "ok", "detail": json.dumps(details, sort_keys=True)})
    except Exception as exc:
        checks.append({"name": f"architecture:{system_id}", "status": "error", "detail": f"{type(exc).__name__}: {exc}"})


def _checkpoint_weight_keys(path: Path) -> tuple[bool, bool, str]:
    if path.suffix == ".safetensors":
        return True, False, "safetensors contains one weight set"
    try:
        import torch
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        return "model_state_dict" in checkpoint, "ema_model_state_dict" in checkpoint, "checkpoint keys inspected"
    except Exception as exc:
        return False, False, f"could not inspect checkpoint: {type(exc).__name__}: {exc}"


def diagnostics(spec_path: Path, spec: dict[str, Any], *, inspect_weights: bool = True) -> list[dict[str, str]]:
    checks: list[dict[str, str]] = []
    strict_contracts = spec["schema_version"] in {SCHEMA, SCHEMA_V2}
    common_assets = [("vocoder", spec["inference"]["vocoder_path"], "dir")]
    if not strict_contracts:
        common_assets.insert(0, ("vocab", spec["inference"]["vocab_path"], "file"))
    for name, raw, kind in common_assets:
        path = resolve_path(spec_path, raw)
        ok = path.is_file() if kind == "file" else path.is_dir()
        checks.append({"name": name, "status": "ok" if ok else "error", "detail": str(path)})
    inspected: dict[Path, tuple[bool, bool, str]] = {}
    for system in spec["systems"]:
        if strict_contracts:
            if inspect_weights:
                _strict_system_diagnostics(system, checks)
            continue
        path = resolve_path(spec_path, system["checkpoint"])
        ok = path.is_file()
        checks.append({"name": f"checkpoint:{system['id']}", "status": "ok" if ok else "error", "detail": str(path)})
        if ok and system.get("checkpoint_sha256") and file_digest(path) != system["checkpoint_sha256"]:
            checks.append({"name": f"hash:{system['id']}", "status": "error", "detail": "checkpoint hash changed after prepare"})
        if ok and inspect_weights:
            inspected.setdefault(path, _checkpoint_weight_keys(path))
            online, ema, detail = inspected[path]
            available = ema if system["weight_source"] == "ema" else online
            checks.append({"name": f"weights:{system['id']}", "status": "ok" if available else "error", "detail": detail})
    for reference in spec["references"]:
        path = resolve_path(spec_path, reference["audio"])
        ok = path.is_file() and (not reference.get("audio_sha256") or file_digest(path) == reference["audio_sha256"])
        checks.append({"name": f"reference:{reference['id']}", "status": "ok" if ok else "error", "detail": str(path)})
    required = int(spec["gates"]["hard"]["total_clips_required"])
    actual = len(jobs(spec))
    checks.append({"name": "case-count", "status": "ok" if actual == required else "error", "detail": f"{actual}/{required}"})
    return checks


def cmd_preflight(args: argparse.Namespace) -> int:
    path = Path(args.manifest).resolve()
    spec = load_manifest(path, prepared=True)
    checks = diagnostics(path, spec)
    for check in checks:
        print(f"[{check['status']}] {check['name']}: {check['detail']}")
    return 1 if any(x["status"] == "error" for x in checks) else 0


def _output_wav(run_dir: Path, job: dict[str, Any]) -> Path:
    return run_dir / job.get("wav", f"audio/{job['system_id']}/{job['text_id']}.wav")


def cmd_generate(args: argparse.Namespace) -> int:
    spec_path = Path(args.manifest).resolve()
    spec = load_manifest(spec_path, prepared=True)
    errors = [x for x in diagnostics(spec_path, spec) if x["status"] == "error"]
    if errors:
        raise ManifestError("preflight failed: " + "; ".join(x["detail"] for x in errors))
    run_dir, log = Path(args.run_dir).resolve(), Path(args.run_dir).resolve() / "provenance.jsonl"
    done = completed_ids(log, "generate")
    import soundfile as sf
    from f5_tts.infer.ja_model import clear_ja_model_cache, load_ja_model
    from f5_tts.infer.utils_infer import infer_process, preprocess_ref_audio_text

    generation_keys = {"nfe_step", "cfg_strength", "sway_sampling_coef", "speed", "cross_fade_duration"}
    infer_args = {k: v for k, v in spec["inference"].items() if k in generation_keys}
    device_arg = None if spec["generation"].get("device", "auto") == "auto" else spec["generation"]["device"]
    all_jobs = jobs(spec)
    requested_systems = set(args.systems.split(",")) if getattr(args, "systems", None) else None
    if requested_systems:
        unknown = requested_systems - {system["id"] for system in spec["systems"]}
        if unknown:
            raise ManifestError(f"unknown generation systems: {sorted(unknown)}")
    for system in spec["systems"]:  # one model at a time
        if requested_systems and system["id"] not in requested_systems:
            continue
        system_jobs = [x for x in all_jobs if x["system_id"] == system["id"]]
        try:
            model, vocoder, device = load_ja_model(
                system["checkpoint"],
                system.get("vocab_path", spec["inference"].get("vocab_path")),
                spec["inference"]["vocoder_path"],
                use_ema=system["weight_source"] == "ema",
                device=device_arg,
                config_path=system.get("config_path"),
                vocab_contract_path=system.get("vocab_contract_path"),
                model_family=system.get("model_family", "legacy_jmica"),
                expected_vocab_identity=system.get("expected_vocab_identity"),
                expected_token_sequence_sha256=system.get("expected_token_sequence_sha256"),
                expected_embedding_rows=system.get("expected_embedding_rows"),
                expected_embedding_width=system.get("expected_embedding_width"),
            )
            for job in system_jobs:
                wav = _output_wav(run_dir, job)
                if job["job_id"] in done and wav.is_file():
                    continue
                started = utc_now()
                try:
                    ref_audio, ref_text_processed = preprocess_ref_audio_text(job["reference"]["audio"], job["reference"]["text_kana"])
                    audio, sample_rate, _ = infer_process(ref_audio, ref_text_processed, job["text"]["text_kana"], model, vocoder, device=device, seed=job["seed"], text_frontend=lambda texts: [list(x) for x in texts], **infer_args)
                    # Vocos 输出偶尔超过 [-1, 1]。直接写 PCM16 会硬削波且不可逆，
                    # 先对每条结果做纯增益峰值归一化；它不改变音色、时长或相对动态。
                    import numpy as np

                    raw_audio = np.asarray(audio)
                    raw_has_nan_inf = not bool(np.isfinite(raw_audio).all())
                    finite_audio = np.nan_to_num(raw_audio, nan=0.0, posinf=1.0, neginf=-1.0)
                    peak_before = float(np.max(np.abs(finite_audio))) if finite_audio.size else 0.0
                    prewrite_overshoot_fraction = (
                        float(np.mean(np.abs(finite_audio) > 1.0)) if finite_audio.size else 0.0
                    )
                    normalization_gain = min(1.0, 0.95 / peak_before) if peak_before > 0 else 1.0
                    if normalization_gain < 1.0:
                        audio = raw_audio * normalization_gain
                    wav.parent.mkdir(parents=True, exist_ok=True)
                    sf.write(wav, audio, sample_rate, subtype="PCM_16")
                    stored_audio, stored_sample_rate = sf.read(wav, always_2d=True, dtype="float32")
                    stored_peak = float(np.max(np.abs(stored_audio))) if stored_audio.size else 0.0
                    stored_clipping_fraction = (
                        float(np.mean(np.abs(stored_audio) >= 0.999)) if stored_audio.size else 0.0
                    )
                    wav_sha256 = file_digest(wav)
                    append_jsonl(log, {"stage": "generate", "status": "ok", "job_id": job["job_id"], "system_id": job["system_id"], "text_id": job["text_id"], "reference_id": job["reference_id"], "seed": job["seed"], "wav": str(wav), "wav_sha256": wav_sha256, "sample_rate": sample_rate, "prewrite_peak": peak_before, "prewrite_overshoot_fraction": prewrite_overshoot_fraction, "prewrite_has_nan_inf": raw_has_nan_inf, "normalization_gain": normalization_gain, "stored_sample_rate": int(stored_sample_rate), "stored_peak": stored_peak, "stored_clipping_fraction": stored_clipping_fraction, "started_at": started, "finished_at": utc_now(), "manifest_sha256": canonical_hash(spec), "checkpoint": str(Path(system["checkpoint"]).resolve()), "checkpoint_sha256": system["checkpoint_sha256"], "loaded_model_state_sha256": getattr(model, "_evaluation_state_sha256", None), "weight_source": system["weight_source"]})
                except Exception as exc:
                    append_jsonl(log, {"stage": "generate", "status": "error", "job_id": job["job_id"], "system_id": job["system_id"], "text_id": job["text_id"], "started_at": started, "finished_at": utc_now(), "error_type": type(exc).__name__, "error": str(exc)})
                    if not args.keep_going:
                        return 1
        finally:
            if "model" in locals():
                del model
            clear_ja_model_cache()
    return 0


def _hard_gate(diag: dict[str, Any], hard: dict[str, Any]) -> list[str]:
    expected = hard["format"]
    tests = [
        (diag["sample_rate"] == expected["sample_rate"], "sample_rate"),
        (diag["channels"] == expected["channels"], "channels"),
        (diag["clipping_fraction"] <= hard["max_clipping_fraction"], "clipping"),
        (hard["min_duration_seconds"] <= diag["duration_seconds"] <= hard["max_duration_seconds"], "duration"),
        (diag["silence_fraction"] <= hard["max_silence_fraction"], "silence"),
        (hard["allow_nan_inf"] or not diag["has_nan_inf"], "nan_inf"),
    ]
    raw_fields = {
        "prewrite_overshoot_fraction": (
            "max_prewrite_overshoot_fraction",
            lambda value: value <= hard["max_prewrite_overshoot_fraction"],
            "raw_overshoot",
        ),
        "normalization_gain": (
            "min_normalization_gain",
            lambda value: value >= hard["min_normalization_gain"],
            "normalization_gain",
        ),
        "stored_clipping_fraction": (
            "max_stored_clipping_fraction",
            lambda value: value <= hard["max_stored_clipping_fraction"],
            "stored_clipping",
        ),
        "prewrite_has_nan_inf": (
            "allow_prewrite_nan_inf",
            lambda value: hard["allow_prewrite_nan_inf"] or not value,
            "prewrite_nan_inf",
        ),
    }
    for field, (gate_field, predicate, failure) in raw_fields.items():
        if gate_field in hard:
            tests.append((field in diag and predicate(diag[field]), failure))
    return [name for ok, name in tests if not ok]


def cmd_score(args: argparse.Namespace) -> int:
    spec = load_manifest(Path(args.manifest).resolve(), prepared=True)
    run_dir, log = Path(args.run_dir).resolve(), Path(args.run_dir).resolve() / "provenance.jsonl"
    done = completed_ids(log, "score")
    latest_generate: dict[str, dict[str, Any]] = {}
    for record in read_jsonl(log):
        if record.get("stage") == "generate" and record.get("job_id"):
            latest_generate[record["job_id"]] = record
    from tools import eval_protocol
    metrics = tuple(x.strip() for x in args.metrics.split(",") if x.strip())
    score_cache: dict[tuple[str, tuple[str, ...], str, str, str], dict[str, Any]] = {}
    for job in jobs(spec):
        if job["job_id"] in done:
            continue
        wav = _output_wav(run_dir, job)
        if not wav.is_file():
            continue
        row = {"stage": "score", "status": "ok", "job_id": job["job_id"], "system_id": job["system_id"], "text_id": job["text_id"], "split": job["text"]["split"], "wav": str(wav), "finished_at": utc_now()}
        try:
            wav_sha256 = file_digest(wav)
            row["wav_sha256"] = wav_sha256
            cache_key = (
                wav_sha256,
                metrics,
                job["text"]["text_kana"],
                file_digest(Path(job["reference"]["audio"])),
                str(job["reference"]["audio"]),
            )
            if cache_key not in score_cache:
                score_cache[cache_key] = eval_protocol.score_audio(str(wav), reference_kana=job["text"]["text_kana"], speaker_reference=job["reference"]["audio"], emotion_reference=job["reference"]["audio"], metrics=metrics)
            values = dict(score_cache[cache_key])
            row.update(values)
            generation = latest_generate.get(job["job_id"])
            raw_metric_fields = (
                "prewrite_peak",
                "prewrite_overshoot_fraction",
                "prewrite_has_nan_inf",
                "normalization_gain",
                "stored_peak",
                "stored_clipping_fraction",
            )
            if generation and generation.get("status") == "ok":
                row.update({field: generation[field] for field in raw_metric_fields if field in generation})
            row["hard_gate_failures"] = _hard_gate(row, spec["gates"]["hard"])
        except Exception as exc:
            row.update(status="error", error_type=type(exc).__name__, error=str(exc))
        append_jsonl(log, row)
        if row["status"] == "error" and not args.keep_going:
            return 1
    return 0


def _quantile(values: list[float], fraction: float) -> float:
    values = sorted(values)
    if not values:
        return math.nan
    return values[round((len(values) - 1) * fraction)]


def _system_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"count": len(rows), "hard_failures": sum(bool(x.get("hard_gate_failures")) for x in rows)}
    for metric in ("cer", "speaker_sim", "emotion_sim", "utmos"):
        values = [float(x[metric]) for x in rows if metric in x]
        if values:
            summary[metric] = {"mean": mean(values), "median": median(values), "p10": _quantile(values, 0.1), "p90": _quantile(values, 0.9), "max": max(values)}
    core = [x for x in rows if x.get("split") == "core" and "cer" in x]
    if core:
        values = [float(x["cer"]) for x in core]
        summary["core_cer"] = {"median": median(values), "p90": _quantile(values, 0.9), "max": max(values)}
    return summary


def _objective_verdict(summary: dict[str, Any], baseline: dict[str, Any], gates: dict[str, float]) -> tuple[bool | None, list[dict[str, Any]]]:
    checks = []
    definitions = [("core_median_cer_delta", summary.get("core_cer", {}).get("median"), baseline.get("core_cer", {}).get("median"), "<=", gates["max_core_median_cer_delta_vs_baseline"]), ("core_p90_cer_delta", summary.get("core_cer", {}).get("p90"), baseline.get("core_cer", {}).get("p90"), "<=", gates["max_core_p90_cer_delta_vs_baseline"]), ("catastrophic_cer", summary.get("cer", {}).get("max"), 0.0, "<=", gates["max_catastrophic_cer_per_text"]), ("speaker_median_delta", summary.get("speaker_sim", {}).get("median"), baseline.get("speaker_sim", {}).get("median"), ">=", gates["min_speaker_similarity_delta_vs_baseline_median"]), ("speaker_p10_delta", summary.get("speaker_sim", {}).get("p10"), baseline.get("speaker_sim", {}).get("p10"), ">=", gates["min_speaker_similarity_delta_vs_baseline_p10"]), ("utmos_mean_delta", summary.get("utmos", {}).get("mean"), baseline.get("utmos", {}).get("mean"), ">=", gates["max_utmos_delta_vs_baseline_mean"])]
    for name, actual, base, op, limit in definitions:
        if actual is None or base is None:
            checks.append({"name": name, "passed": None, "reason": "metric unavailable"})
        else:
            delta = actual - base
            checks.append({"name": name, "actual": delta, "limit": limit, "passed": delta <= limit if op == "<=" else delta >= limit})
    evaluated = [x for x in checks if x["passed"] is not None]
    return (all(x["passed"] for x in evaluated) if len(evaluated) == len(checks) else None), checks


def _pareto(rows: list[dict[str, Any]]) -> list[str]:
    candidates = [x for x in rows if x["passed"]]
    objectives = [("core_cer", "median", False), ("speaker_sim", "median", True), ("utmos", "mean", True)]
    result = []
    for candidate in candidates:
        dominated = False
        for other in candidates:
            if other is candidate:
                continue
            comparable, no_worse, better = False, True, False
            for metric, stat, maximize in objectives:
                a = candidate["summary"].get(metric, {}).get(stat)
                b = other["summary"].get(metric, {}).get(stat)
                if a is None or b is None:
                    continue
                comparable = True
                no_worse &= b >= a if maximize else b <= a
                better |= b > a if maximize else b < a
            if comparable and no_worse and better:
                dominated = True
                break
        if not dominated:
            result.append(candidate["system_id"])
    return result


def cmd_select(args: argparse.Namespace) -> int:
    spec = load_manifest(Path(args.manifest).resolve(), prepared=True)
    run_dir = Path(args.run_dir).resolve()
    scores = [x for x in read_jsonl(run_dir / "provenance.jsonl") if x.get("stage") == "score" and x.get("status") == "ok"]
    expected = len(spec["texts"])
    by_system = {system["id"]: [x for x in scores if x["system_id"] == system["id"]] for system in spec["systems"]}
    summaries = {key: _system_summary(value) for key, value in by_system.items()}
    baseline_id = next(x["id"] for x in spec["systems"] if x["baseline"])
    ranking = []
    for system in spec["systems"]:
        summary = summaries[system["id"]]
        objective_ok, verdicts = _objective_verdict(summary, summaries[baseline_id], spec["gates"]["objective"])
        complete = summary["count"] == expected
        metrics_complete = objective_ok is not None
        passed = complete and summary["hard_failures"] == 0 and (system["baseline"] or objective_ok is True)
        result_status = "complete" if complete and (system["baseline"] or metrics_complete) else "incomplete/provisional"
        ranking.append({"system_id": system["id"], "complete": complete, "metrics_complete": metrics_complete, "result_status": result_status, "passed": passed, "summary": summary, "objective_gates": verdicts})
    shortlist = _pareto(ranking)[: args.limit]
    output = {"generated_at": utc_now(), "baseline": baseline_id, "ranking": ranking, "pareto_shortlist": shortlist, "selected": shortlist}
    (run_dir / "selection.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not getattr(args, "quiet", False):
        print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0 if shortlist else 2


def blind_code(secret: str, job_id: str) -> str:
    return "Y" + hashlib.sha256(f"{secret}\0{job_id}".encode()).hexdigest()[:10].upper()


_LISTENING_HTML = r'''<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Yanase Blind Listening</title>
<style>
:root{color-scheme:light dark;--bg:#eef3f8;--panel:#fff;--ink:#152333;--muted:#617284;--line:#cbd6e2;--accent:#087ea4;--soft:#e7f5fa;--warn:#a85d00}
@media(prefers-color-scheme:dark){:root{--bg:#111820;--panel:#19232d;--ink:#ecf3f8;--muted:#9fb0bf;--line:#354554;--accent:#56c4e8;--soft:#123442;--warn:#f0aa52}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 system-ui,"Yu Gothic UI","Noto Sans JP",sans-serif}.shell{max-width:920px;margin:auto;padding:24px}.top{position:sticky;top:0;z-index:2;background:color-mix(in srgb,var(--bg) 92%,transparent);backdrop-filter:blur(10px);padding:14px 0}.progress{height:8px;background:var(--line);border-radius:8px;overflow:hidden}.progress>i{display:block;height:100%;background:var(--accent);width:0}.meta{display:flex;justify-content:space-between;color:var(--muted);font-size:.9rem;margin:6px 0}.card{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:24px;box-shadow:0 8px 28px #00000012}.tag{display:inline-block;background:var(--soft);color:var(--accent);padding:3px 9px;border-radius:999px;font-size:.8rem;font-weight:700}.sentence{font-size:1.35rem;margin:18px 0;text-wrap:balance}.players{display:grid;gap:12px;margin-bottom:20px}.player label{display:block;color:var(--muted);font-size:.82rem}.player audio{width:100%}.rating{border-top:1px solid var(--line);padding:14px 0}.rating legend{font-weight:700;margin-bottom:8px}.scale{display:grid;grid-template-columns:repeat(5,1fr);gap:8px}.scale label{text-align:center;border:1px solid var(--line);padding:8px 4px;border-radius:8px;cursor:pointer}.scale label:has(input:checked){border-color:var(--accent);background:var(--soft);color:var(--accent);font-weight:700}.flags{display:flex;flex-wrap:wrap;gap:8px}.flags label{border:1px solid var(--line);border-radius:999px;padding:5px 10px;cursor:pointer}.flags label:has(input:checked){border-color:var(--warn);color:var(--warn)}input[type=radio],input[type=checkbox]{accent-color:var(--accent)}textarea{width:100%;min-height:68px;background:transparent;color:var(--ink);border:1px solid var(--line);border-radius:8px;padding:8px}.actions{display:flex;justify-content:space-between;gap:12px;margin-top:18px}button{border:1px solid var(--line);background:var(--panel);color:var(--ink);padding:10px 16px;border-radius:9px;cursor:pointer}button.primary{background:var(--accent);border-color:var(--accent);color:#fff}button:disabled{opacity:.45}.done{text-align:center;padding:60px 10px}.hidden{display:none}@media(max-width:600px){.shell{padding:12px}.card{padding:16px}.sentence{font-size:1.15rem}}
</style>
<div class="shell">
 <div class="top"><h1>Yanase Blind Listening</h1><div class="progress"><i id="bar"></i></div><div class="meta"><span id="count"></span><span>匿名・ローカル保存</span></div></div>
 <main id="card" class="card"></main>
 <section id="done" class="card done hidden"><h2>評価完了</h2><p>結果を JSON または CSV で保存してください。</p><button onclick="download('json')">JSON を保存</button> <button onclick="download('csv')">CSV を保存</button></section>
</div>
<script>
const clips=__CLIPS_JSON__, storeKey='yanase-blind-v2', state=JSON.parse(localStorage.getItem(storeKey)||'{"index":0,"ratings":{}}');
const dims=[['naturalness','自然度'],['intelligibility','発音・可懂度'],['speaker','Yanaseらしさ'],['emotion','感情適合度']];
const flags=[['truncation','吞字/截断'],['repetition','重复'],['misread','错读'],['noise','噪声'],['pitch','音高不稳'],['rhythm','节奏异常']];
function esc(s){return String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
function save(){localStorage.setItem(storeKey,JSON.stringify(state));document.querySelector('#bar').style.width=(100*state.index/clips.length)+'%';document.querySelector('#count').textContent=`${Math.min(state.index+1,clips.length)} / ${clips.length}`}
function render(){save();if(state.index>=clips.length){card.classList.add('hidden');done.classList.remove('hidden');return}const c=clips[state.index],r=state.ratings[c.code]||{flags:[]};let html=`<span class="tag">${esc(c.category)}</span><div class="sentence">${esc(c.text)}</div><div class="players"><div class="player"><label>評価対象</label><audio controls preload="none" src="${esc(c.audio)}"></audio></div><div class="player"><label>声線アンカー（参考）</label><audio controls preload="none" src="${esc(c.reference)}"></audio></div></div>`;for(const [key,label] of dims){if(key==='emotion'&&!c.rate_emotion)continue;html+=`<fieldset class="rating"><legend>${label}</legend><div class="scale">`;for(let n=1;n<=5;n++)html+=`<label><input type="radio" name="${key}" value="${n}" ${r[key]==n?'checked':''}> ${n}</label>`;html+='</div></fieldset>'}html+='<fieldset class="rating"><legend>問題フラグ（任意）</legend><div class="flags">';for(const [key,label] of flags)html+=`<label><input type="checkbox" name="flag" value="${key}" ${r.flags?.includes(key)?'checked':''}> ${label}</label>`;html+=`</div></fieldset><label>メモ<textarea id="note">${esc(r.note||'')}</textarea></label><div class="actions"><button ${state.index===0?'disabled':''} onclick="prev()">戻る</button><button onclick="download('json')">途中保存</button><button class="primary" onclick="next()">${state.index===clips.length-1?'完了':'次へ'}</button></div>`;card.innerHTML=html}
function collect(){const c=clips[state.index],r=state.ratings[c.code]||{};for(const [key] of dims){const el=document.querySelector(`input[name=${key}]:checked`);if(el)r[key]=+el.value}r.flags=[...document.querySelectorAll('input[name=flag]:checked')].map(x=>x.value);r.note=document.querySelector('#note')?.value||'';r.code=c.code;r.text_id=c.text_id;r.category=c.category;state.ratings[c.code]=r}
function next(){collect();const c=clips[state.index],r=state.ratings[c.code];if(!r.naturalness||!r.intelligibility||!r.speaker||(c.rate_emotion&&!r.emotion)){alert('必須項目をすべて評価してください。');return}state.index++;render()}
function prev(){collect();state.index=Math.max(0,state.index-1);render()}
function download(type){if(state.index<clips.length)collect();const rows=Object.values(state.ratings),data=type==='json'?JSON.stringify({exported_at:new Date().toISOString(),ratings:rows},null,2):['code,text_id,category,naturalness,intelligibility,speaker,emotion,flags,note',...rows.map(r=>[r.code,r.text_id,r.category,r.naturalness||'',r.intelligibility||'',r.speaker||'',r.emotion||'',(r.flags||[]).join(';'),JSON.stringify(r.note||'')].join(','))].join('\n');const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([data],{type:type==='json'?'application/json':'text/csv'}));a.download=`yanase_ratings.${type}`;a.click();URL.revokeObjectURL(a.href)}
render();
</script>'''


def _balanced_blind_order(items: list[dict[str, Any]], secret: str) -> list[dict[str, Any]]:
    """确定性打散，避免同一句或同一系统连续出现。"""
    rng = random.Random(int(hashlib.sha256(secret.encode()).hexdigest(), 16))
    pool = list(items)
    rng.shuffle(pool)
    ordered = []
    while pool:
        candidates = [
            item
            for item in pool
            if (not ordered or item["text_id"] != ordered[-1]["text_id"])
            and (
                len(ordered) < 2
                or item["system_id"] != ordered[-1]["system_id"]
                or item["system_id"] != ordered[-2]["system_id"]
            )
        ]
        choices = candidates or pool
        selected = rng.choice(choices)
        pool.remove(selected)
        ordered.append(selected)
    return ordered


def cmd_blind(args: argparse.Namespace) -> int:
    spec = load_manifest(Path(args.manifest).resolve(), prepared=True)
    run_dir = Path(args.run_dir).resolve()
    bundle = run_dir / args.bundle_name
    audio_dir, reference_dir, private_dir = bundle / "audio", bundle / "references", run_dir / "private"
    for directory in (audio_dir, reference_dir, private_dir):
        directory.mkdir(parents=True, exist_ok=True)

    selected_systems = set(args.systems.split(",")) if args.systems else {x["id"] for x in spec["systems"]}
    known_systems = {x["id"] for x in spec["systems"]}
    unknown = selected_systems - known_systems
    if unknown:
        raise ManifestError(f"unknown blind systems: {sorted(unknown)}")

    secret = args.secret or f"{random.SystemRandom().getrandbits(256):064x}"
    text_map = {x["id"]: x for x in spec["texts"]}
    reference_map = {x["id"]: x for x in spec["references"]}
    mapping = []
    public = []
    for job in jobs(spec):
        text = text_map[job["text_id"]]
        if job["system_id"] not in selected_systems or (args.core_only and not text.get("human_listening")):
            continue
        source = _output_wav(run_dir, job)
        if not source.is_file():
            raise ManifestError(f"blind source missing: {source}")
        code = blind_code(secret, job["job_id"])
        target = audio_dir / f"{code}.wav"
        shutil.copyfile(source, target)
        mapping.append(
            {
                "code": code,
                "job_id": job["job_id"],
                "system_id": job["system_id"],
                "text_id": job["text_id"],
                "reference_id": job["reference_id"],
            }
        )
        public.append(
            {
                "code": code,
                "text_id": job["text_id"],
                "category": text["category"],
                "text": text["text_original"],
                "audio": f"audio/{target.name}",
                "reference": f"references/{job['reference_id']}.wav",
                "rate_emotion": text["category"].startswith("emotion"),
            }
        )

    for reference_id in {x["reference_id"] for x in mapping}:
        source = Path(reference_map[reference_id]["audio"])
        if not source.is_file():
            raise ManifestError(f"blind reference missing: {source}")
        shutil.copyfile(source, reference_dir / f"{reference_id}.wav")

    public_by_code = {x["code"]: x for x in public}
    mapping = _balanced_blind_order(mapping, secret)
    public = [public_by_code[x["code"]] for x in mapping]
    key_path = private_dir / f"{args.bundle_name}_key.json"
    key_path.write_text(
        json.dumps({"secret": secret, "systems": sorted(selected_systems), "mapping": mapping}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    (bundle / "manifest.json").write_text(
        json.dumps({"schema": "yanase-blind-listening-v2", "clips": public}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    manifest_json = json.dumps(public, ensure_ascii=False).replace("</", "<\\/")
    page = _LISTENING_HTML.replace("__CLIPS_JSON__", manifest_json)
    (bundle / "index.html").write_text(page, encoding="utf-8")
    print(f"wrote {len(mapping)} clips to {bundle}; keep {key_path} secret")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir).resolve()
    selection = json.loads((run_dir / "selection.json").read_text(encoding="utf-8"))
    summary = {"generated_at": utc_now(), "baseline": selection["baseline"], "selected": selection["selected"], "systems": selection["ranking"]}
    (run_dir / "report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    rows = "".join(f"<tr><td>{html.escape(x['system_id'])}</td><td>{'PASS' if x['passed'] else 'FAIL'}</td><td>{x['summary']['count']}</td><td><pre>{html.escape(json.dumps(x['summary'], ensure_ascii=False, indent=2))}</pre></td></tr>" for x in selection["ranking"])
    page = f"<!doctype html><meta charset='utf-8'><title>Yanase Checkpoint Report</title><style>body{{font:15px system-ui;max-width:1100px;margin:auto;padding:2rem}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #aaa;padding:.5rem;text-align:left}}pre{{white-space:pre-wrap}}</style><h1>Yanase checkpoint report</h1><p>Baseline: {html.escape(selection['baseline'])}; Pareto shortlist: {html.escape(', '.join(selection['selected']) or 'none')}</p><table><tr><th>System</th><th>Gates</th><th>Clips</th><th>Summary</th></tr>{rows}</table>"
    (run_dir / "report.html").write_text(page, encoding="utf-8")
    print(f"wrote {run_dir / 'report.json'} and {run_dir / 'report.html'}")
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare"); p.add_argument("manifest"); p.add_argument("output"); p.add_argument("--reference", action="append", help="ID::AUDIO::TEXT::DATA_RELATION"); p.add_argument("--allow-leakage", action="store_true"); p.set_defaults(func=cmd_prepare)
    p = commands.add_parser("preflight"); p.add_argument("manifest"); p.set_defaults(func=cmd_preflight)
    p = commands.add_parser("generate"); p.add_argument("manifest"); p.add_argument("run_dir"); p.add_argument("--keep-going", action="store_true"); p.add_argument("--systems", help="comma-separated systems to generate into this run directory"); p.set_defaults(func=cmd_generate)
    p = commands.add_parser("score"); p.add_argument("manifest"); p.add_argument("run_dir"); p.add_argument("--metrics", default="cer,speaker_sim,emotion_sim"); p.add_argument("--keep-going", action="store_true"); p.set_defaults(func=cmd_score)
    p = commands.add_parser("select"); p.add_argument("manifest"); p.add_argument("run_dir"); p.add_argument("--limit", type=int, default=3); p.set_defaults(func=cmd_select)
    p = commands.add_parser("blind"); p.add_argument("manifest"); p.add_argument("run_dir"); p.add_argument("--secret"); p.add_argument("--systems", help="comma-separated shortlist"); p.add_argument("--core-only", action="store_true"); p.add_argument("--bundle-name", default="blind"); p.set_defaults(func=cmd_blind)
    p = commands.add_parser("report"); p.add_argument("run_dir"); p.set_defaults(func=cmd_report)
    return root


def main(argv: list[str] | None = None) -> int:
    try:
        args = parser().parse_args(argv)
        return args.func(args)
    except ManifestError as exc:
        print(f"manifest error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
