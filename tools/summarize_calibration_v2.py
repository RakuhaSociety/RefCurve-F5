#!/usr/bin/env python
"""Merge remote embedding scores with locally-computed VNW CER and summarise per system.

Two sources, deliberately:

* Remote `provenance.jsonl` supplies `speaker_sim` / `emotion_sim`, computed on the
  training host with CamPlus and emotion2vec.
* Local `cer_vnw.jsonl` supplies CER, because the remote cannot load any
  transformers Whisper model (transformers 5.15.1 vs torch 2.4.0 -> PyTorch
  backend disabled).

The remote provenance ALSO contains a `cer` field from an earlier paraformer-zh
attempt. Those values are all exactly 1.0 because paraformer-zh is a Mandarin
model and cannot transcribe Japanese; they are dropped here rather than merged,
and the local VNW values are used instead.

On the spec's objective gates: they were written for "does fine-tuning regress
against a competent baseline". Here the baseline is the update-0 initialisation,
which has never seen Japanese, so a CER-delta-vs-baseline gate is close to
vacuous — any model that learned anything scores a large negative delta and
passes trivially. This script therefore reports absolute CER per checkpoint as
the primary signal and labels each gate as informative or vacuous instead of
printing a bare PASS.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import median


def quantile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--remote-provenance", required=True)
    parser.add_argument("--local-cer", required=True)
    parser.add_argument("--out-json", default=None)
    args = parser.parse_args(argv)

    spec = json.loads(Path(args.prepared).read_text(encoding="utf-8"))
    gates = spec["gates"]["objective"]
    baseline_id = next(s["id"] for s in spec["systems"] if s.get("baseline"))
    split_of = {t["id"]: t.get("split") for t in spec["texts"]}
    ordered_ids = [s["id"] for s in spec["systems"]]

    # 远端只取 embedding 指标，明确丢弃其 cer（paraformer-zh 无效值）
    embedding: dict[tuple[str, str], dict] = {}
    for record in read_jsonl(Path(args.remote_provenance)):
        if record.get("stage") != "score" or record.get("status") != "ok":
            continue
        key = (record.get("system_id"), record.get("text_id"))
        entry = embedding.setdefault(key, {})
        for field in ("speaker_sim", "emotion_sim"):
            if field in record:
                entry[field] = float(record[field])

    local_cer: dict[tuple[str, str], float] = {}
    cer_failures = 0
    for record in read_jsonl(Path(args.local_cer)):
        if record.get("status") != "ok" or "cer" not in record:
            cer_failures += 1
            continue
        local_cer[(record["system_id"], record["text_id"])] = float(record["cer"])

    per_system: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"cer_all": [], "cer_core": [], "spk": [], "emo": []}
    )
    for (system_id, text_id), value in local_cer.items():
        bucket = per_system[system_id]
        bucket["cer_all"].append(value)
        if split_of.get(text_id) == "core":
            bucket["cer_core"].append(value)
    for (system_id, text_id), values in embedding.items():
        bucket = per_system[system_id]
        if "speaker_sim" in values:
            bucket["spk"].append(values["speaker_sim"])
        if "emotion_sim" in values:
            bucket["emo"].append(values["emotion_sim"])

    def summarise(system_id: str) -> dict:
        bucket = per_system.get(system_id, {})
        core = bucket.get("cer_core") or []
        every = bucket.get("cer_all") or []
        return {
            "core_cer_median": median(core) if core else None,
            "core_cer_p90": quantile(core, 0.9),
            "cer_max": max(every) if every else None,
            "cer_median_all": median(every) if every else None,
            "speaker_median": median(bucket["spk"]) if bucket.get("spk") else None,
            "speaker_p10": quantile(bucket.get("spk") or [], 0.1),
            "emotion_median": median(bucket["emo"]) if bucket.get("emo") else None,
            "n_cer": len(every),
            "n_embedding": len(bucket.get("spk") or []),
        }

    summaries = {system_id: summarise(system_id) for system_id in ordered_ids}
    base = summaries[baseline_id]

    print(f"CER source: local Visual-novel-whisper   (failed clips: {cer_failures})")
    print(f"embedding source: remote CamPlus + emotion2vec")
    print(f"baseline: {baseline_id}\n")

    header = (
        f"{'system':<22}{'coreCER':>9}{'p90':>8}{'maxCER':>8}"
        f"{'SPK':>8}{'ΔSPK':>8}{'EMO':>8}{'ΔEMO':>8}{'n':>5}"
    )
    print(header)
    print("-" * len(header))
    for system_id in ordered_ids:
        s = summaries[system_id]
        label = system_id.replace("corrected_calibration_", "step").replace("_ema", "")
        label = {"v1_ja_update0": "update-0 (baseline)", "jmica_anchor": "Jmica anchor"}.get(label, label)

        def num(value, digits=3):
            return f"{value:.{digits}f}" if value is not None else "-"

        def delta(value, reference):
            if value is None or reference is None:
                return "-"
            return f"{value - reference:+.3f}"

        print(
            f"{label:<22}{num(s['core_cer_median']):>9}{num(s['core_cer_p90']):>8}{num(s['cer_max']):>8}"
            f"{num(s['speaker_median']):>8}{delta(s['speaker_median'], base['speaker_median']):>8}"
            f"{num(s['emotion_median']):>8}{delta(s['emotion_median'], base['emotion_median']):>8}"
            f"{s['n_cer']:>5}"
        )

    print("\n=== 门禁判定（诚实标注适用性）===")
    print(f"baseline core CER median = {base['core_cer_median']:.3f}" if base["core_cer_median"] is not None else "-")
    print(
        "  注意：baseline 是 update-0 初始化（从未见过日文），CER≈1。"
        "\n  因此 *_delta_vs_baseline 的 CER 门禁在本次评测中近乎失效——"
        "\n  任何学到东西的 checkpoint 都得到大负 delta 而『通过』，不具判别力。"
        "\n  真正有判别力的是绝对 CER 与 catastrophic 门禁（对 0 比较，非对 baseline）。"
    )
    print()
    catastrophic_limit = gates["max_catastrophic_cer_per_text"]
    for system_id in ordered_ids:
        s = summaries[system_id]
        if s["cer_max"] is None:
            continue
        label = system_id.replace("corrected_calibration_", "step").replace("_ema", "")
        verdict = "PASS" if s["cer_max"] <= catastrophic_limit else "FAIL"
        print(f"  catastrophic_cer (max<={catastrophic_limit}): {label:<20} max={s['cer_max']:.3f}  {verdict}")

    print(f"\n  utmos 门禁：无法评估（本次未计算 UTMOS）")
    print(f"  emotion_sim：spec 未定义门禁，仅作观察指标")

    if args.out_json:
        Path(args.out_json).write_text(
            json.dumps(
                {
                    "baseline_id": baseline_id,
                    "cer_source": "local_visual_novel_whisper",
                    "embedding_source": "remote_campplus_emotion2vec",
                    "cer_failures": cer_failures,
                    "gates": gates,
                    "summaries": summaries,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"\nwrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
