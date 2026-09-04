#!/usr/bin/env python
"""Re-score CER clips whose G2P rejected the ASR hypothesis, instead of dropping them.

Why not just drop them: the failures are not randomly distributed. Five of the
seven land on one `core` text across exactly the late checkpoints
(3000..5000), and the objective gates compute `core` median/p90 — so dropping
them would compare 9 texts against 10 and would quietly flatter precisely the
checkpoints under evaluation.

Two distinct causes, handled differently:

* `input contains U+FFFD` (6 clips) — the ASR tokenizer emitted a replacement
  character, in every observed case at position 0, with coherent Japanese after
  it. That is a decode artifact, not unusable audio, so the character is
  stripped and the clip is scored with `asr_replacement_chars` recording how
  many were removed. One stripped character out of ~35 moves CER by ~0.03 and
  cannot flip a conclusion; losing the clip entirely would.
* `G2P output failed strict validation` (1 clip) — pyopenjtalk rejected the
  hypothesis itself. Nothing to sanitize, so it stays failed and is reported as
  such rather than silently patched.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

REPLACEMENT = "�"


def sanitize(text: str) -> tuple[str, int]:
    cleaned = text.replace(REPLACEMENT, "")
    return cleaned.strip(), text.count(REPLACEMENT)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--cer-jsonl", required=True, help="rewritten in place")
    args = parser.parse_args(argv)

    prepared = json.loads(Path(args.prepared).read_text(encoding="utf-8"))
    text_kana = {t["id"]: t["text_kana"] for t in prepared["texts"]}

    path = Path(args.cer_jsonl)
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    failed = [r for r in records if r.get("status") != "ok"]
    print(f"failed clips: {len(failed)}")
    if not failed:
        return 0

    from tools.eval_protocol import cer, normalize_kana

    repaired = 0
    still_failed = 0
    for record in failed:
        hypothesis = record.get("asr_text")
        if not hypothesis:
            still_failed += 1
            continue
        cleaned, removed = sanitize(hypothesis)
        if not cleaned:
            still_failed += 1
            continue
        reference = text_kana.get(record["text_id"])
        if reference is None:
            still_failed += 1
            continue
        try:
            value = cer(normalize_kana(cleaned), normalize_kana(reference, convert=False))
        except Exception as error:  # noqa: BLE001
            record["repair_error_type"] = type(error).__name__
            record["repair_error"] = str(error)[:200]
            still_failed += 1
            print(f"  still failing: {record['clip_id']}  ({type(error).__name__})")
            continue
        record["cer"] = value
        record["status"] = "ok"
        record["asr_text_sanitized"] = cleaned
        record["asr_replacement_chars"] = removed
        record["cer_repaired"] = True
        record.pop("error", None)
        record.pop("error_type", None)
        repaired += 1
        print(f"  repaired: {record['clip_id']}  cer={value:.3f}  stripped={removed}")

    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".jsonl")
    with open(handle, "w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    Path(temporary).replace(path)

    total_ok = sum(1 for r in records if r.get("status") == "ok")
    print(f"\nrepaired={repaired}  still_failed={still_failed}  total_ok={total_ok}/{len(records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
