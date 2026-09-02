from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "src" / "f5_tts" / "train" / "datasets" / "prepare_csv_wavs_ja.py"
SPEC = importlib.util.spec_from_file_location("prepare_csv_wavs_ja", SCRIPT)
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)


class JapaneseDatasetPreparationTests(unittest.TestCase):
    def test_vocab_order_is_preserved_exactly(self):
        with TemporaryDirectory() as tmp:
            vocab_path = Path(tmp) / "vocab.txt"
            vocab_path.write_text(" \nア\n。\n", encoding="utf-8")

            entries, vocab = prepare.load_vocab(vocab_path)

            self.assertEqual(entries, [" ", "ア", "。"])
            self.assertEqual(vocab, {" ", "ア", "。"})

    def test_kana_conversion_reports_oov_characters(self):
        with patch.object(prepare, "ja_to_kana", return_value="ワタシ。"):
            kana, oov = prepare.kana_and_oov("私は。", {"ワ", "タ", "シ", "。"})

        self.assertEqual(kana, "ワタシ。")
        self.assertEqual(oov, [])

        with patch.object(prepare, "ja_to_kana", return_value="ワタシ☆"):
            _, oov = prepare.kana_and_oov("私は☆", {"ワ", "タ", "シ"})
        self.assertEqual(oov, ["☆"])


if __name__ == "__main__":
    unittest.main()
