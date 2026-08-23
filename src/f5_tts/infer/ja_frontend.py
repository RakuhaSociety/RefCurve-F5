"""日文文本前端：把含汉字的日文文本转成 Jmica 日文模型 vocab 兼容的假名序列。

Jmica JA 模型的 vocab（vocab_japanese.txt）覆盖平假名/片假名全集但不含汉字，
输入文本必须先转假名（社区 issue #943 / #1167 的根因）。

用法：
    from f5_tts.infer.ja_frontend import ja_to_kana, check_vocab_coverage
    kana = ja_to_kana("私は日本語のテストです")   # → ワタシワニホンゴノテストデス
"""

from __future__ import annotations

_DICT_READY = False


def _ensure_dict():
    """pyopenjtalk 首次调用会下载 open_jtalk 词典，这里显式触发以便报错清晰。"""
    global _DICT_READY
    if _DICT_READY:
        return
    try:
        import pyopenjtalk  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "缺少 pyopenjtalk。日文推理需要它做汉字→假名：pip install pyopenjtalk"
        ) from e
    _DICT_READY = True


def ja_to_kana(text: str) -> str:
    """汉字混排日文 → 片假名序列（pyopenjtalk g2p kana 模式）。

    标点保留：g2p 会丢弃标点，这里按简单策略把常见句读符号加回，
    保持与参考转写的风格一致（F5 的文本条件对标点敏感）。
    """
    _ensure_dict()
    import pyopenjtalk

    # 按句读切分逐段转换，保住标点位置
    out = []
    buf = ""
    puncts = set("、。！？!?，,．.…「」『』（）()　 ")
    for ch in text:
        if ch in puncts:
            if buf:
                out.append(pyopenjtalk.g2p(buf, kana=True))
                buf = ""
            out.append(ch)
        else:
            buf += ch
    if buf:
        out.append(pyopenjtalk.g2p(buf, kana=True))
    return "".join(out)


def check_vocab_coverage(text: str, vocab_path: str) -> list[str]:
    """返回 text 中不在 vocab 里的字符列表（空列表 = 全覆盖，可安全推理）。"""
    with open(vocab_path, encoding="utf-8") as f:
        vocab = set(f.read().splitlines())
    return sorted({ch for ch in text if ch not in vocab and ch != " "})
