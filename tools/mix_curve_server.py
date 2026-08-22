import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

import torch
import torchaudio
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

# ensure repo root on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from f5_tts.infer.gradio_mix_demo import _load_default_model, _get_paraformer_asr, _clean_cn_text
from f5_tts.infer.utils_infer import infer_process, target_rms, cross_fade_duration, fix_duration

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy globals
_model = None
_vocoder = None
_device = None


def get_model():
    global _model, _vocoder, _device
    if _model is None:
        _model, _vocoder, _device = _load_default_model()
    return _model, _vocoder, _device


def save_upload_tmp(upload: UploadFile) -> str:
    suffix = Path(upload.filename or "ref.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as fp:
        fp.write(upload.file.read())
        return fp.name


@app.post("/infer")
async def infer_endpoint(
    ref_a: UploadFile = File(...),
    ref_b: UploadFile = File(...),
    gen_text: str = Form(...),
    ref_text_a: str = Form(""),
    ref_text_b: str = Form(""),
    steps: int = Form(32),
    cfg: float = Form(2.0),
    sway_coef: float = Form(-1.0),
    speed: float = Form(1.0),
    mix_method: str = Form("lerp"),
    mix_schedule: str = Form("linear"),
    mix_a_start: float = Form(0.5),
    mix_a_end: float = Form(0.5),
    mix_2d_mode: str = Form("t_only"),
    allow_extrapolation: bool = Form(True),
    mix_2d_weights: Optional[str] = Form(None),
    n_schedule: Optional[str] = Form(None),
    n_a_start: Optional[float] = Form(None),
    n_a_end: Optional[float] = Form(None),
    seed: Optional[int] = Form(None),
    use_asr: bool = Form(False),
):
    ref_a_path = save_upload_tmp(ref_a)
    ref_b_path = save_upload_tmp(ref_b)

    mix_grid = None
    if mix_2d_weights:
        mix_grid = torch.tensor(json.loads(mix_2d_weights))

    model, vocoder, device = get_model()

    # Optional ASR when text is empty
    if use_asr and (not ref_text_a.strip() or not ref_text_b.strip()):
        asr = _get_paraformer_asr(device=device)
        if not ref_text_a.strip():
            res = asr.generate(input=str(ref_a_path), batch_size=1)
            ref_text_a = res[0].get("text", "") if isinstance(res, list) else res.get("text", "")
            ref_text_a = _clean_cn_text(ref_text_a)
        if not ref_text_b.strip():
            res = asr.generate(input=str(ref_b_path), batch_size=1)
            ref_text_b = res[0].get("text", "") if isinstance(res, list) else res.get("text", "")
            ref_text_b = _clean_cn_text(ref_text_b)

    # If still empty, give a dot to skip ASR
    if not ref_text_a.strip():
        ref_text_a = "."
    if not ref_text_b.strip():
        ref_text_b = "."

    audio_np, sr, _ = infer_process(
        ref_a_path,
        ref_text_a,
        ref_b_path,
        ref_text_b,
        gen_text,
        model,
        vocoder,
        mel_spec_type="vocos",
        target_rms=target_rms,
        cross_fade_duration=cross_fade_duration,
        nfe_step=steps,
        cfg_strength=cfg,
        sway_sampling_coef=sway_coef,
        speed=speed,
        fix_duration=fix_duration,
        device=device,
        allow_extrapolation=allow_extrapolation,
        seed=seed,
        mix_method=mix_method,
        mix_schedule=mix_schedule,
        mix_a_start=mix_a_start,
        mix_a_end=mix_a_end,
        mix_2d_mode=mix_2d_mode,
        mix_2d_weights=mix_grid,
        n_schedule=None if n_schedule in ("none", "None", None) else n_schedule,
        n_a_start=n_a_start,
        n_a_end=n_a_end,
    )

    # save to wav in-memory
    tensor = torch.tensor(audio_np).unsqueeze(0)  # [1, T]
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as buf:
        tmp_wav = buf.name
    torchaudio.save(tmp_wav, tensor, sample_rate=sr)
    data = Path(tmp_wav).read_bytes()

    # clean temp files
    Path(ref_a_path).unlink(missing_ok=True)
    Path(ref_b_path).unlink(missing_ok=True)
    Path(tmp_wav).unlink(missing_ok=True)

    return Response(content=data, media_type="audio/wav")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("tools.mix_curve_server:app", host="0.0.0.0", port=8002, reload=False)
