import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

import torch
import torchaudio
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

# ensure repo root on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from f5_tts.infer.gradio_mix_demo import _load_default_model, _get_paraformer_asr, _clean_cn_text
from f5_tts.infer.utils_infer import infer_process, target_rms, cross_fade_duration, fix_duration

app = FastAPI()
# 仅供本机使用：默认绑定 127.0.0.1（见文件末尾），CORS 也只放行本地来源。
# 如需局域网访问，设置 MIX_SERVER_HOST=0.0.0.0，并自行加鉴权——该接口无任何认证。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8002", "http://127.0.0.1:8002", "null"],
    allow_credentials=False,
    allow_methods=["POST"],
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


ALLOWED_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".opus", ".webm"}
MAX_UPLOAD_BYTES = 100 * 1024 * 1024  # 单个参考音频上限
MAX_GEN_CHARS = 5000
MAX_GRID_ROWS, MAX_GRID_COLS = 512, 8192


def save_upload_tmp(upload: UploadFile) -> str:
    # 只取客户端扩展名做白名单校验，其余部分（目录、..）由 Path.suffix 天然丢弃
    suffix = Path(upload.filename or "ref.wav").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        suffix = ".wav"
    data = upload.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="reference audio too large")
    if not data:
        raise HTTPException(status_code=400, detail="empty reference audio")
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as fp:
        fp.write(data)
        return fp.name


def parse_mix_grid(raw: str):
    """解析 2D 权重网格，拒绝非法形状与 NaN/Inf。"""

    def _reject(_):
        raise HTTPException(status_code=400, detail="mix_2d_weights must be finite numbers")

    try:
        grid = json.loads(raw, parse_constant=_reject)
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=400, detail="mix_2d_weights is not valid JSON")

    if not isinstance(grid, list) or not grid or not all(isinstance(r, list) and r for r in grid):
        raise HTTPException(status_code=400, detail="mix_2d_weights must be a non-empty 2D array")
    if len(grid) > MAX_GRID_ROWS or max(len(r) for r in grid) > MAX_GRID_COLS:
        raise HTTPException(status_code=400, detail="mix_2d_weights too large")
    if len({len(r) for r in grid}) != 1:
        raise HTTPException(status_code=400, detail="mix_2d_weights must be rectangular")

    tensor = torch.tensor(grid, dtype=torch.float32)
    if not torch.isfinite(tensor).all():
        raise HTTPException(status_code=400, detail="mix_2d_weights contains NaN/Inf")
    return tensor


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
    log_blend_mode: str = Form("logmel"),
    n_normalize_to_ref: bool = Form(True),
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
    if len(gen_text) > MAX_GEN_CHARS:
        raise HTTPException(status_code=400, detail=f"gen_text exceeds {MAX_GEN_CHARS} chars")

    # 客户端滑块不可信，服务端再夹一次范围
    steps = max(1, min(int(steps), 64))
    speed = max(0.1, min(float(speed), 3.0))
    cfg = max(0.0, min(float(cfg), 10.0))
    sway_coef = max(-10.0, min(float(sway_coef), 10.0))

    ref_a_path = save_upload_tmp(ref_a)
    tmp_wav = None
    try:
        ref_b_path = save_upload_tmp(ref_b)
    except Exception:
        Path(ref_a_path).unlink(missing_ok=True)
        raise

    try:
        mix_grid = parse_mix_grid(mix_2d_weights) if mix_2d_weights else None

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
            gen_text,
            model,
            vocoder,
            ref_audio_2=ref_b_path,
            ref_text_2=ref_text_b,
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
            log_blend_mode=log_blend_mode,
            n_normalize_to_ref=n_normalize_to_ref,
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
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"inference failed: {exc}") from exc
    finally:
        # 无论成败都清理临时文件，避免磁盘泄漏
        Path(ref_a_path).unlink(missing_ok=True)
        Path(ref_b_path).unlink(missing_ok=True)
        if tmp_wav:
            Path(tmp_wav).unlink(missing_ok=True)

    return Response(content=data, media_type="audio/wav")


if __name__ == "__main__":
    import uvicorn

    # 默认只监听本机；确需暴露时显式设置 MIX_SERVER_HOST（该接口无鉴权，请自行评估风险）
    host = os.environ.get("MIX_SERVER_HOST", "127.0.0.1")
    uvicorn.run("tools.mix_curve_server:app", host=host, port=8002, reload=False)
