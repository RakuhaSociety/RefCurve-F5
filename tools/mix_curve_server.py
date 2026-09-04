import json
import math
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

from f5_tts.infer.asr_backends import AsrUnavailable, transcribe
from f5_tts.infer.gradio_mix_demo import _load_default_model
from f5_tts.infer.utils_infer import (
    infer_process,
    target_rms,
    cross_fade_duration,
    fix_duration,
    trim_generated_silence,
)

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
    mix_on: str = Form("cond"),
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
    asr_lang: str = Form("zh"),
    trim_edges: bool = Form(True),
    max_internal_silence: float = Form(0.0),
):
    if len(gen_text) > MAX_GEN_CHARS:
        raise HTTPException(status_code=400, detail=f"gen_text exceeds {MAX_GEN_CHARS} chars")

    # 客户端滑块不可信，服务端再夹一次范围
    steps = max(1, min(int(steps), 64))
    speed = max(0.1, min(float(speed), 3.0))
    cfg = max(0.0, min(float(cfg), 10.0))
    sway_coef = max(-10.0, min(float(sway_coef), 10.0))

    # ✅ 白名单 mix_on。不再静默降级成 cond——用户选了 two_stage 却拿到 cond 的结果，
    # 听感差异被归因到曲线参数上，比直接报错更难排查。
    if mix_on not in ("cond", "pred", "two_stage", "output"):
        raise HTTPException(
            status_code=400,
            detail=f"unknown mix_on: {mix_on!r}, expected one of cond / pred / two_stage / output",
        )

    # ✅ NaN/Inf 必须在推理之前挡掉。NaN 的比较恒为 False，min(max(nan,0),2) 会原样
    # 返回 nan，一路到 trim 里的 int(round(nan*sr/hop)) 才抛异常——那时完整推理已经
    # 白跑一遍。float("nan"/"inf") 能合法解析成 float，所以 FastAPI 的类型校验拦不住，
    # 必须在这里显式判。
    if not math.isfinite(float(max_internal_silence)):
        raise HTTPException(status_code=400, detail="max_internal_silence must be finite")

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

        # Optional ASR when text is empty（asr_lang="zh" 走 FunASR，"ja" 走 Visual-novel-whisper）
        if use_asr and (not ref_text_a.strip() or not ref_text_b.strip()):
            try:
                if not ref_text_a.strip():
                    ref_text_a = transcribe(str(ref_a_path), lang=asr_lang, device=device)
                if not ref_text_b.strip():
                    ref_text_b = transcribe(str(ref_b_path), lang=asr_lang, device=device)
            except (AsrUnavailable, ValueError) as e:
                raise HTTPException(status_code=400, detail=str(e))

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
            mix_on=mix_on,
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

        # ✅ 生成长度在推理前就按字节比例算定，模型拿到定长画布必须填满，估不准的
        # 余量成为静音。这里做事后整理，与试验台同一套语义。
        # 上限同样服务端夹一次：客户端表单不可信。
        # 有限性已在推理前校验过（见上面的 math.isfinite），这里只夹取范围。
        cap = min(max(float(max_internal_silence), 0.0), 2.0)
        if trim_edges or cap > 0:
            audio_np, _ = trim_generated_silence(
                audio_np,
                sr,
                trim_edges=bool(trim_edges),
                max_internal_silence_s=cap if cap > 0 else None,
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
