@echo off
chcp 65001 >nul
echo  启动中，请稍等

set PYTHON=%CD%\f5-tts_env\python.exe
set FF_PATH=%CD%\f5-tts_env\ffmpeg\bin
set CU_PATH=%CD%\f5-tts_env\Lib\site-packages\torch\lib
set SC_PATH=%CD%\f5-tts_env\Scripts
set PATH=%FF_PATH%;%CU_PATH%;%SC_PATH%;%PATH%
set HF_ENDPOINT=https://hf-mirror.com
set HF_HOME=%CD%\.huggingface
set TORCH_HOME=%CD%\.huggingface
set XFORMERS_FORCE_DISABLE_TRITON=1
set FFMPEG_PATH=%CD%\f5-tts_env\ffmpeg\bin

%PYTHON% src\f5_tts\infer\gradio_mix_demo.py

pause

