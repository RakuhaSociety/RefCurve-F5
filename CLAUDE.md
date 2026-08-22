# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

这是 **RefCurve-F5** —— [F5-TTS](https://github.com/SWivid/F5-TTS) 的改造版本，核心目标是**多参考音频特征混合与曲线控制**。原版 F5-TTS 是基于流匹配（Flow Matching）+ 扩散 Transformer 的零样本 TTS 系统。

`RefCurve` 是方法名（多参考 + 曲线控制），本仓库是它在 F5-TTS 上的实现；未来可能移植到 GPT-SoVITS（`RefCurve-SoVITS`）。

当前基线：上游 **v1.1.22**（2026-07-23）。上游 remote 为 `upstream`，同步用 `git fetch upstream && git merge upstream/main`。

## 推理管线签名（已与上游兼容）

`infer_process()` 的位置参数与上游一致，第二参考是**关键字可选参数**：

```python
infer_process(ref_audio, ref_text, gen_text, model_obj, vocoder,
              ref_audio_2=None, ref_text_2="", ...)   # 省略即单参考
```

`ref_audio_2=None` 时克隆首参考退化为单参考，所以 `api.py`、`infer_gradio.py`、`socket_server.py` 等上游调用点均正常可用。**改动此签名时务必保持这一兼容性**，否则会再次破坏上游入口。

## 启动方式

**必须用仓库内的便携 Python，不是系统 Python。** 两个 `.bat` 除了选对解释器，还设置了 CUDA DLL 路径、内置 ffmpeg、HF 国内镜像和本地缓存目录：

```bat
启动特征混合试验台.bat     :: → src/f5_tts/infer/gradio_mix_demo.py（Gradio）
启动曲线混合后端.bat       :: → tools/mix_curve_server.py（FastAPI，端口 8002）
```

两者共同设置的环境（照抄自 bat，手动启动时需自行复现）：

```bat
set PYTHON=%CD%\f5-tts_env\python.exe
set PATH=%CD%\f5-tts_env\ffmpeg\bin;%CD%\f5-tts_env\Lib\site-packages\torch\lib;%CD%\f5-tts_env\Scripts;%PATH%
set HF_ENDPOINT=https://hf-mirror.com
set HF_HOME=%CD%\.huggingface
set TORCH_HOME=%CD%\.huggingface
set XFORMERS_FORCE_DISABLE_TRITON=1
```

用 `f5-tts_env\python.exe src\f5_tts\infer\gradio_mix_demo.py` 直接启动通常也行，但缺 `HF_ENDPOINT` 时首次下载 ASR 模型会很慢或超时。

### 曲线混合后端 + 网页编辑器

`tools/mix_curve_server.py` 是独立的 FastAPI 服务，暴露单个 `POST /infer`（multipart 表单，返回 `audio/wav`）。它复用 `gradio_mix_demo.py` 里的 `_load_default_model()` / `_get_paraformer_asr()` / `_clean_cn_text()`，所以模型加载逻辑只有一份。

配套前端是 [tools/mix_curve_editor.html](tools/mix_curve_editor.html)（自包含，无外部脚本，只请求 `localhost:8002`），用于手绘 2D 权重网格再 POST 给后端（对应 `mix_2d_mode="2d_grid"` + `mix_2d_weights` 的 JSON 数组）。

### CLI 推理

`f5_tts` 以可编辑模式装进便携环境（import 直接指向 `src/f5_tts`），所以 `f5-tts_env\Scripts\` 下的 `f5-tts_infer-cli.exe` 等入口跑的就是改造后的代码。裸命令需先把该 Scripts 目录加进 PATH（bat 已代劳），或直接用完整路径调用。

```bash
# 配置文件（推荐，toml 里已带第二参考）
f5-tts_infer-cli -c src/f5_tts/infer/examples/basic/basic.toml

# 命令行：--ref_audio_2 / --ref_text_2 及混合参数均已加入 CLI
f5-tts_infer-cli --model F5TTS_v1_Base \
  --ref_audio "a.wav" --ref_text "转写A" \
  --ref_audio_2 "b.wav" --ref_text_2 "转写B" \
  --gen_text "要合成的文本" \
  --mix_method slerp --mix_schedule cosine --mix_a_start 0.9 --mix_a_end 0.3
```

`--ref_audio_2` 的内置默认值指向 `infer/examples/basic/basic_ref_en_2.wav`，**该文件在仓库中不存在**（[infer_cli.py:323](src/f5_tts/infer/infer_cli.py#L323)）。不显式传第二参考、也不用 toml 时会因文件缺失失败。

`basic.toml` 现在多出 `ref_audio_2` / `ref_text_2` 两个键，且已被改成中文示例音频。自己写 toml 时这两个键必填。

## 混合机制的实现位置

混合发生在 **CFM ODE 采样循环内部**，不是在音频层面预混。数据流：

```
两段参考音频 → 各自 mel → cond_a / cond_b
  → 每个 ODE 步按权重 alpha 融合 → 融合后的条件送入 Transformer
  → mel → vocoder → 音频
```

`CFM.sample()`（[src/f5_tts/model/cfm.py:189](src/f5_tts/model/cfm.py#L189)）新增的关键字参数：

| 参数 | 含义 |
| --- | --- |
| `cond_b` / `lens_b` | 第二参考的 mel 与长度 |
| `mix_method` | `lerp` / `slerp` / `log` |
| `mix_on` | `cond`（融合条件）或 `pred`（融合预测速度场） |
| `mix_schedule`, `mix_a_start`, `mix_a_end` | 时间维度 t 的曲线与起止权重 |
| `n_schedule`, `n_a_start`, `n_a_end` | mel 帧位置维度 n 的曲线与起止权重 |
| `mix_2d_mode` | `t_only`（默认） / `n_only` / `multiply` / `add` / `max` / `min` / `2d_grid` |
| `mix_2d_weights` | `2d_grid` 模式下的权重矩阵 `[steps, n_frames]`，该模式下必传否则抛 `ValueError` |
| `allow_extrapolation` | 为 False 时所有权重 `clamp(0,1)`；为 True 才允许 >1 放大 / <0 反相 |

三个关键实现点：

- `mix_on` 决定融合作用在哪一层：`cond` 分支在 [cfm.py:454](src/f5_tts/model/cfm.py#L454)，`pred` 分支在 [cfm.py:468](src/f5_tts/model/cfm.py#L468)。**注意 `mix_on` 只在 `CFM.sample()` 上存在，`infer_process()` 没有把它透传出来**，所以从 Gradio/CLI 走进来时恒为 `cond`。想试 `pred` 需要自己加透传。
- `alpha_of_t(t)` 返回标量，`alpha_of_n()` 返回 `[1, n, 1]`（在循环外算一次），二者按 `mix_2d_mode` 组合。加新曲线就改这两个函数，或直接传 callable。
- `slerp_with_norm()`（[cfm.py:36](src/f5_tts/model/cfm.py#L36)）方向用 SLERP、幅度用 LERP；`log_domain_blend()`（[cfm.py:84](src/f5_tts/model/cfm.py#L84)）是对数域几何平均，适合能量类信号。

各混合算法/曲线的取舍：`lerp` 直接稳妥；`slerp` 过渡更平滑，适合音色方向性特征；`log` 保持乘性特性。曲线中 `linear` 匀速、`cosine` 中段平缓、`sigmoid` 两端平缓中段陡。

### 三参考情绪迁移

在音频/mel 层面先构造差分再走双参考路径，见 `_prepare_transfer_audio()`（[gradio_mix_demo.py:240](src/f5_tts/infer/gradio_mix_demo.py#L240)）：

```
A = 情绪基准, B = 情绪目标, C = 声线基底
构造 C + diff_scale * (B - A) → 作为参考之一送入混合推理
```

`diff_scale` 常用 0.5–1.5；配合负的 `mix_a_start/end` 可做情绪反向。**注意外推与反相都需要 `allow_extrapolation=True`，否则权重会被静默 clamp 到 [0,1]，看起来像"参数没生效"。**

## 模型与依赖的实际状态

**模型从本地 `ckpts/` 加载，不走 HF 自动下载。** `_load_default_model()`（[gradio_mix_demo.py:52](src/f5_tts/infer/gradio_mix_demo.py#L52)）优先仓库根 `ckpts/`，回退到包内目录，需要：

- `ckpts/F5TTS_v1_Base/model_1250000.safetensors` 与 `vocab.txt`
- `ckpts/vocos-mel-24khz/`（声码器）

缺文件直接报错，不会 fallback 到下载。

**ASR 用 FunASR Paraformer 而非原版 Whisper**（`_get_paraformer_asr()`，中文效果更好且不依赖 FFmpeg），需 `funasr` + `modelscope`——它们在 `pyproject.toml` 里属于 `[eval]` 可选依赖，默认安装不含，需 `pip install funasr modelscope`。转写结果经 `_clean_cn_text()` 清洗。参考文本留空时才触发 ASR；后端里若 ASR 仍得空则填 `"."` 跳过。

**PyTorch 版本注意**：`pyproject.toml` 已随上游放宽为 `torch>=2.0.0`，便携环境 `f5-tts_env` 实装 **torch 2.8.0+cu128**。原先的 cu124 uv 索引已在对齐上游时移除。

## 原版功能（未改造部分）

模型骨架 [src/f5_tts/model/backbones/](src/f5_tts/model/backbones/)：`dit.py`（F5-TTS 用）、`mmdit.py`、`unett.py`（E2 TTS 用）。训练循环在 `trainer.py`（Accelerate + EMA），配置用 Hydra 组合 [src/f5_tts/configs/](src/f5_tts/configs/) 下的 YAML。

```bash
accelerate config   # 首次
accelerate launch src/f5_tts/train/train.py --config-name F5TTS_v1_Base.yaml
accelerate launch src/f5_tts/train/train.py --config-name F5TTS_v1_Base.yaml ++datasets.batch_size_per_gpu=19200
f5-tts_finetune-gradio

python src/f5_tts/train/datasets/prepare_emilia.py     # 或 prepare_csv_wavs.py 等
python src/f5_tts/socket_server.py                     # 实时流式服务
```

`socket_server.py` 调 `infer_batch_process()`（[socket_server.py:125](src/f5_tts/socket_server.py#L125)、[:145](src/f5_tts/socket_server.py#L145)），传的 5 个位置参数正好等于该函数的必需参数个数，其余走关键字，因此未受签名改动影响。

微调要点：从 F5TTS_v1_Base 起步，学习率低于预训练（如 1e-5）；早期微调可设 `use_ema=False` 避免预训练 EMA 主导；`batch_size_type` 可选 `frame` 或 `sample`；WandB 需 `wandb login`，离线用 `WANDB_MODE=offline`。设备自动探测顺序 CUDA → XPU → MPS → CPU。声码器 `vocos`（默认，更快）或 `bigvgan`。

原版推理经验仍适用：参考音频 <12 秒且结尾留约 1 秒静音；单次生成上限约 30 秒（含提示音）；更长文本自动分块；大写字母逐字母读；数字需预先转成目标语言文字。

## 代码规范

`ruff` 统一 lint + format + import 排序（[ruff.toml](ruff.toml)，行宽 120，target py310）。pre-commit 配置了三个 ruff hook（linter `--fix`、formatter、import 排序 `--select I --fix`）加 `check-yaml`：

```bash
pre-commit run --all-files
```

**但 `ruff` 和 `pre-commit` 目前都没装**（便携环境和系统 Python 里都没有），要跑得先 `f5-tts_env\python.exe -m pip install ruff pre-commit`。

改造过的文件（`cfm.py`、`utils_infer.py` 等）没有按 ruff 格式化过（逗号前有空格、成片尾随空白）。对全仓跑 formatter 会把它们整体重排，产生与你本次改动无关的大 diff——若要保持 diff 干净，只对自己动过的文件跑 `ruff format <file> && ruff check --fix <file>`。

改造代码大量使用中文注释与 `✅` 标记新增段落，沿用这一风格便于区分改造点与原版代码。张量注释用爱因斯坦记号：`b` 批次、`n` 序列、`d` 维度、`nt` 文本序列、`nw` 波形长度。部分模型文件带 `# ruff: noqa: F722 F821` 以容纳张量类型注解。

## 测试

仓库无测试套件，无 pytest 配置。验证手段：

- 推理：`f5-tts_infer-cli -c src/f5_tts/infer/examples/basic/basic.toml`，输出到 `tests/infer_cli_basic.wav`
  - **注意**：`infer_cli.py` 的声码器本地路径硬编码为 `../checkpoints/vocos-mel-24khz`（与实际 `ckpts/` 不符，且未开放为命令行参数），走本地声码器会失败。此问题继承自上游，尚未修复。
  - 绕开方式：直接在 Python 里调 `load_vocoder('vocos', is_local=True, local_path='ckpts/vocos-mel-24khz', ...)` + `infer_process()`
- 混合参数：启动试验台，先用 `lerp` + `linear` 建立基线，再改单个参数对比
- 训练：小数据子集跑 1 个 epoch

改动 `cfm.py` 的混合逻辑后，最快的回归验证是跑一次上面的 CLI 命令——它会同时穿过双参考加载、混合采样和声码器三段路径。

## 其他文档

[src/f5_tts/train/README.md](src/f5_tts/train/README.md)（训练）、[src/f5_tts/infer/README.md](src/f5_tts/infer/README.md)（推理，描述的是**原版**签名）、[src/f5_tts/infer/SHARED.md](src/f5_tts/infer/SHARED.md)（社区模型）、[src/f5_tts/eval/README.md](src/f5_tts/eval/README.md)（评测）、[src/f5_tts/runtime/triton_trtllm/README.md](src/f5_tts/runtime/triton_trtllm/README.md)（TensorRT-LLM 部署）。
