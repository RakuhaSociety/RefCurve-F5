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
| `mix_2d_grid_domain` | `"full"`（默认）网格映射到整个 duration（prompt + 生成段）；`"gen"` 网格只映射到生成段，prompt 区用首列填充 |
| `allow_extrapolation` | 为 False 时所有权重 `clamp(0,1)`；为 True 才允许 >1 放大 / <0 反相 |

三个关键实现点：

- `mix_on` 决定融合作用在哪一层：`cond` 分支、`pred` 分支、`two_stage` 模式（先双路独立生成 mel，再混合作为 pred 参考）、`output` 模式（mel 输出层混合）。从 Gradio/曲线后端进来时可选全部四种模式。
- `mix_2d_grid_domain` 控制 `2d_grid` 网格的 n 轴映射：`"full"` 模式下网格列映射到整个 `max_duration`（prompt + 生成段），约 35-50% 的列落在不可听的 prompt 区；`"gen"` 模式下网格列只映射到生成段，prompt 区用首列权重填充，100% 列都对应可听内容。**多块场景下，网格在每块内独立重复**（架构特性）。
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

## 训练与 Scheduler 修复

### Scheduler Bug 与修复（2026-09-05）

**Bug 描述**：原版 `trainer.py` 在 `split_batches=False` 多 GPU 训练时，只对 `num_warmup_updates` 应用了 world-size multiplier，但未对 scheduler 的 `num_training_steps` 应用。导致 AcceleratedScheduler 内部步进速度是实际 optimizer update 的 `world_size` 倍，LR 在约 `global_update = total / world_size` 时提前触底（8 GPU 时是 625 步）。

**修复**（[src/f5_tts/model/trainer.py](src/f5_tts/model/trainer.py)）：
- `_training_update_horizon()` 现在返回的 `num_training_steps` 已经应用 multiplier
- 在 global units 计算完成后，统一把 warmup 和 total 都转换到 inner units
- Multiplier = `1 if accelerator.split_batches else accelerator.num_processes`
- 8 GPU `split_batches=False` 时：inner warmup = 800，inner total = 40000（正确）
- Bug 版本：inner warmup = 800，inner total = 5000（错误，horizon 未缩放）

**验证**：
- `tests/test_trainer_scheduler.py`：模拟 1 GPU vs 8 GPU 的 LR 轨迹一致性
- Smoke runs：120-update 训练验证 LR 符合公式
- 实际 5000-update corrected run：LR 轨迹完全符合预期

### Corrected Calibration Run

**Run 位置**（远端）：`/root/F5-TTS/ckpts/visualnovel_calibration_ja/corrected-calibration-5000-20260905/`

**Run identity**：`2b5705a27a3a5da5a8de85f953d55d04f95e3ed8611bb2ee2343284a0e5e3151`

**关键参数**：
- 数据集：`/data/visualnovel/aggregates/wave-abc-v1-exclude-conflicts`（与 bug run 完全相同）
- Source checkpoint：官方 v1 EMA `model_1250000.safetensors`，vocab remap seed 666
- Scheduler：global warmup=100，global total=5000；inner warmup=800，inner total=40000
- 8×RTX 4090，BF16，`split_batches=False`

**Checkpoints**：9 个完整 EMA（500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500）+ `model_last.pt`（= 5000）

**LR 验证**（关键锚点）：
- update 100: `1.00e-05`（warmup 峰值）✅
- update 500: `9.18e-06`（bug run 的 3.9×）✅
- update 1000: `8.16e-06`（bug run 的 8千万×）✅
- update 2500: `5.10e-06`（仍是峰值的 51%）✅
- update 5000: `1.00e-13`（正确到达 floor）✅

**对比 bug run**：旧 run `abc-v1-calibration-5000-20260829` 在 update 625 左右就到达 `1e-13` floor，之后 4375 个 updates 的权重几乎不再变化。Corrected run 的有效训练时间延长了 4× 以上。

### 训练相关工具

- `src/f5_tts/train/run_adjudication.py`：为完成的 run 创建 immutable forensic verdict sidecar
- `src/f5_tts/train/run_manifest.py`：训练开始时记录 code/dataset/config/environment identity
- `tools/train_visualnovel_calibration_ja.sh`：校准训练的启动脚本（支持环境变量覆盖）
- `tools/train_calibration_remote_data_volume.sh`：远端专用包装，把所有写路径钉在 `/data`
- `tools/smoke_scheduler_world_size.py`：1 GPU vs 8 GPU scheduler 不变性验证脚本

Smoke 验证走 launcher 自带的两个旋钮，不要另写启动脚本——`VISUALNOVEL_CALIBRATION_SMOKE_UPDATES=N`
只提前停步而不改 `optim.max_updates=5000`（scheduler horizon 与正式 run 完全一致，LR 轨迹才有可比性），
`VISUALNOVEL_CALIBRATION_NUM_PROCESSES=N` 切 world size，于是 1 卡与 8 卡走同一条启动路径。
smoke 长度必须 **>100**：warmup 段（≤100）无论 scheduler 修没修都一致，bug 只在第一个 decay step
（update 101）才显形。

### 远端训练环境（183.147.142.130:9000）

三件事会让默认命令直接失败，且报错都不像真正的原因：

- **解释器**：训练用 `/root/f5-tts-env/bin/python`。系统 `python3` 没有 torch 也没有 pytest。
- **磁盘**：根盘 445G 易满，`/data` 才有 3.5T。训练涉及五个写路径（run root、operation lock、
  `$TMPDIR`、numba cache、`hydra.run.dir`），漏一个就失败，且分别伪装成
  `PytorchStreamWriter unexpected pos`（像 torch 序列化 bug）、`No usable temporary directory`
  （挂在 `import torch`）、`cannot cache function: no locator available`（挂在 librosa）、
  `OSError: [Errno 28]`。用 `tools/train_calibration_remote_data_volume.sh` 一次性设好；
  launcher 现在也会在碰 GPU 前预检 run root 空间。
- **代码同步**：远端无法直连 GitHub（`GnuTLS recv error`）。路径是本地 `git push train main`
  → 裸库 `/root/F5-TTS-remote.git` → 远端 `git pull origin main`。

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
