# RefCurve-F5

**多参考音频混合与曲线控制的语音合成** — 基于 [F5-TTS](https://github.com/SWivid/F5-TTS) 实现。

> [!IMPORTANT]
> **本项目不是官方 F5-TTS。** 这是 [SWivid/F5-TTS](https://github.com/SWivid/F5-TTS) 的第三方改造版本，
> 由个人维护，与原作者及其所属机构无关。原版的问题请到上游仓库反馈。

`RefCurve` 是一套**方法**：不再"用一段参考音频克隆一个声音"，而是同时给多段参考，
并用**曲线**精确控制每段参考在生成过程中各处的权重——从而取到各段参考里不同的部分。
本仓库 `RefCurve-F5` 是它在 F5-TTS 上的实现。

---

## 这是什么

标准零样本 TTS 回答的是"让**这个**声音说这句话"。RefCurve 回答的是：

> 让 A 和 B 之间**某个可精确控制的插值点**上的声音说这句话。

三项能力：

| 能力 | 说明 |
| --- | --- |
| **双参考混合** | 同时输入参考 A、B，在扩散采样的每一步融合两者的 mel 条件 |
| **曲线控制** | 权重不是一个常数，而是沿**时间维度 t**（扩散步）和**帧位置维度 n**（音频时间轴）变化的曲线 |
| **情绪迁移** | 给定 A(平静)、B(激动)、C(目标音色)，把 `B-A` 的情绪差分迁移到 C 上 |

关键点：混合发生在 **ODE 采样循环内部**，不是把两段音频在波形层面叠加。

```
参考 A ─┐
        ├─→ mel → cond_a ─┐
参考 B ─┘                  ├─ 每个 ODE 步按 α 融合 → Transformer → mel → vocoder → 音频
             mel → cond_b ─┘
                            ↑
                     α = f(t, n) 由曲线决定
```

## 效果演示

同一段参考音频、同样的生成文本，只改混合参数：

| 样例 | 参数 | 文件 |
| --- | --- | --- |
| 参考 A（温柔） | — | [`samples/ref_A_yasashi.wav`](samples/ref_A_yasashi.wav) |
| 参考 B（平常） | — | [`samples/ref_B_normal.wav`](samples/ref_B_normal.wav) |
| 单参考（等同原版 F5） | 只用 A | [`samples/01_single_ref.wav`](samples/01_single_ref.wav) |
| 双参考混合 | `slerp` + `cosine`，A 权重 0.9→0.3 | [`samples/02_mix_slerp_cosine.wav`](samples/02_mix_slerp_cosine.wav) |
| 2D + 权重外推 | `multiply`，n 维度 1.2→-0.2，开启外推 | [`samples/03_mix_2d_extrapolation.wav`](samples/03_mix_2d_extrapolation.wav) |

> wav 文件需下载后播放，GitHub 不支持在 README 内嵌音频。

## 快速开始

### 环境要求

- Python 3.10+、NVIDIA GPU（CPU 可跑但很慢）
- 模型文件放在 `ckpts/`：
  - `ckpts/F5TTS_v1_Base/model_1250000.safetensors` + `vocab.txt`（[下载](https://huggingface.co/SWivid/F5-TTS)）
  - `ckpts/vocos-mel-24khz/`（[下载](https://huggingface.co/charactr/vocos-mel-24khz)）

### 安装

```bash
git clone https://github.com/RakuhaSociety/RefCurve-F5.git
cd RefCurve-F5
pip install -e .
pip install funasr modelscope   # 可选：中文自动转写
```

### 启动

```bash
python src/f5_tts/infer/gradio_mix_demo.py
```

Windows 用户若使用便携环境，可直接双击 `启动特征混合试验台.bat`（内含 CUDA 路径与 HF 镜像配置）。

界面含两个标签页：**双参考混合** 和 **三参考情绪迁移**。

### 命令行

```bash
f5-tts_infer-cli \
  --ref_audio "A.wav"   --ref_text   "A的转写" \
  --ref_audio_2 "B.wav" --ref_text_2 "B的转写" \
  --gen_text "要合成的文本" \
  --mix_method slerp --mix_schedule cosine \
  --mix_a_start 0.9 --mix_a_end 0.3
```

不传 `--ref_audio_2` 时退化为标准单参考推理，行为与上游一致。

### Python API

```python
from f5_tts.infer.utils_infer import infer_process

wav, sr, _ = infer_process(
    ref_audio="A.wav", ref_text="A的转写",
    gen_text="要合成的文本",
    model_obj=model, vocoder=vocoder,
    ref_audio_2="B.wav", ref_text_2="B的转写",   # 省略即单参考
    mix_method="slerp", mix_schedule="cosine",
    mix_a_start=0.9, mix_a_end=0.3,
)
```

## 混合参数

### 混合算法 `mix_method`

| 值 | 说明 | 适合 |
| --- | --- | --- |
| `lerp` | 线性插值 | 通用，稳妥的起点 |
| `slerp` | 方向用球面插值 + 幅度用线性插值 | 音色类方向性特征，过渡更平滑 |
| `log` | 对数域几何平均 | 能量类信号，保持乘性特性 |

### 权重曲线

权重 α 表示**参考 A 的占比**（`1-α` 即 B 的占比），沿两个维度独立控制：

- **时间维度 t**（扩散步 0→1）：`mix_schedule` + `mix_a_start` / `mix_a_end`
- **帧位置维度 n**（音频开头→结尾）：`n_schedule` + `n_a_start` / `n_a_end`

曲线类型：`linear`（匀速）、`cosine`（中段平缓）、`sigmoid`（两端平缓中段陡），也可传入自定义 callable。

直觉理解：**t 维度**控制"生成过程中何时更像 A"，**n 维度**控制"音频的哪一段更像 A"。
比如 `n_a_start=1.0, n_a_end=0.0` 意味着开头用 A 的音色、结尾过渡到 B。

### 2D 组合 `mix_2d_mode`

两个维度的权重如何合成：

| 值 | 公式 |
| --- | --- |
| `t_only`（默认） | `α = α_t` |
| `n_only` | `α = α_n` |
| `multiply` | `α = α_t × α_n` |
| `add` | `α = (α_t + α_n) / 2` |
| `max` / `min` | 取较大 / 较小值 |
| `2d_grid` | 直接传入 `[steps, n_frames]` 权重矩阵 |

`2d_grid` 可配合网页版曲线编辑器（`tools/mix_curve_editor.html` + `tools/mix_curve_server.py`，FastAPI 后端监听 8002）手绘权重网格。

### 权重外推 `allow_extrapolation`

默认权重被 clamp 到 `[0,1]`。开启后允许越界：

- **α > 1** — 放大该参考的特征（超出 A 本身的程度）
- **α < 0** — 反转特征方向

> 若发现调了参数却"没效果"，先检查是不是没开这个开关被静默 clamp 了。

## 三参考情绪迁移

给定三段音频，构造差分后走双参考路径：

```
A = 情绪基准（如某人平静时）
B = 情绪目标（同一人激动时）
C = 声线基底（目标说话人）

→ C + diff_scale × (B - A)
```

`diff_scale` 界面范围 0.0–2.0，常用 0.5–1.5。配合负权重 + 开启外推可做情绪反向。

## 与上游 F5-TTS 的区别

| 方面 | 上游 F5-TTS | RefCurve-F5 |
| --- | --- | --- |
| 参考音频 | 单段 | 单段或多段，带曲线控制 |
| 混合时机 | — | ODE 采样循环内部融合 mel 条件 |
| 情绪迁移 | — | 三参考差分迁移 |
| 模型加载 | HF 自动下载 | 本地 `ckpts/` 优先 |
| 中文 ASR | Whisper | FunASR Paraformer（不依赖 FFmpeg） |
| 界面 | 标准 Gradio | 额外的混合试验台 + 网页曲线编辑器 |

**权重完全兼容**：改造只发生在推理时，不涉及网络结构与训练。上游发布的 checkpoint 可直接使用。

**上游功能保持可用**：`f5-tts_infer-gradio`、`f5_tts.api.F5TTS`、训练与微调均未受影响。

当前基线：上游 **v1.1.22**（2026-07）。

## 已知限制

- **声码器本地路径硬编码**：`infer_cli.py` 中写死为 `../checkpoints/vocos-mel-24khz`，与实际的 `ckpts/` 不符，且未开放为命令行参数。使用 CLI 时若走本地声码器会失败（此问题继承自上游）。
- **`--ref_audio_2` 默认值**指向不存在的 `basic_ref_en_2.wav`，不显式传第二参考且不用 toml 时会失败。
- **`mix_on` 参数**（融合作用于条件 or 预测速度场）仅存在于 `CFM.sample()`，未从上层透传，实际恒为 `cond`。
- 无自动化测试。

## 路线图

- [x] 双参考混合（lerp / slerp / log）
- [x] t + n 双维度曲线控制、2D 组合模式、权重外推
- [x] 三参考情绪迁移
- [x] 网页版曲线编辑器
- [ ] **基于反向传播的混合模式** — 当前混合是前向的确定性融合；另一种设计是通过梯度优化求解混合参数，尚未实现
- [ ] `RefCurve-SoVITS` — 将方法移植到 [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)

> RefCurve 的思路不绑定 F5-TTS，原则上可移植到任何以参考音频为条件的 TTS 系统。

## 致谢与许可

本项目基于 [SWivid/F5-TTS](https://github.com/SWivid/F5-TTS)（MIT）改造，感谢原作者的工作。

如果你在研究中使用了 F5-TTS 的底层方法，请引用原论文：

```bibtex
@article{chen-etal-2024-f5tts,
  title={F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching},
  author={Yushen Chen and Zhikang Niu and Ziyang Ma and Keqi Deng and Chunhui Wang
          and Jian Zhao and Kai Yu and Xie Chen},
  journal={arXiv preprint arXiv:2410.06885},
  year={2024},
}
```

本仓库沿用 MIT 许可，见 [LICENSE](LICENSE)。上游原始文档保留于 [README_UPSTREAM.md](README_UPSTREAM.md)。
