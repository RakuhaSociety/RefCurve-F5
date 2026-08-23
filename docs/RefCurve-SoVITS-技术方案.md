# RefCurve-SoVITS 技术方案

> **多参考音频混合与双轴曲线控制** 在 GPT-SoVITS 上的移植与重设计。
> 本文档是新项目 `RefCurve-SoVITS` 的启动技术方案，由 `RefCurve-F5`（F5-TTS 实现）的经验直接推导而来。
> 撰写日期：2026-08-23。基于 GPT-SoVITS **v4** 架构（S2 = shortcut-CFM-DiT）。

---

## 0. 落地前必须核对的事实

本方案的架构判断基于官方 wiki 与公开资料，但以下锚点在动手前**必须对照实际代码确认**（clone 后第一件事）：

| # | 待核对项 | 本方案的假设 | 影响 |
| --- | --- | --- | --- |
| 1 | S1 自回归解码入口 | `GPT_SoVITS/AR/models/t2s_model.py` 中 `infer_panel`（或其 v4 变体） | M2 的改造位置 |
| 2 | S2 CFM 采样循环位置 | `GPT_SoVITS/module/models.py` 的 `SynthesizerTrnV3/V4` 内部 | M1 的改造位置 |
| 3 | S2 是否为 F5 式 inpainting（参考 mel 前缀 + 掩码生成区） | 是（与 F5 同范式） | 决定 pred 移植是否近乎 1:1 |
| 4 | S2 是否使用 CFG（无条件分支） | 倾向于无 CFG | 决定每步 forward 次数（2× 或 4×） |
| 5 | shortcut-CFM 的默认采样步数 | 推理界面可选（疑似 4/8/16/32） | **t 维曲线的分辨率上限** |
| 6 | 语义 token 帧率 | ~25 Hz（每 token ≈ 40ms） | S1 位置曲线的时间换算 |
| 7 | 推理管线封装 | `TTS_infer_pack/TTS.py`（v2+ 统一管线） | 上层 API 挂接点 |
| 8 | 日文前端 | pyopenjtalk 音素化 | 无需自己写 G2P |

以上任一项与假设不符时，回到本文档修订对应章节，再继续实施。

---

## 1. 为什么从 F5 迁到 GPT-SoVITS

### 1.1 RefCurve-F5 的结论（经验资产）

RefCurve-F5 验证了两件事、否定了一件事：

**验证成立：**
- **pred 模式**（速度场空间加权混合）是双参考混合的正确层级——每步各跑一次 forward，速度场线性可加有 CFG 同源的理论依据，实测稳定
- **曲线控制**（t 维 = 扩散步、n 维 = 帧位置）能对混合权重做细粒度调度，机制与混合层解耦

**被否定（不要再走）：**
- 波形域差分（`C + scale*(B-A)`）：三段录音不逐样本对齐，差值是干涉不是情绪
- mel 域 DTW 事后混合（output 模式）：mel 帧不足以跨说话人定位音素（对齐帧对中心化余弦仅 0.23），平均 mel 抹平共振峰（谱峰数反而低于任一支路）
- 三参考速度场迁移（`v_C + scale*(v_B - v_A)`）：**F5 的生成区所有参考的音频条件都是零**，速度场差在生成区恒为零，方案在该架构下结构性不可行

### 1.2 F5 的天花板与 GPT-SoVITS 的架构红利

| 维度 | F5-TTS（NAR 流匹配） | GPT-SoVITS（AR + CFM 两段式） |
| --- | --- | --- |
| 韵律/情绪表达 | 倾向抹平（NAR 通病） | S1 自回归天然擅长从 prompt 延续韵律模式，实测强情绪可复刻 |
| 情绪与音色的分离 | 无架构支持，只能用 mel 通道维做粗糙代理 | **架构层面免费分离**：S1 管韵律/情绪，S2 管音色 |
| 时长处理 | 需显式估计 duration（F5 版本一半的 bug 源于此） | S1 自回归隐式决定时长，**整类 duration bug 不存在** |
| 日文 | 官方无，社区模型非商用 | 原生支持 |
| 许可 | 上游 MIT，但可用日文模型 cc-by-nc | MIT，商用无碍 |
| S2 架构 | 流匹配 DiT | v3/v4 的 S2 = **shortcut-CFM-DiT，与 F5 同家族**，pred 混合可直接平移 |

核心判断：**GPT-SoVITS 的两段式把 RefCurve 想做的"分离控制"变成了架构原生能力**。在 F5 上我们试图从一坨 mel 里同时抠出情绪和音色（失败）；在 GPT-SoVITS 上它们本来就在两个不同的阶段流过。

> 注意版本线：**必须基于 v3/v4 线**（S2 = CFM-DiT）。v2/v2Pro 线的 S2 是 VITS-GAN，生成是单次前向而非迭代采样，没有"速度场"可混。v2Pro 的说话人相似度优势与本方案无关。

---

## 2. GPT-SoVITS v4 架构剖析（数据流）

```
                          ┌──────────── S1（GPT，~300M，自回归）────────────┐
  参考音频 ──→ CNHuBERT ──→ 语义 token（离散，~25Hz）──┐                     │
  参考文本 ──→ 音素序列 ─────────────────────────────┼─→ prompt            │
  生成文本 ──→ 音素序列 ─────────────────────────────┘        ↓             │
                                            逐 token 自回归采样 → 生成区语义 token
                          └──────────────────────────────────────────────┘
                                                              ↓
                          ┌──────────── S2（shortcut-CFM-DiT）─────────────┐
  参考音频 ──→ 参考 mel ──→ 条件（音色来源）                                 │
  生成区语义 token ──→ 上采样对齐 ──→ 条件（说什么、怎么说）                  │
                     噪声 ──→ ODE 采样循环 ──→ 生成 mel                     │
                          └──────────────────────────────────────────────┘
                                                              ↓
                                              HiFiGAN vocoder（48kHz 原生）→ 音频
```

关键观察：

1. **参考音频进入两次，角色不同。** 进 S1 时贡献的是韵律延续的 prompt（语义 token 序列）；进 S2 时贡献的是音色条件（mel）。GPT-SoVITS 用户早已发现"换参考音频就换情绪"——那是 S1 在起作用。
2. **语义 token 是两段的唯一接口。** 它编码"说什么 + 怎么说（韵律）"，少量泄漏说话人特征（HuBERT 特征并非完全说话人无关，见 §8 风险 R2）。
3. **S2 的血统是 SVC（歌声转换）。** 它的本职就是"给定内容与韵律，换一个音色渲染"——恰好是交叉组合（B 的情绪 × A 的音色）需要的能力。

---

## 3. 总体设计：双轴曲线控制

RefCurve-SoVITS 的控制面由两条独立的 α 曲线构成：

```
                    情绪/韵律轴（S1）                 音色轴（S2）
参考 A ────┬─→  α₁(pos) 加权 logit 混合  ─┬─→ 共享语义 token ─→  α₂(t, n) 加权速度场混合 ─→ 音频
参考 B ────┘                              │                    ↑
                                          └────────────────────┘
```

| 轴 | 所在阶段 | 混合对象 | 曲线维度 | 控制的感知量 |
| --- | --- | --- | --- | --- |
| **情绪轴 α₁** | S1（AR 解码） | 每步的 logits | token 位置（≈ 音频时间） | 情绪、语调、节奏、停顿 |
| **音色轴 α₂** | S2（CFM 采样） | 每步的速度场 | ODE 时间 t × mel 帧位置 n | 声线、音质、共振峰特征 |

两轴的四个典型工作点：

| α₁（情绪） | α₂（音色） | 效果 |
| --- | --- | --- |
| 偏 A | 偏 A | 纯 A（单参考基线） |
| 偏 B | 偏 A | **情绪迁移**：A 的声线说出 B 的情绪 —— F5 上三参考方案失败的目标，在此为原生能力 |
| 中间值 | 偏 A | A 的声线、A/B 之间的情绪插值（情绪强度旋钮） |
| 中间值 | 中间值 | 完全混合（RefCurve-F5 的双参考混合等价物） |

**这是本项目相对 F5 版本的本质升级**：F5 版本只有一个混合旋钮作用在纠缠的表示上；本项目有两个正交旋钮分别作用在解耦的表示上。

产品形态上默认暴露"两段参考 + 两条曲线"（情绪曲线、音色曲线）。进阶形态允许 S1 与 S2 使用不同的参考对（最多四段参考），实现"情绪取自 C/D 对、音色取自 A/B 对"。

---

## 4. 情绪轴：S1 logit 混合详细设计

### 4.1 解码循环改造

上游单参考解码（示意）：

```python
# prompt = [phones(ref_text) ++ phones(gen_text)] 与 semantic(ref_audio)
# KV cache 初始化后逐 token 采样
for step in range(max_len):
    logits = model.forward_one_step(last_token, kv_cache)
    token  = sample(logits, top_k, top_p, temperature, repetition_penalty)
    if token == EOS: break
    tokens.append(token)
```

双参考混合解码：

```python
# 两套独立 prompt（关键教训：per-branch 文本，各用自己的参考转写）
# prompt_A = [phones(ref_text_A) ++ phones(gen_text)] + semantic(ref_audio_A)
# prompt_B = [phones(ref_text_B) ++ phones(gen_text)] + semantic(ref_audio_B)
# 两套独立 KV cache；生成区 token 共享

for step in range(max_len):
    logits_a = model.forward_one_step(last_token, kv_cache_A)   # 各自条件
    logits_b = model.forward_one_step(last_token, kv_cache_B)
    alpha    = alpha1_of_pos(step)                              # 位置曲线
    mixed    = alpha * logits_a + (1 - alpha) * logits_b        # PoE 混合，见 4.2
    token    = sample(mixed, top_k, top_p, temperature, rep_penalty)
    if token == EOS: break
    tokens.append(token)          # 同一个 token 喂回两个分支 → 单一时间轴
    last_token = token
```

**核心性质：生成的 token 序列只有一条。** 两个分支从下一步起共享全部已生成内容，只在"接下来往哪走"上博弈。这天然解决了 F5 output 模式里 DTW 拼命想解决的问题——**没有两条独立时间轴，就没有对齐问题**。时长由混合分布下的 EOS 采样决定，自动落在两种风格的自然时长之间。

### 4.2 混合语义：PoE（默认）与 MoE（备选）

logit 线性插值后过 softmax，数学上等价于两个分布的**加权几何平均**（Product of Experts）：

```
softmax(α·l_a + (1-α)·l_b) ∝ p_a^α · p_b^(1-α)
```

- **PoE（logit 混合，默认）**：共识型。只在两个分布都认可的 token 上留概率，韵律走"双方都觉得合理"的路径，过渡平滑。风险：两种风格差异极大时交集趋于平庸（中性化）。这是 F5 上"α≈0.5 含混"问题在 AR 域的对应物，但预期更温和——CFG 与 contrastive decoding 都是同类操作，有大量成功先例。
- **MoE（概率混合，备选模式）**：`p = α·p_a + (1-α)·p_b`。实现上等价于**每个 token 先掷 α 硬币选专家、再从被选分布采样**。保留双峰性，风格保真度高，但可能逐 token 在两种风格间抖动。作为可切换模式实现（`mix_mode="poe"|"moe"`），用 Phase 0 评测协议实测比较。

运算顺序（必须固定）：`原始 logits → α 混合 → repetition_penalty（基于共享序列）→ temperature → top-k/top-p → 采样`。惩罚项作用在混合后，因为它依赖的已生成序列是共享的。

### 4.3 位置曲线 α₁(pos)

- token 索引 ≈ 线性对应音频时间（帧率见核对项 #6），位置曲线即时间曲线
- 总长在 EOS 前未知。曲线的归一化位置 `u = step / L_est`，`L_est` 用生成文本音素数 × 参考的 token/音素比估计，`u` clamp 到 [0,1]
- 需要精确对齐时提供 two-pass 模式：第一遍固定 α 生成得到真实长度，第二遍按真实长度走曲线（成本 2×，质量场景可接受）
- 曲线形状复用 F5 版的调度器：`linear / cosine / sigmoid / callable / 逐点数组`（`mix_curve_editor.html` 画出来的就是逐点数组）

### 4.4 工程细节

- **KV cache 显存**：两套 cache，~300M 模型完全无压力
- **EOS**：混合分布下自然采样；保底 max_len = 文本长度估计 × 安全系数
- **单参考快捷路径**：`ref_B is None` 时走上游原始代码路径，**逐位一致**（F5 版的铁律，回归测试点）
- **批处理**：初版不做 batch>1，与 F5 版一致

---

## 5. 音色轴：S2 速度场混合移植

这是 RefCurve-F5 `cfm.py` pred 模式的直接平移。假设核对项 #3 成立（S2 为 F5 式 inpainting），改造与 F5 版几乎同构：

```python
# S2 CFM 采样循环内（每个 ODE 步）
def fn(t, x):
    v_a = dit(x, cond=ref_mel_A, semantic=shared_tokens, t=t)   # 各自音色条件
    v_b = dit(x, cond=ref_mel_B, semantic=shared_tokens, t=t)
    alpha = alpha2_of(t, n)          # [1, n, 1]，t 维 × n 维组合，复用 F5 的 combined_alpha 逻辑
    return alpha * v_a + (1 - alpha) * v_b
```

与 F5 版的差异点：

| 项 | F5 版 | SoVITS 版 |
| --- | --- | --- |
| 文本条件 | 音素序列，需 per-branch 文本 | **语义 token（共享）**——两分支天然同一内容与韵律，per-branch 问题不存在 |
| CFG | 有（每分支 2 次 forward） | 待核对（#4）；若无则恰好 2× 开销 |
| duration | 显式估计（bug 重灾区） | 由共享语义 token 长度决定，**无需估计** |
| prompt 裁剪 | union mask 一堆坑 | 各分支用自己参考 mel 的前缀长度；生成区由语义 token 对齐，核对 #3 后确认细节 |
| 混合算法 | lerp/slerp/log（cond 模式用） | **仅 lerp**——速度场线性可加，slerp/log 对切空间向量无意义（F5 版已论证） |
| t 维曲线分辨率 | 16–32 NFE 步 | 受 shortcut-CFM 步数限制（核对 #5）；若默认 8 步，t 曲线只有 8 个采样点，n 维曲线不受影响 |

**不移植的东西**（在 F5 上已证伪或无对应物）：cond 模式的 mel 条件预混合（S2 条件结构不同，且价值有限）、output/DTW 模式（死路）、三参考速度场迁移（死路，且其目标已被双轴设计原生覆盖）。

---

## 6. 曲线语义映射表（F5 → SoVITS）

| RefCurve-F5 概念 | RefCurve-SoVITS 对应物 | 备注 |
| --- | --- | --- |
| `mix_on="pred"` | S2 速度场混合（音色轴） | 直接移植 |
| `mix_on="cond"` | 无对应，不移植 | — |
| `mix_on="output"` | 无对应，不移植 | — |
| t 维曲线（`mix_schedule`, `mix_a_start/end`） | S2 的 ODE 时间维曲线 | 分辨率受采样步数限制 |
| n 维曲线（`n_schedule`, `n_a_start/end`） | S2 的 mel 帧位置曲线 **和** S1 的 token 位置曲线（两处独立） | S1 的是新增维度 |
| `mix_2d_mode` / `mix_2d_weights` | S2 保留 2D 网格；S1 为 1D 逐点数组 | 编辑器复用 |
| `allow_extrapolation` | 两轴都保留（α 出 [0,1] 即外推/反向） | S1 的 logit 外推 = contrastive decoding，有先例但需实测稳定性 |
| per-branch 文本 | S1 的 prompt_A/prompt_B 各用自己参考转写 | F5 学费直接继承 |
| 单参考快捷路径逐位一致 | 两轴各自保留 | 回归测试点 |

---

## 7. 实施计划（Milestone 分解）

### M0：环境与代码锚点（0.5–1 天）
- clone GPT-SoVITS，跑通 v4 日文 stock 推理（含 pyopenjtalk 前端）
- 逐项核对 §0 的 8 个事实，修订本文档
- 保存一组 stock 输出作为逐位一致基线
- **验收**：核对清单全部落实；单参考日文推理产出正常音频

### M1：音色轴——S2 速度场混合（2–4 天）
- 移植 pred 模式到 S2 CFM 循环；`ref_B=None` 时逐位等于 stock
- 常数 α 扫描 {0, 0.25, 0.5, 0.75, 1}：两端应分别等于纯 A / 纯 B 音色
- 接入 t/n 曲线调度器（从 F5 `cfm.py` 平移 `alpha_of_t/alpha_of_n/combined_alpha`）
- **验收**：α 单调性——说话人 embedding 相似度（对 A）随 α 单调；语义 token 固定时内容完全不变（CER 恒定）
- 先做 M1 的理由：有 F5 现成代码，置信度最高；且单独的音色 morphing 已有独立价值

### M2：情绪轴——S1 logit 混合（3–5 天）
- 双 KV cache 解码循环；PoE 混合 + 采样顺序按 §4.2；`ref_B=None` 逐位等于 stock
- 常数 α₁ 扫描；然后接位置曲线；实现 MoE 备选模式
- **验收**：α₁ 单调性——emotion2vec 情绪相似度（对 B）随 (1-α₁) 单调；固定 α₂=纯A 时说话人相似度保持平坦（见 M3 的轴独立性）
- 风险预案：α₁≈0.5 若出现韵律含混/不稳，切 MoE 模式对比；仍不行则退化为"曲线控制的硬切换"（α 只取 0/1，靠曲线控制切换时机）——表达力略降但依然是 F5 做不到的能力

### M3：双轴联合与交叉组合验证（2–3 天）
- 联调：情绪曲线偏 B + 音色曲线偏 A = 情绪迁移
- **轴独立性量化测试**（本项目方法论的核心主张，必须量化）：
  - 扫 α₁、固定 α₂ → 情绪指标应显著移动，说话人相似度漂移应 < 阈值
  - 扫 α₂、固定 α₁ → 说话人相似度应显著移动，情绪指标漂移应 < 阈值
  - 漂移超阈值的部分即语义 token 的音色泄漏量（风险 R2 的量化）
- **验收**：交叉组合的主观听感 + 上述两条正交性曲线

### M4：产品化（2–3 天）
- 复用 `mix_curve_editor.html` + FastAPI 后端模式：页面画两条曲线（情绪 α₁ 一维、音色 α₂ 支持 2D 网格）
- Gradio 试验台：双参考 + 两条曲线 + 双轴预设（情绪迁移 / 音色 morphing / 完全混合）
- **验收**：从上传参考到出音频的完整闭环

里程碑串行依赖：M0 → M1 → M2 → M3 → M4。M1 与 M2 若两人可并行。

---

## 8. 风险清单

| # | 风险 | 影响 | 缓解 |
| --- | --- | --- | --- |
| R1 | PoE 混合在风格差异大时中性化（α≈0.5 平庸） | 情绪插值中段表达力弱 | MoE 备选模式；温度补偿；最坏退化为曲线控制的硬切换（见 M2 预案） |
| R2 | 语义 token 泄漏音色，S2 覆盖不完全 | 交叉组合时 A 声线带 B 底色 | M3 轴独立性测试量化泄漏量；增大 S2 参考条件权重 / 采样步数；可接受阈值由动漫配音实际需求定 |
| R3 | shortcut-CFM 步数少，t 曲线分辨率低 | S2 的 t 维控制变粗 | n 维曲线不受影响；必要时强制高步数采样（牺牲速度） |
| R4 | AR 解码在混合 logits 下的稳定性（复读、漏字） | 生成失败率上升 | 集成通常增稳而非减稳（先例：CFG）；保留 rep_penalty；MoE 模式做 fallback；失败率纳入评测 |
| R5 | 上游代码版本漂移（v4 → v5 重构） | 改造点漂移 | pin 具体 commit 开发；混合逻辑尽量收敛为独立模块，减少侵入面 |
| R6 | S1/S2 各自的实现假设不成立（核对项 #3/#4） | M1/M2 方案需调整 | M0 强制先行，本文档随核对结果修订 |

---

## 9. 评测协议（与 Phase 0 共享）

所有里程碑的验收使用同一套自动化协议，脚本模式从 RefCurve-F5 的验证脚本平移：

| 指标 | 工具 | 用途 |
| --- | --- | --- |
| 情绪相似度 | emotion2vec（FunASR/ModelScope 生态，F5 项目已有依赖习惯） | 情绪轴有效性 |
| 说话人相似度 | ERes2Net 或 ECAPA-TDNN 余弦 | 音色轴有效性 + 泄漏量化 |
| 可懂度 | 日文 ASR（Whisper large-v3 或 ReazonSpeech）CER | 混合不破坏内容 |
| 单调性 | 上述指标对 α ∈ {0, 0.25, 0.5, 0.75, 1} 的响应曲线 | "曲线即方法论"的可证伪检验 |
| 轴独立性 | 交叉扫描矩阵（见 M3） | 双轴主张的量化证据 |
| 稳定性 | 复读/截断/静音失败率（批量生成统计） | R4 监控 |

固定测试集：从精选日文素材中选 N 组同声优强情绪对（怒/悲/喜 × 平静基线），固定生成文本集（含长短句、疑问/感叹），固定 seed。每次里程碑验收跑全量，结果入库对比。

---

## 10. 从 RefCurve-F5 继承的资产清单

**直接复用（拷贝级）：**
- 曲线调度器逻辑：`cfm.py` 的 `alpha_of_t` / `alpha_of_n` / `combined_alpha`（含 `n_normalize_to_ref` 的经验）
- `tools/mix_curve_editor.html` + `tools/mix_curve_server.py` 的前后端模式
- 评测脚本模式（ASR 字数校验 + 相似度 + α 扫描）

**方法论级继承（原则，不是代码）：**
1. per-branch 文本：每个分支的条件必须与它自己的参考自洽（F5 上最贵的一课）
2. 单参考快捷路径必须与上游逐位一致，作为永久回归测试
3. 速度场只做线性混合；slerp/log 只对静态表示有意义
4. 缓存失效：任何按分支变化的条件都要检查上游缓存是否按内容区分（F5 的 DiT text cache 教训 → 本项目的 KV cache 天然按分支独立，但 S2 若有类似缓存需排查）
5. 先建评测协议再调参数——听感争议用指标裁决
6. 修改混合逻辑后最快的回归 = 单参考路径 + α∈{0,1} 两端点检查

**明确不带过去的（死路清单，避免重复踩坑）：**
- 波形域差分、mel 域 DTW 对齐混合、三参考速度场迁移（失败分析见 RefCurve-F5 仓库 memory 与 commit 历史）

---

## 11. 与整体路线图的关系

本项目是四步路线（见 F5 仓库 memory `japanese-emotion-roadmap`）中的平行轨道：

- **Phase 0**（Jmica 日文 F5 基线测试 + 评测协议搭建）的评测协议与本项目 §9 完全共享——先搭协议，两个项目都受益
- Phase 1/2（F5 侧微调与续训）与本项目独立推进，最终按 M3 的轴独立性数据与 Phase 1 的微调效果对比，决定主力平台
- 万小时日文数据在本项目的潜在用途：GPT-SoVITS 底模微调（官方支持 fine-tune 流程），以及未来"统计均值情绪方向向量"研究（该思路在 AR 的 logit 空间同样成立：对 N 对同角色情绪对的 logit 差取均值，得到可注入的情绪方向——比 F5 上更可行，因为 S1 的生成区不存在"条件全零"问题）

---

*本文档由 RefCurve-F5 项目经验总结生成，随 M0 核对结果滚动修订。*
