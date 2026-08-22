"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""
# ruff: noqa: F722 F821

from __future__ import annotations

from random import random
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torchdiffeq import odeint

from f5_tts.model.modules import MelSpec
from f5_tts.model.utils import (
    default,
    exists,
    get_epss_timesteps,
    lens_to_mask,
    list_str_to_idx,
    list_str_to_tensor,
    mask_from_frac_lengths,
)


# ========== 阶段一：两种 cond 混合方法 ==========

def slerp_with_norm(a: torch.Tensor, b: torch.Tensor, alpha: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    SLERP-with-norm: 方向用 SLERP，幅度用 LERP。
    适用于将两个 mel 条件嵌入沿球面插值（方向），同时线性插值它们的能量/幅度。
    
    Args:
        a: [b, n, d] 第一个条件
        b: [b, n, d] 第二个条件
        alpha: [1,1,1] 或标量，a 的权重 (0~1)
    Returns:
        融合结果 [b, n, d]
    """
    # 计算范数
    norm_a = a.norm(dim=-1, keepdim=True).clamp(min=eps)  # [b, n, 1]
    norm_b = b.norm(dim=-1, keepdim=True).clamp(min=eps)
    
    # 归一化方向
    dir_a = a / norm_a  # [b, n, d]
    dir_b = b / norm_b
    
    # 计算夹角的余弦
    cos_theta = (dir_a * dir_b).sum(dim=-1, keepdim=True).clamp(-1.0 + eps, 1.0 - eps)  # [b, n, 1]
    theta = torch.acos(cos_theta)  # 夹角
    
    # 处理接近平行的情况 (theta ≈ 0)，退化为 LERP
    sin_theta = torch.sin(theta).clamp(min=eps)
    
    # SLERP 方向插值: (sin((1-t)*θ)/sin(θ)) * a + (sin(t*θ)/sin(θ)) * b
    # 这里 t = 1 - alpha (因为 alpha 是 a 的权重)
    t = 1.0 - alpha
    w_a = torch.sin((1 - t) * theta) / sin_theta
    w_b = torch.sin(t * theta) / sin_theta
    
    # 对于 theta 很小的情况，使用线性插值
    small_angle_mask = theta.abs() < 1e-4
    w_a = torch.where(small_angle_mask, alpha, w_a)
    w_b = torch.where(small_angle_mask, 1.0 - alpha, w_b)
    
    # 插值后的方向
    dir_fused = w_a * dir_a + w_b * dir_b
    dir_fused = dir_fused / dir_fused.norm(dim=-1, keepdim=True).clamp(min=eps)
    
    # LERP 幅度
    norm_fused = alpha * norm_a + (1 - alpha) * norm_b
    
    return dir_fused * norm_fused


def log_domain_blend(a: torch.Tensor, b: torch.Tensor, alpha: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Log-domain 融合（几何平均）：在对数域进行线性插值，等价于几何加权平均。
    适用于能量/幅度信号的融合，保持乘法特性。
    
    注意：mel 频谱可能包含负值（log mel），所以这里处理的是"符号 + 幅度"模式。
    
    Args:
        a: [b, n, d] 第一个条件
        b: [b, n, d] 第二个条件  
        alpha: [1,1,1] 或标量，a 的权重 (0~1)
    Returns:
        融合结果 [b, n, d]
    """
    # 对于 mel 条件可能含负值的情况，我们取绝对值在 log 域插值，然后恢复符号
    # 另一种做法：直接把负值当作 log-scale 的值处理
    
    # 方法1: 如果 a, b 都是 log-mel (可能为负)，直接线性插值就是 log-domain blend
    # 因为 log(x^α * y^(1-α)) = α*log(x) + (1-α)*log(y)
    # 如果输入已经是 log-mel，直接 LERP 就是几何平均的效果
    
    # 但如果输入是线性 mel，则需要：
    # result = exp(α * log(a) + (1-α) * log(b)) = a^α * b^(1-α)
    
    # F5-TTS 使用的是 log-mel，所以直接 LERP 就是 log-domain blend
    # 这里提供两种模式供用户选择
    
    # 模式1: 假设输入是 log-mel，直接线性插值（等价于几何平均）
    # return alpha * a + (1 - alpha) * b
    
    # 模式2: 假设输入是线性 mel（正值），在 log 域插值
    # 为了鲁棒性，处理符号问题
    sign_a = torch.sign(a)
    sign_b = torch.sign(b)
    
    abs_a = a.abs().clamp(min=eps)
    abs_b = b.abs().clamp(min=eps)
    
    # 在 log 域插值幅度
    log_a = torch.log(abs_a)
    log_b = torch.log(abs_b)
    log_fused = alpha * log_a + (1 - alpha) * log_b
    abs_fused = torch.exp(log_fused)
    
    # 符号处理：当两者同号时保持符号，异号时按权重决定
    # 简单策略：取主导权重的符号，或者按权重插值符号
    sign_fused = torch.where(alpha > 0.5, sign_a, sign_b)
    # 更平滑的策略：同号保持，异号时取加权
    same_sign = (sign_a == sign_b)
    sign_fused = torch.where(same_sign, sign_a, 
                              torch.where(alpha > 0.5, sign_a, sign_b))
    
    return sign_fused * abs_fused


class CFM(nn.Module):
    def __init__(
        self,
        transformer: nn.Module,
        sigma=0.0,
        odeint_kwargs: dict = dict(
            # atol = 1e-5,
            # rtol = 1e-5,
            method="euler"  # 'midpoint'
        ),
        audio_drop_prob=0.3,
        cond_drop_prob=0.2,
        num_channels=None,
        mel_spec_module: nn.Module | None = None,
        mel_spec_kwargs: dict = dict(),
        frac_lengths_mask: tuple[float, float] = (0.7, 1.0),
        vocab_char_map: dict[str:int] | None = None,
    ):
        super().__init__()

        self.frac_lengths_mask = frac_lengths_mask

        # mel spec
        self.mel_spec = default(mel_spec_module, MelSpec(**mel_spec_kwargs))
        num_channels = default(num_channels, self.mel_spec.n_mel_channels)
        self.num_channels = num_channels

        # classifier-free guidance
        self.audio_drop_prob = audio_drop_prob
        self.cond_drop_prob = cond_drop_prob

        # transformer
        self.transformer = transformer
        dim = transformer.dim
        self.dim = dim

        # conditional flow related
        self.sigma = sigma

        # sampling related
        self.odeint_kwargs = odeint_kwargs

        # vocab map for tokenization
        self.vocab_char_map = vocab_char_map

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def sample(
        self,
        cond: float["b n d"] | float["b nw"],
        text: int["b nt"] | list[str],
        duration: int | int["b"],
        *,
        cond_b=None,          # 新增：第二参考
        lens: int["b"] | None = None,
        lens_b=None,          # 新增：第二参考长度（mel 帧数）
        steps=32,
        cfg_strength=1.0,
        sway_sampling_coef=None,
        seed: int | None = None,
        max_duration=65536,
        vocoder: Callable[[float["b d n"]], float["b nw"]] | None = None,
        use_epss=True,
        no_ref_audio=False,
        duplicate_test=False,
        t_inter=0.1,
        edit_mask=None,

        # 新增：融合控制
        mix_schedule="linear",   # "linear" | "cosine" | "sigmoid" | callable
        mix_a_start=0.9,         # t=0 时 a 的权重
        mix_a_end=0.9,           # t=1 时 a 的权重（比如从A渐变到B）
        mix_on="cond",           # "cond" 或 "pred"
        mix_method="lerp",       # "lerp" | "slerp" | "log"  阶段一：三种混合算法
        dynamic_disable_cache=True,
        allow_extrapolation=False,
        
        # 阶段二：mel 帧位置维度控制
        # n_schedule: 沿 mel 帧位置 n 的混合曲线 (可选)
        # 如果为 None，则所有帧使用相同权重
        # 可以是: "linear" | "cosine" | "sigmoid" | callable(n_ratio) -> weight
        # 也可以是 Tensor [n] 直接指定每帧权重
        n_schedule=None,
        n_a_start=None,          # n=0 (音频开头) 时 a 的权重，None 表示不启用 n 维度
        n_a_end=None,            # n=1 (音频结尾) 时 a 的权重
        
        # 2D 混合模式: 如何结合 t 维度和 n 维度的权重
        # "multiply": alpha = alpha_t * alpha_n
        # "add": alpha = (alpha_t + alpha_n) / 2
        # "t_only": 只用 t 维度 (向后兼容)
        # "n_only": 只用 n 维度
        # "2d_grid": 使用 2D 权重网格 [steps, n_frames]
        mix_2d_mode="t_only",
        mix_2d_weights=None,     # 可选: 直接传入 2D 权重矩阵 [steps, n_frames]
    ):
        self.eval()
        # raw wave

        # ---------- 0) 允许 cond 直接传 tuple/list ----------
        if isinstance(cond, (tuple, list)):
            assert len(cond) == 2
            cond, cond_b = cond
        # 未提供第二参考时退化为单参考：cond_b 复用 cond，混合权重恒等于 A，
        # 结果与上游单参考行为一致。这样 trainer.log_samples / speech_edit /
        # eval_infer_batch / benchmark 等上游调用点无需改动即可正常工作。
        single_ref = cond_b is None
        if single_ref:
            cond_b = cond
            if lens_b is None:
                lens_b = lens

        def to_mel(x):
            # x: [b, nw] raw wave or [b, n, d] mel
            if x.ndim == 2:
                x = self.mel_spec(x)      # [b, d, n]
                x = x.permute(0, 2, 1)    # [b, n, d]
                assert x.shape[-1] == self.num_channels
            return x.to(next(self.parameters()).dtype)
        
        cond_a = to_mel(cond)
        cond_b = to_mel(cond_b)

        batch, len_a, device = cond_a.shape[0], cond_a.shape[1], cond_a.device
        len_b = cond_b.shape[1]
        assert cond_b.shape[0] == batch

        if lens is None:
            lens = torch.full((batch,), len_a, device=device, dtype=torch.long)
        if lens_b is None:
            lens_b = torch.full((batch,), len_b, device=device, dtype=torch.long)

        # 两个参考的 mask（之后会 pad 到 max_duration）
        mask_a = lens_to_mask(lens)       # [b, len_a]
        mask_b = lens_to_mask(lens_b)     # [b, len_b]

        # 统一参考长度：用“更长的参考长度”参与 duration 下限约束
        lens_union = torch.maximum(lens, lens_b)

        # text
        if isinstance(text, list):
            if exists(self.vocab_char_map):
                text = list_str_to_idx(text, self.vocab_char_map).to(device)
            else:
                text = list_str_to_tensor(text).to(device)
            assert text.shape[0] == batch

        if isinstance(duration, int):
            duration = torch.full((batch,), duration, device=device, dtype=torch.long)

        duration = torch.maximum(
            torch.maximum((text != -1).sum(dim=-1), lens) + 1, duration
        )  # duration at least text/audio prompt length plus one token, so something is generated
        duration = duration.clamp(max=max_duration)
        max_duration = duration.amax()

        # pad cond_a / cond_b 到 max_duration
        cond_a = F.pad(cond_a, (0, 0, 0, max_duration - len_a), value=0.0)
        cond_b = F.pad(cond_b, (0, 0, 0, max_duration - len_b), value=0.0)

        # pad masks
        mask_a = F.pad(mask_a, (0, max_duration - mask_a.shape[-1]), value=False).unsqueeze(-1)  # [b,n,1]
        mask_b = F.pad(mask_b, (0, max_duration - mask_b.shape[-1]), value=False).unsqueeze(-1)  # [b,n,1]

        # edit_mask 逻辑：对 union mask 生效
        cond_mask = (mask_a | mask_b).squeeze(-1)  # [b,n]
        if edit_mask is not None:
            cond_mask = cond_mask & edit_mask
        cond_mask = cond_mask.unsqueeze(-1)        # [b,n,1]

        if no_ref_audio:
            cond_a = torch.zeros_like(cond_a)
            cond_b = torch.zeros_like(cond_b)

        if batch > 1:
            mask = lens_to_mask(duration)
        else:
            mask = None

        # ---------- 阶段二：构建 n 维度的位置比例 [1, n, 1] ----------
        n_ratio = torch.linspace(0, 1, max_duration, device=device, dtype=cond_a.dtype)
        n_ratio = n_ratio.view(1, -1, 1)  # [1, n, 1] 用于广播

        # ---------- 1) 定义随 t 变化的 alpha_t(t) ----------
        def alpha_of_t(t):
            """计算 t 维度的权重，返回标量"""
            if single_ref:
                # 单参考模式：权重恒为 1（纯 A），与上游行为一致
                return torch.ones((), device=device, dtype=cond_a.dtype)
            if callable(mix_schedule):
                a01 = mix_schedule(t)  # 期望返回 [0,1]
            else:
                if mix_schedule == "linear":
                    a01 = t
                elif mix_schedule == "cosine":
                    a01 = 0.5 - 0.5 * torch.cos(torch.pi * t)
                elif mix_schedule == "sigmoid":
                    a01 = torch.sigmoid(12.0 * (t - 0.5))
                else:
                    raise ValueError(f"unknown mix_schedule: {mix_schedule}")

            # 映射到 [mix_a_start, mix_a_end]
            a = mix_a_start * (1 - a01) + mix_a_end * a01
            return a if allow_extrapolation else a.clamp(0.0, 1.0)

        # ---------- 阶段二：定义随 n 变化的 alpha_n(n_ratio) ----------
        def alpha_of_n():
            """计算 n 维度的权重，返回 [1, n, 1]"""
            if single_ref or n_a_start is None or n_a_end is None:
                # 单参考模式，或未启用 n 维度：返回全 1（不影响最终权重）
                return torch.ones_like(n_ratio)
            
            if callable(n_schedule):
                a01_n = n_schedule(n_ratio)  # [1, n, 1]
            elif isinstance(n_schedule, torch.Tensor):
                # 直接使用传入的权重向量
                a01_n = n_schedule.view(1, -1, 1).to(device=device, dtype=cond_a.dtype)
                # 如果长度不匹配，插值
                if a01_n.shape[1] != max_duration:
                    a01_n = F.interpolate(a01_n.permute(0, 2, 1), size=max_duration, mode='linear', align_corners=True)
                    a01_n = a01_n.permute(0, 2, 1)
                return a01_n if allow_extrapolation else a01_n.clamp(0.0, 1.0)
            else:
                if n_schedule is None or n_schedule == "linear":
                    a01_n = n_ratio
                elif n_schedule == "cosine":
                    a01_n = 0.5 - 0.5 * torch.cos(torch.pi * n_ratio)
                elif n_schedule == "sigmoid":
                    a01_n = torch.sigmoid(12.0 * (n_ratio - 0.5))
                else:
                    raise ValueError(f"unknown n_schedule: {n_schedule}")
            
            # 映射到 [n_a_start, n_a_end]
            a_n = n_a_start * (1 - a01_n) + n_a_end * a01_n
            return a_n if allow_extrapolation else a_n.clamp(0.0, 1.0)
        
        # 预计算 n 维度权重（因为它不随 ODE step 变化）
        alpha_n = alpha_of_n()  # [1, n, 1]

        # ---------- 阶段二：组合 t 和 n 维度的权重 ----------
        def combined_alpha(t):
            """
            根据 mix_2d_mode 组合 t 维度和 n 维度的权重。
            返回 [1, n, 1] 形状的权重（每帧可能不同）。
            """
            alpha_t = alpha_of_t(t)  # 标量
            
            if mix_2d_mode == "t_only":
                # 向后兼容：只用 t 维度
                return alpha_t.view(1, 1, 1).expand(1, max_duration, 1)
            elif mix_2d_mode == "n_only":
                # 只用 n 维度
                return alpha_n
            elif mix_2d_mode == "multiply":
                # 相乘：两个维度都为1时才为1
                out = (alpha_t.view(1, 1, 1) * alpha_n)
                return out if allow_extrapolation else out.clamp(0.0, 1.0)
            elif mix_2d_mode == "add":
                # 平均：两个维度的加权平均
                out = (alpha_t.view(1, 1, 1) + alpha_n) / 2
                return out if allow_extrapolation else out.clamp(0.0, 1.0)
            elif mix_2d_mode == "max":
                # 取较大值
                out = torch.maximum(alpha_t.view(1, 1, 1).expand_as(alpha_n), alpha_n)
                return out if allow_extrapolation else out.clamp(0.0, 1.0)
            elif mix_2d_mode == "min":
                # 取较小值
                out = torch.minimum(alpha_t.view(1, 1, 1).expand_as(alpha_n), alpha_n)
                return out if allow_extrapolation else out.clamp(0.0, 1.0)
            elif mix_2d_mode == "2d_grid":
                # 使用预定义的 2D 权重网格
                if mix_2d_weights is None:
                    raise ValueError("mix_2d_mode='2d_grid' requires mix_2d_weights")
                # mix_2d_weights: [steps, n_frames]
                # 需要根据当前 t 找到对应的行
                # t 在 [0, 1]，steps 行
                weights = mix_2d_weights.to(device=device, dtype=cond_a.dtype)
                n_steps = weights.shape[0]
                step_idx = (t * (n_steps - 1)).long().clamp(0, n_steps - 1)
                row = weights[step_idx]  # [n_frames]
                # 如果帧数不匹配，插值
                if row.shape[0] != max_duration:
                    row = F.interpolate(row.view(1, 1, -1), size=max_duration, mode='linear', align_corners=True)
                    row = row.view(-1)
                out = row.view(1, -1, 1)
                return out if allow_extrapolation else out.clamp(0.0, 1.0)
            else:
                raise ValueError(f"unknown mix_2d_mode: {mix_2d_mode}")

        # ---------- 2) 融合 cond（按 mask 处理 padding 区） ----------
        def fused_cond(t):
            # 获取组合后的 alpha [1, n, 1]
            a = combined_alpha(t).to(cond_a.dtype)
            
            # 根据 mix_method 选择不同的混合算法
            if mix_method == "lerp":
                # 经典线性插值
                fused = a * cond_a + (1 - a) * cond_b
            elif mix_method == "slerp":
                # 方向用 SLERP、幅度用 LERP
                fused = slerp_with_norm(cond_a, cond_b, a)
            elif mix_method == "log":
                # Log-domain 融合（几何平均）
                fused = log_domain_blend(cond_a, cond_b, a)
            else:
                raise ValueError(f"unknown mix_method: {mix_method}, expected 'lerp', 'slerp', or 'log'")
            
            # 只有 A 有数据：直接用 A；只有 B 有数据：直接用 B
            # 使用单次 where 避免顺序依赖
            only_a = mask_a & (~mask_b)
            only_b = mask_b & (~mask_a)
            fused = torch.where(only_a, cond_a, fused)
            fused = torch.where(only_b, cond_b, fused)
            return fused

        # ---------- 3) ODE fn：每个 t 动态算 step_cond ----------
        def fn(t, x):
            # 动态 step_cond
            step_cond = torch.where(cond_mask, fused_cond(t), torch.zeros_like(cond_a))

            # 动态 cond 时，建议关 cache（避免 transformer 复用旧 cond 特征）
            use_cache = not (dynamic_disable_cache and (mix_schedule is not None))

            if mix_on == "cond":
                # 只做一次 forward：便宜
                if cfg_strength < 1e-5:
                    return self.transformer(
                        x=x, cond=step_cond, text=text, time=t, mask=mask,
                        drop_audio_cond=False, drop_text=False, cache=use_cache
                    )
                pred_cfg = self.transformer(
                    x=x, cond=step_cond, text=text, time=t, mask=mask,
                    cfg_infer=True, cache=use_cache
                )
                pred, null_pred = torch.chunk(pred_cfg, 2, dim=0)
                return pred + (pred - null_pred) * cfg_strength

            elif mix_on == "pred":
                # 更“强”的方式：分别用 cond_a / cond_b 算 pred，再按 a(t) 混 pred
                # 代价：每个 step 多一次 transformer forward
                a = alpha_of_t(t).to(cond_a.dtype)

                def guided_pred(step_cond_local):
                    if cfg_strength < 1e-5:
                        return self.transformer(
                            x=x, cond=step_cond_local, text=text, time=t, mask=mask,
                            drop_audio_cond=False, drop_text=False, cache=use_cache
                        )
                    pred_cfg = self.transformer(
                        x=x, cond=step_cond_local, text=text, time=t, mask=mask,
                        cfg_infer=True, cache=use_cache
                    )
                    p, n = torch.chunk(pred_cfg, 2, dim=0)
                    return p + (p - n) * cfg_strength

                step_a = torch.where(cond_mask, cond_a, torch.zeros_like(cond_a))
                step_b = torch.where(cond_mask, cond_b, torch.zeros_like(cond_b))

                pa = guided_pred(step_a)
                pb = guided_pred(step_b)

                return a * pa + (1 - a) * pb

            else:
                raise ValueError(f"unknown mix_on: {mix_on}")

        # noise input
        # to make sure batch inference result is same with different batch size, and for sure single inference
        # still some difference maybe due to convolutional layers
        # ---------- 4) 后面 noise init / odeint / vocoder 基本沿用 ----------
        y0 = []
        for dur in duration:
            if exists(seed):
                torch.manual_seed(seed)
            y0.append(torch.randn(dur, self.num_channels, device=self.device, dtype=cond_a.dtype))
        y0 = pad_sequence(y0, padding_value=0, batch_first=True)

        t_start = 0
        if duplicate_test:
            t_start = t_inter
            # 如果你也想支持 duplicate_test，需要自己决定 test_cond 该用 fused_cond(t_start) 还是某个固定参考
            test_cond = fused_cond(torch.tensor(t_start, device=self.device, dtype=cond_a.dtype))
            y0 = (1 - t_start) * y0 + t_start * test_cond
            steps = int(steps * (1 - t_start))

        if t_start == 0 and use_epss:
            t = get_epss_timesteps(steps, device=self.device, dtype=cond_a.dtype)
        else:
            t = torch.linspace(t_start, 1, steps + 1, device=self.device, dtype=cond_a.dtype)
        if sway_sampling_coef is not None:
            t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)

        trajectory = odeint(fn, y0, t, **self.odeint_kwargs)
        self.transformer.clear_cache()

        out = trajectory[-1]

        # 最终把参考段“钉死”为 t=1 的融合结果（避免参考段被采样扰动）
        final_cond = fused_cond(torch.tensor(1.0, device=self.device, dtype=cond_a.dtype))
        out = torch.where(cond_mask, final_cond, out)

        if exists(vocoder):
            out = out.permute(0, 2, 1)
            out = vocoder(out)

        return out, trajectory

    def forward(
        self,
        inp: float["b n d"] | float["b nw"],  # mel or raw wave
        text: int["b nt"] | list[str],
        *,
        lens: int["b"] | None = None,
        noise_scheduler: str | None = None,
    ):
        # handle raw wave
        if inp.ndim == 2:
            inp = self.mel_spec(inp)
            # inp1 = self.mel_spec(inp1)
            # inp2 = self.mel_spec(inp2)
            # inp = func(inp1, inp2)
            

            inp = inp.permute(0, 2, 1)
            assert inp.shape[-1] == self.num_channels

        batch, seq_len, dtype, device, _σ1 = *inp.shape[:2], inp.dtype, self.device, self.sigma

        # handle text as string
        if isinstance(text, list):
            if exists(self.vocab_char_map):
                text = list_str_to_idx(text, self.vocab_char_map).to(device)
            else:
                text = list_str_to_tensor(text).to(device)
            assert text.shape[0] == batch

        # lens and mask
        if not exists(lens):  # if lens not acquired by trainer from collate_fn
            lens = torch.full((batch,), seq_len, device=device)
        mask = lens_to_mask(lens, length=seq_len)

        # get a random span to mask out for training conditionally
        frac_lengths = torch.zeros((batch,), device=self.device).float().uniform_(*self.frac_lengths_mask)
        rand_span_mask = mask_from_frac_lengths(lens, frac_lengths)

        if exists(mask):
            rand_span_mask &= mask

        # mel is x1
        x1 = inp

        # x0 is gaussian noise
        x0 = torch.randn_like(x1)

        # time step
        time = torch.rand((batch,), dtype=dtype, device=self.device)
        # TODO. noise_scheduler

        # sample xt (φ_t(x) in the paper)
        t = time.unsqueeze(-1).unsqueeze(-1)
        φ = (1 - t) * x0 + t * x1
        flow = x1 - x0

        # only predict what is within the random mask span for infilling
        cond = torch.where(rand_span_mask[..., None], torch.zeros_like(x1), x1)

        # transformer and cfg training with a drop rate
        drop_audio_cond = random() < self.audio_drop_prob  # p_drop in voicebox paper
        if random() < self.cond_drop_prob:  # p_uncond in voicebox paper
            drop_audio_cond = True
            drop_text = True
        else:
            drop_text = False

        # apply mask will use more memory; might adjust batchsize or batchsampler long sequence threshold
        pred = self.transformer(
            x=φ, cond=cond, text=text, time=time, drop_audio_cond=drop_audio_cond, drop_text=drop_text, mask=mask
        )

        # flow matching loss
        loss = F.mse_loss(pred, flow, reduction="none")
        loss = loss[rand_span_mask]

        return loss.mean(), cond, pred
