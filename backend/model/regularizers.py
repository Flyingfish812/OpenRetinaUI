import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable

class L1Smooth2DRegularizer:
    def __init__(
        self,
        sparsity_factor=1e-4,
        smoothness_factor=1e-4,
        padding_mode='constant',
        center_mass_factor=1e-2,  # 质心正则强度
        target_center=(0.5, 0.5)  # 目标中心位置 (cy, cx)，单位为比例
    ):
        self.sparsity_factor = sparsity_factor
        self.smoothness_factor = smoothness_factor
        self.padding_mode = padding_mode
        self.target_center = target_center

        if isinstance(center_mass_factor, Iterable):
            factors = list(center_mass_factor)
            assert len(factors) == 3, "center_mass_factor must be float or iterable of 3 floats"
            self.cm_center, self.cm_compact, self.cm_peak = factors
        else:
            self.cm_center = self.cm_compact = self.cm_peak = center_mass_factor

        # Laplacian kernel (3x3)
        self.registered_kernel = torch.tensor(
            [[0.25, 0.5, 0.25],
             [0.5, -3.0, 0.5],
             [0.25, 0.5, 0.25]],
            dtype=torch.float32
        ).view(1, 1, 3, 3)  # shape: (1, 1, 3, 3)

    def __call__(self, weights: torch.Tensor) -> torch.Tensor:
        # weights shape: [out_channels, in_channels, H, W]
        reg = torch.tensor(0.0, device=weights.device)

        # L1 sparsity
        if self.sparsity_factor:
            reg += self.sparsity_factor * torch.sum(torch.abs(weights))

        # Smoothness via Laplacian
        if self.smoothness_factor:
            w = weights.permute(1, 0, 2, 3)  # [in, out, H, W]
            B, C, H, W = w.shape
            w = w.reshape(-1, 1, H, W)  # [B*C, 1, H, W]

            pad = nn.ReflectionPad2d(1) if self.padding_mode == 'symmetric' else nn.ConstantPad2d(1, 0.0)
            w_pad = pad(w).expand(-1, w.shape[0], -1, -1)

            lap_kernel = self.registered_kernel.to(weights.device)
            lap_kernel = lap_kernel.repeat(w.shape[0], 1, 1, 1)

            x_lap = F.conv2d(w_pad, lap_kernel, groups=w.shape[0])
            tmp1 = torch.sum(x_lap ** 2, dim=(1, 2, 3))
            tmp2 = 1e-8 + torch.sum(w ** 2, dim=(1, 2, 3))
            smoothness_reg = torch.sum(tmp1 / tmp2)
            # reg += self.smoothness_factor * smoothness_reg.sqrt()
            reg += self.smoothness_factor * smoothness_reg

        # Center of mass regularization
        if any(f > 0 for f in [self.cm_center, self.cm_compact, self.cm_peak]):
            norm_weights = weights.pow(2).sum(dim=1)  # [B, H, W]
            B, H, W = norm_weights.shape

            y = torch.linspace(0, 1, H, device=weights.device).view(1, H, 1)
            x = torch.linspace(0, 1, W, device=weights.device).view(1, 1, W)

            total = norm_weights.sum(dim=(1, 2), keepdim=True) + 1e-8
            mass = norm_weights / total

            # --- Center deviation ---
            cy = (mass * y).sum(dim=(1, 2))
            cx = (mass * x).sum(dim=(1, 2))
            d_center = (cy - self.target_center[0]) ** 2 + (cx - self.target_center[1]) ** 2
            reg += self.cm_center * d_center.mean()

            # --- Compactness ---
            dist2 = ((y - cy.view(-1, 1, 1)) ** 2 + (x - cx.view(-1, 1, 1)) ** 2)
            compactness = (mass * dist2).sum(dim=(1, 2))
            reg += self.cm_compact * compactness.mean()

            # --- Peakness ---
            max_val = norm_weights.view(B, -1).max(dim=1)[0]
            mean_val = norm_weights.view(B, -1).mean(dim=1)
            peak_ratio = mean_val / (max_val + 1e-8)
            reg += self.cm_peak * peak_ratio.mean()

        return reg