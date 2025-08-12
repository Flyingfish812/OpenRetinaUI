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
    
class L1Smooth3DRegularizer:
    def __init__(
        self,
        sparsity_factor: float = 1e-4,
        smoothness_factor: float = 1e-4,
        padding_mode: str = 'constant',
        center_mass_factor: float | Iterable[float] = 1e-2,
        target_center: tuple[float, float, float] = (0.5, 0.5, 0.5)
    ):
        self.sparsity_factor = sparsity_factor
        self.smoothness_factor = smoothness_factor
        self.padding_mode = padding_mode
        self.target_center = target_center

        # 支持对质心三部分（中心偏移、紧凑度、峰值）分别加权
        if isinstance(center_mass_factor, Iterable):
            factors = list(center_mass_factor)
            assert len(factors) == 3, "center_mass_factor must be float or iterable of 3 floats"
            self.cm_center, self.cm_compact, self.cm_peak = factors
        else:
            self.cm_center = self.cm_compact = self.cm_peak = center_mass_factor

        # 构造 3x3x3 的离散 Laplacian 核（面邻居 6 个，中心 -6）
        kernel3d = torch.zeros((3, 3, 3), dtype=torch.float32)
        kernel3d[1, 1, 1] = -6.0
        # depth 方向
        kernel3d[0, 1, 1] = 1.0
        kernel3d[2, 1, 1] = 1.0
        # height 方向
        kernel3d[1, 0, 1] = 1.0
        kernel3d[1, 2, 1] = 1.0
        # width 方向
        kernel3d[1, 1, 0] = 1.0
        kernel3d[1, 1, 2] = 1.0

        self.registered_kernel = kernel3d.view(1, 1, 3, 3, 3)

    def __call__(self, weights: torch.Tensor) -> torch.Tensor:
        """
        weights: [out_channels, in_channels, D, H, W]
        返回一个标量张量，表示所有正则项的加权和。
        """
        reg = torch.tensor(0.0, device=weights.device)

        # 1) L1 稀疏
        if self.sparsity_factor:
            reg = reg + self.sparsity_factor * torch.sum(torch.abs(weights))

        # 2) 平滑度（3D Laplacian）
        if self.smoothness_factor:
            # 将每个输入-输出通道对展开成单独的“图像”
            w = weights.permute(1, 0, 2, 3, 4)   # [in, out, D, H, W]
            B, C, D, H, W = w.shape
            w = w.reshape(-1, 1, D, H, W)       # [B*C, 1, D, H, W]

            # 边界填充
            if self.padding_mode == 'symmetric':
                pad = nn.ReflectionPad3d(1)
            else:
                pad = nn.ConstantPad3d(1, 0.0)
            w_pad = pad(w).expand(-1, w.shape[0], -1, -1, -1)

            # 拉到设备并复制为每个组
            lap = self.registered_kernel.to(weights.device)
            lap = lap.repeat(w.shape[0], 1, 1, 1, 1)

            # 分组卷积
            x_lap = F.conv3d(w_pad, lap, groups=w.shape[0])
            num = torch.sum(x_lap**2, dim=(1, 2, 3, 4))
            den = 1e-8 + torch.sum(w**2,   dim=(1, 2, 3, 4))
            smooth_reg = torch.sum(num / den)

            reg = reg + self.smoothness_factor * smooth_reg

        # 3) 质心相关：中心偏移、紧凑度、峰值比
        if any(f > 0 for f in (self.cm_center, self.cm_compact, self.cm_peak)):
            # 按输出通道合并输入通道后计算
            norm_w = weights.pow(2).sum(dim=1)  # [out, D, H, W]
            B, D, H, W = norm_w.shape

            # 坐标归一化到 [0,1]
            z = torch.linspace(0, 1, D, device=weights.device).view(1, D, 1, 1)
            y = torch.linspace(0, 1, H, device=weights.device).view(1, 1, H, 1)
            x = torch.linspace(0, 1, W, device=weights.device).view(1, 1, 1, W)

            total = norm_w.sum(dim=(1, 2, 3), keepdim=True) + 1e-8
            mass  = norm_w / total  # 质量分布

            # ——— 中心偏移 ———
            cz = (mass * z).sum(dim=(1, 2, 3))
            cy = (mass * y).sum(dim=(1, 2, 3))
            cx = (mass * x).sum(dim=(1, 2, 3))
            d_center = ((cz - self.target_center[0])**2 +
                        (cy - self.target_center[1])**2 +
                        (cx - self.target_center[2])**2)
            reg = reg + self.cm_center * d_center.mean()

            # ——— 紧凑度 ———
            dist2 = ((z - cz.view(-1, 1, 1, 1))**2 +
                     (y - cy.view(-1, 1, 1, 1))**2 +
                     (x - cx.view(-1, 1, 1, 1))**2)
            compactness = (mass * dist2).sum(dim=(1, 2, 3))
            reg = reg + self.cm_compact * compactness.mean()

            # ——— 峰值比 ———
            max_val  = norm_w.view(B, -1).max(dim=1)[0]
            mean_val = norm_w.view(B, -1).mean(dim=1)
            peak_ratio = mean_val / (max_val + 1e-8)
            reg = reg + self.cm_peak * peak_ratio.mean()

        return reg