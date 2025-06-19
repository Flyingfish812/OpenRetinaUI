import torch
import torch.nn as nn
import torch.nn.functional as F
from openretina.modules.core.base_core import Core
from openretina.modules.readout.base import Readout
from openretina.models.core_readout import BaseCoreReadout
from utils.activations import ParametricSoftplus
from typing import Iterable, Optional, Any


class L1Smooth3DRegularizer:
    def __init__(self, sparsity_factor=1e-4, smoothness_factor=1e-4, padding_mode='constant'):
        self.sparsity_factor = sparsity_factor
        self.smoothness_factor = smoothness_factor
        self.padding_mode = padding_mode

        # 3D Laplacian kernel (3x3x3)
        lap3d = torch.ones(3, 3, 3, dtype=torch.float32)
        lap3d[1, 1, 1] = -32.0
        lap3d[2, 1, 1] = 2.0
        lap3d[0, 1, 1] = 2.0
        lap3d[1, 0, 1] = 2.0
        lap3d[1, 2, 1] = 2.0
        lap3d[1, 1, 0] = 2.0
        lap3d[1, 1, 2] = 2.0
        lap3d = lap3d / lap3d.norm()

        self.registered_kernel = lap3d.view(1, 1, 3, 3, 3)  # shape: (out=1, in=1, D, H, W)

    def __call__(self, weights: torch.Tensor) -> torch.Tensor:
        # weights: [out_channels, in_channels, D, H, W]
        reg = torch.tensor(0.0, device=weights.device)

        # L1 sparsity
        if self.sparsity_factor:
            reg += self.sparsity_factor * torch.sum(torch.abs(weights))

        # Smoothness regularization
        if self.smoothness_factor:
            w = weights.permute(1, 0, 2, 3, 4)  # [in, out, D, H, W]
            B, C, D, H, W = w.shape
            w = w.reshape(-1, 1, D, H, W)  # [B*C, 1, D, H, W]

            # Padding
            if self.padding_mode == 'symmetric':
                pad = nn.ReplicationPad3d(1)
            else:
                pad = nn.ConstantPad3d(1, 0.0)
            w_pad = pad(w)

            # Convolution
            lap_kernel = self.registered_kernel.to(weights.device)
            lap_kernel = lap_kernel.repeat(w.shape[0], 1, 1, 1, 1)  # depthwise convolution
            x_lap = F.conv3d(w_pad, lap_kernel, groups=w.shape[0])

            tmp1 = torch.sum(x_lap ** 2, dim=(1, 2, 3, 4))
            tmp2 = 1e-8 + torch.sum(w ** 2, dim=(1, 2, 3, 4))
            smoothness_reg = torch.sum(tmp1 / tmp2).sqrt()

            reg += self.smoothness_factor * smoothness_reg

        return reg

class LNCore3D(Core):
    def __init__(
        self,
        in_channels: int,
        kernel_size: int | Iterable[int],
        kernel_initializer: Optional[str] = 'truncated_normal',
        sta: Optional[torch.Tensor] = None,
        sparsity_factor: float = 0.001,
        smoothness_factor: float = 0.001,
        kernel_constraint: Optional[str] = 'maxnorm',
        seed: Optional[int] = None
    ):
        super().__init__()

        if seed is not None:
            torch.manual_seed(seed)

        if isinstance(kernel_size, int):
            self.kernel_size = (1, kernel_size, kernel_size)
        else:
            if len(kernel_size) == 1:
                self.kernel_size = (1, kernel_size[0], kernel_size[0])
            elif len(kernel_size) == 2:
                self.kernel_size = (1, kernel_size[0], kernel_size[1])
            elif len(kernel_size) == 3:
                self.kernel_size = kernel_size
            else:
                raise ValueError(f"Invalid kernel size: {kernel_size}")
        self.kernel_constraint = kernel_constraint

        self.conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=1,
            kernel_size=kernel_size,
            bias=False
        )

        # Weight initialization
        if kernel_initializer == 'truncated_normal':
            nn.init.trunc_normal_(self.conv.weight, std=0.05)
        elif kernel_initializer == 'STA + truncated normal' and sta is not None:
            with torch.no_grad():
                assert sta.shape == self.conv.weight.shape, "STA shape mismatch"
                noise = torch.normal(0.0, 0.05, size=sta.shape)
                self.conv.weight.data.copy_(sta + noise)
        else:
            raise ValueError(f"Unsupported initializer: {kernel_initializer}")

        self.regularizer_module = L1Smooth3DRegularizer(
            sparsity_factor=sparsity_factor,
            smoothness_factor=smoothness_factor
        )

    def forward(self, x):
        # x: [B, C, T, H, W]
        return self.conv(x)

    def regularizer(self) -> torch.Tensor:
        reg = self.regularizer_module(self.conv.weight)

        if self.kernel_constraint == 'maxnorm':
            with torch.no_grad():
                norms = self.conv.weight.data.norm(2, dim=(2, 3), keepdim=True)
                norms = torch.clamp(norms, min=1e-6)
                self.conv.weight.data /= norms
                self.conv.weight.data *= (norms < 1.0).float() + (norms >= 1.0).float() * 1.0

        return reg

class LNReadout3D(Readout): 
    def __init__(
        self,
        mask_size: Iterable[int],  # (H, W)
        num_neurons: int,
        activation: nn.Module,
    ):
        super().__init__()
        self.mask = nn.Linear(mask_size[0] * mask_size[1], num_neurons)
        self.activation = activation
        self.mask_size = mask_size

    def forward(self, x, **kwarg: Any):
        # x: [B, C, T, H, W]
        B, C, T, H, W = x.shape
        h, w = self.mask_size
        assert H == h and W == w, f"Input spatial size ({H}, {W}) must match mask size ({h}, {w})"

        # Step 1: sum over C → [B, T, H, W]
        x_reduced = x.sum(dim=1)

        # Step 2: flatten spatial dims → [B, T, H*W]
        x_flat = x_reduced.view(B, T, H * W)

        # Step 3: apply mask (Linear) time step by time step
        x_masked = self.mask(x_flat)  # [B, T, N]

        # Step 4: apply activation
        output = self.activation(x_masked)  # [B, T, N]
        return output

    def regularizer(self, data_key: Optional[str] = None) -> torch.Tensor:
        return torch.tensor(0.0, device=next(self.parameters()).device)

class LNCoreReadout3D(BaseCoreReadout):
    def __init__(
        self,
        in_channels: int,
        input_size: int,
        num_neurons: int,
        kernel_size: int | Iterable[int],
        activation: nn.Module,
        kernel_initializer: Optional[str] = 'truncated_normal',
        sta: Optional[torch.Tensor] = None,
        sparsity_factor: float = 0.001,
        smoothness_factor: float = 0.001,
        kernel_constraint: Optional[str] = 'maxnorm',
        loss: Optional[nn.Module] = None,
        correlation_loss: nn.Module | None = None,
        learning_rate: float = 0.002,
        data_info: Optional[dict] = None,
        seed: Optional[int] = None,
    ):
        # Step 1: Core
        core = LNCore3D(
            in_channels=in_channels,
            kernel_size=kernel_size,
            kernel_initializer=kernel_initializer,
            sta=sta,
            sparsity_factor=sparsity_factor,
            smoothness_factor=smoothness_factor,
            kernel_constraint=kernel_constraint,
            seed=seed
        )

        # Step 2: Dummy forward to get output shape
        dummy_input = torch.zeros(1, in_channels, 3, input_size, input_size)
        with torch.no_grad():
            core_out = core(dummy_input)
        _, _, _, H, W = core_out.shape
        mask_size = (H, W)
        self.mask_size = mask_size

        # Step 3: Readout
        readout = LNReadout3D(mask_size=mask_size, activation=activation, num_neurons=num_neurons)

        # Step 4: Assemble
        super().__init__(
            core=core,
            readout=readout,
            learning_rate=learning_rate,
            loss=loss,
            correlation_loss=correlation_loss,
            data_info=data_info
        )

        self.save_hyperparameters()