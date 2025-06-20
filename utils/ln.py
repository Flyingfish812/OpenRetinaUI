import torch
import torch.nn as nn
import torch.nn.functional as F
from openretina.modules.core.base_core import Core
from openretina.modules.readout.base import Readout
from openretina.models.core_readout import BaseCoreReadout
from utils.activations import build_activation_layer
from utils.model import CorrelationLoss2D
from typing import Iterable, Optional, Any


class L1Smooth2DRegularizer:
    def __init__(self, sparsity_factor=1e-4, smoothness_factor=1e-4, padding_mode='constant'):
        self.sparsity_factor = sparsity_factor
        self.smoothness_factor = smoothness_factor
        self.padding_mode = padding_mode

        # Laplacian kernel (3x3)
        self.registered_kernel = torch.tensor(
            [[0.25, 0.5, 0.25],
             [0.5, -3.0, 0.5],
             [0.25, 0.5, 0.25]],
            dtype=torch.float32
        ).view(1, 1, 3, 3)  # shape: (out_channels=1, in_channels=1, H, W)

    def __call__(self, weights: torch.Tensor) -> torch.Tensor:
        reg = torch.tensor(0.0, device=weights.device)

        # L1 sparsity
        if self.sparsity_factor:
            reg += self.sparsity_factor * torch.sum(torch.abs(weights))

        # Smoothness via Laplacian
        if self.smoothness_factor:
            # weights shape: [out_channels, in_channels, H, W] (PyTorch)
            w = weights.permute(1, 0, 2, 3)  # [in, out, H, W]
            B, C, H, W = w.shape

            w = w.reshape(-1, 1, H, W)  # [B*C, 1, H, W]

            pad = nn.ReflectionPad2d(1) if self.padding_mode == 'symmetric' else nn.ConstantPad2d(1, 0.0)
            w_pad = pad(w)

            lap_kernel = self.registered_kernel.to(weights.device)
            lap_kernel = lap_kernel.repeat(w.shape[0], 1, 1, 1)  # depthwise conv style

            x_lap = F.conv2d(w_pad, lap_kernel, groups=w.shape[0])
            tmp1 = torch.sum(x_lap ** 2, dim=(1, 2, 3))
            tmp2 = 1e-8 + torch.sum(w ** 2, dim=(1, 2, 3))

            smoothness_reg = torch.sum(tmp1 / tmp2)
            reg += self.smoothness_factor * smoothness_reg.sqrt()

        return reg

class LNCore2D(Core):
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
            self.kernel_size = (kernel_size, kernel_size)
        else:
            if len(kernel_size) == 1:
                self.kernel_size = (kernel_size[0], kernel_size[0])
            elif len(kernel_size) == 2:
                self.kernel_size = kernel_size
        self.kernel_constraint = kernel_constraint

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=1,
            kernel_size=self.kernel_size,
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

        self.regularizer_module = L1Smooth2DRegularizer(
            sparsity_factor=sparsity_factor,
            smoothness_factor=smoothness_factor
        )

    def forward(self, x):
        # x: [B, C, H, W]
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

class LNReadout2D(Readout):
    def __init__(
        self,
        mask_size: Iterable[int],  # (H, W)
        num_neurons: int,
        activation: str,
    ):
        super().__init__()
        self.mask = nn.Linear(mask_size[0] * mask_size[1], num_neurons)
        self.activation = build_activation_layer(activation)
        self.mask_size = mask_size

    def forward(self, x, **kwarg: Any):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        h, w = self.mask_size
        assert H == h and W == w, f"Input spatial size ({H}, {W}) must match mask size ({h}, {w})"

        # Flatten spatial dims and reshape
        x_flat = x.view(B, C, -1)  # [B, C, H*W] -> [B, H*W]

        masked = self.mask(x_flat)  # [B, N]
        masked = masked.sum(dim=1)
        output = self.activation(masked)  # [B, N]
        return output

    # 定义一个名为regularizer的函数，该函数接受一个可选的data_key参数，返回一个torch.Tensor类型的值
    def regularizer(self, data_key: Optional[str] = None) -> torch.Tensor:
        # 返回一个值为0.0的torch.Tensor，并将其放置在self.parameters()返回的参数所在的设备上
        return torch.tensor(0.0, device=next(self.parameters()).device)

class LNCoreReadout2D(BaseCoreReadout):
    def __init__(
        self,
        in_channels: int,
        input_size: int | Iterable[int],
        num_neurons: int,
        kernel_size: int | Iterable[int],
        activation: str,
        kernel_initializer: Optional[str] = 'truncated_normal',
        sta: Optional[torch.Tensor] = None,
        sparsity_factor: float = 1e-4,
        smoothness_factor: float = 1e-4,
        kernel_constraint: Optional[str] = 'maxnorm',
        loss: Optional[nn.Module] = nn.PoissonNLLLoss(log_input=False),
        correlation_loss: nn.Module | None = CorrelationLoss2D(),
        learning_rate: float = 1e-4,
        data_info: Optional[dict] = None,
        seed: Optional[int] = None,
    ):
        # Step 1: Core
        core = LNCore2D(
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
        if isinstance(input_size, int):
            dummy_input = torch.zeros(1, in_channels, input_size, input_size)
        elif isinstance(input_size, Iterable) and len(input_size) == 2:
            dummy_input = torch.zeros(1, in_channels, *input_size)
        else:
            raise ValueError(f"Invalid image_size: {input_size}")
        with torch.no_grad():
            core_out = core(dummy_input)
        _, _, H, W = core_out.shape
        mask_size = (H, W)
        self.mask_size = mask_size

        # Step 3: Readout
        readout = LNReadout2D(mask_size=mask_size, activation=activation, num_neurons=num_neurons)

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