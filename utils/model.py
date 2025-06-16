import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Iterable, Optional, Any
from utils.activations import build_activation_layer
from openretina.modules.core.base_core import Core
from openretina.modules.readout.base import Readout
from openretina.models.core_readout import BaseCoreReadout

# 2D Loss Functions
class CorrelationLoss2D(nn.Module):
    def __init__(self, eps: float = 1e-16, per_neuron: bool = False, avg: bool = False):
        super().__init__()
        self.eps = eps
        self.per_neuron = per_neuron
        self.avg = avg

    def forward(self, output, target):
        # output: [B, N]
        # target: [B, N]
        delta_out = output - output.mean(dim=0, keepdim=True)
        delta_target = target - target.mean(dim=0, keepdim=True)

        var_out = delta_out.pow(2).mean(dim=0, keepdim=True)
        var_target = delta_target.pow(2).mean(dim=0, keepdim=True)

        corrs = (delta_out * delta_target).mean(dim=0, keepdim=True) / (
            (var_out + self.eps) * (var_target + self.eps)
        ).sqrt()

        if not self.per_neuron:
            return -corrs.mean() if self.avg else -corrs.sum()
        else:
            return -corrs.view(-1)

class ZeroLoss(nn.Module):
    def forward(self, output, target):
        return torch.tensor(0.0, device=output.device)

class KlindtCoreWrapper2D(Core):
    def __init__(
        self,
        # image_size: int,
        image_channels: int,
        kernel_sizes: Iterable[int | Iterable[int]],
        num_kernels: Iterable[int],
        act_fns: Iterable[str],
        reg: Iterable[float],
        init_scales: np.ndarray,
        kernel_constraint: Optional[str] = None,
        batch_norm: bool = True,
        bn_cent: bool = False,
        # dropout_rate: float = 0.0,
        seed: Optional[int] = None,
    ):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        self.kernel_sizes = []
        for k in kernel_sizes:
            if isinstance(k, int):
                self.kernel_sizes.append((k, k))
            elif isinstance(k, Iterable) and len(k) == 2:
                self.kernel_sizes.append(k)
            else:
                raise ValueError(f"Invalid kernel size format: {k}")
        
        self.num_kernels = num_kernels
        self.act_fns = act_fns
        self.reg = reg
        self.kernel_constraint = kernel_constraint

        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList() if batch_norm else None
        self.activation_layers = nn.ModuleList()
        self.dropout = nn.Dropout(p=self.reg[3])

        input_channels = image_channels
        # build 2D conv stack
        for i, (k_size, k_num, act_fn) in enumerate(zip(kernel_sizes, num_kernels, act_fns)):
            conv = nn.Conv2d(
                in_channels=input_channels,
                out_channels=k_num,
                kernel_size=k_size,
                stride=1,
                padding=0,
                bias=not batch_norm
            )

            nn.init.normal_(conv.weight, mean=init_scales[0][0], std=init_scales[0][1])

            if kernel_constraint == 'norm':
                with torch.no_grad():
                    norm = torch.sqrt(torch.sum(conv.weight**2, dim=(2, 3), keepdim=True) + 1e-5)
                    conv.weight.data = conv.weight.data / norm

            self.conv_layers.append(conv)

            if self.bn_layers is not None:
                self.bn_layers.append(
                    nn.BatchNorm2d(
                        num_features=k_num,
                        affine=bn_cent,
                        track_running_stats=True,
                        momentum=0.002
                    )
                )
            input_channels = k_num

            self.activation_layers.append(build_activation_layer(act_fn))

    def apply_constraints(self):
        if self.kernel_constraint == 'norm':
            with torch.no_grad():
                for conv in self.conv_layers:
                    norm = torch.sqrt(torch.sum(conv.weight**2, dim=(2, 3), keepdim=True) + 1e-5)
                    conv.weight.data = conv.weight.data / norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.apply_constraints()
        # x: [B, C, H, W]
        x = self.dropout(x)
        for i, conv in enumerate(self.conv_layers):
            x = conv(x)
            if self.bn_layers is not None:
                x = self.bn_layers[i](x)
            # if self.act_fns[i] == 'relu':
                # x = F.relu(x)
            x = self.activation_layers[i](x)
        return x

    def regularizer(self) -> torch.Tensor:
        kernel_reg = sum(torch.sum(conv.weight**2) for conv in self.conv_layers)
        return kernel_reg * self.reg[0]

class KlindtReadoutWrapper2D(Readout):
    def __init__(
        self,
        num_kernels: Iterable[int],
        num_neurons: int,
        reg: Iterable[float],
        mask_size: int | Iterable[int], # int or (h,w)
        final_relu: bool = False,
        weights_constraint: Optional[str] = None,
        mask_constraint: Optional[str] = None,
        init_mask: Optional[torch.Tensor] = None,
        init_weights: Optional[torch.Tensor] = None,
        init_scales: Optional[torch.Tensor] = None,
    ):
        super().__init__()

        self.num_neurons = num_neurons
        self.reg = reg
        self.mask_size = mask_size
        self.final_relu = final_relu
        self.weights_constraint = weights_constraint
        self.mask_constraint = mask_constraint

        if isinstance(mask_size, int):
            assert mask_size > 0, "Mask size must be positive integer(s)"
            self.num_mask_pixels = mask_size **2
        elif isinstance(mask_size, Iterable) and len(mask_size) == 2:
            h, w = mask_size
            assert h > 0 and w > 0, "Mask size must be positive integer(s)"
            self.mask_size = (h, w)
            self.num_mask_pixels = h * w

        # Initialize mask weights
        if init_mask is not None:
            self.mask_weights = nn.Parameter(init_mask)
        else:
            assert init_scales is not None, "Either init_mask or init_scales must be provided"
            mean, std = init_scales[1]
            mask_init = torch.normal(
                mean=mean,
                std=std,
                size=(self.num_mask_pixels, num_neurons)
            )
            self.mask_weights = nn.Parameter(mask_init)

        # Initialize readout weights
        if init_weights is not None:
            self.readout_weights = nn.Parameter(init_weights)
        else:
            assert init_scales is not None, "Either init_weights or init_scales must be provided"
            mean, std = init_scales[2]
            self.readout_weights = nn.Parameter(
                torch.normal(
                    mean=mean,
                    std=std,
                    size=(num_kernels[-1], num_neurons)  # We will infer input channel count at first forward
                ),
                # requires_grad=True
            )

        self.bias = nn.Parameter(torch.zeros(num_neurons)) if final_relu else None

    def apply_constraints(self):
        if self.mask_constraint == 'abs':
            with torch.no_grad():
                self.mask_weights.data = torch.abs(self.mask_weights.data)

        if self.weights_constraint == 'abs':
            with torch.no_grad():
                self.readout_weights.data = torch.abs(self.readout_weights.data)
        elif self.weights_constraint == 'norm':
            with torch.no_grad():
                norm = torch.sqrt(torch.sum(self.readout_weights ** 2, dim=0, keepdim=True) + 1e-5)
                self.readout_weights.data = self.readout_weights.data / norm
        elif self.weights_constraint == 'absnorm':
            with torch.no_grad():
                self.readout_weights.data = torch.abs(self.readout_weights.data)
                norm = torch.sqrt(torch.sum(self.readout_weights ** 2, dim=0, keepdim=True) + 1e-5)
                self.readout_weights.data = self.readout_weights.data / norm

    def forward(self, x: torch.Tensor, **kwarg: Any) -> torch.Tensor:
        # x: [B, C, H, W]
        self.apply_constraints()

        # batch_size = x.shape[0]
        # channels = x.shape[1]
        B, C, H, W = x.shape
        h, w = self.mask_size
        assert H == h and W == w, f"Input spatial size ({H}, {W}) must match mask size ({h}, {w})"

        # Flatten spatial dims and reshape
        x_flat = x.view(B, C, -1)  # [B, C, H*W]

        # Apply spatial mask: [HW, N]
        masked = torch.matmul(x_flat, self.mask_weights)  # → [B, C, N]
        masked = masked.permute(0, 2, 1)  # → [B, N, C]

        # Apply readout weights: [C, N] → [N, C] → [B, N, C] ( = [1, N, C] )
        output = (masked * self.readout_weights.T.unsqueeze(0)).sum(dim=2)  # → [B, N]

        if self.final_relu:
            output = F.relu(output + self.bias)  # → [B, N]

        return output

    def regularizer(self, data_key: str) -> torch.Tensor:
        mask_reg = torch.mean(torch.sum(torch.abs(self.mask_weights), dim=0)) * self.reg[1]
        weights_reg = torch.mean(torch.sum(torch.abs(self.readout_weights), dim=0)) * self.reg[2]
        return mask_reg + weights_reg

class KlindtCoreReadout2D(BaseCoreReadout):
    def __init__(
        self,
        # Core parameters
        image_size: int,
        image_channels: int,
        kernel_sizes: Iterable[int | Iterable[int]],
        num_kernels: Iterable[int],
        act_fns: Iterable[str],
        reg: Iterable[float],
        init_scales: Iterable[Iterable[float]],  # 3x2 matrix: [kernel, mask, weights] x [mean, std]
        kernel_constraint: Optional[str] = None,
        batch_norm: bool = True,
        bn_cent: bool = False,
        # dropout_rate: float = 0.0,
        seed: Optional[int] = None,
        # Readout parameters
        num_neurons: int = 1,
        final_relu: bool = False,
        weights_constraint: Optional[str] = None,
        mask_constraint: Optional[str] = None,
        init_mask: Optional[torch.Tensor] = None,
        init_weights: Optional[torch.Tensor] = None,
        # Common parameters
        loss: nn.Module | None = nn.PoissonNLLLoss(log_input = False),
        correlation_loss: nn.Module | None = CorrelationLoss2D(),
        learning_rate: float = 0.01,
        data_info: Optional[dict[str, Any]] = None,
    ):
        # Step 1: Instantiate core
        core = KlindtCoreWrapper2D(
            image_channels=image_channels,
            kernel_sizes=kernel_sizes,
            num_kernels=num_kernels,
            act_fns=act_fns,
            reg=reg,
            init_scales=init_scales,
            kernel_constraint=kernel_constraint,
            batch_norm=batch_norm,
            bn_cent=bn_cent,
            # dropout_rate=dropout_rate,
            seed=seed,
        )

        # Step 2: Get the spatial dimensions of the core output (h, w)
        dummy_input = torch.zeros(1, image_channels, image_size, image_size)
        with torch.no_grad():
            core_out = core(dummy_input)
        _, _, h, w = core_out.shape  # Spatial dimensions
        mask_size = (h, w)

        # Step 3: Instantiate readout, with mask size inferred from core output
        readout = KlindtReadoutWrapper2D(
            num_kernels=num_kernels,
            num_neurons=num_neurons,
            reg=reg,
            mask_size=mask_size,
            final_relu=final_relu,
            weights_constraint=weights_constraint,
            mask_constraint=mask_constraint,
            init_mask=init_mask,
            init_weights=init_weights,
            init_scales=init_scales,
        )

        # Step 4: Initialize BaseCoreReadout
        super().__init__(
            core=core,
            readout=readout,
            learning_rate=learning_rate,
            loss=loss,
            correlation_loss=correlation_loss,
            data_info=data_info,
        )

        # Save all hyperparameters for reproducibility
        self.save_hyperparameters()
