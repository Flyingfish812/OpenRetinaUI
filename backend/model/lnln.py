import torch
import torch.nn as nn
from typing import Iterable, Optional, Any
from backend.model.activations import build_activation_layer
from backend.model.losses_2d import build_loss_2d
from openretina.modules.core.base_core import Core
from openretina.modules.readout.base import Readout
from openretina.models.core_readout import BaseCoreReadout


class LNLNCore2D(Core):
    def __init__(
        self,
        in_channels: int,
        kernel_size: int | Iterable[int],
        use_bipolar: bool = True,
        use_feature_layer: bool = True,
        use_batchnorm: bool = False,
        flatten_merge: bool = True,
        activation: str = "parametric_softplus",
        bias: bool = False,
        gaussian_sigma: float = 0.2,
    ):
        super().__init__()

        # Save settings
        self.use_bipolar = use_bipolar
        self.use_feature_layer = use_feature_layer
        self.use_batchnorm = use_batchnorm
        self.flatten_merge = flatten_merge

        # Determine kernel dimensions
        if isinstance(kernel_size, int):
            kh, kw = kernel_size, kernel_size
        elif isinstance(kernel_size, Iterable) and len(kernel_size) == 2:
            kh, kw = kernel_size
        else:
            raise ValueError(f"Invalid kernel_size: {kernel_size}")

        # Output channels: ON & OFF or single polarity
        out_channels = 2 if use_bipolar else 1

        # Convolution filter
        self.filter = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(kh, kw),
            bias=bias,
        )

        # Gaussian initialization for ON/OFF centers
        with torch.no_grad():
            weight = self.filter.weight  # shape (out_c, in_c, H, W)
            out_c, in_c, H, W = weight.shape
            # Create 2D Gaussian
            yy, xx = torch.meshgrid(
                torch.linspace(-1, 1, H, device=weight.device),
                torch.linspace(-1, 1, W, device=weight.device),
                indexing='ij',
            )
            gaussian = torch.exp(-(xx**2 + yy**2) / (2 * gaussian_sigma**2))
            gaussian = gaussian / gaussian.max()
            # Assign to weights
            if use_bipolar:
                for c in range(in_c):
                    weight[0, c] = -gaussian
                    weight[1, c] = gaussian
            else:
                for c in range(in_c):
                    weight[0, c] = gaussian
        # Freeze filter parameters (fixed Gaussian)
        self.filter.weight.requires_grad_(False)
        if bias and self.filter.bias is not None:
            self.filter.bias.requires_grad_(False)

        # Optional batch normalization
        self.bn = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()

        # Activation layer
        self.activation = build_activation_layer(activation)

        # Optional ON/OFF merging feature layer
        if use_feature_layer and use_bipolar:
            self.feature_conv = nn.Conv2d(
                in_channels=2, out_channels=1, kernel_size=(1, 2), bias=True
            )
        else:
            self.feature_conv = None

    def apply_constraints(self):
        # No additional constraints for fixed Gaussian filters
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply any kernel constraints
        self.apply_constraints()

        # Convolution + BN + activation
        x = self.filter(x)
        x = self.bn(x)
        x = self.activation(x)

        # Flatten ON/OFF channels if requested
        if self.use_bipolar and self.flatten_merge:
            B, C, H, W = x.shape
            x = x.view(B, 2, -1)
            x = x.permute(0, 2, 1).reshape(B, H, W * 2).unsqueeze(1)

        # Feature merging
        if self.feature_conv is not None:
            x = self.feature_conv(x)

        return x

    def regularizer(self) -> torch.Tensor:
        # L1 regularization on filter weights
        return torch.norm(self.filter.weight, p=1)


class LNLNReadout2D(Readout):
    def __init__(
        self,
        in_shape: Iterable[int],  # (C, H, W)
        out_features: int,
        bias: bool = False,
        activation: str = "parametric_softplus",
    ):
        super().__init__()
        C, H, W = in_shape
        self.mask = nn.Linear(C * H * W, out_features, bias=bias)
        self.activation = build_activation_layer(activation)

    def apply_constraints(self):
        # MaxNorm constraint on mask weights
        with torch.no_grad():
            w = self.mask.weight  # shape (out_features, in_dim)
            norm = w.norm(dim=1, keepdim=True).clamp(min=1e-5)
            self.mask.weight.div_(norm)

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        # Apply mask constraints
        self.apply_constraints()
        # Flatten spatial dimensions and apply mask
        x = x.view(x.size(0), -1)
        x = self.mask(x)
        x = self.activation(x)
        return x

    def regularizer(self, data_key: Optional[str] = None) -> torch.Tensor:
        # L2 regularization on mask weights
        return torch.norm(self.mask.weight, p=2)


class LNLNCoreReadout2D(BaseCoreReadout):
    def __init__(
        self,
        image_channels: int,
        image_size: int | tuple[int, int],
        num_neurons: int,
        kernel_size: int | Iterable[int],
        use_bipolar: bool = True,
        use_feature_layer: bool = True,
        use_batchnorm: bool = False,
        flatten_merge: bool = False,
        core_activation: str = "parametric_softplus",
        core_bias: bool = False,
        readout_bias: bool = False,
        readout_activation: str = "parametric_softplus",
        loss: Optional[str] = "poisson",
        correlation_loss: Optional[str] = "correlation",
        data_info: dict = None,
        learning_rate: float = 1e-3,
    ):
        # Instantiate core
        core = LNLNCore2D(
            in_channels=image_channels,
            kernel_size=kernel_size,
            use_bipolar=use_bipolar,
            use_feature_layer=use_feature_layer,
            use_batchnorm=use_batchnorm,
            flatten_merge=flatten_merge,
            activation=core_activation,
            bias=core_bias,
        )

        # Infer readout spatial dimensions
        if isinstance(image_size, int):
            dummy = torch.zeros(1, image_channels, image_size, image_size)
        else:
            dummy = torch.zeros(1, image_channels, *image_size)
        with torch.no_grad():
            out = core(dummy)
        _, c, h, w = out.shape

        # Instantiate readout
        readout = LNLNReadout2D(
            in_shape=(c, h, w),
            out_features=num_neurons,
            bias=readout_bias,
            activation=readout_activation,
        )

        # Initialize BaseCoreReadout
        super().__init__(
            core=core,
            readout=readout,
            loss=build_loss_2d(loss),
            correlation_loss=build_loss_2d(correlation_loss),
            data_info=data_info or {},
            learning_rate=learning_rate,
        )
        self.save_hyperparameters()
