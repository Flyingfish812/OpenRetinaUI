import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_, xavier_uniform_
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
        kernel_size: int | Iterable[int],  # (H, W)
        use_bipolar: bool = True,
        use_feature_layer: bool = True,
        use_batchnorm: bool = False,
        flatten_merge: bool = True,
        activation: str = "parametric_softplus",
        bias: bool = False,
    ):
        super().__init__()

        self.use_bipolar = use_bipolar
        self.use_feature_layer = use_feature_layer
        self.use_batchnorm = use_batchnorm
        self.flatten_merge = flatten_merge
        self.bias = bias

        out_channels = 2 if use_bipolar else 1

        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            if len(kernel_size) == 1:
                self.kernel_size = (kernel_size[0], kernel_size[0])
            elif len(kernel_size) == 2:
                self.kernel_size = kernel_size

        self.filter = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=self.kernel_size,
            bias=bias
        )

        if use_bipolar:
            with torch.no_grad():
                # One ON center, one OFF center
                self.filter.weight[0].copy_(torch.ones_like(self.filter.weight[0]))
                self.filter.weight[1].copy_(-torch.ones_like(self.filter.weight[1]))
                if bias:
                    constant_(self.filter.bias, 0.0)

        self.bn = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()

        # Activation
        self.activation = build_activation_layer(activation)

        # Optional ON/OFF merging conv layer
        if use_feature_layer and use_bipolar:
            self.feature_conv = nn.Conv2d(
                in_channels=2,
                out_channels=1,
                kernel_size=(1, 2),
                bias=True
            )
        else:
            self.feature_conv = None

    def forward(self, x):
        # x: [B, C, H, W] or [B, C, T, F]
        x = self.filter(x)
        x = self.bn(x)
        x = self.activation(x)

        if self.use_bipolar and self.flatten_merge:
            B, C, H, W = x.shape
            x = x.view(B, 2, -1)  # [B, 2, H*W]
            x = x.permute(0, 2, 1).reshape(B, H, W * 2).unsqueeze(1)  # [B, 1, H, 2W]

        if self.feature_conv is not None:
            x = self.feature_conv(x)

        return x

    def regularizer(self):
        # Default: L1 norm on filter weights (no smoothness assumed)
        return torch.norm(self.filter.weight, p=1)
    
class LNLNReadout2D(Readout):
    def __init__(
        self,
        in_shape: Iterable[int],  # (C, H, W)
        out_features: int,
        bias: bool = True,
        activation: str = "parametric_softplus",
    ):
        super().__init__()

        self.in_shape = in_shape
        self.out_features = out_features

        C, H, W = in_shape
        self.mask = nn.Linear(C * H * W, out_features, bias=bias)
        self.activation = build_activation_layer(activation)

    def forward(self, x, **kwarg: Any):
        # x: [B, C, H, W]
        x = x.view(x.size(0), -1)  # Flatten
        x = self.mask(x)
        x = self.activation(x)
        return x

    def regularizer(self, data_key: Optional[str] = None):
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
        readout_bias: bool = True,
        readout_activation: str = "parametric_softplus",
        # loss: nn.Module | None = nn.PoissonNLLLoss(log_input = False),
        # correlation_loss: nn.Module | None = CorrelationLoss2D(),
        loss: Optional[str] = "poisson",
        correlation_loss: Optional[str] = "correlation",
        data_info: dict = None,
        learning_rate: float = 1e-3,
    ):
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

        if isinstance(image_size, int):
            dummy_input = torch.zeros(1, image_channels, image_size, image_size)
        elif isinstance(image_size, Iterable) and len(image_size) == 2:
            dummy_input = torch.zeros(1, image_channels, *image_size)
        else:
            raise ValueError(f"Invalid image_size: {image_size}")

        with torch.no_grad():
            core_out = core(dummy_input)
        _, c, h, w = core_out.shape
        self.mask_size = (h, w)

        readout = LNLNReadout2D(
            in_shape=(c, h, w),
            out_features=num_neurons,
            bias=readout_bias,
            activation=readout_activation,
        )

        super().__init__(
            core=core,
            readout=readout,
            loss=build_loss_2d(loss),
            correlation_loss=build_loss_2d(correlation_loss),
            data_info=data_info or {},
            learning_rate=learning_rate,
        )

        self.save_hyperparameters()
