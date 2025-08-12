import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Iterable, Any, Optional
from backend.model.activations import build_activation_layer
from backend.model.losses_3d import build_loss_3d
from backend.model.regularizers import L1Smooth3DRegularizer
from openretina.modules.core.base_core import Core
from openretina.modules.readout.base import Readout
from openretina.models.core_readout import BaseCoreReadout

class KlindtCoreWrapper3D(Core):
    def __init__(
        self,
        image_channels: int,
        kernel_sizes: Iterable[int | Iterable[int]],
        num_kernels: Iterable[int],
        act_fns: Iterable[str],
        smoothness_reg: float,
        sparsity_reg: float,
        center_mass_reg: float,
        init_scales: Iterable[Iterable[float]],
        init_kernels: Optional[str] = None,
        kernel_constraint: Optional[str] = None,
        batch_norm: bool = True,
        bn_cent: bool = False,
        dropout_rate: float = 0.0,
        seed: Optional[int] = None,
    ):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # 规范 kernel_sizes 到 [T, H, W]
        self.kernel_sizes = []
        for k in kernel_sizes:
            if isinstance(k, int):
                self.kernel_sizes.append([1, k, k])
            elif isinstance(k, Iterable) and len(k) == 2:
                self.kernel_sizes.append([1, k[0], k[1]])
            elif isinstance(k, Iterable) and len(k) == 3:
                self.kernel_sizes.append(list(k))
            else:
                raise ValueError(f"Invalid kernel size: {k}")

        self.num_kernels = num_kernels
        self.act_fns = act_fns
        self.reg = [smoothness_reg, sparsity_reg, center_mass_reg]
        self.kernel_constraint = kernel_constraint

        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList() if batch_norm else None
        self.activation_layers = nn.ModuleList()
        self.dropout = nn.Dropout3d(p=dropout_rate) if dropout_rate > 0 else nn.Identity()

        in_ch = image_channels
        for k_size, out_ch, act_fn in zip(self.kernel_sizes, num_kernels, act_fns):
            conv = nn.Conv3d(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=k_size,
                stride=1,
                padding=0,
                # bias=True
            )

            # 权重初始化
            if init_kernels and init_kernels.startswith("gaussian"):
                with torch.no_grad():
                    w = conv.weight
                    _, _, D, H, W = w.shape

                    try:
                        sigma = float(init_kernels.split(":")[1])
                    except:
                        sigma = 0.2

                    zz = torch.linspace(-1,1,D,device=w.device).view(D,1,1)
                    yy = torch.linspace(-1,1,H,device=w.device).view(1,H,1)
                    xx = torch.linspace(-1,1,W,device=w.device).view(1,1,W)
                    gauss = torch.exp(-((zz**2+yy**2+xx**2)/(2*sigma**2)))
                    gauss /= gauss.max()
                    w.copy_(torch.randn_like(w)*gauss*init_scales[0][1])
            else:
                with torch.no_grad():
                    nn.init.normal_(conv.weight, mean=init_scales[0][0], std=init_scales[0][1])

            # 约束
            if self.kernel_constraint == 'norm':
                with torch.no_grad():
                    norm = torch.sqrt(torch.sum(conv.weight**2, dim=(2,3,4), keepdim=True)+1e-5)
                    conv.weight.data /= norm

            self.conv_layers.append(conv)
            if self.bn_layers is not None:
                self.bn_layers.append(
                    nn.BatchNorm3d(
                        num_features=out_ch,
                        affine=bn_cent,
                        track_running_stats=True,
                        momentum=0.02
                    )
                )
            in_ch = out_ch
            self.activation_layers.append(build_activation_layer(act_fn))
        
        # 初始化 3D 正则模块，仅对最后一层卷积生效
        self.regularizer_module = L1Smooth3DRegularizer(
            sparsity_factor=sparsity_reg,
            smoothness_factor=smoothness_reg,
            center_mass_factor=center_mass_reg,
        )

    def apply_constraints(self):
        if self.kernel_constraint == 'norm':
            with torch.no_grad():
                for conv in self.conv_layers:
                    norm = torch.sqrt(torch.sum(conv.weight**2, dim=(2,3,4), keepdim=True)+1e-5)
                    conv.weight.data /= norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.apply_constraints()
        # x: [B, C, T, H, W]
        x = self.dropout(x)
        for conv, act, bn in zip(self.conv_layers, self.activation_layers, self.bn_layers or []):
            x = conv(x)
            x = act(x)
            if bn is not None:
                x = bn(x)
        return x

    def regularizer(self) -> torch.Tensor:
        return self.regularizer_module(self.conv_layers[-1].weight)
    
class KlindtReadoutWrapper3D(Readout):
    def __init__(
        self,
        num_kernels: Iterable[int],
        num_neurons: int,
        mask_reg: float,
        weights_reg: float,
        mask_size: int | Iterable[int],
        final_relu: bool = False,
        weights_constraint: Optional[str] = None,
        mask_constraint: Optional[str] = None,
        init_mask: Optional[torch.Tensor] = None,
        init_weights: Optional[torch.Tensor] = None,
        init_scales: Optional[Iterable[Iterable[float]]] = None,
    ):
        super().__init__()
        
        self.num_neurons = num_neurons
        self.reg = [mask_reg, weights_reg]
        self.mask_size = mask_size
        self.final_relu = final_relu
        self.weights_constraint = weights_constraint
        self.mask_constraint = mask_constraint

        if isinstance(mask_size, int):
            self.num_mask_pixels = mask_size**2
        else:
            h,w = mask_size
            self.num_mask_pixels = h*w

        if init_mask is not None:
            num_neurons = init_mask.shape[0]
            h, w = self.mask_size
            H, W = init_mask.shape[2], init_mask.shape[3]
            h_offset = (H - h) // 2
            w_offset = (W - w) // 2

            # Crop center region
            cropped = init_mask[:, :, h_offset:h_offset + h, w_offset:w_offset + w]  # shape: (num_neurons, 1, h, w)

            # Reshape to (num_mask_pixels, num_neurons)
            reshaped = cropped.reshape(num_neurons, -1).T  # shape: (h*w, num_neurons)

            # Convert to tensor and register as parameter
            self.mask_weights = nn.Parameter(torch.tensor(reshaped, dtype=torch.float32))
        else:
            assert init_scales is not None
            mean,std = init_scales[1]
            self.mask_weights = nn.Parameter(torch.normal(mean=mean, std=std, size=(self.num_mask_pixels,num_neurons)))

        if init_weights is not None:
            self.readout_weights = nn.Parameter(init_weights)
        else:
            assert init_scales is not None
            mean,std = init_scales[2]
            self.readout_weights = nn.Parameter(torch.normal(mean=mean, std=std, size=(num_kernels[-1],num_neurons)))

        self.bias = nn.Parameter(torch.full((num_neurons,),0.5)) if final_relu else None

    def apply_constraints(self):
        if self.mask_constraint=='abs':
            with torch.no_grad(): self.mask_weights.data.abs_()
        if self.weights_constraint=='abs':
            with torch.no_grad(): self.readout_weights.data.abs_()
        elif self.weights_constraint=='norm':
            with torch.no_grad():
                norm = torch.sqrt(torch.sum(self.readout_weights**2,dim=0,keepdim=True)+1e-5)
                self.readout_weights.data /= norm
        elif self.weights_constraint=='absnorm':
            with torch.no_grad():
                self.readout_weights.data.abs_()
                norm = torch.sqrt(torch.sum(self.readout_weights**2,dim=0,keepdim=True)+1e-5)
                self.readout_weights.data /= norm

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        self.apply_constraints()
        B, C, T, H, W = x.shape
        h, w = self.mask_size
        assert H == h and W == w
        x_flat = x.view(B,C,T,-1).permute(0,2,1,3)
        masked = torch.matmul(x_flat, self.mask_weights)
        masked = masked.permute(0,1,3,2)
        out = (masked * self.readout_weights.T.unsqueeze(0).unsqueeze(0)).sum(dim=3)
        return F.softplus(out + self.bias) if self.final_relu else out

    def regularizer(self) -> torch.Tensor:
        mask_r = self.reg[0]*torch.mean(torch.sum(torch.abs(self.mask_weights),dim=0))
        wt_r   = self.reg[1]*torch.mean(torch.sum(torch.abs(self.readout_weights),dim=0))
        return mask_r + wt_r
    
class KlindtCoreReadout3D(BaseCoreReadout):
    def __init__(
        self,
        image_size: int | Iterable[int],
        image_channels: int,
        kernel_sizes: Iterable[int | Iterable[int]],
        num_kernels: Iterable[int],
        act_fns: Iterable[str],
        smoothness_reg: float = 1.0,
        sparsity_reg: float = 0.1,
        center_mass_reg: float | Iterable[float] = 0.0,
        init_scales: Iterable[Iterable[float]] = None,
        init_kernels: Optional[str] = None,
        kernel_constraint: Optional[str] = None,
        batch_norm: bool = True,
        bn_cent: bool = False,
        dropout_rate: float = 0.0,
        seed: Optional[int] = None,
        num_neurons: int = 1,
        final_relu: bool = False,
        weights_constraint: Optional[str] = None,
        mask_constraint: Optional[str] = None,
        init_mask: Optional[torch.Tensor] = None,
        init_weights: Optional[torch.Tensor] = None,
        mask_reg: float = 1e-3,
        weights_reg: float = 1e-1,
        loss: str = "poisson",
        correlation_loss: str = "correlation",
        learning_rate: float = 0.01,
        data_info: Optional[dict[str, Any]] = None,
    ):
        core = KlindtCoreWrapper3D(
            image_channels=image_channels,
            kernel_sizes=kernel_sizes,
            num_kernels=num_kernels,
            act_fns=act_fns,
            smoothness_reg=smoothness_reg,
            sparsity_reg=sparsity_reg,
            center_mass_reg=center_mass_reg,
            init_scales=init_scales,
            init_kernels=init_kernels,
            kernel_constraint=kernel_constraint,
            batch_norm=batch_norm,
            bn_cent=bn_cent,
            dropout_rate=dropout_rate,
            seed=seed
        )
        # infer spatial dims
        if isinstance(image_size,int): 
            dummy = torch.zeros(1,image_channels,3,image_size,image_size)
        else:
            dummy = torch.zeros(1,image_channels,3,*image_size)
        with torch.no_grad():
            out = core(dummy)
        _, _, _, h, w = out.shape
        self.mask_size = (h, w)

        readout = KlindtReadoutWrapper3D(
            num_kernels=num_kernels,
            num_neurons=num_neurons,
            mask_reg=mask_reg,
            weights_reg=weights_reg,
            mask_size=(h, w),
            final_relu=final_relu,
            weights_constraint=weights_constraint,
            mask_constraint=mask_constraint,
            init_mask=init_mask,
            init_weights=init_weights,
            init_scales=init_scales,
        )
        
        super().__init__(
            core=core,
            readout=readout,
            learning_rate=learning_rate,
            loss=build_loss_3d(loss),
            correlation_loss=build_loss_3d(correlation_loss),
            data_info=data_info,
        )
        self.save_hyperparameters()
