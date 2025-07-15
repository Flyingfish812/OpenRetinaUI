import re
from typing import Iterable, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

class ReQU(nn.Module):
    """
    Rectified Quadratic Unit:
    f(x) = 0, if x < thresh
    f(x) = (x - thresh)^2, if x >= thresh
    """
    def __init__(self, thresh: float = 0.0):
        super().__init__()
        self.thresh = thresh

    def forward(self, x):
        return torch.square(F.relu(x - self.thresh))


class ParametricReLU(nn.Module):
    """
    Parametric ReLU with learnable positive and negative slope, and optional threshold.
    When no parameters are specified, behaves like standard ReLU.
    """
    def __init__(
        self,
        shape=1,
        learn_thresh: bool = False,
        learn_neg_slope: bool = False,
        learn_pos_slope: bool = False,
        init_thresh: float = 0.0,
        init_neg_slope: float = 0.0,
        init_pos_slope: float = 1.0,
    ):
        super().__init__()
        self.thresh = nn.Parameter(torch.full((shape,), init_thresh), requires_grad=learn_thresh)
        self.neg_slope = nn.Parameter(torch.full((shape,), init_neg_slope), requires_grad=learn_neg_slope)
        self.pos_slope = nn.Parameter(torch.full((shape,), init_pos_slope), requires_grad=learn_pos_slope)

    def forward(self, x):
        shifted = x - self.thresh
        return torch.where(shifted < 0, self.neg_slope * shifted, self.pos_slope * shifted)


class ParametricSoftplus(nn.Module):
    """
    Parametric softplus activation:
    f(x) = alpha * ln(1 + exp(beta * x + gamma))
    Switches to linear approx when value exceeds threshold (for stability).
    When no parameters are specified, behaves like standard softplus.
    """
    def __init__(
        self,
        shape=1,
        threshold: float = 20.0,
        minimum: float = None,
        learn_alpha: bool = False,
        learn_beta: bool = False,
        learn_gamma: bool = False,
        init_alpha: float = 1.0,
        init_beta: float = 1.0,
        init_gamma: float = 0.0,
    ):
        super().__init__()
        assert threshold is None or threshold >= 10.0
        self.threshold = threshold
        self.minimum = minimum

        self.alpha = nn.Parameter(torch.full((shape,), init_alpha), requires_grad=learn_alpha)
        self.beta = nn.Parameter(torch.full((shape,), init_beta), requires_grad=learn_beta)
        self.gamma = nn.Parameter(torch.full((shape,), init_gamma), requires_grad=learn_gamma)

    def forward(self, x):
        shifted = self.beta * x + self.gamma
        sp = F.softplus(shifted)

        if self.threshold is not None:
            sp = torch.where(sp > self.threshold, shifted, sp)
        out = self.alpha * sp

        if self.minimum is not None:
            out = torch.maximum(out, torch.tensor(self.minimum, device=out.device))

        return out

def build_activation_layer(activation: str) -> nn.Module:
    if activation in ('', 'same', None):
        return nn.Identity()
    
    # 标准ReLU
    if activation == 'relu':
        return nn.ReLU(inplace=True)

    # 简写ReQU
    if activation == 'requ':
        return ReQU()
    
    if activation == 'elu':
        return nn.ELU(inplace=True)

    # 支持参数化格式，如 'parametric_relu:neg=1,pos=0.5'
    if ':' in activation:
        name, args_str = activation.split(':', 1)
        args = {}
        for part in args_str.split(','):
            k, v = part.split('=')
            try:
                args[k.strip()] = float(v)
            except ValueError:
                raise ValueError(f"Invalid value for {k} in activation '{activation}'")
    else:
        name, args = activation, {}

    # 注册表实现
    registry = {
        'parametric_relu': lambda: ParametricReLU(
            learn_neg_slope='neg' in args,
            learn_pos_slope='pos' in args,
            init_neg_slope=args.get('neg', 0.0),
            init_pos_slope=args.get('pos', 1.0),
        ),
        'parametric_softplus': lambda: ParametricSoftplus(
            minimum=args.get('min', None),
        ),
    }

    if name not in registry:
        raise ValueError(f"Unknown activation function '{name}'")

    return registry[name]()

def build_activation_layers(activations: Iterable[Union[str, None]]) -> list[nn.Module]:
    return [build_activation_layer(act) for act in activations]