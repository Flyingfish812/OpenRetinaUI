from openretina.modules.losses.poisson import PoissonLoss3d, CelltypePoissonLoss3d
from openretina.modules.losses.correlation import (
    CorrelationLoss3d,
    CelltypeCorrelationLoss3d,
    L1CorrelationLoss3d,
    ScaledCorrelationLoss3d,
)
from openretina.modules.losses.mse import MSE3d

import torch.nn as nn

class ZeroLoss3d(nn.Module):
    def forward(self, output, target, *args, **kwargs):
        return output.sum() * 0

def build_loss_3d(loss: str) -> nn.Module:
    """
    Construct 3D loss function from string.
    Supported values:
        "poisson", "poisson:avg", "poisson:per_neuron"
        "corr", "correlation", "correlation3d"
        "cellcorr", "celltypecorr", "celltypecorrelation"
        "l1corr", "scaledcorr"
        "mse"
        "zero"
    """
    if not loss or loss.lower() in ("none", "identity"):
        return nn.Identity()

    if ':' in loss:
        name, args_str = loss.split(':', 1)
        args = {k: True for k in args_str.split(',')}
    else:
        name = loss
        args = {}

    name = name.lower()
    if name == "poisson":
        return PoissonLoss3d(avg=args.get("avg", False), per_neuron=args.get("per_neuron", False))
    elif name in ("cellpoisson", "celltypepoisson"):
        return CelltypePoissonLoss3d(avg=args.get("avg", False), per_neuron=args.get("per_neuron", False))
    elif name in ("corr", "correlation", "correlation3d"):
        return CorrelationLoss3d(avg=args.get("avg", False), per_neuron=args.get("per_neuron", False))
    elif name in ("cellcorr", "celltypecorr", "celltypecorrelation"):
        return CelltypeCorrelationLoss3d(avg=args.get("avg", False), per_neuron=args.get("per_neuron", False))
    elif name == "l1corr":
        return L1CorrelationLoss3d(avg=args.get("avg", False), per_neuron=args.get("per_neuron", False))
    elif name == "scaledcorr":
        return ScaledCorrelationLoss3d(avg=args.get("avg", False), per_neuron=args.get("per_neuron", False))
    elif name == "mse":
        return MSE3d()
    elif name == "zero":
        return ZeroLoss3d()

    raise ValueError(f"Unknown loss function '{loss}'")
