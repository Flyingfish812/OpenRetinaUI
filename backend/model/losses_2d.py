import torch
import torch.nn as nn
from typing import Iterable

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
    
def build_loss_2d(loss: str) -> nn.Module:
    """
    Construct 2D loss function from string.
    Example inputs:
        "poisson"
        "poisson:log"
        "zero"
        "corr"
        "correlation"
        "correlation2d"
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
        log_input = args.get("log", False)
        return nn.PoissonNLLLoss(log_input=log_input)

    if name == "zero":
        return ZeroLoss()

    if name in ("corr", "correlation", "correlation2d"):
        return CorrelationLoss2D()

    raise ValueError(f"Unknown loss function '{loss}'")

def build_loss_layers_2d(losses: Iterable[str]) -> list[nn.Module]:
    return [build_loss_2d(l) for l in losses]
