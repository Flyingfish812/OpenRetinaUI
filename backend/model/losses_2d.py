import torch
import torch.nn as nn

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