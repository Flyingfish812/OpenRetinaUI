import torch
import matplotlib.pyplot as plt
from math import sqrt, ceil
from backend.viz.utils import fig_to_buffer

def plot_response_curves(predictions, targets, cell_indices=None):
    """
    绘制每个神经元的响应曲线：目标值（targets）和模型输出值（predictions）。

    参数:
        predictions (Tensor): 模型输出，形状为 [num_samples, num_neurons]，例如 [30, 41]
        targets (Tensor): 真实目标值，形状与 predictions 相同
        cell_indices (list[int], optional): 选择要可视化的神经元索引，默认显示全部神经元

    返回:
        matplotlib.figure.Figure: 绘图对象
    """
    if not isinstance(predictions, torch.Tensor):
        predictions = torch.tensor(predictions)
    if not isinstance(targets, torch.Tensor):
        targets = torch.tensor(targets)

    num_samples, total_neurons = predictions.shape

    if cell_indices is None:
        cell_indices = list(range(total_neurons))
    else:
        cell_indices = [i for i in cell_indices if 0 <= i < total_neurons]

    selected_predictions = predictions[:, cell_indices]  # [30, N]
    selected_targets = targets[:, cell_indices]  # [30, N]
    n_cells = len(cell_indices)

    grid_cols = min(8, ceil(sqrt(n_cells)))
    grid_rows = ceil(n_cells / grid_cols)

    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(2.5 * grid_cols, 2 * grid_rows))

    if n_cells == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i < n_cells:
            ax.plot(selected_targets[:, i], label='Target', linestyle='--', marker='o', alpha=0.7, markersize=3)
            ax.plot(selected_predictions[:, i], label='Output', linestyle='-', marker='x', alpha=0.7, markersize=3)
            ax.set_title(f"Neuron {cell_indices[i]}")
            ax.set_xlabel("Sample")
            ax.set_ylabel("Response")
            ax.legend(fontsize=8)
        else:
            ax.axis('off')

    fig.suptitle("Neuron Response Curves", fontsize=14)
    fig.tight_layout()

    return fig_to_buffer(fig)