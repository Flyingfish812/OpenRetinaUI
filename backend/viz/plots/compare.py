import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple
from backend.viz.utils import fig_to_buffer

def plot_corrected_r2_boxplot(metrics_list: List[Tuple[str, dict]]):
    """
    Draws a boxplot comparing corrected R^2 distributions across multiple models.
    Each model must provide "corrected_r2": List[float] in its metrics.
    """
    model_names = [name for name, _ in metrics_list]
    data = [metrics["corrected_r2"] for _, metrics in metrics_list]

    fig, ax = plt.subplots(figsize=(1.5 * len(data), 6))
    box = ax.boxplot(data, patch_artist=True, showfliers=False)

    for patch in box['boxes']:
        patch.set_facecolor('none')
        patch.set_edgecolor('black')

    for element in ['whiskers', 'caps', 'medians']:
        for line in box[element]:
            line.set_color('black')

    # Add scatter points with horizontal jitter
    for i, y in enumerate(data):
        x_jittered = np.random.normal(loc=i + 1, scale=0.05, size=len(y))
        ax.scatter(x_jittered, y, alpha=0.6, color='black', s=10)

    ax.set_xticks(np.arange(1, len(model_names) + 1))
    ax.set_xticklabels(model_names)
    ax.set_ylabel("Corrected $R^2$")
    ax.set_title("Model-wise corrected $R^2$ distribution")
    ax.axhline(0, color='gray', linestyle='dashed', linewidth=1)
    plt.tight_layout()
    return fig_to_buffer(fig)

def plot_corrected_r2_barchart(metrics_list: List[Tuple[str, dict]]):
    """
    Draws a bar chart comparing median corrected R^2 values across models,
    with individual neuron scatter points overlaid, and colored bars.
    """
    model_names = [name for name, _ in metrics_list]
    data = [metrics["corrected_r2"] for _, metrics in metrics_list]
    medians = [np.nanmedian(values) for values in data]

    colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(model_names)))

    fig, ax = plt.subplots(figsize=(1.5 * len(medians), 6))

    bar_positions = np.arange(len(model_names))
    bars = ax.bar(bar_positions, medians, color=colors, edgecolor='black', width=0.6)

    # Add individual scatter points (no jitter)
    for i, y in enumerate(data):
        x_vals = np.full(len(y), bar_positions[i])
        ax.scatter(x_vals, y, color='black', s=10, zorder=5, alpha=0.8)

    ax.set_xticks(bar_positions)
    ax.set_xticklabels(model_names)
    ax.set_ylabel("Corrected $R^2$")
    ax.set_title("Median Corrected $R^2$ with Cell-wise Distribution")
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    return fig_to_buffer(fig)

def plot_corrected_r2_scatter(model1_name: str, model2_name: str,
                               model1_r2: List[float], model2_r2: List[float]):
    """
    Draws a scatter plot comparing corrected R^2 values from two models for the same cells.
    """
    model1_r2 = np.array(model1_r2)
    model2_r2 = np.array(model2_r2)
    assert model1_r2.shape == model2_r2.shape

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(model1_r2, model2_r2, alpha=0.9, color='navy', s=30)
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel(f"{model1_name} Corrected $R^2$")
    ax.set_ylabel(f"{model2_name} Corrected $R^2$")
    ax.set_title(f"Per-cell Performance: {model2_name} vs {model1_name}")
    plt.tight_layout()
    return fig_to_buffer(fig)