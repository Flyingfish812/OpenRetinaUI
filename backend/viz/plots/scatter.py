import numpy as np
import matplotlib.pyplot as plt
from backend.viz.utils import fig_to_buffer

# Fig 1-3 Correlation vs. Reliability
def plot_corr_vs_reliability(reliability, correlations):
    fig, ax = plt.subplots()
    ax.scatter(reliability, correlations, alpha=0.7)
    max_val = max(np.nanmax(reliability), np.nanmax(correlations))
    ax.plot([0, max_val], [0, max_val], 'k--', label='Ceiling')
    reliability = np.array(reliability)
    correlations = np.array(correlations)
    mask = ~np.isnan(reliability) & ~np.isnan(correlations)
    if np.sum(mask) > 1:
        z = np.polyfit(reliability[mask], correlations[mask], 1)
        p = np.poly1d(z)
        x_range = np.linspace(np.min(reliability[mask]), np.max(reliability[mask]), 100)
        ax.plot(x_range, p(x_range), 'r--', label=f"Trend: y={z[0]:.2f}x+{z[1]:.2f}")
    ax.set_title("Correlation vs. Reliability")
    ax.set_xlabel("Reliability")
    ax.set_ylabel("Correlation")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig_to_buffer(fig)

# Fig 1-4 Correlation vs. Reliability with indices
def plot_corr_vs_reliability_with_indice(reliability, correlations, top_n=5):
    reliability = np.array(reliability)
    correlations = np.array(correlations)

    # fig, ax = plt.subplots(figsize=(12, 10))
    fig, ax = plt.subplots()
    ax.scatter(reliability, correlations, alpha=0.7)

    max_val = max(np.nanmax(reliability), np.nanmax(correlations))
    ax.plot([0, max_val], [0, max_val], 'k--', label='Ceiling')

    # Best by correlation
    top_by_corr = np.argsort(correlations)[-top_n:]
    for idx in top_by_corr:
        ax.text(reliability[idx], correlations[idx], str(idx), color='green')

    # Worst by correlation
    bottom_by_corr = np.argsort(correlations)[:top_n]
    for idx in bottom_by_corr:
        ax.text(reliability[idx], correlations[idx], str(idx), color='red')

    ax.set_title('Model Correlation vs. Reliability with Neuron Indices')
    ax.set_xlabel('Reliability')
    ax.set_ylabel('Model Correlation')
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig_to_buffer(fig)