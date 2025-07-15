import numpy as np
import matplotlib.pyplot as plt
from backend.viz.utils import fig_to_buffer

# Fig 1-1 Model Correlation Distribution
def plot_correlation_distribution(correlations, mean_corr, median_corr):
    fig, ax = plt.subplots()
    ax.hist(correlations, bins=20, alpha=0.7)
    ax.axvline(mean_corr, color='r', linestyle='--', label=f"Mean: {mean_corr:.4f}")
    ax.axvline(median_corr, color='g', linestyle='--', label=f"Median: {median_corr:.4f}")
    ax.set_title("Model Correlation Distribution")
    ax.set_xlabel("Correlation")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig_to_buffer(fig)

# Fig 1-2 Neuron Reliability Distribution
def plot_reliability_distribution(reliability):
    mean_rel = np.nanmean(reliability)
    median_rel = np.nanmedian(reliability)
    fig, ax = plt.subplots()
    ax.hist(reliability, bins=20, alpha=0.7)
    ax.axvline(mean_rel, color='r', linestyle='--', label=f"Mean: {mean_rel:.4f}")
    ax.axvline(median_rel, color='g', linestyle='--', label=f"Median: {median_rel:.4f}")
    ax.set_title("Neuron Reliability Distribution")
    ax.set_xlabel("Reliability")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig_to_buffer(fig)

# Fig 1-5 Fraction of Ceiling Distribution
def plot_fraction_of_ceiling(foc, mean_foc, median_foc):
    foc = [f for f in foc if not np.isnan(f)]
    fig, ax = plt.subplots()
    ax.hist(foc, bins=20, alpha=0.7)
    ax.axvline(mean_foc, color='r', linestyle='--', label=f"Mean: {mean_foc:.4f}")
    ax.axvline(median_foc, color='g', linestyle='--', label=f"Median: {median_foc:.4f}")
    ax.axvline(1.0, color='k', linestyle='--', label='Ceiling')
    ax.set_title("Fraction of Ceiling Distribution")
    ax.set_xlabel("Model Correlation / Reliability")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig_to_buffer(fig)