import io
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from ui.global_settings import global_state, METRICS_SAVE_DIR
from utils.data_io import Unpickler

def calculate_reliability(data, axis=2):
    """
    Calculate reliability using leave-one-out correlation.
    data shape: (time, neuron, repetitions)
    """
    n_reps = data.shape[axis]
    correlations = []
    
    for i in range(n_reps):
        # Leave one repetition out
        single_rep = data[:, :, i]
        other_reps = np.delete(data, i, axis=axis)
        mean_others = np.mean(other_reps, axis=axis)
        
        # Calculate correlation for each neuron
        for n in range(data.shape[1]):
            corr = np.corrcoef(single_rep[:, n], mean_others[:, n])[0, 1]
            correlations.append(corr)
    
    # Average all correlations per neuron
    reliability = np.mean(np.array(correlations).reshape(-1, data.shape[1]), axis=0)
    return reliability

#2. Bootstrapped reliability estimate:

def bootstrap_reliability(data, n_bootstrap=1000, axis=2):
    """
    Calculate reliability using bootstrap sampling.
    data shape: (time, neuron, repetitions)
    """
    n_reps = data.shape[axis]
    reliabilities = []
    
    for _ in range(n_bootstrap):
        # Sample repetitions with replacement
        idx1 = np.random.choice(n_reps, size=n_reps//2, replace=True)
        idx2 = np.random.choice(list(set(range(n_reps)) - set(idx1)), 
                              size=min(n_reps//2, len(set(range(n_reps)) - set(idx1))), 
                              replace=True)
        
        m1 = np.mean(data[:, :, idx1], axis=axis)
        m2 = np.mean(data[:, :, idx2], axis=axis)
        
        # Calculate correlation for each neuron
        corrs = [np.corrcoef(m1[:, n], m2[:, n])[0, 1] for n in range(data.shape[1])]
        reliabilities.append(corrs)
    
    # Get mean and confidence intervals
    reliability = np.mean(reliabilities, axis=0)
    reliability_ci = np.percentile(reliabilities, [2.5, 97.5], axis=0)
    return reliability, reliability_ci

def compute_evaluation_metrics(model, dataloader_dict, response_data, is_2d: bool, device='cuda'):
    """
    Compute model prediction and evaluation metrics given a test dataloader.

    Args:
        model: Trained torch.nn.Module
        dataloader_dict: dict[str, torch.utils.data.DataLoader]
        is_2d: Whether input is 2D (True) or 3D (False)
        device: Device to run model on

    Returns:
        dict[str, Any]: evaluation results including metrics and intermediate tensors
    """
    model = model.to(device)
    model.eval()

    all_preds = []
    all_targets = []

    for session_name, test_loader in dataloader_dict.items():
        for data_point in test_loader:
            inputs = data_point.inputs  # shape depends on 2D/3D
            targets = data_point.targets  # shape: [num_neurons]

            if is_2d:
                # inputs: [C, H, W] → [1, C, H, W]
                input_tensor = inputs.to(device)
            else:
                # inputs: [C, B, H, W] → [B, C, H, W]
                input_tensor = inputs.permute(1, 0, 2, 3).to(device)

            with torch.no_grad():
                preds = model(input_tensor).cpu()  # shape: [B, N] or [1, N]

            if preds.ndim == 1:
                preds = preds.unsqueeze(0)
            if targets.ndim == 1:
                targets = targets.unsqueeze(0)

            all_preds.append(preds)    # shape: [B, N]
            all_targets.append(targets)  # shape: [B, N]

    # Concatenate along batch axis
    predictions = torch.cat(all_preds, dim=0).numpy()    # [total_samples, N]
    targets = torch.cat(all_targets, dim=0).numpy()      # [total_samples, N]

    # Compatibility, response_data = (stimuli, neurons, trials)
    reliability, reliability_ci = bootstrap_reliability(response_data, axis=2)

    mse = np.mean((predictions - targets) ** 2)
    rmse = np.sqrt(mse)

    correlations = [
        np.corrcoef(predictions[:, i], targets[:, i])[0, 1]
        for i in range(predictions.shape[1])
    ]

    foc = [
        correlations[i] / reliability[i] if reliability[i] > 0 and not np.isnan(reliability[i]) else np.nan
        for i in range(len(correlations))
    ]

    metrics = {
        "predictions": predictions,
        "targets": targets,
        "reliability": reliability,
        "reliability_ci": reliability_ci,
        "mse": mse,
        "rmse": rmse,
        "correlations": correlations,
        "fraction_of_ceiling": foc,
        "mean_correlation": np.nanmean(correlations),
        "median_correlation": np.nanmedian(correlations),
        "mean_fraction_of_ceiling": np.nanmean(foc),
        "median_fraction_of_ceiling": np.nanmedian(foc)
    }

    global_state["metrics"] = metrics

    return metrics

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
    return fig

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
    return fig

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
    return fig

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

    return fig

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
    return fig

# Fig 2 Example Prediction
def plot_example_prediction(predictions, targets, reliability, correlations, foc):
    sorted_indices = np.argsort(correlations)
    # Select a range of neurons from worst to best
    neurons_to_plot = [
        sorted_indices[0],                         # Worst
        sorted_indices[len(sorted_indices)//4],    # 25th percentile
        sorted_indices[len(sorted_indices)//2],    # Median
        sorted_indices[3*len(sorted_indices)//4],  # 75th percentile
        sorted_indices[-1]                         # Best
    ]
    
    fig, axes = plt.subplots(len(neurons_to_plot), 1, figsize=(10, 3*len(neurons_to_plot)))
    
    for i, n_idx in enumerate(neurons_to_plot):
        ax = axes[i]
        ax.scatter(targets[:, n_idx], predictions[:, n_idx], alpha=0.7)
        
        # Add identity line
        min_val = min(np.min(targets[:, n_idx]), np.min(predictions[:, n_idx]))
        max_val = max(np.max(targets[:, n_idx]), np.max(predictions[:, n_idx]))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        # Add correlation and reliability info
        rel_str = f'{reliability[n_idx]:.4f}' if not np.isnan(reliability[n_idx]) else 'N/A'
        foc_str = f'{foc[n_idx]:.4f}' if not np.isnan(foc[n_idx]) else 'N/A'
        percentile = np.round(100 * i / (len(neurons_to_plot)-1))
        
        ax.set_title(f'Neuron #{n_idx} ({percentile}th %ile) - Corr: {correlations[n_idx]:.4f}, Rel: {rel_str}, FoC: {foc_str}')
        ax.set_xlabel('Actual Response')
        ax.set_ylabel('Predicted Response')
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    return fig

# Fig 3 Grid Predictions of Neurons
def plot_grid_predictions(predictions, targets, correlations, n_per_plot=16):
    n_neurons = targets.shape[1]
    figs = []

    for start_idx in range(0, n_neurons, n_per_plot):
        end_idx = min(start_idx + n_per_plot, n_neurons)
        n_to_plot = end_idx - start_idx
        grid_size = int(np.ceil(np.sqrt(n_to_plot)))

        fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
        axes = axes.flatten()

        for i, n_idx in enumerate(range(start_idx, end_idx)):
            ax = axes[i]
            ax.scatter(targets[:, n_idx], predictions[:, n_idx], alpha=0.5, s=15)

            lims = [
                min(np.min(targets[:, n_idx]), np.min(predictions[:, n_idx])),
                max(np.max(targets[:, n_idx]), np.max(predictions[:, n_idx]))
            ]
            ax.plot(lims, lims, 'r--', alpha=0.5)
            ax.set_title(f'Neuron {n_idx}: r={correlations[n_idx]:.2f}')

            if i % grid_size == 0:
                ax.set_ylabel('Predicted')
            if i >= (grid_size**2 - grid_size):
                ax.set_xlabel('Actual')

            ax.tick_params(axis='both', which='major', labelsize=8)

        for j in range(n_to_plot, grid_size**2):
            axes[j].axis('off')

        plt.tight_layout()
        figs.append(fig)
    return figs

# 将matplotlib图像保存为BytesIO以供gr.Image使用
def fig_to_buffer(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    return Image.open(buf)