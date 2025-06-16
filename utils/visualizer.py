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

def bootstrap_reliability(data, n_bootstrap=100, axis=2):
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

    all_inputs = []
    all_preds = []
    all_targets = []
    lsta_per_neuron = None

    for session_name, test_loader in dataloader_dict.items():
        for data_point in test_loader:
            inputs = data_point.inputs.to(device)  # 2d: [B, C, H, W]; 3d: [1, C, T, H, W]
            inputs.requires_grad_(True)
            outputs = model(inputs)
            preds = outputs.detach().cpu()
            targets = data_point.targets  # 2d: [B, num_neurons]; 3d: [1, T, num_neurons]

            # with torch.no_grad():
            #     preds = model(inputs).cpu()  # shape: [B, N] or [1, T, N]

            
            
            # print(inputs.shape, targets.shape, preds.shape)

            # [1, T, N] -> [T, N]
            # if not is_2d:
                # preds = preds.squeeze(0)
                # targets = targets.squeeze(0)

            if lsta_per_neuron is None:
                num_neurons = outputs.shape[-1]
                lsta_per_neuron = [[] for _ in range(num_neurons)]
            
            if is_2d:
                for neuron_idx in range(outputs.shape[1]):
                    grad_outputs = torch.zeros_like(outputs)
                    grad_outputs[:, neuron_idx] = 1.0
                    grads = torch.autograd.grad(
                        outputs=outputs,
                        inputs=inputs,
                        grad_outputs=grad_outputs,
                        retain_graph=True,
                        only_inputs=True,
                    )[0]  # [B, C, H, W]
                    lsta_per_neuron[neuron_idx].append(grads.detach().cpu())
            else:
                # 3D case: [1, T, N] -> [T, N]
                preds = preds.squeeze(0)
                outputs = outputs.squeeze(0)  # [T, N]
                targets = targets.squeeze(0)
                for neuron_idx in range(outputs.shape[1]):
                    grad_outputs = torch.zeros_like(outputs)
                    grad_outputs[:, neuron_idx] = 1.0
                    grads = torch.autograd.grad(
                        outputs=outputs,
                        inputs=inputs,
                        grad_outputs=grad_outputs,
                        retain_graph=True,
                        only_inputs=True,
                    )[0]  # [1, C, T, H, W]
                    lsta_per_neuron[neuron_idx].append(grads.squeeze(0).permute(1, 0, 2, 3).detach().cpu())

            # shape: [B, N] or [T, N]
            all_inputs.append(inputs.detach().cpu())
            all_preds.append(preds)
            all_targets.append(targets)  

    # Concatenate along batch axis
    if is_2d:
        images = torch.cat(all_inputs, dim=0).numpy()  # [B, C, H, W] -> [total_samples, C, H, W]
        predictions = torch.cat(all_preds, dim=0).numpy()    # [B, N] -> [total_samples, N]
        targets = torch.cat(all_targets, dim=0).numpy()      # [B, N] -> [total_samples, N]
    else:
        images = all_inputs[0].squeeze(0).permute(1, 0, 2, 3).numpy()
        predictions = all_preds[0].numpy()    # [T, N]
        targets = all_targets[0].numpy()      # [T, N]
    
    lsta_array = [
        torch.cat(lsta_per_neuron[n], dim=0).numpy()
        for n in range(len(lsta_per_neuron))
    ]
    lsta_array = np.stack(lsta_array, axis=0)  # shape: [N, B, C, H, W] or [N, T, C, H, W]

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
        "images": images,
        "predictions": predictions,
        "targets": targets,
        "lsta": lsta_array,
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

def plot_lsta(images, lstas, cell_indexs=None, nb_columns=10):
    """
    Plot LSTA images for all cells. Each cell gets a figure containing
    the raw stimuli and corresponding LSTA responses.

    Args:
        images: Tensor or ndarray of shape [N, C, H, W]
        lstas: Tensor or ndarray of shape [num_cells, N, C, H, W]
        nb_columns: int, number of columns in subplot grid

    Returns:
        figs: List[matplotlib.figure.Figure], one for each cell
    """
    if isinstance(images, torch.Tensor):
        images = images.cpu().numpy()
    if isinstance(lstas, torch.Tensor):
        lstas = lstas.cpu().numpy()

    nb_images = images.shape[0]
    nb_cells = lstas.shape[0]
    nb_rows = 2 * ((nb_images - 1) // nb_columns + 1)
    figs = []

    if cell_indexs:
        valid_cell_indices = [
            i for i in cell_indexs if isinstance(i, int) and i >= 0 and i < nb_cells
        ]
    else:
        valid_cell_indices = list(range(nb_cells))

    for cell_idx in valid_cell_indices:
        fig, axes = plt.subplots(
            nrows=nb_rows,
            ncols=nb_columns,
            figsize=(nb_columns * 1.5, nb_rows * 1.5),
            squeeze=False
        )

        for ax in axes.flat:
            ax.set_axis_off()

        for k in range(nb_images):
            row_img = 2 * (k // nb_columns)
            col = k % nb_columns

            # 显示原始图像
            ax_img = axes[row_img, col]
            ax_img.imshow(images[k, 0], cmap='Greys_r',
                          vmin=np.min(images), vmax=np.max(images))

            # 显示LSTA
            ax_lsta = axes[row_img + 1, col]
            ax_lsta.imshow(lstas[cell_idx, k, 0], cmap='RdBu_r',
                           vmin=-np.max(np.abs(lstas[cell_idx])),
                           vmax=+np.max(np.abs(lstas[cell_idx])))

        fig.suptitle(f"LSTA Visualization for Cell {cell_idx}", fontsize=14)
        fig.tight_layout()
        figs.append(fig)

    return figs

def fig_to_buffer(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    return Image.open(buf)