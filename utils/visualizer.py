import io
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from math import sqrt, ceil
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

def compute_evaluation_metrics(model, dataloader_dict, response_data, is_2d: bool, test_std=1):
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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
            print("Mean response:", outputs.mean().item())
            print("Std response:", outputs.std().item())
            preds = outputs.detach().cpu()
            targets = data_point.targets  # 2d: [B, num_neurons]; 3d: [1, T, num_neurons]

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
                    # grads *= test_std
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
                    # grads *= test_std
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
    if response_data is not None:
        reliability, reliability_ci = bootstrap_reliability(response_data, axis=2)
    else:
        reliability = None
        reliability_ci = None


    mse = np.mean((predictions - targets) ** 2)
    rmse = np.sqrt(mse)

    correlations = [
        np.corrcoef(predictions[:, i], targets[:, i])[0, 1]
        for i in range(predictions.shape[1])
    ]

    if reliability is not None:
        foc = [
            correlations[i] / reliability[i] if reliability[i] > 0 and not np.isnan(reliability[i]) else np.nan
            for i in range(len(correlations))
        ]
        mean_foc = np.nanmean(foc)
        median_foc = np.nanmedian(foc)
    
        corrected_r2 = []
        num_images, num_neurons = predictions.shape
        assert response_data.shape[:2] == (num_images, num_neurons)

        for i in range(num_neurons):
            pred = predictions[:, i]
            # target = targets[:, i]
            # 获取所有 trial 原始数据：shape [30, trials]
            full_trials = response_data[:, i, :]  # shape: [30, num_trials]
            odd_mean = full_trials[:, ::2].mean(axis=1)
            even_mean = full_trials[:, 1::2].mean(axis=1)
            reliability_corr = np.corrcoef(odd_mean, even_mean)[0, 1]

            # 预测与奇偶 trial 的相关性
            r_odd = np.corrcoef(pred, odd_mean)[0, 1]
            r_even = np.corrcoef(pred, even_mean)[0, 1]

            if reliability_corr > 0 and not np.isnan(r_odd) and not np.isnan(r_even):
                r_nc = 0.5 * (r_odd + r_even) / np.sqrt(reliability_corr)
                r2_nc = r_nc ** 2
            else:
                r2_nc = np.nan

            corrected_r2.append(r2_nc)
        
        mean_corrected_r2 = np.nanmean(corrected_r2)
        median_corrected_r2 = np.nanmedian(corrected_r2)
    else:
        foc = None
        mean_foc = None
        median_foc = None
        corrected_r2 = None
        mean_corrected_r2 = None
        median_corrected_r2 = None

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
        "corrected_r2": corrected_r2,
        "mean_correlation": np.nanmean(correlations),
        "median_correlation": np.nanmedian(correlations),
        "mean_fraction_of_ceiling": mean_foc,
        "median_fraction_of_ceiling": median_foc,
        "mean_corrected_r2": mean_corrected_r2,
        "median_corrected_r2": median_corrected_r2,
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

def plot_lsta(images, lsta_data, lsta_model, ellipses, image_indices, cell_indexs=None):
    """
    Plot cropped regions around ellipses for raw images, data-based LSTA, and model-predicted LSTA.

    Args:
        images: np.ndarray, shape [N, C, H, W]
        lsta_data: np.ndarray, shape [num_cells, 8, C, H, W]
        lsta_model: np.ndarray, shape [num_cells, 8, C, H, W]
        ellipses: np.ndarray, shape [num_cells, 2, 360]
        image_indices: list[int], length 8
        cell_indexs: list[int] or None

    Returns:
        figs: List[matplotlib.figure.Figure]
    """
    if isinstance(images, torch.Tensor):
        images = images.cpu().numpy()
    if isinstance(lsta_data, torch.Tensor):
        lsta_data = lsta_data.cpu().numpy()
    if isinstance(lsta_model, torch.Tensor):
        lsta_model = lsta_model.cpu().numpy()
    if isinstance(ellipses, torch.Tensor):
        ellipses = ellipses.cpu().numpy()

    figs = []
    nb_images = 8
    nb_rows = 4
    nb_cols = 6
    factor = 108 / 72  # 缩放因子

    if cell_indexs is None:
        valid_indices = list(range(lsta_model.shape[0]))
    else:
        valid_indices = [i for i in cell_indexs if 0 <= i < lsta_model.shape[0]]

    def get_crop_bounds(x, y, padding, shape_limit):
        x_min = max(int(np.floor(x.min()) - padding), 0)
        x_max = min(int(np.ceil(x.max()) + padding), shape_limit)
        y_min = max(int(np.floor(y.min()) - padding), 0)
        y_max = min(int(np.ceil(y.max()) + padding), shape_limit)
        return x_min, x_max, y_min, y_max

    for cell_idx in valid_indices:
        fig, axes = plt.subplots(nb_rows, nb_cols, figsize=(nb_cols * 1.8, nb_rows * 1.8), squeeze=False)
        for ax in axes.flat:
            ax.set_axis_off()

        x72 = ellipses[cell_idx, 0]
        y72 = ellipses[cell_idx, 1]
        x108 = x72 * factor
        y108 = y72 * factor

        for i in range(nb_images):
            row = i // (nb_cols // 3)
            col_group = (i % (nb_cols // 3)) * 3

            # === 原图 crop ===
            x0, x1, y0, y1 = get_crop_bounds(x108, y108, padding=10, shape_limit=108)
            raw_crop = images[image_indices[i], 0, y0:y1, x0:x1]
            extent_raw = [x0, x1, y1, y0]

            ax1 = axes[row, col_group]
            ax1.imshow(raw_crop, interpolation='bicubic', cmap='gray', extent=extent_raw, vmin=np.min(images), vmax=np.max(images))
            ax1.plot(x108, y108, 'y', lw=0.8)
            ax1.set_title(f"Stim {i}", fontsize=6)

            # === 实验 LSTA crop ===
            x0, x1, y0, y1 = get_crop_bounds(x72, y72, padding=5, shape_limit=72)
            data = lsta_data[cell_idx, i, y0:y1, x0:x1]
            data = data ** 2 * np.sign(data)
            vmax = np.max([np.amax(data), -np.amin(data)]) * 0.75
            extent_lsta = [x0, x1, y1, y0]

            ax2 = axes[row, col_group + 1]
            ax2.imshow(data, interpolation='bicubic', cmap='RdBu_r', vmin=-vmax, vmax=vmax, extent=extent_lsta)
            ax2.plot(x72, y72, 'y', lw=0.8)
            ax2.set_title("Exp", fontsize=6)

            # === 模型 LSTA crop ===
            x0, x1, y0, y1 = get_crop_bounds(x108, y108, padding=10, shape_limit=108)
            data = lsta_model[cell_idx, i, 0, y0:y1, x0:x1]
            data = data ** 2 * np.sign(data)
            vmax = np.max([np.amax(data), -np.amin(data)]) * 0.75
            extent_model = [x0, x1, y1, y0]

            ax3 = axes[row, col_group + 2]
            ax3.imshow(data, interpolation='bicubic', cmap='RdBu_r', vmin=-vmax, vmax=vmax, extent=extent_model)
            ax3.plot(x108, y108, 'y', lw=0.8)
            ax3.set_title("Model", fontsize=6)

        fig.suptitle(f"LSTA Comparison for Cell {cell_idx}", fontsize=12)
        fig.tight_layout()
        figs.append(fig)

    return figs

def plot_convolutional_kernels(model, is_2d=False, channels=None, time_frames=None):
    """
    Plot convolutional kernels in a per-(channel, time_frame) basis.

    Returns:
        A list of fig.
    """
    core = model.core
    if hasattr(core, 'conv_layers'):
        weight = dict(core.named_parameters())['conv_layers.0.weight'].detach().cpu()
    elif hasattr(core, 'conv'):
        weight = dict(core.named_parameters())['conv.weight'].detach().cpu()
    elif hasattr(core, 'filter'):
        weight = dict(core.named_parameters())['filter.weight'].detach().cpu()
    else:
        raise AttributeError("Model does not have 'conv_layers' or 'conv'.")

    out_channels = weight.shape[0]
    in_channels = weight.shape[1]

    if is_2d:
        # Shape: [out_channels, in_channels, H, W]
        T = 1
        time_frames = [None]
        if channels is None:
            channels = list(range(in_channels))
    else:
        # Shape: [out_channels, in_channels, T, H, W]
        T = weight.shape[2]
        if channels is None:
            channels = list(range(in_channels))
        if time_frames is None:
            time_frames = [T // 2]

    figs = []

    for c in channels:
        for t in time_frames:
            if is_2d:
                # [out_channels, H, W]
                kernel_slices = weight[:, c]
                tag = f"Channel {c}"
            else:
                # [out_channels, H, W] from 3D kernel at specific t
                kernel_slices = weight[:, c, t]
                tag = f"Channel {c}, Time {t}"

            # if max_kernels:
                # kernel_slices = kernel_slices[:max_kernels]

            n_kernels = kernel_slices.shape[0]
            grid_cols = min(8, ceil(sqrt(n_kernels)))
            grid_rows = ceil(n_kernels / grid_cols)

            fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(1.5 * grid_cols, 1.5 * grid_rows))
            axes = axes.flatten() if n_kernels > 1 else [axes]

            vlim = torch.max(torch.abs(kernel_slices)).item()

            for i, ax in enumerate(axes):
                if i < n_kernels:
                    ax.imshow(kernel_slices[i], cmap="RdBu_r", vmin=-vlim, vmax=+vlim)
                    ax.set_xticks([])
                    ax.set_yticks([])
                else:
                    ax.axis('off')

            fig.suptitle(f"Convolutional Kernels ({tag})", fontsize=12)
            fig.tight_layout()
            figs.append(fig)

    return figs

def plot_spatial_masks(model, cell_indices=None):
    if hasattr(model, 'mask_size'):
        H, W = model.mask_size
    else:
        raise AttributeError("Model does not have attribute 'mask_size'.")
    
    readout = model.readout
    if hasattr(readout, 'mask_weights'):  # nn.Parameter & matmul
        weights = dict(readout.named_parameters())['mask_weights'].detach().cpu()
        weights = weights.T.view(-1, H, W)  # [num_neurons, H, W]
    elif hasattr(readout, 'mask'):  # nn.Linear
        weights = dict(readout.named_parameters())['mask.weight'].detach().cpu()
        weights = weights.view(-1, H, W)
    else:
        raise AttributeError("Model does not have 'mask_weights' or 'readout_weights'.")

    total_neurons = weights.shape[0]
    if not cell_indices:
        cell_indices = list(range(total_neurons))
    else:
        cell_indices = [i for i in cell_indices if 0 <= i < total_neurons]

    selected_weights = weights[cell_indices]
    n_cells = selected_weights.shape[0]

    grid_cols = min(8, ceil(sqrt(n_cells)))
    grid_rows = ceil(n_cells / grid_cols)

    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(1.5 * grid_cols, 1.5 * grid_rows))

    if n_cells == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    vmin = torch.min(selected_weights).item()
    vmax = torch.max(selected_weights).item()

    for i, ax in enumerate(axes):
        if i < n_cells:
            ax.imshow(selected_weights[i], cmap='gray', vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.axis('off')

    fig.suptitle("Spatial Masks", fontsize=14)
    fig.tight_layout()
    return fig

def plot_feature_weights(model):
    weights = dict(model.named_parameters())['readout.readout_weights'].detach().cpu()
    weights = weights.numpy()

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(weights, aspect='auto', cmap='gray')
    ax.set_title("Feature Weights (channels × neurons)")
    ax.set_xlabel("Neuron")
    ax.set_ylabel("Feature Channel")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig

def fig_to_buffer(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return Image.open(buf)