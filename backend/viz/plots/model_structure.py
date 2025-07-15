import torch
import matplotlib.pyplot as plt
from math import sqrt, ceil
from backend.viz.utils import fig_to_buffer

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
                    ax.imshow(kernel_slices[i], interpolation='bicubic', cmap="RdBu_r", vmin=-vlim, vmax=+vlim)
                    ax.set_xticks([])
                    ax.set_yticks([])
                else:
                    ax.axis('off')

            fig.suptitle(f"Convolutional Kernels ({tag})", fontsize=12)
            fig.tight_layout()
            figs.append(fig)

    images = [fig_to_buffer(fig) for fig in figs]

    return images

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
            ax.imshow(selected_weights[i], interpolation='bicubic', cmap='gray', vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.axis('off')

    fig.suptitle("Spatial Masks", fontsize=14)
    fig.tight_layout()

    return fig_to_buffer(fig)

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

    return fig_to_buffer(fig)