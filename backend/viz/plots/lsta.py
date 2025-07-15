import numpy as np
import torch
import matplotlib.pyplot as plt
from backend.viz.utils import fig_to_buffer

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
            data = lsta_model[cell_idx, image_indices[i], 0, y0:y1, x0:x1]
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

    images = [fig_to_buffer(fig) for fig in figs]

    return images
