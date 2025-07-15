import numpy as np
import matplotlib.pyplot as plt
from backend.viz.utils import fig_to_buffer

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

    return fig_to_buffer(fig)

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
    
    images = [fig_to_buffer(fig) for fig in figs]
    
    return images