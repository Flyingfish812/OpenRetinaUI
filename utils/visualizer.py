import torch
import numpy as np
import matplotlib.pyplot as plt

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

def evaluate_model_final(model, dataset, device='cuda', n_bootstrap=100, 
                         save_plots=True, plot_prefix="final_"):
    """
    Evaluate model using the correct interpretation (dim1=trials)
    
    Args:
        model: Trained neural network model
        dataset: Data module containing test dataset
        device: Device to run model on ('cuda' or 'cpu')
        n_bootstrap: Number of bootstrap samples for reliability estimation
        save_plots: Whether to save plots to disk
        plot_prefix: Prefix for saved plot filenames
        
    Returns:
        Dictionary with evaluation results
    """
    model = model.to(device)
    model.eval()
    
    # Get test data
    images_test = dataset["images_test"]
    responses_test = dataset["responses_test"]
    
    # Print data shapes
    print(f"Test images shape: {images_test.shape}")
    print(f"Test responses shape: {responses_test.shape}")
    
    # Interpret dimensions correctly
    n_trials, n_stimuli, n_neurons = responses_test.shape
    print(f"Correct interpretation: {n_trials} trials, {n_stimuli} stimuli, {n_neurons} neurons")
    
    # Average across trials to get mean response for each stimulus
    mean_responses = np.mean(responses_test, axis=0)  # Shape: (n_stimuli, n_neurons)
    print(f"Mean responses shape: {mean_responses.shape}")
    
    # Reordered data for reliability calculation: (stimuli, neurons, trials)
    reordered_data = np.transpose(responses_test, (1, 2, 0))
    
    # Calculate bootstrap reliability
    print(f"Calculating bootstrap reliability with {n_bootstrap} samples...")
    reliability, reliability_ci = bootstrap_reliability(reordered_data, n_bootstrap=n_bootstrap, axis=2)
    print(f"Reliability: min={np.nanmin(reliability):.4f}, max={np.nanmax(reliability):.4f}, mean={np.nanmean(reliability):.4f}")
    
    # Get model predictions for each stimulus
    print("Getting model predictions...")
    predictions = []
    with torch.no_grad():
        # Create a single batch for faster prediction if it fits in memory
        if len(images_test) <= 100:  # Arbitrary limit - adjust based on your GPU memory
            batch_images = torch.from_numpy(images_test).permute(0, 3, 1, 2).to(device)
            batch_preds = model(batch_images).cpu().numpy()
            predictions = batch_preds
        else:
            # Otherwise predict stimulus by stimulus
            for i in range(len(images_test)):
                image = torch.from_numpy(images_test[i:i+1]).permute(0, 3, 1, 2).to(device)
                pred = model(image).cpu().numpy()
                predictions.append(pred[0])  # Remove batch dimension
            predictions = np.array(predictions)
    
    print(f"Model predictions shape: {predictions.shape}")
    
    # Calculate evaluation metrics
    # 1. Mean squared error
    mse = np.mean((predictions - mean_responses)**2)
    rmse = np.sqrt(mse)
    
    # 2. Correlation for each neuron
    correlations = []
    for n in range(n_neurons):
        corr = np.corrcoef(predictions[:, n], mean_responses[:, n])[0, 1]
        correlations.append(corr)
    
    # 3. Fraction of ceiling (model performance / reliability)
    fraction_of_ceiling = []
    for n in range(n_neurons):
        if not np.isnan(reliability[n]) and reliability[n] > 0:
            foc = correlations[n] / reliability[n]
        else:
            foc = np.nan
        fraction_of_ceiling.append(foc)
    
    # 4. Aggregated metrics
    mean_correlation = np.nanmean(correlations)
    median_correlation = np.nanmedian(correlations)
    mean_foc = np.nanmean(fraction_of_ceiling)
    median_foc = np.nanmedian(fraction_of_ceiling)
    
    # Print results
    print("\nModel Evaluation Results:")
    print(f"  MSE: {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  Mean Correlation: {mean_correlation:.4f}")
    print(f"  Median Correlation: {median_correlation:.4f}")
    print(f"  Mean Fraction of Ceiling: {mean_foc:.4f}")
    print(f"  Median Fraction of Ceiling: {median_foc:.4f}")
    
    # Create visualizations
    if save_plots:
        # 1. Correlation and reliability distributions
        plt.figure(figsize=(15, 12))
        
        # Correlation distribution
        plt.subplot(2, 2, 1)
        plt.hist(correlations, bins=20, alpha=0.7)
        plt.axvline(mean_correlation, color='r', linestyle='--', 
                   label=f'Mean: {mean_correlation:.4f}')
        plt.axvline(median_correlation, color='g', linestyle='--', 
                   label=f'Median: {median_correlation:.4f}')
        plt.title('Model Correlation Distribution')
        plt.xlabel('Correlation')
        plt.ylabel('Count')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Reliability distribution
        plt.subplot(2, 2, 2)
        plt.hist(reliability, bins=20, alpha=0.7)
        plt.axvline(np.nanmean(reliability), color='r', linestyle='--', 
                   label=f'Mean: {np.nanmean(reliability):.4f}')
        plt.axvline(np.nanmedian(reliability), color='g', linestyle='--', 
                   label=f'Median: {np.nanmedian(reliability):.4f}')
        plt.title('Neuron Reliability Distribution')
        plt.xlabel('Reliability')
        plt.ylabel('Count')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Correlation vs Reliability scatter
        plt.subplot(2, 2, 3)
        plt.scatter(reliability, correlations, alpha=0.7)
        
        # Add diagonal line (ceiling)
        max_val = max(np.nanmax(reliability), np.nanmax(correlations))
        plt.plot([0, max_val], [0, max_val], 'k--', label='Ceiling')
        
        # Add trend line - FIXED HERE
        mask = ~np.isnan(np.array(reliability)) & ~np.isnan(np.array(correlations))
        if np.sum(mask) > 1:
            rel_filtered = np.array(reliability)[mask]
            corr_filtered = np.array(correlations)[mask]
            z = np.polyfit(rel_filtered, corr_filtered, 1)
            p = np.poly1d(z)
            x_range = np.linspace(np.min(rel_filtered), np.max(rel_filtered), 100)
            plt.plot(x_range, p(x_range), 'r--', 
                     label=f'Trend: y = {z[0]:.2f}x + {z[1]:.2f}')
        
        plt.title('Model Correlation vs. Neuron Reliability')
        plt.xlabel('Reliability')
        plt.ylabel('Model Correlation')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Fraction of ceiling distribution
        plt.subplot(2, 2, 4)
        plt.hist([f for f in fraction_of_ceiling if not np.isnan(f)], bins=20, alpha=0.7)
        plt.axvline(mean_foc, color='r', linestyle='--', 
                   label=f'Mean: {mean_foc:.4f}')
        plt.axvline(median_foc, color='g', linestyle='--', 
                   label=f'Median: {median_foc:.4f}')
        plt.axvline(1.0, color='k', linestyle='--', 
                   label='Reliability Ceiling')
        plt.title('Fraction of Ceiling Distribution')
        plt.xlabel('Model Correlation / Reliability')
        plt.ylabel('Count')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{plot_prefix}evaluation_summary.png')
        plt.show()
        
        # 2. Example neuron predictions vs actual
        # Sort neurons by correlation
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
            ax.scatter(mean_responses[:, n_idx], predictions[:, n_idx], alpha=0.7)
            
            # Add identity line
            min_val = min(np.min(mean_responses[:, n_idx]), np.min(predictions[:, n_idx]))
            max_val = max(np.max(mean_responses[:, n_idx]), np.max(predictions[:, n_idx]))
            ax.plot([min_val, max_val], [min_val, max_val], 'r--')
            
            # Add correlation and reliability info
            rel_str = f'{reliability[n_idx]:.4f}' if not np.isnan(reliability[n_idx]) else 'N/A'
            foc_str = f'{fraction_of_ceiling[n_idx]:.4f}' if not np.isnan(fraction_of_ceiling[n_idx]) else 'N/A'
            percentile = np.round(100 * i / (len(neurons_to_plot)-1))
            
            ax.set_title(f'Neuron #{n_idx} ({percentile}th %ile) - Corr: {correlations[n_idx]:.4f}, Rel: {rel_str}, FoC: {foc_str}')
            ax.set_xlabel('Actual Response')
            ax.set_ylabel('Predicted Response')
            ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{plot_prefix}prediction_examples.png')
        plt.show()
        
        # 3. Correlation vs reliability with neuron indices
        plt.figure(figsize=(12, 10))
        plt.scatter(reliability, correlations, alpha=0.5)
        
        # Add diagonal line
        max_val = max(np.nanmax(reliability), np.nanmax(correlations))
        plt.plot([0, max_val], [0, max_val], 'k--', label='Ceiling')
        
        # Add index labels for best and worst neurons
        top_n = 5  # Number of neurons to label
        
        # Best by correlation
        top_by_corr = np.argsort(correlations)[-top_n:]
        for idx in top_by_corr:
            plt.text(reliability[idx], correlations[idx], str(idx), 
                    fontsize=12, color='green')
        
        # Worst by correlation
        bottom_by_corr = np.argsort(correlations)[:top_n]
        for idx in bottom_by_corr:
            plt.text(reliability[idx], correlations[idx], str(idx), 
                    fontsize=12, color='red')
        
        plt.title('Model Correlation vs. Reliability with Neuron Indices')
        plt.xlabel('Reliability')
        plt.ylabel('Model Correlation')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{plot_prefix}corr_vs_reliability_labeled.png')
        plt.show()
        
        # 4. Grid of prediction plots for all neurons
        # Create a grid of scatter plots for all neurons
        n_per_plot = 16  # Number of neurons per plot
        
        for start_idx in range(0, n_neurons, n_per_plot):
            end_idx = min(start_idx + n_per_plot, n_neurons)
            n_to_plot = end_idx - start_idx
            
            # Determine grid dimensions
            grid_size = int(np.ceil(np.sqrt(n_to_plot)))
            
            fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
            axes = axes.flatten()
            
            for i, n_idx in enumerate(range(start_idx, end_idx)):
                ax = axes[i]
                
                # Plot scatter of predictions vs actual
                ax.scatter(mean_responses[:, n_idx], predictions[:, n_idx], alpha=0.5, s=15)
                
                # Add identity line
                lims = [
                    min(np.min(mean_responses[:, n_idx]), np.min(predictions[:, n_idx])),
                    max(np.max(mean_responses[:, n_idx]), np.max(predictions[:, n_idx]))
                ]
                ax.plot(lims, lims, 'r--', alpha=0.5)
                
                # Add correlation as title
                ax.set_title(f'Neuron {n_idx}: r={correlations[n_idx]:.2f}')
                
                # Only add axis labels on the edge plots
                if i % grid_size == 0:  # Left edge
                    ax.set_ylabel('Predicted')
                if i >= (grid_size**2 - grid_size):  # Bottom edge
                    ax.set_xlabel('Actual')
                    
                # Remove most ticks for cleaner appearance
                ax.tick_params(axis='both', which='major', labelsize=8)
            
            # Turn off any unused subplots
            for j in range(n_to_plot, grid_size**2):
                axes[j].axis('off')
                
            plt.tight_layout()
            plt.savefig(f'{plot_prefix}neuron_grid_{start_idx}_{end_idx-1}.png')
            plt.show()
    
    # Return results dictionary
    return {
        'mse': mse,
        'rmse': rmse,
        'correlations': correlations,
        'reliability': reliability,
        'reliability_ci': reliability_ci,
        'fraction_of_ceiling': fraction_of_ceiling,
        'mean_correlation': mean_correlation,
        'median_correlation': median_correlation,
        'mean_fraction_of_ceiling': mean_foc,
        'median_fraction_of_ceiling': median_foc,
        'predictions': predictions,
        'targets': mean_responses
    }