import numpy as np
import torch
from backend.utils import global_state

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
            preds = outputs.detach().cpu()
            targets = data_point.targets  # 2d: [B, num_neurons]; 3d: [1, T, num_neurons]
            print("Mean response:", outputs.mean().item())
            print("Std response:", outputs.std().item())
            print("Mean target:", targets.mean().item())
            print("Std target:", targets.std().item())

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