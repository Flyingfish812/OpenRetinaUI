import torch
import numpy as np

# Backup functions
def load_tf_weights_into_model(model, core_path, mask_path, readout_path, frozen, indices=None):
    core_tf = np.load(core_path)
    core_torch = core_tf.transpose(2, 0, 1)[..., np.newaxis]  # → (4, 31, 31, 1)
    core_torch = np.transpose(core_torch, (0, 3, 1, 2))       # → (4, 1, 31, 31)
    core_tensor = torch.tensor(core_torch, dtype=torch.float32)
    print("core_tensor.shape:", core_tensor.shape)

    mask = torch.tensor(np.load(mask_path), dtype=torch.float32)
    if indices is not None:
        mask = mask[indices]
        num_neurons = len(indices)
    else:
        num_neurons = 41
    mask_flat = mask.reshape(num_neurons, -1).T  # shape: (6084, 41)
    mask_tensor = torch.tensor(mask_flat, dtype=torch.float32)
    print("mask.shape:", mask.shape)

    readout = torch.tensor(np.load(readout_path), dtype=torch.float32)
    if indices is not None:
        readout = readout[indices]
    readout_tensor = readout.T  # shape: (4, 41)
    print("readout.shape:", readout.shape)

    with torch.no_grad():
        # 注入 core 权重并冻结
        model.core.conv_layers[0].weight.copy_(core_tensor)
        if frozen:
            model.core.conv_layers[0].weight.requires_grad = False
        print("Core weight insert complete and frozen")

        # 注入 mask 权重并冻结
        model.readout.mask_weights.copy_(mask_tensor)
        if frozen:
            model.readout.mask_weights.requires_grad = False
        print("Mask weight insert complete and frozen")

        # 注入 readout 权重并冻结
        model.readout.readout_weights.copy_(readout_tensor)
        if frozen:
            model.readout.readout_weights.requires_grad = False
        print("Readout weight insert complete and frozen")

def backup(model, frozen = True, indices=None):
    load_tf_weights_into_model(model, 
                               "plots_ans/convolutional_kernels.npy", 
                               "plots_ans/spatial_masks.npy", 
                               "plots_ans/feature_weights.npy", 
                               frozen, 
                               indices)
    print("Backup complete")