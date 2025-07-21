import numpy as np
from typing import Tuple, Dict, Optional

def convert_format(data: dict, neuron_indices: Optional[list] = None) -> dict:
    """
    Transform the TensorFlow-style dataset into an Open-retina-style dataset,
    with optional neuron selection.

    Input:
        data: a dictionary with the following keys:
            - images_train, images_val, images_test: [B, H, W, C]
            - responses_train, responses_val: [B, N]
            - responses_test: [T, N] or [trial, T, N]
        neuron_indices (list[int], optional): selected neuron indices to keep

    Output:
        dict: converted dictionary
            - images_{split}: [C, T, H, W]
            - responses_{split}: [N, T] (selected neurons only)
            - responses_test_by_trial: [T, N, trial] (selected neurons only, if applicable)
    """
    converted = {}

    # 图像部分：BHWC → CBHW（B=T）
    for split in ["train", "val", "test"]:
        key = f"images_{split}"
        if key in data:
            img = data[key]  # [B, H, W, C]
            img = np.transpose(img, (3, 0, 1, 2))  # → [C, B, H, W]
            img = img.astype(np.float32)
            converted[key] = img

    # 响应部分：转置为 [N, T]，并裁剪神经元
    for split in ["train", "val"]:
        key = f"responses_{split}"
        if key in data:
            resp = data[key]  # [B, N]
            if neuron_indices is not None:
                resp = resp[:, neuron_indices]  # 保留指定神经元
            resp = resp.T.astype(np.float32)  # → [N, B]
            converted[key] = resp

    # 测试响应：支持 [T, N] 或 [trial, T, N] → [N, T] 或 [T, N, trial]
    if "responses_test" in data:
        r_test = data["responses_test"]
        if r_test.ndim == 2:
            if neuron_indices is not None:
                r_test = r_test[:, neuron_indices]  # [T, N'] ← [T, N]
            converted["responses_test"] = r_test.T.astype(np.float32)  # → [N, T]
        elif r_test.ndim == 3:
            if neuron_indices is not None:
                r_test = r_test[:, :, neuron_indices]  # [trial, T, N'] ← [trial, T, N]
            r_test = np.transpose(r_test, (1, 2, 0))  # [T, N, trial]
            converted["responses_test_by_trial"] = r_test.astype(np.float32)
        else:
            raise ValueError("Unsupported test response shape.")

    return converted

def normalize_data(
    data: dict,
    mode: str = "None"
) -> Tuple[dict, dict, dict]:
    """
    Normalization:
    - Image: [C, B, H, W], normalize each channel (C) independently
    - Response: [N, B], normalize all responses together

    Input:
        data: dict, with images_train, images_val, images_test,
                     responses_train, responses_val, responses_test, etc.

    Output:
        - normalized_data: normalized dictionary with the same structure
        - mean_dict: mean values before normalization for each field
        - std_dict: standard deviations before normalization for each field
    """
    normalized = {}
    mean_dict: Dict[str, np.ndarray] = {}
    std_dict: Dict[str, np.ndarray] = {}

    for key, arr in data.items():
        arr = arr.astype(np.float32)

        if mode == "None":
            normalized[key] = arr

        elif mode == "Normalize by total":
            # 图像：按通道归一化
            if key.startswith("images_") and arr.ndim == 4:
                C, B, H, W = arr.shape
                flat = arr.reshape(C, -1)  # [C, B*H*W]
                means = flat.mean(axis=1)  # [C]
                stds = flat.std(axis=1)
                stds = np.where(stds > 1e-6, stds, 1.0)
                normed = (arr - means[:, None, None, None]) / stds[:, None, None, None]
                normalized[key] = normed
                mean_dict[key] = means
                std_dict[key] = stds

            # 响应：按神经元归一化
            elif key.startswith("responses_") and arr.ndim == 2:
                N, B = arr.shape
                means = arr.mean(axis=1)  # [N]
                stds = arr.std(axis=1)
                stds = np.where(stds > 1e-6, stds, 1.0)
                normed = (arr - means[:, None]) / stds[:, None]
                normalized[key] = normed
                mean_dict[key] = means
                std_dict[key] = stds

            # 多 trial 测试响应
            elif key == "responses_test_by_trial":
                T, N, R = arr.shape
                flat = arr.transpose(1, 0, 2).reshape(N, -1)  # [N, T*R]
                means = flat.mean(axis=1)
                stds = flat.std(axis=1)
                stds = np.where(stds > 1e-6, stds, 1.0)
                normed = (arr - means[None, :, None]) / stds[None, :, None]
                normalized[key] = normed
                # 平均后生成 responses_test
                mean_resp = normed.mean(axis=2)  # [T,N]
                normalized["responses_test"] = mean_resp.T.astype(np.float32)
                mean_dict[key] = means
                std_dict[key] = stds

            else:
                normalized[key] = arr

        elif mode == "Normalize by Frame":
            # 图像：每帧单独归一化
            if key.startswith("images_") and arr.ndim == 4:
                C, B, H, W = arr.shape
                reshaped = arr.transpose(1, 0, 2, 3).reshape(B, -1)  # [B, C*H*W]
                means = reshaped.mean(axis=1)  # [B]
                stds = reshaped.std(axis=1)
                stds = np.where(stds > 1e-6, stds, 1.0)
                normed = (arr - means[None, :, None, None]) / stds[None, :, None, None]
                normalized[key] = normed
                mean_dict[key] = means
                std_dict[key] = stds

            # 响应：每时刻单独归一化
            elif key.startswith("responses_") and arr.ndim == 2:
                N, B = arr.shape
                means = arr.mean(axis=0)  # [B]
                stds = arr.std(axis=0)
                stds = np.where(stds > 1e-6, stds, 1.0)
                normed = (arr - means[None, :]) / stds[None, :]
                normalized[key] = normed
                mean_dict[key] = means
                std_dict[key] = stds

            # 多 trial 测试响应
            elif key == "responses_test_by_trial":
                T, N, R = arr.shape
                flat = arr.transpose(2, 0, 1).reshape(R * T, -1)  # [R*T, N]
                means = flat.mean(axis=1)  # [R*T]
                stds = flat.std(axis=1)
                stds = np.where(stds > 1e-6, stds, 1.0)
                # 回写归一化
                normed = np.empty_like(arr, dtype=np.float32)
                for t in range(T):
                    for r in range(R):
                        idx = r * T + t
                        normed[t, :, r] = (arr[t, :, r] - means[idx]) / stds[idx]
                normalized[key] = normed
                mean_resp = normed.mean(axis=2)  # [T,N]
                normalized["responses_test"] = mean_resp.T.astype(np.float32)
                mean_dict[key] = means
                std_dict[key] = stds

            else:
                normalized[key] = arr

        else:
            raise ValueError(f"Invalid normalization mode: {mode}")

    return normalized, mean_dict, std_dict

def prepare_data_and_metadata(normalized_data: dict, train_chunk_size: int = 1, batch_size: int = 32, seed: int = 42, clip_length: int = 1):
    # Original Data
    images_train = normalized_data["images_train"]  # [1, 2910, 108, 108]
    responses_train = normalized_data["responses_train"]  # [41, 2910]

    images_val = normalized_data["images_val"]  # [1, 250, 108, 108]
    responses_val = normalized_data["responses_val"]  # [41, 250]

    images_test = normalized_data["images_test"]  # [1, 30, 108, 108]
    responses_test = normalized_data["responses_test"]  # [41, 30]

    # Merge training and validation sets
    images_train_combined = np.concatenate([images_train, images_val], axis=1)  # [1, 3160, 108, 108]
    responses_train_combined = np.concatenate([responses_train, responses_val], axis=1)  # [41, 3160]

    # Construct val_clip_indices
    train_size = images_train.shape[1]  # Train frame number = 2910
    val_size = images_val.shape[1]  # Validation frame number = 250
    val_clip_indices = list(range(train_size // clip_length, (train_size + val_size) // clip_length))  # [2910, ..., 3159]

    # Dimensions
    c, _, h, w = images_train.shape

    # Metadata Info
    metadata = {
        "train_size": train_size,
        "train_chunk_size": train_chunk_size,
        "val_size": val_size,
        "test_size": images_test.shape[1],
        "image_width": w,
        "image_height": h,
        "channel": c,
        "batch_size": batch_size,
        "seed": seed,
        "clip_length": clip_length,
        "num_val_clips": len(val_clip_indices),
        "val_clip_indices": val_clip_indices,
        "flattened": False,
    }

    # Return the merged data and metadata
    merged_data = {
        "images_train": images_train_combined,
        "responses_train": responses_train_combined,
        "images_test": images_test,
        "responses_test": responses_test,
    }

    return merged_data, metadata