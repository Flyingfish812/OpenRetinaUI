import pickle
import torch
import numpy as np
from typing import Tuple, Dict, Any, Optional
from scipy.signal import convolve2d
from torch.utils.data import Dataset, DataLoader
from openretina.data_io.base_dataloader import DataPoint
from openretina.data_io.base import MoviesTrainTestSplit, ResponsesTrainTestSplit

class Unpickler(pickle.Unpickler):
    """
    Unpickler that can handle torch.utils.data.Dataset objects.
    """
    def find_class(self, module, name):
        if module == 'data' and name == 'Dataset':
            return torch.utils.data.Dataset
        elif name == 'Dataset':
            return torch.utils.data.Dataset
        elif name == 'DataPoint':
            return DataPoint
        return super().find_class(module, name)

def load_dataloader(path: str) -> dict[str, dict[str, DataLoader]]:
    """
    Load a nested dataloader structure from a pickled file,
    including handling of custom DataPoint objects and Dataset.
    """
    with open(path, 'rb') as f:
        return Unpickler(f).load()

def load_metadata(path: str) -> dict[str, Any]:
    """
    Load a nested dataloader structure from a pickled file,
    including handling of custom DataPoint objects and Dataset.
    """
    with open(path, 'rb') as f:
        return Unpickler(f).load()


class DatasetLoader:
    """
    A generic loader for structured numpy arrays from a .pkl file.
    The dataset is now packed by torch.utils.data.Dataset,
    and the loader can handle both Dataset objects and dictionaries.
    The .pkl dataset should be converted to dictionary format later.
    """

    def __init__(self):
        self.expected_keys = [
            "images_train", "responses_train",
            "images_val", "responses_val",
            "images_test", "responses_test"
        ]

    def _load_pickle(self, path: str) -> Any:
        """
        Compatible loading of .pkl files that may contain torch Dataset objects.
        """
        with open(path, 'rb') as f:
            return Unpickler(f).load()

    def _extract_from_object(self, obj: Any) -> Dict[str, np.ndarray]:
        """
        Extract all ndarray from a torch Dataset object.
        """
        data = {}
        for key in self.expected_keys:
            if hasattr(obj, key):
                val = getattr(obj, key)
                if isinstance(val, torch.Tensor):
                    val = val.cpu().numpy()
                if isinstance(val, np.ndarray):
                    data[key] = val
        return data

    def _extract_from_dict(self, d: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Extract all ndarray from a dictionary.
        """
        data = {}
        for key in self.expected_keys:
            if key in d:
                val = d[key]
                if isinstance(val, torch.Tensor):
                    val = val.cpu().numpy()
                if isinstance(val, np.ndarray):
                    data[key] = val
        return data

    def load_numpy_arrays(self, path: str) -> Dict[str, np.ndarray]:
        """
        External interface to load numpy arrays from a .pkl file.
        """
        obj = self._load_pickle(path)

        if isinstance(obj, dict):
            return self._extract_from_dict(obj)
        else:
            return self._extract_from_object(obj)
        
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

def normalize_data(data: dict) -> Tuple[dict, dict, dict]:
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

    for key, value in data.items():
        arr = value.astype(np.float32)

        # 图像数据处理（按通道归一化）
        if key.startswith("images_") and arr.ndim == 4:
            C, B, H, W = arr.shape
            # normed = np.empty_like(arr, dtype=np.float32)
            # means = np.zeros(C, dtype=np.float32)
            # stds = np.ones(C, dtype=np.float32)
            # for c in range(C):
            #     mean = arr[c].mean()
            #     std = arr[c].std()
            #     std = std if std > 1e-6 else 1.0
            #     normed[c] = (arr[c] - mean) / std
            #     means[c] = mean
            #     stds[c] = std
            # normalized[key] = normed
            # mean_dict[key] = means
            # std_dict[key] = stds

            # means = arr.mean()
            # std = arr.std(ddof=1)
            # normalized[key] = (arr - means) / std
            # mean_dict[key] = means
            # std_dict[key] = std

            normalized[key] = arr

        # 响应数据处理（整体归一化）
        elif key.startswith("responses_") and arr.ndim == 2:
            # Per-dimension std
            # std = arr.std(axis=1, keepdims=True, ddof=1)
            # mean_std = std.mean()
            # std[std < (mean_std / 100)] = 1.0

            # # Clamp to positive and normalize
            # arr = np.clip(arr, a_min=0, a_max=None)
            # normalized[key] = (arr / std).astype(np.float32)

            # mean_dict[key] = np.zeros_like(std, dtype=np.float32)  # 不减 mean
            # std_dict[key] = std.astype(np.float32)

            normalized[key] = arr

        elif key == "responses_test_by_trial":
            # [T, N, trial]
            # arr = arr.astype(np.float32)
            # # T, N, R = arr.shape

            # # Step 1: Clamp to positive
            # arr = np.clip(arr, a_min=0, a_max=None)

            # # Step 2: Normalize across trials (each trial treated independently)
            # std = arr.std(axis=2, keepdims=True, ddof=1)  # shape: [T, N]
            # mean_std = std.mean()
            # std[std < (mean_std / 100)] = 1.0

            # arr_norm = arr / std  # shape [T, N, trial]

            # normalized[key] = arr_norm.astype(np.float32)

            # # Step 3: After normalization, compute mean across trials → [T, N]
            # mean_resp = arr_norm.mean(axis=2)  # → [T, N]
            # normalized["responses_test"] = mean_resp.T.astype(np.float32)  # [N, T]

            # mean_dict["responses_test"] = np.zeros_like(std.T, dtype=np.float32)
            # std_dict["responses_test"] = std.T.astype(np.float32)

            mean_resp = arr.mean(axis=2)  # [T, N]
            normalized[key] = arr
            normalized["responses_test"] = mean_resp.T.astype(np.float32)

        # 其他数据直接保留
        else:
            normalized[key] = arr

    return normalized, mean_dict, std_dict

def build_train_test_splits(normalized_data: dict) -> tuple[MoviesTrainTestSplit, ResponsesTrainTestSplit]:
    movie = MoviesTrainTestSplit(
        train=normalized_data["images_train"],  # [C, B, H, W]
        test=normalized_data["images_test"],    # [C, B_test, H, W]
        stim_id="klindt2017"
    )

    response = ResponsesTrainTestSplit(
        train=normalized_data["responses_train"],  # [N, B]
        test=normalized_data["responses_test"],    # [N, B_test]
        test_by_trial=normalized_data.get("responses_test_by_trial", None),  # 可选
        stim_id="klindt2017",
        session_kwargs={
            "roi_ids": np.arange(normalized_data["responses_train"].shape[0]),
            "roi_coords": np.zeros((normalized_data["responses_train"].shape[0], 2)),
            "group_assignment": None,
            "eye": "unknown",
            "scan_sequence_idx": np.arange(normalized_data["responses_train"].shape[1]),
        },
    )

    return movie, response

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

def print_dataloader_info(dataloaders: dict[str, dict[str, DataLoader]]) -> None:
    """
    打印每个DataLoader中包含的样本数量和每个样本的尺寸信息
    
    Args:
        dataloaders: 嵌套字典结构，外层键为分割类型(train/validation/test)，
                    内层键为session_id，值为DataLoader对象
    """
    for split_name, session_loaders in dataloaders.items():
        print(f"Split: {split_name}")
        print("-" * 40)
        
        for session_id, dataloader in session_loaders.items():
            dataset = dataloader.dataset
            sample_count = len(dataset)
            
            # 获取第一个样本来确定数据形状
            first_sample = next(iter(dataset))
            
            # 处理DataPoint namedtuple结构
            if isinstance(first_sample, tuple) and hasattr(first_sample, '_fields'):
                inputs_shape = first_sample.inputs.shape
                targets_shape = first_sample.targets.shape
                
                print(f"  Session: {session_id}")
                print(f"    Samples: {sample_count}")
                print(f"    Input shape: {inputs_shape}")
                print(f"    Target shape: {targets_shape}")
                print()
            else:
                print(f"  Session: {session_id} - Unexpected data format")
                print()
        print()

class SplitDataset(Dataset):
    """Flatten dataset: [C,T,H,W] → T * [C,H,W] and [N,T] → T * [N]"""
    def __init__(self, base_dataset):
        self.samples = []

        for i in range(len(base_dataset)):
            inputs, targets = base_dataset[i]
            T = inputs.shape[1]

            for t in range(T):
                x = inputs[:, t, :, :].clone()  # [C, H, W]
                y = targets[t].clone()  # [N]
                self.samples.append(DataPoint(inputs=x, targets=y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def strip_all_dataloaders(dataloaders_dict: dict[str, dict[str, DataLoader]]):
    wrapped = {}

    for split, phase_loaders in dataloaders_dict.items():
        wrapped[split] = {}

        for session, loader in phase_loaders.items():
            base_dataset = loader.dataset

            new_dataset = SplitDataset(base_dataset)
            
            new_loader = DataLoader(
                new_dataset,
                batch_size=loader.batch_size,
                shuffle=(split == 'train'),
                num_workers=loader.num_workers,
                pin_memory=True,
            )

            wrapped[split][session] = new_loader

    return wrapped