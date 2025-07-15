import pickle
import torch
import numpy as np
from openretina.data_io.base_dataloader import DataPoint
from torch.utils.data import DataLoader
from typing import Dict, Any

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