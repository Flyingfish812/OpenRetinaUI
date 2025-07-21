import os
import pickle
import numpy as np
from backend.dataio import DatasetLoader

# 1. Load the original dataset
loader = DatasetLoader()
raw_data = loader.load_numpy_arrays("data/raw/data2_41mixed_tr28.pkl")

# 2. Prepare a new dict to hold the transformed data
new_data = {}

# 3. Helper to upgrade channels from 1 to 2 according to your scheme
def upgrade_channels(images: np.ndarray) -> np.ndarray:
    """
    images: np.ndarray of shape (N, H, W, 1)
    returns: np.ndarray of shape (N, H, W, 2)
    - first N//3: fill channel 0 only
    - next N//3: fill channel 1 only
    - last N - 2*(N//3): fill both channels
    """
    N, H, W, _ = images.shape
    new_imgs = np.zeros((N, H, W, 2), dtype=images.dtype)
    third = N // 3

    # 1st third → channel 0
    new_imgs[:third, ..., 0] = images[:third, ..., 0]

    # 2nd third → channel 1
    start = third
    end = 2 * third
    new_imgs[start:end, ..., 1] = images[start:end, ..., 0]

    # last part → both channels
    new_imgs[end:, ..., 0] = images[end:, ..., 0]
    new_imgs[end:, ..., 1] = images[end:, ..., 0]

    return new_imgs

# 4. Process each split
for split in ("train", "val", "test"):
    img_key = f"images_{split}"
    resp_key = f"responses_{split}"

    images = raw_data[img_key]        # e.g. (2910,108,108,1)
    responses = raw_data[resp_key]    # e.g. (2910,41) or (30,30,41)

    # upgrade channels
    new_data[img_key] = upgrade_channels(images)
    # keep responses as-is
    new_data[resp_key] = responses

    print(f"{split.capitalize()} images: {images.shape} → {new_data[img_key].shape}")
    print(f"{split.capitalize()} responses: {responses.shape}")

# 5. Save to a new pickle
out_path = "data/raw/data2_41mixed_tr28_2channel.pkl"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "wb") as f:
    pickle.dump(new_data, f)

print(f"\nSaved upgraded dataset to: {out_path}")
