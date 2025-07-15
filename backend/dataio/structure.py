import numpy as np
from torch.utils.data import Dataset, DataLoader
from openretina.data_io.base_dataloader import DataPoint
from openretina.data_io.base import MoviesTrainTestSplit, ResponsesTrainTestSplit

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