from torch.utils.data import DataLoader
from .datasets_utils import ImageFolderDataset


def create_dataloader(opt, phase, idx):
    # set params
    data_opt = opt["dataset"].get(idx)

    if phase == "test":
        loader = DataLoader(
            dataset=ImageFolderDataset(data_opt),
            batch_size=1,
            shuffle=False,
            num_workers=data_opt["num_worker_per_gpu"],
            pin_memory=data_opt["pin_memory"],
        )
    else:
        raise ValueError(f"Unrecognized phase: {phase}")

    return loader
